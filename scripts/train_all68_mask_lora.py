#!/usr/bin/env python
"""Fresh joint SFT/mask LoRA on all available successful screened cases."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, default_collate
from megatron.core import parallel_state
from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init
from train_lrscene5_mask_lora import MaskMetrics

SUBSET = REPO / 'training/tta_sft_all68_mask_65'
EXCLUDED = ['LRSCENE2-init004', 'KSCENE4-init023', 'LRSCENE5-init011']


def prepare_data():
    import h5py
    from eval_sft_68 import LANGS
    latest = json.loads((REPO / 'experiments/tta/mask_view/latest/manifest.json').read_text())
    cases = sorted([case for case in latest['cases']
                    if case['tag'] not in EXCLUDED and 'success=True' in Path(case['source_h5']).name],
                   key=lambda case: case['tag'])
    if len(cases) != 65 or len({case['tag'] for case in cases}) != 65:
        raise ValueError('Expected 65 unique successful trajectory/mask pairs')
    SUBSET.mkdir(parents=True, exist_ok=True)
    steps, expected = 0, set()
    for case in cases:
        source = Path(case['source_h5']).resolve(strict=True)
        info = json.loads(Path(case['case_json']).read_text())
        if not info['success'] or Path(info['h5']).resolve() != source:
            raise ValueError(f"Not a matching successful trajectory: {case['tag']}")
        with h5py.File(source, 'r') as h5:
            n = len(h5['actions'])
            if not bool(h5.attrs['success']) or h5['actions'].shape != (n, 7):
                raise ValueError(f'Invalid success/action metadata: {source}')
            if len(h5['primary_images_jpeg']) != n or len(h5['wrist_images_jpeg']) != n:
                raise ValueError(f'Image/action length mismatch: {source}')
            if not np.isfinite(h5['actions'][:]).all() or not np.isfinite(h5['proprio'][:]).all():
                raise ValueError(f'Non-finite trajectory: {source}')
        with np.load(case['mask_npz'], allow_pickle=False) as masks:
            bits = masks['agentview_bits']
            if bits.dtype != np.uint8 or bits.shape != (n, 256, 256) or case['frames'] != n:
                raise ValueError(f"Mask/action length mismatch: {case['tag']}")
            if not ((bits & 3) != 0).any():
                raise ValueError(f"All-empty mask: {case['tag']}")
        target = SUBSET / source.name
        if not target.exists():
            target.symlink_to(source)
        if target.resolve() != source:
            raise ValueError(f'Trajectory conflict: {target}')
        expected.add(source.name)
        steps += n
    if {p.name for p in SUBSET.glob('*.hdf5')} != expected or steps != 16818:
        raise ValueError('Unexpected full training subset')
    manifest = SUBSET / 'mask_manifest.json'
    manifest.write_text(json.dumps({'total':65, 'complete':65, 'steps':steps,
                                   'excluded_from_training':EXCLUDED, 'cases':cases}, indent=2))

    # Snapshot eval configuration; the deleted episode/actions/masks stay deleted.
    meta = json.loads((REPO / 'memory_system/pointcloud_action/results/tta_screening_meta.json').read_text())
    for case in latest['cases']:
        info = json.loads(Path(case['case_json']).read_text())
        row = meta[case['tag']]
        if row['census_abs_init'] != info['init'] or row['suite_task_name'] != info['pert']:
            raise ValueError(f"Eval/mask source state mismatch: {case['tag']}")
    if 'LRSCENE2-init004' not in meta:
        row = dict(meta['LRSCENE2-init005'])
        row.update(init='init004', census_abs_init=4)
        meta['LRSCENE2-init004'] = row
    if len(meta) != 68:
        raise ValueError(f'Expected the original 68 eval states, got {len(meta)}')
    for tag, row in meta.items():
        row.update(base_task=row['task'], pert_task=row['suite_task_name'], seed=7,
                   language=LANGS[tag.split('-')[0]], pert_name='robot_initial_states',
                   pert_category='Robot Initial States')
    (SUBSET / 'eval_meta_68.json').write_text(json.dumps(dict(sorted(meta.items())), indent=2))
    report = {'status':'PREPARED', 'training_trajectories':65, 'training_steps':steps,
              'eval_initial_states':68, 'excluded_from_training':EXCLUDED,
              'mask_manifest':str(manifest), 'eval_meta':str(SUBSET/'eval_meta_68.json')}
    (SUBSET / 'preflight.json').write_text(json.dumps(report, indent=2))
    return report


def fixed_batches(dataset):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    batches = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            random.seed(20261002); np.random.seed(20261002); torch.manual_seed(20261002)
            manifest = json.loads((SUBSET / 'mask_manifest.json').read_text())
            for task in sorted({c['tag'].split('-')[0] for c in manifest['cases']}):
                case = next(c for c in manifest['cases'] if c['tag'].startswith(task + '-'))
                tag, step = case['tag'], case['frames'] // 2
                ep = next(i for i, info in dataset.rollout_episode_metadata.items()
                          if f'--{tag}--' in Path(info['file_path']).name)
                index = next(i for i, pair in dataset._step_to_episode_map.items() if pair == (ep, step))
                batches.append((tag, step, default_collate([dataset[index]])))
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-name')
    parser.add_argument('--max-iter', type=int, default=2742)
    parser.add_argument('--probe-every', type=int, default=100)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare_data()), flush=True)
        return
    if not args.job_name or args.max_iter < 1 or args.probe_every < 1:
        parser.error('A new job name and positive iteration counts are required')
    run_dir = REPO / 'scripts/experiments/all68_mask_lora' / args.job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TTA_REPAIR_SFT_METADATA_DIR'] = str(REPO / 'training/tta_sft_metadata')
    os.environ['TTA_REPAIR_SFT_ROLLOUT_DIR'] = str(SUBSET)
    os.environ.pop('TTA_REPAIR_SFT_DROP_TAIL', None)
    os.environ.pop('DETERMINISTIC', None)
    os.environ.setdefault('IMAGINAIRE_OUTPUT_ROOT', '/data1/liu/exp/counterfactual/checkpoints')
    with distributed_init(): distributed.init()
    if distributed.is_rank0(): prepare_data()
    dist.barrier()
    spec = importlib.util.spec_from_file_location('archived_fixed_sft',
        REPO / 'tests/tta/archived_lorafix/train_tta_sft_fixed.py')
    fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)
    config = load_config('cosmos_policy/config/config.py',
                         ['--','experiment=tta_repair_sft'], enable_one_logger=False)
    fixed.mutate_config(config, SimpleNamespace(max_iter=args.max_iter, job_name=args.job_name))
    config.model.config.fsdp_shard_size = dist.get_world_size()
    config.model.config.use_mask_attention_loss = True
    config.model.config.mask_attention_lambda = 1.0
    config.model.config.mask_attention_layers = 4
    config.dataloader_train.dataset.agentview_mask_manifest = str(SUBSET / 'mask_manifest.json')
    config.validate(); config.freeze()
    if Path(config.job.path_local, 'checkpoints/latest_checkpoint.txt').exists():
        raise ValueError('Use a new job name: this run must start with a fresh LoRA')
    trainer = config.trainer.type(config)
    with model_init(): model = instantiate(config.model)
    n_lora = fixed.reinit_lora(model.net)
    dataset = instantiate(config.dataloader_train.dataset)
    if dataset.num_episodes != 65 or dataset.num_steps != 16818:
        raise ValueError(f'Unexpected dataset: {dataset.num_episodes} episodes, {dataset.num_steps} steps')
    probes = fixed_batches(dataset)
    monitor = MaskMetrics(run_dir, probes, args.probe_every)
    monitor.config, monitor.trainer = config, trainer
    trainer.callbacks._callbacks.append(monitor)
    sampler = DistributedSampler(dataset, num_replicas=parallel_state.get_data_parallel_world_size(),
                                 rank=parallel_state.get_data_parallel_rank(), shuffle=True, seed=0)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, drop_last=True,
                           num_workers=0, persistent_workers=False, pin_memory=True)
    if distributed.is_rank0():
        record = {'job_name':args.job_name, 'world_size':dist.get_world_size(), 'batch_size_per_gpu':1,
                  'global_batch_size':dist.get_world_size(), 'max_iter':args.max_iter, 'learning_rate':1e-5,
                  'lambda':1.0, 'attention_layers':4, 'lora_rank':8, 'lora_alpha':16,
                  'fresh_lora_modules':n_lora, 'trajectories':65, 'steps':16818,
                  'excluded_from_training':EXCLUDED, 'eval_initial_states':68,
                  'data_directory':str(SUBSET), 'mask_manifest':str(SUBSET / 'mask_manifest.json'),
                  'base_checkpoint':config.checkpoint.load_path,
                  'checkpoint_directory':str(Path(config.job.path_local) / 'checkpoints'),
                  'fixed_probe_every':args.probe_every,
                  'fixed_probe_samples':[{'tag':tag,'relative_step':step} for tag,step,_ in probes],
                  'started_at':time.strftime('%Y-%m-%d %H:%M:%S')}
        (run_dir / 'run.json').write_text(json.dumps(record, indent=2))
        print('[MASK_RUN] ' + json.dumps(record), flush=True)
    trainer.train(model, dataloader, None)
    if distributed.is_rank0():
        (run_dir / 'completed.json').write_text(json.dumps({'iteration':args.max_iter}))
    dist.destroy_process_group()


if __name__ == '__main__': main()
