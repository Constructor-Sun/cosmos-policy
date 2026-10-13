#!/usr/bin/env python3
"""Fresh SFT with LoRA or full policy parameters on successful task-P95 episodes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
POLICY = REPO.parents[1] / 'checkpoints/Cosmos-Policy-LIBERO-Predict2-2B'
BASE = POLICY / 'Cosmos-Policy-LIBERO-Predict2-2B.pt'
MANIFEST = REPO / 'experiments/tta/repair_libero_plus_435/success_task_p95/selected_manifest.json'
DATA = REPO / 'training/tta_sft_libero_plus_190_p95'


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare_data(manifest_path, data_root, expected_cases=190):
    import h5py
    import numpy as np
    import torch

    manifest = json.loads(manifest_path.read_text())
    cases = manifest['cases']
    if manifest['num_cases'] != expected_cases or len(cases) != expected_cases or len({row['task_id'] for row in cases}) != expected_cases:
        raise ValueError(f'Expected exactly {expected_cases} distinct P95-filtered successful trajectories')
    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    failure_manifest = json.loads((REPO / 'experiments/tta/repair_libero_plus_435/inputs/failed_435.json').read_text())
    baseline_cases = {row['task_id']: row for row in failure_manifest['cases']}
    baseline_root = Path(failure_manifest['source_run'])
    evaluation = json.loads((Path(failure_manifest['source_run']) / 'run_config.json').read_text())
    cache = {}
    for path in [POLICY / 'libero_t5_embeddings.pkl', Path(evaluation['extra_t5'])]:
        with path.open('rb') as handle:
            cache.update(pickle.load(handle))
    metadata, rollouts = data_root / 'metadata', data_root / 'rollouts'
    metadata.mkdir(parents=True, exist_ok=True)
    rollouts.mkdir(parents=True, exist_ok=True)
    previous = data_root / 'preflight.json'
    if previous.is_file() and json.loads(previous.read_text())['manifest_sha256'] != checksum:
        raise ValueError('Existing training data belongs to a different selection')
    records, commands, expected = [], set(), set()
    total, changed_commands = 0, 0
    trajectory_sources = {}
    for row in cases:
        if row['success'] is not True or row['action_steps'] > row['training_p95']:
            raise ValueError(f"Invalid selected trajectory: {row['case_id']}")
        source_kind = row.get('trajectory_source', 'memory_repair')
        if source_kind == 'memory_repair':
            source_row = baseline_cases[row['task_id']]
            if source_row['success'] is not False:
                raise ValueError(f"Memory repair must originate from a baseline failure: {row['case_id']}")
        elif source_kind == 'baseline_success':
            result = baseline_root / row['group'] / 'episodes' / row['case_id'] / 'result.json'
            source_row = json.loads(result.read_text())
            if (source_row['status'] != 'completed' or source_row['success'] is not True
                    or Path(row['episode_hdf5']).resolve() != Path(source_row['episode_hdf5']).resolve()
                    or Path(row['actions']).resolve() != Path(source_row['actions_path']).resolve()):
                raise ValueError(f"Invalid directly successful baseline source: {row['case_id']}")
        else:
            raise ValueError(f'Unknown trajectory source: {source_kind}')
        if (source_row['task_id'] != row['task_id'] or source_row['group'] != row['group']
                or source_row['task_name'] != row['task_name']
                or source_row.get('base_task', row['base_task']) != row['base_task']
                or not source_row['task_name'].startswith(row['base_task'] + '_')):
            raise ValueError(f"Base task mismatch: {row['case_id']}")
        trajectory_sources[source_kind] = trajectory_sources.get(source_kind, 0) + 1
        # Use the very instruction supplied to the policy during evaluation.
        # Environment descriptions may append visual-perturbation suffixes.
        command = source_row['instruction']
        if not command or command not in cache:
            raise ValueError(f'Missing exact T5 embedding: {command!r}')
        commands.add(command)
        source = Path(row['episode_hdf5']).resolve(strict=True)
        target = rollouts / f"episode_data--{row['case_id']}--suite=libero_10--success=True--p95.hdf5"
        with h5py.File(source, 'r') as original:
            steps = original['actions'].shape[0]
            if not bool(original.attrs['success']) or steps != row['action_steps']:
                raise ValueError(f'Incorrect success/length: {source}')
            if original['actions'].shape != (steps, 7) or original['proprio'].shape != (steps, 9):
                raise ValueError(f'Incorrect action/proprio shape: {source}')
            for key in ['primary_images_jpeg', 'wrist_images_jpeg']:
                if len(original[key]) != steps:
                    raise ValueError(f'Image/action mismatch: {source}')
            for key in ['actions', 'proprio']:
                if not np.isfinite(original[key][:]).all():
                    raise ValueError(f'Non-finite {key}: {source}')
            original_command = str(original.attrs['task_description'])
            changed_commands += int(original_command != command)
            if not target.exists():
                temporary = target.with_name(target.name + '.tmp')
                with h5py.File(temporary, 'w') as view:
                    for key in original:
                        view[key] = h5py.ExternalLink(str(source), '/' + key)
                    for key, value in original.attrs.items():
                        view.attrs[key] = value
                    view.attrs['task_description'] = command
                    view.attrs['original_task_description'] = original_command
                    view.attrs['source_hdf5'] = str(source)
                    view.attrs['source_manifest_sha256'] = checksum
                temporary.replace(target)
        with h5py.File(target, 'r') as view:
            if (view.attrs['source_hdf5'] != str(source)
                    or view.attrs['source_manifest_sha256'] != checksum
                    or view.attrs['task_description'] != command
                    or view['actions'].shape != (steps, 7)):
                raise ValueError(f'Existing training view conflicts: {target}')
        expected.add(target.name)
        total += steps
        records.append({**row, 'trajectory_source': source_kind,
                        'training_hdf5': str(target), 'instruction': command,
                        'original_task_description': original_command})
    if {path.name for path in rollouts.glob('*.hdf5')} != expected:
        raise ValueError('Unexpected extra/missing training episodes')
    # Preserve the checkpoint's action/proprio normalization for later evaluation.
    shutil.copyfile(POLICY / 'libero_dataset_statistics.json', metadata / 'dataset_statistics.json')
    shutil.copyfile(REPO / 'training/tta_sft_metadata/dataset_statistics_post_norm.json',
                    metadata / 'dataset_statistics_post_norm.json')
    embeddings = {command: cache[command].detach().cpu() if torch.is_tensor(cache[command])
                  else cache[command] for command in sorted(commands)}
    temporary = metadata / 't5_embeddings.pkl.tmp'
    with temporary.open('wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(metadata / 't5_embeddings.pkl')
    atomic_json(data_root / 'manifest.json', {**manifest, 'cases': records})
    report = dict(status='PREPARED', trajectories=len(records), steps=total,
                  unique_instructions=len(commands), aligned_instruction_descriptions=changed_commands,
                  manifest=str(manifest_path), manifest_sha256=checksum,
                  metadata_directory=str(metadata), rollout_directory=str(rollouts),
                  training_manifest=str(data_root / 'manifest.json'),
                  trajectory_sources=trajectory_sources,
                  base_checkpoint=str(BASE), full_trajectories=True, mask_supervision=False)
    atomic_json(data_root / 'preflight.json', report)
    return report


def fixed_batches(dataset, cases):
    import numpy as np
    import torch
    from torch.utils.data import default_collate

    python_state, numpy_state = random.getstate(), np.random.get_state()
    batches = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            random.seed(20261002)
            np.random.seed(20261002)
            torch.manual_seed(20261002)
            for task in sorted({row['base_task'] for row in cases}):
                case = next(row for row in cases if row['base_task'] == task)
                step = case['action_steps'] // 2
                matches = [ep for ep, meta in dataset.rollout_episode_metadata.items()
                           if f"--{case['case_id']}--" in Path(meta['file_path']).name]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous probe episode: {case['case_id']}")
                index = next(index for index, pair in dataset._step_to_episode_map.items()
                             if pair == (matches[0], step))
                batches.append((case['case_id'], step, default_collate([dataset[index]])))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-name')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--epochs', type=int, default=1)
    duration.add_argument('--max-iter', type=int, help='Override epoch duration for a short smoke check')
    parser.add_argument('--probe-every', type=int, default=100)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--expected-cases', type=int, default=190)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--full-finetune', action='store_true',
                        help='Disable LoRA and train all policy backbone parameters')
    parser.add_argument('--batch-size-per-gpu', type=int, default=1)
    parser.add_argument('--checkpoint-every-epoch', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.expected_cases < 1:
        parser.error('Expected case count must be positive')
    if not args.full_finetune and (args.lora_rank < 1 or args.lora_alpha < 1):
        parser.error('LoRA rank and alpha must be positive')
    if args.batch_size_per_gpu < 1:
        parser.error('Batch size per GPU must be positive')
    if args.prepare_only:
        print(json.dumps(prepare_data(args.manifest, args.data_root, args.expected_cases)), flush=True)
        return
    if not args.job_name or args.epochs < 1 or (args.max_iter is not None and args.max_iter < 1) or args.probe_every < 1:
        parser.error('A new job name and positive iteration counts are required')

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from cosmos_policy._src.imaginaire.config import load_config
    from cosmos_policy._src.imaginaire.lazy_config import instantiate
    from cosmos_policy._src.imaginaire.utils import distributed
    from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init
    from train_lrscene5_sft_only import SFTMetrics

    run = REPO / 'scripts/experiments/libero_plus_sft190' / args.job_name
    run.mkdir(parents=True, exist_ok=True)
    os.environ['TTA_REPAIR_SFT_METADATA_DIR'] = str(args.data_root / 'metadata')
    os.environ['TTA_REPAIR_SFT_ROLLOUT_DIR'] = str(args.data_root / 'rollouts')
    os.environ['TTA_REPAIR_SFT_BASE_CHECKPOINT'] = str(BASE)
    os.environ.pop('TTA_REPAIR_SFT_DROP_TAIL', None)
    os.environ.pop('DETERMINISTIC', None)
    os.environ.setdefault('IMAGINAIRE_OUTPUT_ROOT', str(REPO.parents[1] / 'checkpoints'))
    with distributed_init():
        distributed.init()
    if distributed.is_rank0():
        report = prepare_data(args.manifest, args.data_root, args.expected_cases)
        print('[SFT190_DATA] ' + json.dumps(report), flush=True)
    dist.barrier()
    report = json.loads((args.data_root / 'preflight.json').read_text())
    cases = json.loads(Path(report['training_manifest']).read_text())['cases']
    spec = importlib.util.spec_from_file_location('archived_fixed_sft',
        REPO / 'tests/tta/archived_lorafix/train_tta_sft_fixed.py')
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    config = load_config('cosmos_policy/config/config.py',
                         ['--', 'experiment=tta_repair_sft'], enable_one_logger=False)
    config.model.config.fsdp_shard_size = dist.get_world_size()
    config.model.config.use_mask_attention_loss = False
    config.dataloader_train.dataset.agentview_mask_manifest = ''
    config.trainer.grad_accum_iter = 1
    config.dataloader_train.batch_size = args.batch_size_per_gpu
    config.dataloader_train.drop_last = False
    dataset = instantiate(config.dataloader_train.dataset)
    if dataset.num_episodes != args.expected_cases or dataset.num_steps != report['steps'] or len(dataset) != dataset.num_steps:
        raise ValueError(f'Unexpected dataset: {dataset.num_episodes} episodes, {dataset.num_steps} steps, {len(dataset)} samples')
    if set(dataset.unique_commands) != {case['instruction'] for case in cases}:
        raise ValueError('Training language conditions differ from the evaluation instructions')
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank(), shuffle=True, seed=0)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_per_gpu, drop_last=False,
                           num_workers=0, persistent_workers=False, pin_memory=True)
    iterations_per_epoch = len(dataloader)
    max_iter = args.max_iter if args.max_iter is not None else args.epochs * iterations_per_epoch
    fixed.mutate_config(config, SimpleNamespace(max_iter=max_iter, job_name=args.job_name))
    config.model.config.use_lora = not args.full_finetune
    config.model.config.lora_rank = args.lora_rank
    config.model.config.lora_alpha = args.lora_alpha
    lora_targets = [name.strip() for name in config.model.config.lora_target_modules.split(',')]
    if args.checkpoint_every_epoch:
        config.checkpoint.save_iter = iterations_per_epoch
    config.validate()
    config.freeze()
    if Path(config.job.path_local, 'checkpoints/latest_checkpoint.txt').exists():
        raise ValueError('Use a new job name: fresh training is required')
    if Path(config.checkpoint.load_path).resolve() != BASE.resolve():
        raise ValueError('Training must initialize from the original policy checkpoint')
    trainer = config.trainer.type(config)
    with model_init():
        model = instantiate(config.model)
    initialized = 0 if args.full_finetune else fixed.reinit_lora(model.net)
    adapters = [module for module in model.net.modules()
                if hasattr(module, 'lora_A') and 'default' in module.lora_A]
    if args.full_finetune:
        if adapters or any('lora_' in name for name, _ in model.net.named_parameters()):
            raise ValueError('Full-parameter training must not contain LoRA adapters')
        model.net.requires_grad_(True)
    elif initialized != 280 or len(adapters) != initialized or any(
            module.r['default'] != args.lora_rank
            or module.lora_alpha['default'] != args.lora_alpha
            for module in adapters):
        raise ValueError('Instantiated LoRA adapters differ from the requested rank/alpha')
    policy_parameters = sum(parameter.numel() for parameter in model.net.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.net.parameters()
                               if parameter.requires_grad)
    if args.full_finetune and (policy_parameters == 0 or trainable_parameters != policy_parameters):
        raise ValueError('Every policy backbone parameter must be trainable')
    probes = fixed_batches(dataset, cases)
    class FullParameterMetrics(SFTMetrics):
        audited_optimizer = False

        def on_before_optimizer_step(self, model_ddp, optimizer, scheduler, grad_scaler, iteration=0):
            super().on_before_optimizer_step(model_ddp, optimizer, scheduler, grad_scaler, iteration)
            if self.audited_optimizer:
                return
            parameters = list(model.net.named_parameters())
            expected = {id(parameter) for _, parameter in parameters}
            actual = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
            if expected != actual or any(not parameter.requires_grad for _, parameter in parameters):
                raise ValueError('The optimizer must update every policy backbone parameter')
            gradients = [parameter.grad for _, parameter in parameters if parameter.grad is not None]
            if not gradients:
                raise ValueError('Full-parameter training produced no policy gradients')
            dtypes = {}
            for _, parameter in parameters:
                key = str(parameter.dtype)
                dtypes[key] = dtypes.get(key, 0) + parameter.numel()
            if distributed.is_rank0():
                atomic_json(run / 'full_parameter_audit.json', dict(
                    use_lora=False, policy_parameters=policy_parameters,
                    trainable_parameters=trainable_parameters,
                    optimizer_parameter_tensors=len(actual), policy_parameter_tensors=len(expected),
                    parameters_with_gradients=len(gradients), parameter_dtypes=dtypes,
                    optimizer_covers_all_policy_parameters=True))
            self.audited_optimizer = True

    monitor_type = FullParameterMetrics if args.full_finetune else SFTMetrics
    monitor = monitor_type(run, probes, args.probe_every)
    monitor.config, monitor.trainer = config, trainer
    trainer.callbacks._callbacks.append(monitor)
    if distributed.is_rank0():
        record = dict(job_name=args.job_name, world_size=dist.get_world_size(),
                      batch_size_per_gpu=args.batch_size_per_gpu,
                      global_batch_size=dist.get_world_size()*args.batch_size_per_gpu,
                      last_batch_size_per_gpu=len(sampler)%args.batch_size_per_gpu or args.batch_size_per_gpu,
                      dataloader_drop_last=False,
                      max_iter=max_iter, epochs=args.epochs if args.max_iter is None else None,
                      iterations_per_epoch=iterations_per_epoch,
                      samples_per_epoch=len(sampler)*dist.get_world_size(),
                      sampler_padding_samples=len(sampler)*dist.get_world_size()-len(dataset),
                      checkpoint_save_iter=config.checkpoint.save_iter,
                      sampler_epoch_updates=True,
                      learning_rate=1e-5, full_finetune=args.full_finetune,
                      training_mode='full' if args.full_finetune else 'lora',
                      use_lora=not args.full_finetune,
                      policy_parameters=policy_parameters, trainable_parameters=trainable_parameters,
                      lora_rank=None if args.full_finetune else args.lora_rank,
                      lora_alpha=None if args.full_finetune else args.lora_alpha,
                      lora_target_modules=[] if args.full_finetune else lora_targets,
                      lora_scaling=None if args.full_finetune else args.lora_alpha/args.lora_rank,
                      use_mask_attention_loss=False, attention_capture_enabled=False, fresh_lora_modules=initialized,
                      trajectories=dataset.num_episodes, steps=dataset.num_steps,
                      trajectory_sources=report['trajectory_sources'],
                      data_directory=str(args.data_root), source_manifest=report['training_manifest'],
                      selected_manifest_sha256=report['manifest_sha256'],
                      base_checkpoint=config.checkpoint.load_path,
                      checkpoint_directory=str(Path(config.job.path_local) / 'checkpoints'),
                      fixed_probe_every=args.probe_every,
                      fixed_probe_samples=[{'tag':tag,'relative_step':step} for tag,step,_ in probes],
                      started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        atomic_json(run / 'run.json', record)
        print('[SFT190_RUN] ' + json.dumps(record), flush=True)
    trainer.train(model, dataloader, None)
    free, total = torch.cuda.mem_get_info()
    memory = dict(rank=dist.get_rank(), device=torch.cuda.current_device(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  free_bytes=free, total_bytes=total)
    memories = [None] * dist.get_world_size()
    dist.all_gather_object(memories, memory)
    if distributed.is_rank0():
        atomic_json(run / 'cuda_memory.json', dict(per_rank=memories))
        atomic_json(run / 'completed.json', {'iteration': max_iter, 'iterations_per_epoch': iterations_per_epoch})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
