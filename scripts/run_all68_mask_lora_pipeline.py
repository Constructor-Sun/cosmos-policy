#!/usr/bin/env python
"""Run fresh joint LoRA training, merge the final checkpoint, and evaluate 68 states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data1/liu/miniconda3/envs/cosmospolicy/bin/python'
BASE = Path('/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt')


def collect_evaluation(work, meta, checkpoint, training_tags):
    rows, errors = [], []
    for tag in sorted(meta):
        try:
            summaries = list((work / 'results' / tag).glob('*summary.json'))
            if len(summaries) != 1:
                raise ValueError(f'Expected one actual episode summary, found {len(summaries)}')
            summary = json.loads(summaries[0].read_text())
            conditions = summary['conditions']
            if len(conditions) != 1:
                raise ValueError('Expected one evaluation condition')
            condition = conditions[0]
            if condition['condition'] != 'robot_initial_states' or condition['num_trials'] != 1:
                raise ValueError('Unexpected condition/trial count')
            if condition['task_name'] != meta[tag]['pert_task']:
                raise ValueError('Perturbed task mismatch')
            episodes = json.loads(Path(condition['episodes_path']).read_text())
            if isinstance(episodes, dict): episodes = episodes['episodes']
            if len(episodes) != 1: raise ValueError('Expected one episode')
            episode = episodes[0]
            if episode['init_state_index'] != meta[tag]['census_abs_init'] or episode['seed'] != 7:
                raise ValueError('Initial state/seed mismatch')
            success = bool(episode['success'])
            if success != (condition['successes'] == 1):
                raise ValueError('Episode/summary success mismatch')
            rows.append({'tag':tag, 'success':success, 'was_in_training':tag in training_tags,
                         'init_state_index':episode['init_state_index'], 'seed':episode['seed'],
                         'summary_path':str(summaries[0]), 'episodes_path':condition['episodes_path'],
                         'log_path':condition['log_path']})
        except Exception as error:
            errors.append({'tag':tag, 'error':f'{type(error).__name__}: {error}'})
    groups = defaultdict(list)
    for row in rows: groups[row['tag'].split('-')[0]].append(row)
    by_task = {task:{'completed':len(group), 'successes':sum(row['success'] for row in group),
                     'success_rate':sum(row['success'] for row in group)/len(group)}
               for task,group in sorted(groups.items())}
    successes = sum(row['success'] for row in rows)
    report = {'status':'COMPLETED' if len(rows) == 68 and not errors else 'INCOMPLETE',
              'checked_at':time.strftime('%Y-%m-%d %H:%M:%S'), 'checkpoint':str(checkpoint),
              'training_trajectories':65, 'expected':68, 'completed':len(rows), 'successes':successes,
              'success_rate_completed':successes/len(rows) if rows else None,
              'scope':'Original 68 screened initial states; 65 have successful training demonstrations',
              'by_task':by_task, 'rows':rows, 'errors':errors, 'work_directory':str(work)}
    (work / 'evaluation_report.json').write_text(json.dumps(report, indent=2))
    Path('/tmp/ALL68_mask_lora_eval_report.json').write_text(json.dumps(report, indent=2))
    return report


def verify_checkpoint(directory):
    from torch.distributed.checkpoint import FileSystemReader
    groups = {}
    for name in ['model', 'optim', 'scheduler', 'trainer']:
        group = directory / name
        metadata = FileSystemReader(str(group)).read_metadata()
        if not metadata.state_dict_metadata: raise ValueError(f'Empty checkpoint group: {name}')
        for location in metadata.storage_data.values():
            file = group / location.relative_path
            if not file.is_file() or file.stat().st_size < location.offset + location.length:
                raise ValueError(f'Missing/truncated checkpoint shard: {file}')
        groups[name] = {'keys':len(metadata.state_dict_metadata),
                        'files':len({loc.relative_path for loc in metadata.storage_data.values()})}
    return groups


def verify_merge(checkpoint):
    import torch
    merged = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    base = torch.load(BASE, map_location='cpu', mmap=True, weights_only=False)
    if not isinstance(merged,dict) or any('lora_' in key or 'base_layer' in key for key in merged):
        raise ValueError('Merged checkpoint still contains unmerged LoRA keys')
    changed = 0
    for key, value in merged.items():
        if not isinstance(value,torch.Tensor): continue
        if not torch.isfinite(value).all(): raise ValueError(f'Non-finite merged weight: {key}')
        if key not in base or value.shape != base[key].shape:
            raise ValueError(f'Merged weight does not match original model schema: {key}')
        if not torch.equal(value,base[key]): changed += 1
    missing = [key for key in base if key not in merged and not key.endswith('_extra_state')]
    if missing or changed < 280:
        raise ValueError(f'Merge mismatch: {len(missing)} missing weights, {changed} changed weights')
    return {'merged_keys':len(merged), 'changed_weights':changed, 'all_weights_finite':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--max-iter', type=int, default=2742)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    run = REPO / 'scripts/experiments/all68_mask_lora' / args.job_name
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / 'pipeline_status.json'
    latest_path = Path('/tmp/all68_mask_lora_latest_run.json')
    state = {'job_name':args.job_name, 'pid':os.getpid(), 'run_directory':str(run),
             'training_trajectories':65, 'evaluation_initial_states':68, 'max_iter':args.max_iter,
             'gpu_ids':args.gpus, 'lambda':1.0, 'base_checkpoint':str(BASE),
             'started_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    def update(status, **extra):
        state.update(status=status, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)
        data = json.dumps(state, indent=2)
        for path in [status_path, latest_path]:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(data); temporary.replace(path)
        print('[ALL68_PIPELINE] ' + json.dumps(state), flush=True)
    env = dict(os.environ)
    env.update(TTA_SFT_JOB_NAME=args.job_name, TTA_SFT_MAX_ITER=str(args.max_iter),
               TTA_MASK_PROBE_EVERY='100', HF_HUB_OFFLINE='1',
               HF_HUB_CACHE='/tmp/tta_hf_cache/huggingface-hub', OMP_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH','')
    try:
        update('TRAINING', train_log=str(run / 'train.log'))
        with (run / 'train.log').open('w') as log:
            subprocess.run(['bash',str(REPO / 'scripts/run_all68_mask_lora.sh'),args.gpus],
                           cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        completion = json.loads((run / 'completed.json').read_text())
        if completion['iteration'] != args.max_iter: raise ValueError('Training ended at unexpected step')
        recipe = json.loads((run / 'run.json').read_text())
        checkpoint_dir = Path(recipe['checkpoint_directory']) / f'iter_{args.max_iter:09d}'
        verification = verify_checkpoint(checkpoint_dir)
        merged = REPO / 'training' / f'{args.job_name}_iter{args.max_iter}_merged.pt'
        update('MERGING', checkpoint_directory=str(checkpoint_dir), checkpoint_verification=verification,
               merged_checkpoint=str(merged))
        merge_env = dict(env); merge_env['CUDA_VISIBLE_DEVICES'] = args.gpus.split(',')[2]
        with (run / 'merge.log').open('w') as log:
            subprocess.run([PYTHON, str(REPO / 'scripts/merge_lora_ckpt.py'),
                            '--ckpt',str(checkpoint_dir/'model'),'--out',str(merged)],
                           cwd=REPO, env=merge_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        merge_verification = verify_merge(merged)
        work = REPO / 'scripts/experiments/all68_mask_lora_eval' / args.job_name
        work.mkdir(parents=True, exist_ok=True)
        meta_path = REPO / 'training/tta_sft_all68_mask_65/eval_meta_68.json'
        meta = json.loads(meta_path.read_text())
        training_tags = {c['tag'] for c in json.loads(Path(recipe['mask_manifest']).read_text())['cases']}
        eval_env = dict(env)
        for key in list(eval_env):
            if key.startswith(('COSMOS_TTA_','TTA_REPAIR_')): eval_env.pop(key)
        eval_env.update(SMOKE_LATENT_ADAPTER='', SMOKE_DATA_COLLECTION='false', COSMOS_HELD_OBJECT='',
                        COSMOS_INITIAL_ALIGNMENT='0', COSMOS_CUROBO_JOINT_EXECUTION='0',
                        COSMOS_SKILL_READY_MEMORY='0', COSMOS_SKILL_COMPLETION_SHADOW='0',
                        SMOKE_DETERMINISTIC_RESET='true', SMOKE_DETERMINISTIC_RESET_SEED='0', SMOKE_SEED='7')
        update('EVALUATING', merge_verification=merge_verification, eval_work_directory=str(work),
               eval_meta=str(meta_path), eval_log=str(work/'eval.log'))
        with (work / 'eval.log').open('w') as log:
            subprocess.run([PYTHON,str(REPO/'scripts/eval_sft_68.py'),'--gpus',args.gpus,
                            '--ckpt',str(merged),'--work-dir',str(work),'--meta',str(meta_path)],
                           cwd=REPO, env=eval_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # eval_sft_68 permits failed children; success requires 68 verified episode JSONs.
        report = collect_evaluation(work, meta, merged, training_tags)
        if report['status'] != 'COMPLETED':
            raise ValueError(f"Evaluation incomplete: {report['completed']}/68; errors={report['errors']}")
        update('COMPLETED', evaluation_report=str(work/'evaluation_report.json'),
               successes=report['successes'], success_rate=report['success_rate_completed'],
               by_task=report['by_task'])
    except Exception as error:
        update('FAILED', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise


if __name__ == '__main__': main()
