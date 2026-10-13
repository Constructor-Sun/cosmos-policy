#!/usr/bin/env python3
"""Evaluate the 29 background training cases with existing epoch checkpoints."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import run_libero_plus_batch as batch
from run_libero_plus_sft190_pipeline import BASELINE, read_results

PROTOCOL_KEYS = ['manifest_index_sha256', 'source_sha256', 'action_seed',
                 'environment_seed', 'settle_steps', 'max_steps', 'rng_mode',
                 'num_trials_per_task', 'init_state_index']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[6, 7])
    parser.add_argument('--epochs', type=int, nargs='+', default=[4, 8, 1, 2])
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('Require distinct nonnegative GPU IDs')
    if len(set(args.epochs)) != len(args.epochs) or not set(args.epochs) <= {1, 2, 4, 8}:
        parser.error('Select distinct trained epochs from 1, 2, 4, 8')

    source_run = args.source_run.resolve()
    original = batch.read_json(source_run / 'pipeline_status.json')
    if original['status'] != 'COMPLETED':
        raise ValueError('The source training and heldout evaluation must be complete')
    selection = batch.read_json(original['source_manifest'])
    baseline = read_results(BASELINE)
    trained = {row['task_id'] for row in selection['cases']}
    heldout = set(batch.read_json(source_run / 'evaluation_cases.json')['task_ids'])
    if (selection['num_cases'] != 29 or len(selection['cases']) != 29
            or len(trained) != 29 or len(heldout) != 41 or trained & heldout):
        raise ValueError('Require the original disjoint 29 training / 41 heldout split')
    failed_background = {task_id for task_id, row in baseline.items()
                         if row['group'] == 'background_textures' and row['success'] is False}
    if trained | heldout != failed_background:
        raise ValueError('Training and heldout IDs must cover all 70 background baseline failures')
    if any(row['group'] != 'background_textures' or row['success'] is not True
           or row.get('trajectory_source', 'memory_repair') != 'memory_repair'
           for row in selection['cases']):
        raise ValueError('Expected successful background Memory repair training trajectories')
    epoch_reports = {}
    for epoch in args.epochs:
        report = batch.read_json(source_run / 'epoch_reports' / f'epoch_{epoch:02d}.json')
        if report['epoch'] != epoch or not Path(report['checkpoint']).is_file():
            raise ValueError(f'Missing trained checkpoint for epoch {epoch}')
        epoch_reports[epoch] = report
    print(json.dumps(dict(status='SELECTION_VALIDATED',training_cases=29,
                         heldout_cases=41,overlap=0,epochs=args.epochs,gpus=args.gpus)), flush=True)
    if args.validate_only:
        return

    work = source_run / 'training_set_evaluation'
    work.mkdir(exist_ok=True)
    selection_file = work / 'training_case_ids.json'
    selection_data = dict(task_ids=sorted(trained), source_manifest=original['source_manifest'],
                          evaluation_scope='background-training-cases',num_cases=29)
    if selection_file.exists() and batch.read_json(selection_file) != selection_data:
        raise ValueError('Existing evaluation has a different training selection')
    batch.atomic_json(selection_file, selection_data)
    status_file = work / 'status.json'
    state = dict(pid=os.getpid(),source_run=str(source_run),gpu_ids=args.gpus,
                 training_cases=29,evaluation_epochs=args.epochs,completed_epochs=[],
                 scope='Policy-only rollouts on the 29 training case identities; Memory disabled')
    curve = []

    def update(status, **extra):
        state.update(status=status,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**extra)
        batch.atomic_json(status_file,state)
        print(json.dumps(state),flush=True)

    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('COSMOS_TTA_', 'TTA_REPAIR_')):
            env.pop(key)
    env.update(HF_HUB_OFFLINE='1',HF_HUB_CACHE='/tmp/tta_hf_cache/huggingface-hub',
               OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
               COSMOS_INITIAL_ALIGNMENT='0',COSMOS_SKILL_COMPLETION_SHADOW='0',
               COSMOS_SKILL_COMPLETION_ACTIVE='0',COSMOS_SKILL_READY_MEMORY='0',
               COSMOS_CUROBO_JOINT_EXECUTION='0')
    old_config = batch.read_json(BASELINE / 'run_config.json')
    try:
        for epoch in args.epochs:
            checkpoint = Path(epoch_reports[epoch]['checkpoint'])
            output = REPO / 'experiments/libero_plus/sft190_p95' / source_run.name / 'training_cases' / f'epoch_{epoch:02d}'
            command = [sys.executable,str(REPO / 'scripts/run_libero_plus_batch.py'),
                       '--ckpt',str(checkpoint),'--output-root',str(output),
                       '--gpus',*map(str,args.gpus),'--task-ids-file',str(selection_file),
                       '--action-seed','42','--environment-seed','10000',
                       '--settle-steps','30','--max-steps','700','--save-episode-data']
            if (output / 'run_config.json').exists():
                command.append('--resume')
            update('EVALUATING',current_epoch=epoch,evaluation_directory=str(output),checkpoint=str(checkpoint))
            log = work / f'eval_epoch{epoch:02d}.log'
            with log.open('a') as handle:
                result = subprocess.run(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                summary = batch.read_json(output / 'summary.json') if (output / 'summary.json').exists() else {}
                if not summary.get('errors'):
                    raise RuntimeError(f'Epoch {epoch} evaluation exited with code {result.returncode}; see {log}')
                retry = [c for c in command if c != '--resume'] + ['--resume','--retry-errors']
                update('RETRYING_ERRORS',errors=summary['errors'])
                with log.open('a') as handle:
                    subprocess.run(retry,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
            rows = read_results(output,trained)
            config = batch.read_json(output / 'run_config.json')
            if Path(config['ckpt_path']).resolve() != checkpoint.resolve():
                raise ValueError('Evaluated checkpoint differs from the trained epoch checkpoint')
            for key in PROTOCOL_KEYS:
                if config[key] != old_config[key]:
                    raise ValueError(f'Evaluation protocol mismatch: {key}')
            for task_id,row in rows.items():
                if row['task_name'] != baseline[task_id]['task_name'] or row['success'] not in (True,False):
                    raise ValueError(f'Invalid task identity/outcome: {task_id}')
                for artifact in [*row['videos'],row['actions_path'],row['episode_hdf5']]:
                    if not Path(artifact).is_file():
                        raise ValueError(f'Missing evaluation artifact: {artifact}')
            successful = sorted(task_id for task_id,row in rows.items() if row['success'] is True)
            failed = sorted(trained-set(successful))
            record = dict(epoch=epoch,training_successes=len(successful),training_failures=len(failed),
                          training_total=29,training_success_rate=len(successful)/29,
                          heldout_successes=epoch_reports[epoch]['overall']['successes'],heldout_total=41)
            record['combined_successes'] = record['training_successes']+record['heldout_successes']
            record['combined_total'] = 70
            record['combined_success_rate'] = record['combined_successes']/70
            report = dict(status='COMPLETED',**record,successful_training_case_ids=successful,
                          failed_training_case_ids=failed,checkpoint=str(checkpoint),
                          evaluation_directory=str(output),protocol={key:config[key] for key in PROTOCOL_KEYS})
            batch.atomic_json(work / f'epoch_{epoch:02d}_report.json',report)
            curve.append(record)
            ordered = sorted(curve,key=lambda row:row['epoch'])
            batch.atomic_json(work / 'evaluation_curve.json',dict(epochs=ordered))
            table = io.StringIO();writer=csv.DictWriter(table,fieldnames=list(ordered[0]))
            writer.writeheader();writer.writerows(ordered)
            (work / 'evaluation_curve.csv').write_text(table.getvalue())
            update('EPOCH_COMPLETED',completed_epochs=[row['epoch'] for row in ordered],results=ordered)
        update('COMPLETED',results=ordered,evaluation_curve=str(work / 'evaluation_curve.json'))
    except Exception as error:
        update('FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
