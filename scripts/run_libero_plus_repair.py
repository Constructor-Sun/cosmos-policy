#!/usr/bin/env python3
"""Reuse the legacy failure-stage Memory repair on saved LIBERO-plus failures.

Default: four smoke cases; --all-cases repairs every saved failure and
--exclude-completed-from preserves earlier completed attempts. Diagnostics
replay the recorded actions and validate proprio before any intervention. The
existing phase recorder, cut-in rule, Memory selector and repair runner are used.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_libero_plus_batch as batch


def proprio(obs):
    import numpy as np
    return np.concatenate([obs['robot0_gripper_qpos'], obs['robot0_eef_pos'], obs['robot0_eef_quat']])


def diagnose(row, config, suite, initial, actions, reference, evaluator, directory):
    import numpy as np
    from PIL import Image
    from libero.libero import get_libero_path
    from memory_system.execute.plan import plan_from_bddl, drop_presatisfied_turnon
    from memory_system.tta.phase_record import PhaseEventRecorder, compute_t_star, load_task_sequence
    from memory_system.tta.object_query import SceneObjectQuery
    from memory_system.execute.skill_completion.shadow import PickTargetPointCloud

    task = suite.get_task(row['task_id'])
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    if '_view_' in str(bddl) and '_initstate_' in str(bddl):
        bddl = Path(str(bddl).split('_view_', 1)[0] + '.bddl')
    phases = plan_from_bddl(bddl)
    _, _, memory_phases = load_task_sequence(row['base_task'])
    # Retain the existing Memory IDs when the BDDL phase has the same identity.
    normalized = []
    for phase in phases:
        matches = [p for p in memory_phases if p.skill == phase.skill and p.arguments == phase.arguments]
        normalized.append(replace(phase, planner_step_id=matches[0].planner_step_id) if len(matches) == 1 else phase)
    evaluator.set_seed_everywhere(config['environment_seed'])
    env, _ = evaluator.get_libero_env(task, 'cosmos', resolution=256, camera_depths=[True, True], env_seed=None)
    try:
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(config['settle_steps']):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        phases = drop_presatisfied_turnon(tuple(normalized), env)
        recorder = PhaseEventRecorder(phases, task_name=task.name, episode_id=row['case_id'], object_query=SceneObjectQuery(env, resolution=256))
        recorder.begin(step=0, initial_gripper_state=0)
        frames, observed, deviations = [], [], []
        success = False
        for step, action in enumerate(actions):
            current = proprio(obs)
            observed.append(current)
            deviation = float(np.max(np.abs(current - reference[step])))
            deviations.append(deviation)
            if deviation > 1e-5:
                raise RuntimeError(f"Baseline replay mismatch at step {step}: max proprio error={deviation}")
            frames.append(np.flipud(np.asarray(obs['agentview_image'])).copy())
            obs, _, done, _ = env.step(action.tolist())
            phase = recorder.active_phase
            with evaluator._policy_rng_scope(config['action_seed']):
                points = None
                if phase is not None and phase.skill == 'Pick':
                    points = PickTargetPointCloud(env, 256).points(obs, phase.arguments['item'])
                decision = recorder.observe(step=step, action=action, obs=obs, target_points=points)
                if not done and not (decision is not None and decision.advance) and (step + 1) % 16 == 0:
                    recorder.finish_chunk(step=step)
            if done:
                success = True
                break
        if success:
            raise RuntimeError('Recorded failed baseline succeeded during action replay')
        if len(observed) != len(actions):
            raise RuntimeError('Baseline replay terminated before the saved action sequence ended')
        record = recorder.finalize(total_steps=len(actions), success=False)
        candidate = record.get('candidate')
        numeric_tstar, event = compute_t_star(candidate)
        tstar = 10 if numeric_tstar is None else int(numeric_tstar)
        if not 0 <= tstar < len(actions):
            raise RuntimeError(f'Invalid cut-in index {tstar}')
        plan = [dict(phase_index=i, planner_step_id=p.planner_step_id, skill=p.skill, arguments=dict(p.arguments)) for i, p in enumerate(phases)]
        repair_phase = dict(plan[0])
        if numeric_tstar is not None:
            diagnosed = candidate.get('repair_phase') or candidate.get('runtime_phase') or candidate.get('span') or {}
            if all(k in diagnosed for k in ['planner_step_id', 'skill']):
                repair_phase = dict(phase_index=int(diagnosed.get('phase_index', 0)), planner_step_id=int(diagnosed['planner_step_id']), skill=diagnosed['skill'], arguments=dict(diagnosed.get('arguments') or {}))
        request = dict(task=task.name, tag=row['case_id'], phase=repair_phase, plan=plan, t_star=tstar, actions=actions.tolist())
        record.update(census_abs_init=0, task_name_perturbed=task.name, suite_task_name=task.name, base_task=row['base_task'], suite='libero_10', perturbation_category=row['category'], t_star=tstar, numeric_t_star=numeric_tstar, event_step=event)
        batch.atomic_json(directory / 'phase_record.json', record)
        batch.atomic_json(directory / 'repair_request.json', request)
        batch.atomic_json(directory / 'replay_verification.json', dict(steps=len(observed), maximum_proprio_error=max(deviations), success=False, verified=True))
        Image.fromarray(frames[tstar]).save(directory / 'tstar.png')
        print(f"DIAGNOSED task={row['task_id']} t_star={tstar} skill={repair_phase['skill']} replay_error={max(deviations):.3g}", flush=True)
        return request
    finally:
        env.close()


def save_video(data, destination):
    import imageio.v2 as imageio
    from PIL import Image
    import numpy as np
    writer = imageio.get_writer(destination, fps=30)
    try:
        if 'primary_images_jpeg' in data:
            for encoded in data['primary_images_jpeg']:
                with Image.open(io.BytesIO(np.asarray(encoded, dtype=np.uint8).tobytes())) as image:
                    writer.append_data(np.asarray(image.convert('RGB')))
        else:
            for frame in data['primary_images']:
                writer.append_data(frame)
    finally:
        writer.close()


def repair_worker(job):
    import numpy as np
    import h5py
    import torch
    batch.configure_sensor_dependencies()
    config = batch.read_json(job['source_config'])
    from cosmos_policy._src.imaginaire.utils import checkpoint_db
    original_checkpoint = checkpoint_db.get_checkpoint_path
    checkpoint_db.get_checkpoint_path = lambda uri: config['ckpt_path'] if uri.rstrip('/') == 'hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt' else original_checkpoint(uri)
    from cosmos_policy.experiments.robot.libero import run_libero_eval as evaluator
    from cosmos_policy.experiments.robot import cosmos_utils
    torch.cuda.set_device(0)
    suite, _, cache, _ = batch.load_catalog(config, check_states=False)
    cosmos_utils.t5_text_embeddings_cache.update(cache)
    model = selector = None
    original_env = evaluator.get_libero_env
    evaluator.save_rollout_video = lambda *args, **kwargs: None
    evaluator.save_rollout_video_with_future_image_predictions = lambda *args, **kwargs: None
    cfg = evaluator.PolicyEvalConfig(
        config='cosmos_predict2_2b_480p_libero__inference_only', ckpt_path=config['ckpt_path'],
        dataset_stats_path=str(Path(config['policy_dir']) / 'libero_dataset_statistics.json'),
        t5_text_embeddings_path=str(Path(config['policy_dir']) / 'libero_t5_embeddings.pkl'),
        task_suite_name='libero_10', unnorm_key='libero_10', num_trials_per_task=1,
        seed=config['action_seed'], environment_seed=config['environment_seed'], rng_mode='openwam',
        deterministic=True, deterministic_reset=False, randomize_seed=False,
        data_collection=True, jpeg_compress=True, enable_initial_alignment=True,
        enable_collision_aware_initial_alignment=True, enable_curobo_joint_execution=False,
    )
    evaluator.validate_config(cfg)
    os.environ['DETERMINISTIC'] = 'True'
    stats = evaluator.load_dataset_stats(cfg.dataset_stats_path)
    for row in job['cases']:
        directory = Path(job['work_dir']) / row['group'] / row['case_id']
        diagnosis = directory / 'diagnosis'
        diagnosis.mkdir(parents=True, exist_ok=True)
        result = dict(task_id=row['task_id'], case_id=row['case_id'], group=row['group'], task_name=row['task_name'], baseline_success=False, baseline_video=row['videos'][0], baseline_actions=row['actions_path'], status='error', success=None, protocol={k:config[k] for k in ['action_seed','environment_seed','settle_steps','max_steps','rng_mode']})
        started = time.time()
        recordings = []
        print(f"START task={row['task_id']} group={row['group']}", flush=True)
        batch.atomic_json(directory / 'result.json', {**result, 'status': 'running'})
        try:
            os.environ['COSMOS_TTA_REPAIR'] = ''
            evaluator.get_libero_env = original_env
            with h5py.File(row['episode_hdf5'], 'r') as baseline:
                initial = baseline['initial_state'][:]
                actions = baseline['actions'][:]
                reference = baseline['proprio'][:]
            request = diagnose(row, config, suite, initial, actions, reference, evaluator, diagnosis)
            result.update(t_star=request['t_star'], phase=request['phase'])
            if model is None:
                evaluator.set_seed_everywhere(cfg.seed)
                model, model_config = evaluator.get_model(cfg)
                if cfg.chunk_size != model_config.dataloader_train.dataset.chunk_size:
                    raise RuntimeError('Model chunk size mismatch')
                selector = evaluator._create_initial_alignment_selector(cfg)
                original_select = selector.select
                def select_with_rng(*args, **kwargs):
                    with evaluator._policy_rng_scope(cfg.seed):
                        return original_select(*args, **kwargs)
                selector.select = select_with_rng
                # The repair-only VAE encoding also stays out of the environment
                # RNG stream. Policy queries retain their existing RNG behavior.
                query = {'active': False}
                original_action = evaluator.get_action
                original_encode = model.get_data_and_condition
                def action_query(*args, **kwargs):
                    query['active'] = True
                    try:
                        return original_action(*args, **kwargs)
                    finally:
                        query['active'] = False
                def encode(*args, **kwargs):
                    if query['active']:
                        return original_encode(*args, **kwargs)
                    with evaluator._policy_rng_scope(cfg.seed):
                        return original_encode(*args, **kwargs)
                evaluator.get_action = action_query
                model.get_data_and_condition = encode
            key = row['instruction']
            cosmos_utils.get_t5_embedding_from_cache = lambda label: cosmos_utils.t5_text_embeddings_cache[key]
            native_wait = batch.native_wait_steps(evaluator.run_episode)
            def make_env(*args, **kwargs):
                env, description = original_env(*args, **kwargs)
                wrapped = batch.RecordedEnv(env, config['settle_steps'] - native_wait, native_wait)
                recordings.append(wrapped)
                return wrapped, description
            evaluator.get_libero_env = make_env
            # Keep the full executed trajectory, including the replayed prefix,
            # within the same 700-action budget as the baseline.
            evaluator.TASK_MAX_STEPS['libero_10'] = config['max_steps'] - request['t_star']
            class SavedSuite:
                def get_task(self, task_id):
                    return suite.get_task(task_id)
                def get_task_init_states(self, task_id):
                    return np.asarray([initial])
            cfg.local_log_dir = str(directory)
            cfg.run_id_note = row['case_id']
            os.environ['COSMOS_TTA_REPAIR'] = str(diagnosis / 'repair_request.json')
            os.environ['COSMOS_TTA_REPAIR_OUT'] = str(directory / 'memory_result.json')
            os.environ['COSMOS_TTA_REPAIR_TAG'] = row['case_id']
            with (directory / 'eval.log').open('w') as handle:
                log = batch.EpisodeLog(handle)
                evaluator.set_seed_everywhere(cfg.environment_seed)
                episodes, successes = evaluator.run_task(cfg, SavedSuite(), row['task_id'], model, None, stats, None, evaluator.get_image_resize_size(cfg.model_family), log_file=log, initial_alignment_selector=selector)
                if log.errors:
                    raise RuntimeError('\n'.join(log.errors))
            if episodes != 1:
                raise RuntimeError(f'Expected one trial, got {episodes}')
            files = list((directory / 'rollout_data').glob('*.hdf5'))
            if len(files) != 1 or len(recordings) != 1:
                raise RuntimeError('Missing or ambiguous repair recording')
            recording = recordings[0]
            executed = np.asarray(recording.actions)
            with h5py.File(files[0], 'a') as data:
                corrected = data['actions'][:]
                if not np.array_equal(corrected, executed[config['settle_steps']:]):
                    raise RuntimeError('Repair actions differ from executed actions')
                if len(corrected) > config['max_steps']:
                    raise RuntimeError('Repair exceeded the baseline action budget')
                # The legacy repair loader explicitly casts the request's
                # prefix to float32 before execution. Check that exact stream.
                legacy_prefix = actions[:request['t_star']].astype(np.float32)
                if not np.array_equal(corrected[:request['t_star']], legacy_prefix):
                    raise RuntimeError('Repair prefix differs from the saved baseline')
                result['baseline_prefix_max_precision_delta'] = float(np.max(np.abs(corrected[:request['t_star']] - actions[:request['t_star']]))) if request['t_star'] else 0.0
                data.create_dataset('initial_state', data=recording.initial_state)
                data.create_dataset('executed_actions', data=executed)
                data.create_dataset('executed_action_is_settle', data=np.arange(len(executed)) < config['settle_steps'])
                data.create_dataset('environment_step_indices', data=np.arange(len(corrected)) + config['settle_steps'])
                data.attrs['repair_request'] = json.dumps({k:v for k,v in request.items() if k!='actions'})
                data.attrs['baseline_episode_hdf5'] = row['episode_hdf5']
                save_video(data, directory / 'video.mp4')
            np.save(directory / 'actions.npy', corrected)
            memory = batch.read_json(directory / 'memory_result.json')
            result.update(status='completed', success=bool(successes), memory_status=memory.get('status'), memory_result=memory, action_steps=len(corrected), video=str(directory/'video.mp4'), actions=str(directory/'actions.npy'), episode_hdf5=str(files[0]))
            print(f"REPAIRED task={row['task_id']} memory={result['memory_status']} success={result['success']}", flush=True)
        except Exception:
            result['error'] = traceback.format_exc()
            print(result['error'], flush=True)
        finally:
            for recording in recordings:
                try:
                    recording.env.close()
                except Exception:
                    pass
        result['elapsed_seconds'] = round(time.time()-started, 3)
        batch.atomic_json(directory / 'result.json', result)


def main():
    if os.environ.get('LIBERO_PLUS_REPAIR_JOB'):
        repair_worker(batch.read_json(os.environ['LIBERO_PLUS_REPAIR_JOB']))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'experiments/tta/repair_libero_plus_435/inputs/failed_435.json')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'experiments/tta/repair_libero_plus_435/smoke_4')
    parser.add_argument('--gpus', type=int, nargs='+', default=[2,3,5,7])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--task-ids', type=int, nargs='+')
    selection.add_argument('--all-cases', action='store_true')
    parser.add_argument('--exclude-completed-from', type=Path, nargs='+', default=[])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = batch.read_json(args.manifest)
    source_config = Path(manifest['source_run'])/'run_config.json'
    config = batch.read_json(source_config)
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError('GPU IDs must be distinct and nonnegative')
    if args.retry_errors and not args.resume:
        raise ValueError('--retry-errors requires --resume')
    if args.all_cases:
        cases = list(manifest['cases'])
    elif args.task_ids:
        wanted=set(args.task_ids)
        cases=[row for row in manifest['cases'] if row['task_id'] in wanted]
        if len(cases)!=len(wanted):
            raise ValueError('Selected IDs must be recorded baseline failures')
    else:
        cases=[]
        for group in ['robot_initial_states','camera_viewpoints','background_textures','language_instructions']:
            rows=[row for row in manifest['cases'] if row['group']==group]
            cases.append(next((row for row in rows if row['base_task'].startswith('LIVING_ROOM_SCENE5_')), rows[0]))
    manifest_ids = {row['task_id'] for row in manifest['cases']}
    excluded = {}
    for previous in args.exclude_completed_from:
        for path in previous.glob('*/task*/result.json'):
            result = batch.read_json(path)
            if result['status'] != 'completed' or result['task_id'] not in manifest_ids:
                continue
            for key in ['video', 'actions', 'episode_hdf5']:
                if not Path(result[key]).is_file():
                    raise ValueError(f'Missing completed artifact: {result[key]}')
            excluded[result['task_id']] = result
    cases = [row for row in cases if row['task_id'] not in excluded]
    args.work_dir = args.work_dir.resolve()
    identity = dict(manifest=str(args.manifest.resolve()),
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    source_config_sha256=hashlib.sha256(source_config.read_bytes()).hexdigest(),
                    task_ids=[row['task_id'] for row in cases],
                    excluded_task_ids=sorted(excluded))
    pending = []
    for row in cases:
        result_path = args.work_dir / row['group'] / row['case_id'] / 'result.json'
        result = batch.read_json(result_path) if args.resume and result_path.is_file() else None
        if result and result['status'] == 'completed':
            for key in ['video', 'actions', 'episode_hdf5']:
                if not Path(result[key]).is_file():
                    raise ValueError(f'Missing completed artifact: {result[key]}')
            continue
        if result and result['status'] == 'error' and not args.retry_errors:
            continue
        pending.append(row)
    plan = dict(total_failures=len(manifest['cases']), already_completed=len(excluded),
                selected=len(cases), pending=len(pending), gpus=args.gpus, work_dir=str(args.work_dir))
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if args.resume:
        if not (args.work_dir / 'run_config.json').is_file():
            raise ValueError('--resume requires an existing run_config.json')
        if batch.read_json(args.work_dir / 'run_config.json') != identity:
            raise ValueError('Resume inputs differ from the original repair run')
    elif args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise ValueError('Choose an empty --work-dir to preserve previous attempts')
    (args.work_dir/'jobs').mkdir(parents=True, exist_ok=True)
    with (args.work_dir / '.driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch.atomic_json(args.work_dir / 'run_config.json', identity)
        batch.atomic_json(args.work_dir / 'selected_cases.json', cases)
        batch.atomic_json(args.work_dir / 'previous_completed.json', list(excluded.values()))
        # Preserve unfinished attempts before rerunning their case directories.
        for row in pending:
            directory = args.work_dir / row['group'] / row['case_id']
            if directory.exists():
                archive = args.work_dir / 'previous_attempts' / row['group'] / f"{row['case_id']}_{time.time_ns()}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                directory.rename(archive)
        children = []
        def summarize(finished=False):
            results = {row['task_id']: row for row in excluded.values()}
            selected_ids = set(identity['task_ids'])
            for path in args.work_dir.glob('*/task*/result.json'):
                row = batch.read_json(path)
                if row['task_id'] in selected_ids:
                    results[row['task_id']] = row
            rows = [results[key] for key in sorted(results)]
            completed = sum(row['status'] == 'completed' for row in rows)
            errors = sum(row['status'] == 'error' for row in rows)
            summary = dict(planned=len(cases), previous_completed=len(excluded),
                           total_planned=len(cases)+len(excluded),
                           completed=completed, successes=sum(row.get('success') is True for row in rows),
                           errors=errors, unfinished=len(cases)+len(excluded)-completed-errors,
                           status='finished' if finished else 'running',
                           workers=[dict(gpu=gpu, pid=process.pid, returncode=process.poll()) for gpu,process,_ in children],
                           cases=rows)
            batch.atomic_json(args.work_dir / 'summary.json', summary)
            return summary
        try:
            for index, gpu in enumerate(args.gpus):
                assigned = pending[index::len(args.gpus)]
                if not assigned:
                    continue
                path = args.work_dir / 'jobs' / f'gpu{gpu}.json'
                batch.atomic_json(path, dict(cases=assigned, source_config=str(source_config), work_dir=str(args.work_dir)))
                env = batch.worker_env(config, gpu, 'repair', source_config)
                env.update(SMOKE_PYTHON_SCRIPT=str(ROOT/'scripts/run_libero_plus_repair.py'), LIBERO_PLUS_REPAIR_JOB=str(path), COSMOS_SKILL_COMPLETION_SHADOW='1', COSMOS_SKILL_COMPLETION_ACTIVE='0', COSMOS_SKILL_COMPLETION_ITEM='', COSMOS_TTA_RGBD='1', COSMOS_TTA_PHASE_LOOP='0', COSMOS_INITIAL_ALIGNMENT='1', MUJOCO_EGL_DEVICE_ID=str(gpu))
                handle = (args.work_dir / f'gpu{gpu}.log').open('a' if args.resume else 'w')
                process = subprocess.Popen(['sh', str(ROOT/'scripts/run_libero_smoke_test.sh')], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((gpu, process, handle))
            summary = summarize()
            last_counts = None
            while any(process.poll() is None for _,process,_ in children):
                summary = summarize()
                counts = (summary['completed'], summary['successes'], summary['errors'])
                if counts != last_counts:
                    print(json.dumps({key:summary[key] for key in ['total_planned','completed','successes','errors','unfinished']}), flush=True)
                    last_counts = counts
                time.sleep(15)
        finally:
            for _,process,handle in children:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                handle.close()
        summary = summarize(finished=True)
        print(json.dumps({key:summary[key] for key in ['total_planned','completed','successes','errors','unfinished','status','workers']}, indent=2), flush=True)
        if summary['errors'] or summary['unfinished'] or any(row['returncode'] for row in summary['workers']):
            raise SystemExit(1)


if __name__=='__main__':
    main()
