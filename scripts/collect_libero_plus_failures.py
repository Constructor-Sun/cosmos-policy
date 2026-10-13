#!/usr/bin/env python3
"""Build the Memory-repair failure manifest from a saved LIBERO-plus baseline.

Only completed, unsuccessful trials are failures. Missing/running/error trials
are rejected unless --allow-incomplete is explicit; they are never relabeled.
This entry point uses the Python standard library and does not load a policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_FIELDS = (
    'action_seed', 'environment_seed', 'settle_steps', 'max_steps',
    'rng_mode', 'init_state_index', 'num_trials_per_task',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_from_config(config):
    return {key: config[key] for key in PROTOCOL_FIELDS}


def recorded_file(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError(f'Invalid artifact path: {value!r}')
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def catalog_from_config(config):
    catalog = {}
    groups = set()
    for group in config['groups']:
        key = group['key']
        if not isinstance(key, str) or not key or '/' in key or key in groups:
            raise ValueError(f'Invalid/duplicate group: {key!r}')
        groups.add(key)
        for raw in group['tasks']:
            task_id = raw['task_id']
            if type(task_id) is not int or task_id < 0 or task_id in catalog:
                raise ValueError(f'Invalid/duplicate task_id: {task_id!r}')
            if raw.get('group', key) != key or raw['category'] != group['category']:
                raise ValueError(f'Task category differs from its group: {task_id}')
            catalog[task_id] = dict(raw, group=key)
    return catalog


def validate_identity(result, expected, path):
    for key in ('task_id', 'task_name', 'group'):
        if result.get(key) != expected[key]:
            raise ValueError(f'Case identity mismatch ({key}): {path}')
    if type(result['task_id']) is not int:
        raise ValueError(f'Noninteger task_id: {path}')
    case_id = f"task{expected['task_id']:04d}"
    if result.get('case_id', case_id) != case_id:
        raise ValueError(f'Case ID mismatch: {path}')


def validate_protocol(result, protocol, path, keys=PROTOCOL_FIELDS):
    for key in keys:
        if result.get(key) != protocol[key]:
            raise ValueError(f'Protocol mismatch ({key}): {path}')


def json_payload(path, value, overwrite=False):
    """Preserve an existing equivalent manifest, including its historical hash."""
    path = Path(path)
    data = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if path.exists():
        if not path.is_file():
            raise FileExistsError(path)
        old = path.read_bytes()
        try:
            equivalent = json.loads(old) == value
        except (ValueError, UnicodeDecodeError):
            equivalent = False
        if equivalent:
            return old
        if not overwrite:
            raise FileExistsError(f'Different output exists: {path}; use --overwrite or a new output path')
    return data


def write_payloads(payloads, overwrite=False):
    # Check every destination before writing any output.
    for path, data in payloads.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != data) and not overwrite:
            raise FileExistsError(f'Different output exists: {path}; use --overwrite or a new output path')
    for path, data in payloads.items():
        if path.is_file() and path.read_bytes() == data:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.tmp.', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def build_failure_manifest(baseline_root, base_tasks_path=None, allow_incomplete=False):
    baseline_root = Path(baseline_root).resolve()
    config_path = baseline_root / 'run_config.json'
    config = read_json(config_path)
    protocol = protocol_from_config(config)
    catalog = catalog_from_config(config)
    base_tasks_path = recorded_file(str(base_tasks_path or config['base_tasks']), baseline_root)
    bases = read_json(base_tasks_path)['tasks']
    if not bases or len({row['name'] for row in bases}) != len(bases):
        raise ValueError('Base tasks must be nonempty and distinct')
    expected_paths = {
        baseline_root / row['group'] / 'episodes' / f"task{task_id:04d}" / 'result.json'
        for task_id, row in catalog.items()
    }
    extras = set(baseline_root.glob('*/episodes/task*/result.json')) - expected_paths
    if extras:
        raise ValueError(f'Unexpected/duplicate baseline result: {sorted(extras)[0]}')
    cases, incomplete = [], []
    counts = Counter()
    for task_id, expected in sorted(catalog.items(), key=lambda item: (item[1]['group'], item[0])):
        path = baseline_root / expected['group'] / 'episodes' / f'task{task_id:04d}' / 'result.json'
        if not path.is_file():
            incomplete.append({'task_id': task_id, 'status': 'missing'})
            continue
        result = read_json(path)
        validate_identity(result, expected, path)
        for key in ('category', 'source_id', 'difficulty_level'):
            if key in expected and result.get(key) != expected[key]:
                raise ValueError(f'Catalog mismatch ({key}): {path}')
        validate_protocol(result, protocol, path)
        if config.get('fingerprint') is not None and result.get('fingerprint') != config['fingerprint']:
            raise ValueError(f'Baseline fingerprint mismatch: {path}')
        if result.get('status') != 'completed':
            incomplete.append({'task_id': task_id, 'status': result.get('status')})
            continue
        if type(result.get('success')) is not bool:
            raise ValueError(f'Completed trial requires a Boolean success: {path}')
        counts['completed'] += 1
        if result['success']:
            counts['successes'] += 1
            continue
        counts['failures'] += 1
        matches = [base for base in bases if result['task_name'].startswith(base['name'] + '_')]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely resolve base task: {path}')
        if not isinstance(result.get('instruction'), str) or not result['instruction']:
            raise ValueError(f'Missing exact policy instruction: {path}')
        case = dict(result, case_id=f'task{task_id:04d}', base_task=matches[0]['name'],
                    clean_language=matches[0]['language'], baseline_result=str(path))
        for key in ('episode_hdf5', 'actions_path'):
            case[key] = str(recorded_file(case.get(key), baseline_root))
        if not isinstance(case.get('videos'), list) or not case['videos']:
            raise ValueError(f'Missing baseline video: {path}')
        case['videos'] = [str(recorded_file(video, baseline_root)) for video in case['videos']]
        cases.append(case)
    if incomplete and not allow_incomplete:
        raise ValueError(f'Baseline incomplete: {len(incomplete)} missing/running/error trials; '
                         'finish the run or explicitly use --allow-incomplete')
    manifest = dict(
        version=1, source_run=str(baseline_root), source_config_sha256=digest(config_path),
        num_cases=len(cases), protocol=protocol,
        counts_by_category=dict(sorted(Counter(row['group'] for row in cases).items())), cases=cases,
    )
    if incomplete:
        manifest['incomplete'] = True
    generation = dict(
        source_run=str(baseline_root), source_config_sha256=digest(config_path),
        base_tasks=str(base_tasks_path), base_tasks_sha256=digest(base_tasks_path),
        planned_cases=len(catalog), completed_cases=counts['completed'],
        successes=counts['successes'], failures=counts['failures'],
        complete=not incomplete, incomplete_cases=incomplete,
        rule='status == completed and success is false; runtime errors are not failures',
    )
    return manifest, generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=REPO / 'experiments/libero_plus/cosmospolicy_full_7dims')
    parser.add_argument('--base-tasks', type=Path, help='Override the base-task mapping in baseline run_config.json')
    parser.add_argument('--output', type=Path, default=REPO / 'experiments/tta/repair_libero_plus_435/inputs/failed_435.json')
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicitly omit unfinished trials; never count them as failures')
    parser.add_argument('--overwrite', action='store_true', help='Replace different existing outputs')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print counts without writing files')
    args = parser.parse_args()
    manifest, generation = build_failure_manifest(args.baseline_root, args.base_tasks, args.allow_incomplete)
    output = args.output.resolve()
    sidecar = output.with_name(output.stem + '_generation.json')
    if not args.dry_run:
        payloads = {path: json_payload(path, value, args.overwrite)
                    for path, value in [(output, manifest), (sidecar, generation)]}
        write_payloads(payloads, args.overwrite)
    print(json.dumps(dict(output=str(output), generation=str(sidecar), dry_run=args.dry_run,
                         num_cases=manifest['num_cases'], counts_by_category=manifest['counts_by_category'],
                         complete=generation['complete']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
