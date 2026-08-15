#!/usr/bin/env python3
"""
Resolve experiment configuration from libero10_experiment_tasks.json.

Given a task identifier, perturbation type, and variant value, this script:
  1. Looks up the base task + language instruction
  2. Validates the variant against the official list
  3. Constructs the correct SMOKE_PAIR_PERT_TASK name
  4. Prints shell variable assignments to stdout

Usage:
  python bin/resolve_experiment_config.py \
      --task-idx 1 \
      --perturbation robot_initial_states \
      --variant 274

  python bin/resolve_experiment_config.py \
      --task "KITCHEN_SCENE4_put_the_black_bowl_..." \
      --perturbation background_textures \
      --variant 5 \
      --bg-kind table

Output (shell sourceable):
  SMOKE_PAIR_SUITE=libero_10
  SMOKE_PAIR_BASE_TASK=KITCHEN_SCENE4_...
  SMOKE_PAIR_CLEAN_LANGUAGE=put the black bowl in...
  SMOKE_PAIR_PERT_NAME=robot_initial_states
  SMOKE_PAIR_PERT_CATEGORY=Robot Initial States
  SMOKE_PAIR_PERT_TASK=KITCHEN_SCENE4_..._view_0_0_100_0_0_initstate_274
"""

import argparse
import json
import os
import shlex
import sys


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'configs', 'libero10_experiment_tasks.json')
    with open(config_path) as f:
        return json.load(f)


def find_task(config, task_idx=None, task_name=None):
    """Find a task entry by index or name."""
    tasks = config['tasks']
    if task_name is not None:
        for t in tasks:
            if t['name'] == task_name:
                return t
        raise ValueError(f"Task '{task_name}' not found in config. Available: {[t['name'] for t in tasks]}")
    if task_idx is not None:
        if 0 <= task_idx < len(tasks):
            return tasks[task_idx]
        raise ValueError(f"Task index {task_idx} out of range [0, {len(tasks) - 1}]")
    raise ValueError("Must specify --task or --task-idx")


def validate_and_build_variant(task_entry, perturbation, variant_str, bg_kind=None):
    """Validate the variant and construct the PERT_TASK name."""
    pert = task_entry['perturbations'].get(perturbation)
    if pert is None:
        available = list(task_entry['perturbations'].keys())
        raise ValueError(f"Perturbation '{perturbation}' not found. Available: {available}")

    pattern = pert['task_name_pattern']

    if perturbation == 'background_textures':
        # variant_str is the background_id, bg_kind is 'table' or 'tb'
        try:
            bg_id = int(variant_str)
        except ValueError:
            raise ValueError(f"Background variant must be an integer ID, got: {variant_str}")

        # Find matching entry in variants
        match = None
        for v in pert['variants']:
            if v['id'] == bg_id:
                match = v
                break
        if match is None:
            valid_ids = [v['id'] for v in pert['variants']]
            raise ValueError(f"Background ID {bg_id} is not official for this task. Official IDs: {valid_ids}")

        if bg_kind is None:
            bg_kind = match['kinds'][0]  # default to first available kind
        elif bg_kind not in match['kinds']:
            raise ValueError(f"Background kind '{bg_kind}' not available for ID {bg_id}. Available: {match['kinds']}")

        pert_task = pattern.format(task=task_entry['name'], kind=bg_kind, id=bg_id)
        variant_display = f"{bg_id} ({bg_kind})"

    elif perturbation == 'sensor_noise':
        try:
            noise_id = int(variant_str)
        except ValueError:
            raise ValueError(f"Noise variant must be an integer ID, got: {variant_str}")
        match = None
        for v in pert['variants']:
            if v['id'] == noise_id:
                match = v
                break
        if match is None:
            valid_ids = [v['id'] for v in pert['variants']]
            raise ValueError(f"Noise ID {noise_id} is not official for this task. Official IDs: {valid_ids}")
        pert_task = pattern.format(task=task_entry['name'], variant=noise_id)
        variant_display = f"{noise_id} (family={match['family']}, severity={match['severity']})"

    elif perturbation == 'camera_viewpoints':
        # variant_str is a camera spec: "h_v_s_yaw_pitch" (e.g., "50_0_100_0_0")
        specs = pert.get('specs', {})
        if variant_str not in specs:
            available = list(specs.keys())[:10]
            raise ValueError(
                f"Camera spec '{variant_str}' is not official for this task. "
                f"First 10 official specs: {available}"
            )
        params = specs[variant_str]
        pert_task = pattern.format(
            task=task_entry['name'],
            h=params['h'], v=params['v'], s=params['s'],
            yaw=params['yaw'], pitch=params['pitch']
        )
        variant_display = f"h={params['h']} v={params['v']} s={params['s']} yaw={params['yaw']} pitch={params['pitch']}"

    elif perturbation in ('light_conditions', 'language_instructions', 'robot_initial_states'):
        try:
            var_id = int(variant_str)
        except ValueError:
            raise ValueError(f"Variant must be an integer for {perturbation}, got: {variant_str}")
        if var_id not in pert['variants']:
            raise ValueError(f"Variant {var_id} is not official for this task. Official: {pert['variants'][:10]}...")
        pert_task = pattern.format(task=task_entry['name'], variant=var_id)
        variant_display = str(var_id)

    else:
        raise ValueError(f"Unsupported perturbation type: {perturbation}")

    return pert_task, pert, variant_display


def main():
    parser = argparse.ArgumentParser(description='Resolve experiment config for libero smoke test')
    parser.add_argument('--task-idx', type=int, help='Task index (0-9) in the JSON config')
    parser.add_argument('--task', type=str, help='Full base task name')
    parser.add_argument('--perturbation', type=str, required=True,
                        help='Perturbation type key (e.g., robot_initial_states, background_textures)')
    parser.add_argument('--variant', type=str, required=True,
                        help='Variant parameter value')
    parser.add_argument('--bg-kind', type=str, choices=['table', 'tb'],
                        help='For background_textures: "table" or "tb"')
    parser.add_argument('--dry-run', action='store_true',
                        help='Also print a human-readable summary to stderr')
    args = parser.parse_args()

    config = load_config()
    task_entry = find_task(config, task_idx=args.task_idx, task_name=args.task)

    try:
        pert_task, pert_info, variant_display = validate_and_build_variant(
            task_entry, args.perturbation, args.variant, bg_kind=args.bg_kind
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Print shell variable assignments (shell-safe quoting via shlex)
    lines = [
        ('SMOKE_PAIR_SUITE', config['suite']),
        ('SMOKE_PAIR_BASE_TASK', task_entry['name']),
        ('SMOKE_PAIR_CLEAN_LANGUAGE', task_entry['language']),
        ('SMOKE_PAIR_PERT_NAME', args.perturbation),
        ('SMOKE_PAIR_PERT_CATEGORY', pert_info['category']),
        ('SMOKE_PAIR_PERT_TASK', pert_task),
    ]
    for key, val in lines:
        print(f'{key}={shlex.quote(val)}')

    if args.dry_run:
        print(f'\n# --- Dry-run info ---', file=sys.stderr)
        print(f'# Task:          {task_entry["name"]}', file=sys.stderr)
        print(f'# Language:      {task_entry["language"]}', file=sys.stderr)
        print(f'# Perturbation:  {args.perturbation}', file=sys.stderr)
        print(f'# Variant:       {variant_display}', file=sys.stderr)
        print(f'# PERT_TASK:     {pert_task}', file=sys.stderr)
        print(f'# Suite:         {config["suite"]}', file=sys.stderr)


if __name__ == '__main__':
    main()
