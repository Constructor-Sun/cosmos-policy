#!/usr/bin/env python3
"""Full offline migration verification: compare old skill_memory vs new skill_memory_test.

Run after generating the new memory with:

    python tests/generate_memory_system_test.py

Then verify all artifacts with:

    python tests/compare_memory_system_offline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = REPO_ROOT / "skill_memory" / "libero_10"
NEW_DIR = REPO_ROOT / "skill_memory_test" / "libero_10"

TORCH_ARTIFACTS = [
    "phase_targets.pt",
    "wrist_completion_targets.pt",
    "feasible_wrist_targets.pt",
    "recovery_targets.pt",
    "feasible_recovery_targets.pt",
]

JSON_ARTIFACTS = [
    "segments.json",
    "segments_ready_fixed16.json",
]


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _values_equal(left, right, path: str) -> list[str]:
    errors: list[str] = []
    left = _to_numpy(left)
    right = _to_numpy(right)

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        if left_arr.shape != right_arr.shape:
            errors.append(f"{path}: shape {left_arr.shape} != {right_arr.shape}")
        elif not np.array_equal(left_arr, right_arr):
            errors.append(f"{path}: array values differ")
        return errors

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            errors.append(f"{path}: length {len(left)} != {len(right)}")
            return errors
        for index, (lv, rv) in enumerate(zip(left, right)):
            errors.extend(_values_equal(lv, rv, f"{path}[{index}]"))
        return errors

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            errors.append(f"{path}: keys {sorted(left.keys())} != {sorted(right.keys())}")
            return errors
        for key in sorted(left.keys()):
            errors.extend(_values_equal(left[key], right[key], f"{path}.{key}"))
        return errors

    if isinstance(left, (int, float, str, bool)) or left is None:
        if type(left) is not type(right) or left != right:
            errors.append(f"{path}: {left!r} != {right!r}")
        return errors

    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def compare_json(old_path: Path, new_path: Path) -> list[str]:
    errors: list[str] = []
    if not old_path.exists():
        errors.append(f"missing old {old_path.name}")
        return errors
    if not new_path.exists():
        errors.append(f"missing new {new_path.name}")
        return errors
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    errors.extend(_values_equal(old, new, old_path.name))
    return errors


def compare_torch(old_path: Path, new_path: Path) -> list[str]:
    errors: list[str] = []
    if not old_path.exists():
        errors.append(f"missing old {old_path.name}")
        return errors
    if not new_path.exists():
        errors.append(f"missing new {new_path.name}")
        return errors
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    errors.extend(_values_equal(old, new, old_path.name))
    return errors


def compare_skill_memory_demos() -> list[str]:
    errors: list[str] = []
    old_files = sorted(OLD_DIR.glob("skill_memory_*.pt"))
    new_files = sorted(NEW_DIR.glob("skill_memory_*.pt"))

    def index_by_demo(paths: list[Path]) -> dict[tuple[str, str], Path]:
        result: dict[tuple[str, str], Path] = {}
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            key = (str(payload.get("task_name")), str(payload.get("demo_id")))
            result[key] = path
        return result

    old_by_demo = index_by_demo(old_files)
    new_by_demo = index_by_demo(new_files)

    if set(old_by_demo) != set(new_by_demo):
        errors.append(
            f"skill_memory demo keys differ: old={len(old_by_demo)} new={len(new_by_demo)}"
        )
        return errors

    for key in sorted(old_by_demo):
        old_path = old_by_demo[key]
        new_path = new_by_demo[key]
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        new = torch.load(new_path, map_location="cpu", weights_only=False)
        errors.extend(_values_equal(old, new, f"skill_memory/{key[0]}/{key[1]}"))

    return errors


def main() -> int:
    print(f"Comparing old: {OLD_DIR}")
    print(f"Comparing new: {NEW_DIR}")
    all_errors: list[str] = []

    for name in JSON_ARTIFACTS:
        print(f"  [json] {name}")
        all_errors.extend(compare_json(OLD_DIR / name, NEW_DIR / name))

    for name in TORCH_ARTIFACTS:
        print(f"  [torch] {name}")
        all_errors.extend(compare_torch(OLD_DIR / name, NEW_DIR / name))

    print("  [torch] skill_memory_*.pt")
    all_errors.extend(compare_skill_memory_demos())

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} difference(s)", file=sys.stderr)
        for error in all_errors[:200]:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("\nOK: all offline artifacts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
