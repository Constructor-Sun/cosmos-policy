#!/usr/bin/env python3
"""Verify recovery_targets.pt / feasible_recovery_targets.pt match old after patch.

Run:

    python tests/verify_recovery_patch.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = REPO_ROOT / "skill_memory/libero_10"
NEW_DIR = REPO_ROOT / "skill_memory_test/libero_10"

ARTIFACTS = [
    "recovery_targets.pt",
    "feasible_recovery_targets.pt",
]


def _norm(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _equal(a, b) -> bool:
    a = _norm(a)
    b = _norm(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a)
        b = np.asarray(b)
        return a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


COMMON_FIELDS = [
    "task_name",
    "demo_id",
    "planner_step_id",
    "skill",
    "arguments",
    "ee_states",
    "action_scale",
    "action_chunk_raw",
]

RECOVERY_FIELDS = COMMON_FIELDS + ["recovery_frame"]
FEASIBLE_FIELDS = COMMON_FIELDS + ["ready_frame"]


def main() -> int:
    failed = False
    checks = [
        ("recovery_targets.pt", RECOVERY_FIELDS),
        ("feasible_recovery_targets.pt", FEASIBLE_FIELDS),
    ]
    for name, fields in checks:
        old_path = OLD_DIR / name
        new_path = NEW_DIR / name
        if not old_path.exists() or not new_path.exists():
            print(f"{name}: missing file")
            failed = True
            continue
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        new = torch.load(new_path, map_location="cpu", weights_only=False)
        mismatches = 0
        for a, b in zip(old.get("targets", []), new.get("targets", [])):
            for field in fields:
                if not _equal(a.get(field), b.get(field)):
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"{name}: {field} differs")
        if mismatches:
            print(f"{name}: DIFF ({mismatches} field mismatches)")
            failed = True
        else:
            print(f"{name}: OK (action/pose fields match, VAE fields skipped)")

    if failed:
        print("\nFAILED")
        return 1
    print("\nRecovery action/pose fields match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
