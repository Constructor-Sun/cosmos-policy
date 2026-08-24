"""Test-only harness for Stage 1 (depth correctness) tests.

NOT part of test-time / production code.  Formal runtime code must never
import this module: the memory system consumes metric depth + calibrated
camera parameters only.  This module exists solely so the Stage-1 tests can
create LIBERO environments with the main camera's depth enabled and measure
the depth observation chain (obs key -> normalized depth -> metric depth).

Run with the `cosmospolicy` conda env:

    conda activate cosmospolicy
    cd <cosmos-policy repo>
    pytest tests/stage1_depth/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/tests/stage1_depth -> <repo>
LIBERO_PLUS = REPO_ROOT.parent / "LIBERO-plus"

for _p in (str(REPO_ROOT), str(LIBERO_PLUS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESOLUTION = 256

# Two representative LIBERO-10 tasks (kitchen + living room) with exact bddl files.
TASKS = (
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
)


def load_task_names() -> list[str]:
    """Task names used by the memory system (from the segments manifest)."""
    import json

    manifest_path = REPO_ROOT / "skill_memory/libero_10/segments_ready_fixed16.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    return sorted(set(r["task_name"] for r in manifest["records"]))


def resolve_bddl(task_name: str) -> Path:
    """Resolve the exact LIBERO-10 bddl file for a task (test-only)."""
    candidate = LIBERO_PLUS / "libero/libero/bddl_files/libero_10" / f"{task_name}.bddl"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def create_env(task_name: str, resolution: int = RESOLUTION):
    """LIBERO env with main-camera depth enabled, wrist camera RGB only."""
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(resolve_bddl(task_name)),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=[True, False],  # agentview depth on, eye_in_hand stays RGB
    )
    return env
