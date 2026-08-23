#!/usr/bin/env python3
"""Generate a complete LIBERO-10 memory into skill_memory_test using memory_system.

This is the full offline validation pipeline:

1. Label skill segments from success HDF5 demos.
2. Add ready/terminal boundaries.
3. Build phase/wrist target artifacts.
4. Build recovery / feasible-recovery / per-demo skill memory.

Run from the repository root:

    /data1/liu/miniconda3/envs/cosmospolicy/bin/python \
        tests/generate_memory_system_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

from memory_system.offline.build_recovery import build_recovery
from memory_system.offline.build_targets import build_phase_targets
from memory_system.offline.label_boundaries import enrich_manifest
from memory_system.offline.label_segments import label_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
TASK_CONFIG = REPO_ROOT / "configs/libero10_experiment_tasks.json"
OUT_DIR = REPO_ROOT / "skill_memory_test" / "libero_10"

DEVICE = "cuda:0"

MAX_PER_TASK = 10
RESOLUTION = 256
STABLE_FRAMES = 3
LIFT_THRESHOLD = 0.02
FALLBACK_LENGTH = 16
MIN_RUN = 3
MAX_SUFFIX = 64


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1/4: labeling skill segments ...", flush=True)
    segments_json = OUT_DIR / "segments.json"
    label_manifest(
        input_dir=INPUT_DIR,
        output=segments_json,
        task_config=TASK_CONFIG,
        max_per_task=MAX_PER_TASK,
        stable_frames=STABLE_FRAMES,
        lift_threshold=LIFT_THRESHOLD,
    )

    print("Step 2/4: adding ready/terminal boundaries ...", flush=True)
    manifest = json.loads(segments_json.read_text())
    enrich_manifest(
        manifest,
        input_dir=INPUT_DIR,
        boundary_mode="fixed",
        fallback_length=FALLBACK_LENGTH,
        min_run=MIN_RUN,
        max_suffix=MAX_SUFFIX,
    )
    ready_manifest = OUT_DIR / "segments_ready_fixed16.json"
    ready_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print("Step 3/4: building phase/wrist targets ...", flush=True)
    build_phase_targets(
        input_dir=INPUT_DIR,
        segments_manifest=ready_manifest,
        output=OUT_DIR / "phase_targets.pt",
        wrist_completion_output=OUT_DIR / "wrist_completion_targets.pt",
        wrist_feasible_output=OUT_DIR / "feasible_wrist_targets.pt",
        max_per_task=MAX_PER_TASK,
        resolution=RESOLUTION,
    )

    print("Step 4/4: building recovery/skill memory ...", flush=True)
    build_recovery(
        input_dir=INPUT_DIR,
        segments_manifest=ready_manifest,
        output_dir=OUT_DIR,
        feasible_recovery_output=OUT_DIR / "feasible_recovery_targets.pt",
        max_per_task=MAX_PER_TASK,
        device=DEVICE,
    )

    print(f"\nDone. Generated memory under: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
