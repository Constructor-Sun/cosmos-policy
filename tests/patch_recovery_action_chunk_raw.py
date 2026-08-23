#!/usr/bin/env python3
"""Patch skill_memory_test recovery_targets.pt action_chunk_raw to match old data.

Old recovery_targets.pt stores the first 16-step raw action chunk from each
segment start:

    raw_action_chunk(actions, start, start + 16)

The earlier memory_system implementation used:

    raw_action_chunk(actions, start, recovery_frame)

which produced zero-padded chunks when recovery_frame < start + 16.  This
script repairs only that field in the already-generated new artifact.

Run:

    python tests/patch_recovery_action_chunk_raw.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
MANIFEST = REPO_ROOT / "skill_memory_test/libero_10/segments_ready_fixed16.json"
RECOVERY_PT = REPO_ROOT / "skill_memory_test/libero_10/recovery_targets.pt"


def raw_action_chunk(acts: np.ndarray, start: int, end: int, chunk_size: int = 16) -> np.ndarray:
    start = max(0, int(start))
    end = min(int(end), acts.shape[0])
    raw = acts[start:end, :7].astype(np.float32)
    if len(raw) < chunk_size:
        pad_row = raw[-1] if len(raw) else np.zeros(7, dtype=np.float32)
        pad = np.tile(pad_row, (chunk_size - len(raw), 1))
        raw = np.concatenate([raw, pad], axis=0)
    return raw


def main() -> None:
    payload = torch.load(RECOVERY_PT, map_location="cpu", weights_only=False)
    manifest = json.loads(MANIFEST.read_text())

    starts: dict[tuple[str, str, int], int] = {}
    for record in manifest["records"]:
        if not record.get("valid"):
            continue
        for segment in record["segments"]:
            starts[(record["task_name"], record["demo_id"], int(segment["planner_step_id"]))] = int(
                segment["start"]
            )

    cache: dict[tuple[str, str], np.ndarray] = {}
    changed = 0
    for target in payload.get("targets", []):
        task = str(target["task_name"])
        demo = str(target["demo_id"])
        step = int(target["planner_step_id"])
        key = (task, demo, step)
        if key not in starts:
            continue
        start = starts[key]
        h5_key = (task, demo)
        if h5_key not in cache:
            h5_path = INPUT_DIR / f"{task}_demo.hdf5"
            with h5py.File(h5_path, "r") as handle:
                cache[h5_key] = handle["data"][demo]["actions"][:]
        acts = cache[h5_key]
        target["action_chunk_raw"] = raw_action_chunk(acts, start, start + 16)
        changed += 1

    torch.save(payload, RECOVERY_PT)
    print(f"Patched {changed} recovery targets in {RECOVERY_PT}")


if __name__ == "__main__":
    main()
