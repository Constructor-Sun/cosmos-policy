#!/usr/bin/env python
"""Compare full initial-alignment planned path lengths from logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_full_plans(log_path: Path):
    plans = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        marker = "[INIT_ALIGN_FULL_PLAN] "
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        plans[int(data["episode"])] = np.asarray(data["waypoints"], dtype=np.float64)
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, required=True)
    args = parser.parse_args()

    plans = load_full_plans(Path(args.log_path))
    print(f"found {len(plans)} full-plan records")

    for ep in sorted(plans):
        wp = plans[ep]
        pos = wp[:, :3]
        diffs = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        total = float(diffs.sum())
        straight = float(np.linalg.norm(pos[-1] - pos[0]))
        max_step = float(diffs.max())
        print(
            f"episode={ep + 1} waypoints={len(wp)} "
            f"path_length={total:.4f} straight={straight:.4f} "
            f"detour_ratio={total / max(straight, 1e-9):.3f} "
            f"max_step={max_step:.4f}"
        )


if __name__ == "__main__":
    main()
