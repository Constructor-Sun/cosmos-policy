#!/usr/bin/env python
"""Check whether the straight start->target path would hit the bottle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def point_segment_distance(p, a, b):
    ab = b - a
    t = 0.0 if np.allclose(ab, 0) else np.dot(p - a, ab) / np.dot(ab, ab)
    t = float(np.clip(t, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


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
    bottles = {
        0: np.array([-0.151, 0.060, 0.899]),
        1: np.array([-0.171, 0.057, 0.899]),
        2: np.array([-0.171, 0.057, 0.899]),
    }

    for ep in sorted(plans):
        wp = plans[ep]
        start = wp[0, :3]
        end = wp[-1, :3]
        bottle = bottles.get(ep)
        if bottle is None:
            continue
        d3d = point_segment_distance(bottle, start, end)
        # horizontal distance along the straight segment (ignore z)
        start_xy = start[:2]
        end_xy = end[:2]
        dxy = point_segment_distance(bottle[:2], start_xy, end_xy)
        print(
            f"episode={ep + 1} start={np.round(start, 3)} end={np.round(end, 3)} "
            f"bottle={np.round(bottle, 3)} "
            f"straight_min_d3d_to_bottle={d3d:.4f} straight_min_dxy_to_bottle={dxy:.4f}"
        )


if __name__ == "__main__":
    main()
