#!/usr/bin/env python
"""Print the geom-name distribution of the 'other' point-cloud category.

This is a standalone diagnostic script.  It does not modify any production code.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.disable(logging.CRITICAL)

from memory_system.offline.build_targets import patch_numpy2_segmentation
patch_numpy2_segmentation()

from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.geometry import camera_params as build_camera_params
from memory_system.geometry import pixel_to_world

DEFAULT_TASK = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "_view_0_0_100_0_0_initstate_274"
)


def classify(name: str | None) -> str:
    if not name:
        return "other"
    name = name.lower()
    if "robot" in name or "gripper" in name or name.startswith("panda"):
        return "robot"
    if "bowl" in name:
        return "bowl"
    if "cabinet" in name:
        return "cabinet"
    if "table" in name:
        return "table"
    if "floor" in name or "wall" in name or "visual" in name and "robot" not in name:
        return "floor"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-state-index", type=int, default=1)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_10"](
        category_value="Robot Initial States"
    )
    matches = [
        (i, t) for i, t in enumerate(suite.tasks) if t.name == args.task_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"task not found or ambiguous: {args.task_name}")
    task_index, task = matches[0]
    states = suite.get_task_init_states(task_index)

    env, _ = get_libero_env(
        task,
        "cosmos",
        resolution=256,
        camera_depths=[True, False],
    )

    try:
        env.reset()
        obs = env.set_init_state(states[args.init_state_index])
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        height, width = obs["agentview_image"].shape[:2]
        cam = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, cam, flip_images=True)

        seg_render, _ = env.sim.render(
            height, width, camera_name="agentview", depth=True, segmentation=True
        )
        seg_canon = np.flipud(seg_render)

        rows, cols = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        pixels = np.stack([rows.ravel(), cols.ravel()], axis=-1)
        valid = (np.isfinite(depth) & (depth > 0.0)).ravel()
        pixels = pixels[valid]
        _ = pixel_to_world(pixels, depth, cam)

        seg_flat = seg_canon.reshape(-1, 2)[valid]
        name_counter = Counter()
        category_counter = Counter()
        for t, oid in zip(seg_flat[:, 0], seg_flat[:, 1]):
            t = int(t)
            oid = int(oid)
            if t == 5:
                name = env.sim.model.geom_id2name(oid)
            elif t == 6:
                name = env.sim.model.site_id2name(oid) if hasattr(env.sim.model, "site_id2name") else None
            else:
                name = None
            name_counter[name] += 1
            category_counter[classify(name)] += 1

        print("category counts:")
        for cat, count in category_counter.most_common():
            print(f"  {cat}: {count}")

        print("\ntop 40 geom names overall:")
        for name, count in name_counter.most_common(40):
            print(f"  {count:6d}  {name}")

        print("\ntop 30 'other' geom names:")
        other_counter = Counter({
            name: count for name, count in name_counter.items() if classify(name) == "other"
        })
        for name, count in other_counter.most_common(30):
            print(f"  {count:6d}  {name}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
