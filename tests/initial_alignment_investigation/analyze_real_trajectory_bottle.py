#!/usr/bin/env python
"""Analyze real initial-alignment trajectories relative to the bottle.

This is a standalone diagnostic script.  It does not modify any production code.
It reads [INIT_ALIGN_DEBUG] logs and compares the real EEF trajectory and the
logged waypoint path against the bottle position in the scene.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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

DEFAULT_TASK = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "_view_0_0_100_0_0_initstate_274"
)


def parse_debug_records(log_path: Path, episode_index: int) -> list[dict]:
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        marker = "[INIT_ALIGN_DEBUG] "
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if int(data.get("episode", -1)) == episode_index:
            records.append(data)
    records.sort(key=lambda r: (int(r["t"]), int(r["correction_step_index"])))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--init-state-index", type=int, required=True)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--episodes-json", type=str,
                        default="scripts/experiments/libero10_robotinit_single/robotinit/robot_initial_states/episodes.json")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    records = parse_debug_records(log_path, args.episode - 1)
    print(f"episode={args.episode} init_state_index={args.init_state_index} records={len(records)}")

    actual_path = [np.asarray(records[0]["eef_before"], dtype=np.float64)]
    for r in records:
        actual_path.append(np.asarray(r["eef_after"], dtype=np.float64))
    actual_path = np.asarray(actual_path)

    waypoint_by_index = {}
    for r in records:
        idx = r.get("waypoint_index")
        wp = r.get("waypoint")
        if idx is not None and wp is not None:
            waypoint_by_index[int(idx)] = np.asarray(wp, dtype=np.float64)
    waypoints = (
        np.stack([waypoint_by_index[i] for i in sorted(waypoint_by_index)])
        if waypoint_by_index
        else np.zeros((0, 6), dtype=np.float64)
    )

    # Get scene anchors.
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

        body_names = list(env.sim.model.body_names)
        def body_pos(name):
            if name in body_names:
                return env.sim.data.body_xpos[env.sim.model.body_name2id(name)].copy()
            return None

        bottle = body_pos("wine_bottle_1_main")
        if bottle is None:
            # Try common bottle body names.
            for name in body_names:
                if "wine_bottle" in name and name.endswith("_main"):
                    bottle = env.sim.data.body_xpos[env.sim.model.body_name2id(name)].copy()
                    break
        bowl = body_pos("akita_black_bowl_1_main")
        cabinet = body_pos("white_cabinet_1_main")
        print(f"bottle={np.round(bottle, 3) if bottle is not None else None}")
        print(f"bowl={np.round(bowl, 3) if bowl is not None else None}")
        print(f"cabinet={np.round(cabinet, 3) if cabinet is not None else None}")

        def path_metrics(path, label):
            if len(path) == 0:
                print(f"{label}: empty")
                return
            pos = path[:, :3]
            if bottle is not None:
                d3d = np.linalg.norm(pos - bottle, axis=1)
                dxy = np.linalg.norm(pos[:, :2] - bottle[:2], axis=1)
                over = pos[:, 2] > bottle[2] + 0.02
                print(f"{label}:")
                print(f"  min_d3d_to_bottle={d3d.min():.4f}")
                print(f"  min_dxy_to_bottle={dxy.min():.4f}")
                print(f"  max_z_rel_bottle={(pos[:, 2] - bottle[2]).max():.4f}")
                print(f"  n_over_bottle={int(over.sum())}/{len(pos)}")
                print(f"  end_d3d_to_bottle={d3d[-1]:.4f} end_dxy_to_bottle={dxy[-1]:.4f} end_z_rel_bottle={pos[-1,2]-bottle[2]:.4f}")
            else:
                print(f"{label}: bottle not found")

        path_metrics(waypoints, "logged_waypoint_path")
        path_metrics(actual_path, "actual_eef_path")
    finally:
        env.close()


if __name__ == "__main__":
    main()
