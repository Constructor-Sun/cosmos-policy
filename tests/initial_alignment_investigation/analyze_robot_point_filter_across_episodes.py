#!/usr/bin/env python
"""Analyze robot-arm point removal across all 20 robot-init episodes.

This is a standalone diagnostic script.  It does not modify any production code.
For each init state it computes:

- how many depth points belong to the robot (via MuJoCo segmentation);
- how many of those are removed by CuroboPlanner's filter_robot_points();
- how many robot points remain after filtering.

It then prints a table alongside the recorded success label.
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
from memory_system.execute.curobo_planner import CuroboPlanner
from memory_system.geometry import camera_params as build_camera_params
from memory_system.geometry import pixel_to_world

DEFAULT_TASK = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "_view_0_0_100_0_0_initstate_274"
)


def classify_robot(name: str | None) -> bool:
    if not name:
        return False
    name = name.lower()
    return (
        "robot" in name
        or "gripper" in name
        or name.startswith("panda")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--episodes-json", type=str,
                        default="scripts/experiments/libero10_robotinit_single/robotinit/robot_initial_states/episodes.json")
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/robot_point_filter_across_episodes.csv")
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

    if Path(args.episodes_json).exists():
        episodes = json.loads(Path(args.episodes_json).read_text())
        success = {e["init_state_index"]: e["success"] for e in episodes}
    else:
        success = {}

    env, _ = get_libero_env(
        task,
        "cosmos",
        resolution=256,
        camera_depths=[True, False],
    )

    planner = CuroboPlanner(joint_execution=False)
    try:
        import torch
        from curobo._src.geom.types import SceneCfg
        from curobo._src.state.state_joint import JointState

        env.reset()
        motion_planner = planner._ensure_planner(SceneCfg())

        rows = []
        for init_index in range(min(20, len(states))):
            obs = env.set_init_state(states[init_index])
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

            height, width = obs["agentview_image"].shape[:2]
            cam = build_camera_params(env.sim, "agentview", height, width)
            depth = _make_main_depth(obs, cam, flip_images=True)

            # Segmentation oracle.
            seg_render, _ = env.sim.render(
                height, width, camera_name="agentview", depth=True, segmentation=True
            )
            seg_canon = np.flipud(seg_render)

            rows_idx, cols_idx = np.meshgrid(
                np.arange(height), np.arange(width), indexing="ij"
            )
            pixels = np.stack([rows_idx.ravel(), cols_idx.ravel()], axis=-1)
            valid = (np.isfinite(depth) & (depth > 0.0)).ravel()
            pixels = pixels[valid]
            raw_world = pixel_to_world(pixels, depth, cam)

            seg_flat = seg_canon.reshape(-1, 2)[valid]
            names = []
            for t, oid in zip(seg_flat[:, 0], seg_flat[:, 1]):
                t = int(t)
                oid = int(oid)
                if t == 5:
                    names.append(env.sim.model.geom_id2name(oid))
                elif t == 6:
                    names.append(
                        env.sim.model.site_id2name(oid)
                        if hasattr(env.sim.model, "site_id2name")
                        else None
                    )
                else:
                    names.append(None)

            robot_mask = np.array([classify_robot(n) for n in names], dtype=bool)
            robot_points_total = int(robot_mask.sum())

            # Filter in the same robot-base frame as CuroboPlanner.plan().
            base_pos = np.asarray(env.robots[0].base_pos, dtype=np.float64)
            base_ori = np.asarray(env.robots[0].base_ori, dtype=np.float64)
            base_rot = Rotation.from_quat(base_ori).as_matrix()
            raw_base = (base_rot.T @ (raw_world - base_pos).T).T

            joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
            start = JointState.from_position(
                torch.as_tensor(
                    joint_positions, device=planner.device, dtype=torch.float32
                ).reshape(1, -1),
                joint_names=motion_planner.joint_names,
            )
            start_kin = motion_planner.compute_kinematics(start)
            robot_spheres = start_kin.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
            clearance = np.linalg.norm(
                raw_base[:, None, :] - robot_spheres[None, :, :3], axis=2
            ) - robot_spheres[None, :, 3]
            keep = np.all(clearance > float(planner.robot_padding), axis=1)

            robot_removed = int((~keep & robot_mask).sum())
            robot_remaining = int((keep & robot_mask).sum())

            print(
                f"init={init_index:2d} success={str(success.get(init_index, '?')):5s} "
                f"robot_total={robot_points_total:5d} robot_removed={robot_removed:5d} "
                f"robot_remaining={robot_remaining:5d}"
            )
            rows.append(
                {
                    "init_state_index": init_index,
                    "success": success.get(init_index),
                    "robot_points_total": robot_points_total,
                    "robot_removed": robot_removed,
                    "robot_remaining": robot_remaining,
                }
            )

        # Save CSV.
        import csv

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "init_state_index",
                    "success",
                    "robot_points_total",
                    "robot_removed",
                    "robot_remaining",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved CSV to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
