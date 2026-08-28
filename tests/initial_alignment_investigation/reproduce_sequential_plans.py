#!/usr/bin/env python
"""Reproduce sequential initial-alignment planning like the real rollout.

This is a standalone diagnostic script.  It does not modify any production code.
It mimics the real run by reusing one environment and one CuroboPlanner across
episodes, and by calling env.reset() before each set_init_state().
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.execute.curobo_planner import CuroboPlanner
from memory_system.geometry import camera_params as build_camera_params

DEFAULT_TASK = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "_view_0_0_100_0_0_initstate_274"
)
DEFAULT_TARGET = np.array(
    [
        -0.05882002040743828,
        -0.08573316782712936,
        0.9660174250602722,
        2.6380581855773926,
        -1.9302700757980347,
        0.28322678804397583,
    ],
    dtype=np.float64,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--init-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/sequential_plan_lengths.txt")
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

    planner = CuroboPlanner(
        joint_execution=False,
        enable_urdf_robot_filter=os.environ.get(
            "COSMOS_URDF_ROBOT_FILTER", ""
        ).lower() in {"1", "true", "yes"},
    )
    lines = []
    try:
        for init_index in args.init_indices:
            # Mimic run_episode: reset the env before each set_init_state.
            env.reset()
            obs = env.set_init_state(states[init_index])
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

            height, width = obs["agentview_image"].shape[:2]
            cam = build_camera_params(env.sim, "agentview", height, width)
            depth = _make_main_depth(obs, cam, flip_images=True)
            current_ee = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
                ]
            ).astype(np.float32)
            joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
            robot_base_pose = np.concatenate(
                [env.robots[0].base_pos, env.robots[0].base_ori]
            ).astype(np.float32)

            gripper_qpos = obs.get("robot0_gripper_qpos")
            plan = planner.plan(
                current_ee_states=current_ee,
                target_ee_states=DEFAULT_TARGET,
                depth=depth,
                camera_params=cam,
                joint_positions=joint_positions,
                gripper_joint_positions=(
                    np.asarray(gripper_qpos, dtype=np.float32)
                    if gripper_qpos is not None
                    else None
                ),
                robot_base_pose=robot_base_pose,
            )
            if plan is None or plan.controller is None:
                msg = f"init={init_index} plan=None"
                print(msg, flush=True)
                lines.append(msg)
                continue

            wp = np.asarray(plan.controller.waypoints, dtype=np.float64)
            pos = wp[:, :3]
            path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
            straight = float(np.linalg.norm(pos[-1] - pos[0]))
            msg = (
                f"init={init_index} waypoints={len(wp)} "
                f"path_length={path_len:.4f} straight={straight:.4f} "
                f"last_dpos={np.linalg.norm(pos[-1] - DEFAULT_TARGET[:3]):.4f}"
            )
            print(msg, flush=True)
            lines.append(msg)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"saved to {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
