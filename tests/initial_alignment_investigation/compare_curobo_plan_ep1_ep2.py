#!/usr/bin/env python
"""Compare CuroboPlanner planning behavior for episode 1 vs episode 2.

This is a standalone diagnostic script.  It does not modify any production code.
It prints the planner logs (surface counts, backoff, conflicts) and the
resulting full waypoint path length for two init states.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

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
    parser.add_argument("--init-indices", type=int, nargs="+", default=[0, 1])
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

    planner = CuroboPlanner(joint_execution=False)
    try:
        env.reset()
        for init_index in args.init_indices:
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

            print(f"\n===== init_state_index={init_index} =====", flush=True)
            print(f"joint_positions={np.round(joint_positions, 3)}", flush=True)
            plan = planner.plan(
                current_ee_states=current_ee,
                target_ee_states=DEFAULT_TARGET,
                depth=depth,
                camera_params=cam,
                joint_positions=joint_positions,
                robot_base_pose=robot_base_pose,
            )
            if plan is None or plan.controller is None:
                print("plan=None", flush=True)
                continue
            wp = np.asarray(plan.controller.waypoints, dtype=np.float64)
            pos = wp[:, :3]
            path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
            straight = float(np.linalg.norm(pos[-1] - pos[0]))
            print(
                f"correction_steps={plan.correction_steps} waypoints={len(wp)} "
                f"path_length={path_len:.4f} straight={straight:.4f} "
                f"last_dpos={np.linalg.norm(pos[-1] - DEFAULT_TARGET[:3]):.4f}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
