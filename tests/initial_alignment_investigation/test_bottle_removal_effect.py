#!/usr/bin/env python
"""Test whether removing bottle points changes the Curobo plan length.

This is a standalone diagnostic script.  It does not modify any production code.
It runs the same sequential 0,1,2 setup, then at init_state_index=3 plans twice:
once with the original depth and once with the bottle depth points removed.
"""

from __future__ import annotations

import argparse
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


def plan_length(plan):
    wp = np.asarray(plan.controller.waypoints, dtype=np.float64)
    pos = wp[:, :3]
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()), wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--target-init", type=int, default=3)
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
        # Run earlier init states to reach the same planner state as the long-path run.
        for init_index in range(args.target_init):
            env.reset()
            obs = env.set_init_state(states[init_index])
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
            h, w = obs["agentview_image"].shape[:2]
            cam = build_camera_params(env.sim, "agentview", h, w)
            depth = _make_main_depth(obs, cam, flip_images=True)
            current_ee = np.concatenate([
                obs["robot0_eef_pos"],
                Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
            ]).astype(np.float32)
            jpos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
            base = np.concatenate([env.robots[0].base_pos, env.robots[0].base_ori]).astype(np.float32)
            plan = planner.plan(
                current_ee_states=current_ee,
                target_ee_states=DEFAULT_TARGET,
                depth=depth,
                camera_params=cam,
                joint_positions=jpos,
                robot_base_pose=base,
            )
            length, _ = plan_length(plan)
            print(f"warmup init={init_index} path_length={length:.4f}", flush=True)

        # Target init: first original, then bottle removed.
        env.reset()
        obs = env.set_init_state(states[args.target_init])
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
        h, w = obs["agentview_image"].shape[:2]
        cam = build_camera_params(env.sim, "agentview", h, w)
        depth = _make_main_depth(obs, cam, flip_images=True)
        current_ee = np.concatenate([
            obs["robot0_eef_pos"],
            Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
        ]).astype(np.float32)
        jpos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        base = np.concatenate([env.robots[0].base_pos, env.robots[0].base_ori]).astype(np.float32)

        # Build bottle and robot masks in canonical space.
        seg_render, _ = env.sim.render(h, w, camera_name="agentview", depth=True, segmentation=True)
        seg_canon = np.flipud(seg_render)
        bottle_mask = np.zeros((h, w), dtype=bool)
        robot_mask = np.zeros((h, w), dtype=bool)
        for yy in range(h):
            for xx in range(w):
                t, oid = seg_canon[yy, xx]
                if int(t) == 5:
                    name = env.sim.model.geom_id2name(int(oid))
                    if not name:
                        continue
                    lower = name.lower()
                    if "wine_bottle" in lower:
                        bottle_mask[yy, xx] = True
                    if ("robot" in lower or "gripper" in lower or "panda" in lower
                            or "mount" in lower):
                        robot_mask[yy, xx] = True
        print(f"bottle_mask_pixels={int(bottle_mask.sum())}", flush=True)
        print(f"robot_mask_pixels={int(robot_mask.sum())}", flush=True)

        # Original plan.
        plan_orig = planner.plan(
            current_ee_states=current_ee,
            target_ee_states=DEFAULT_TARGET,
            depth=depth,
            camera_params=cam,
            joint_positions=jpos,
            robot_base_pose=base,
        )
        len_orig, _ = plan_length(plan_orig)
        print(f"original init={args.target_init} path_length={len_orig:.4f}", flush=True)

        # Bottle-removed plan.
        depth_no_bottle = depth.copy()
        depth_no_bottle[bottle_mask] = 0.0
        plan_mod = planner.plan(
            current_ee_states=current_ee,
            target_ee_states=DEFAULT_TARGET,
            depth=depth_no_bottle,
            camera_params=cam,
            joint_positions=jpos,
            robot_base_pose=base,
        )
        len_mod, _ = plan_length(plan_mod)
        print(f"bottle_removed init={args.target_init} path_length={len_mod:.4f}", flush=True)

        # Robot-removed plan.
        depth_no_robot = depth.copy()
        depth_no_robot[robot_mask] = 0.0
        plan_robot = planner.plan(
            current_ee_states=current_ee,
            target_ee_states=DEFAULT_TARGET,
            depth=depth_no_robot,
            camera_params=cam,
            joint_positions=jpos,
            robot_base_pose=base,
        )
        len_robot, _ = plan_length(plan_robot)
        print(f"robot_removed init={args.target_init} path_length={len_robot:.4f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
