#!/usr/bin/env python
"""Overlay the Curobo initial-alignment point cloud onto the original RGB image.

This is a standalone diagnostic script.  It does not modify any production code.
It reproduces the same depth back-projection / surface sampling as
``CuroboPlanner.plan``, then projects the sampled obstacle points back into the
canonical (flipped) agentview RGB image and saves an overlay PNG.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logging.disable(logging.CRITICAL)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.execute.curobo_planner import CuroboPlanner
from memory_system.execute.surface_obstacles import (
    filter_robot_points,
    select_surface_points,
)
from memory_system.geometry import camera_params as build_camera_params
from memory_system.geometry import world_to_pixel

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


def to_world(points_base, robot_base_pose):
    base_pos = np.asarray(robot_base_pose[:3], dtype=np.float64)
    base_rot = Rotation.from_quat(
        np.asarray(robot_base_pose[3:], dtype=np.float64)
    ).as_matrix()
    return (base_rot @ np.asarray(points_base, dtype=np.float64).T).T + base_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-state-index", type=int, default=1)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/initial_alignment_pointcloud_overlay.png")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--no-show", action="store_true")
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

        # Canonical RGB image: same orientation as the rollout video / policy input.
        rgb = np.flipud(obs["agentview_image"]).copy()
        height, width = rgb.shape[:2]
        cam = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, cam, flip_images=True)

        joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        robot_base_pose = np.concatenate(
            [env.robots[0].base_pos, env.robots[0].base_ori]
        ).astype(np.float32)

        planner = CuroboPlanner(joint_execution=False)
        raw_world = planner._points_from_depth(depth, cam)
        print(f"raw_depth_points={len(raw_world)}")

        base_pos = np.asarray(robot_base_pose[:3], dtype=np.float64)
        base_rot = Rotation.from_quat(
            np.asarray(robot_base_pose[3:], dtype=np.float64)
        ).as_matrix()
        raw_base = (base_rot.T @ (raw_world - base_pos).T).T

        import torch
        from curobo._src.geom.types import SceneCfg
        from curobo._src.state.state_joint import JointState

        motion_planner = planner._ensure_planner(SceneCfg())
        start = JointState.from_position(
            torch.as_tensor(
                joint_positions, device=planner.device, dtype=torch.float32
            ).reshape(1, -1),
            joint_names=motion_planner.joint_names,
        )
        start_kin = motion_planner.compute_kinematics(start)
        robot_spheres = start_kin.robot_spheres.detach().cpu().numpy()

        filtered_base = filter_robot_points(
            raw_base, robot_spheres, planner.robot_padding
        )
        sample_target = base_rot.T @ (DEFAULT_TARGET[:3] - base_pos)
        selection = select_surface_points(
            filtered_base,
            sample_target,
            max_points=args.max_points,
            voxel_size=planner.voxel_size,
        )
        selected_world = to_world(selection.points, robot_base_pose)
        print(f"selected_obstacle_points={len(selected_world)}")

        # Project the sampled obstacle points and the memory target back to pixels.
        selected_px = world_to_pixel(selected_world, cam, flip=True)
        target_px = world_to_pixel(DEFAULT_TARGET[:3].reshape(1, 3), cam, flip=True).reshape(-1)

        # Keep only points that fall inside the image.
        inside = (
            (selected_px[:, 0] >= 0) & (selected_px[:, 0] < height) &
            (selected_px[:, 1] >= 0) & (selected_px[:, 1] < width)
        )
        selected_px = selected_px[inside]
        print(f"selected_points_visible_in_image={len(selected_px)}")

        # Plot overlay.
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)
        ax.scatter(
            selected_px[:, 1], selected_px[:, 0],
            s=6, c="cyan", alpha=0.7, linewidths=0,
            label=f"sampled obstacle points ({len(selected_px)} visible)",
        )
        ax.scatter(
            target_px[1], target_px[0],
            s=160, c="lime", marker="*", edgecolors="black",
            label="memory target",
        )
        ax.set_title(
            f"init_state_index={args.init_state_index} point cloud overlay\n"
            f"cyan=sampled Curobo obstacle points, green star=memory target"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.axis("off")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"saved overlay to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
