#!/usr/bin/env python
"""Visualize whether robot-arm points are fully removed before Curobo planning.

This is a standalone diagnostic script.  It does not modify any production code.
It projects the raw depth point cloud, the robot-removed points, and the
remaining filtered point cloud back onto the original agentview image.
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


def project_to_image(points_world, cam, height, width, max_points=20000):
    rng = np.random.default_rng(0)
    if len(points_world) > max_points:
        idx = rng.choice(len(points_world), max_points, replace=False)
        points_world = points_world[idx]
    px = world_to_pixel(points_world, cam, flip=True)
    inside = (
        (px[:, 0] >= 0) & (px[:, 0] < height) &
        (px[:, 1] >= 0) & (px[:, 1] < width)
    )
    return px[inside]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-state-index", type=int, default=1)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/robot_point_filter.png")
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

        # Recompute the same keep/remove mask used by filter_robot_points().
        spheres = np.asarray(robot_spheres, dtype=np.float64).reshape(-1, 4)
        clearance = np.linalg.norm(
            raw_base[:, None, :] - spheres[None, :, :3], axis=2
        ) - spheres[None, :, 3]
        keep = np.all(clearance > float(planner.robot_padding), axis=1)
        removed_base = raw_base[~keep]
        filtered_base = raw_base[keep]

        removed_world = to_world(removed_base, robot_base_pose)
        filtered_world = to_world(filtered_base, robot_base_pose)

        raw_px = project_to_image(raw_world, cam, height, width, max_points=20000)
        removed_px = project_to_image(removed_world, cam, height, width, max_points=20000)
        filtered_px = project_to_image(filtered_world, cam, height, width, max_points=20000)

        print(f"raw_points={len(raw_world)}")
        print(f"removed_robot_points={len(removed_base)}")
        print(f"remaining_points_after_filter={len(filtered_base)}")
        print(f"visible_raw_px={len(raw_px)}")
        print(f"visible_removed_px={len(removed_px)}")
        print(f"visible_remaining_px={len(filtered_px)}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        # Left: before filtering.
        ax = axes[0]
        ax.imshow(rgb)
        ax.scatter(raw_px[:, 1], raw_px[:, 0], s=1, c="gray", alpha=0.3,
                   linewidths=0, label="raw depth points")
        ax.scatter(removed_px[:, 1], removed_px[:, 0], s=3, c="red", alpha=0.8,
                   linewidths=0, label="removed as robot points")
        ax.set_title(f"Before robot filter\nraw={len(raw_world)}, removed={len(removed_base)}")
        ax.legend(loc="upper right", fontsize=8, markerscale=3)
        ax.axis("off")

        # Right: after filtering.
        ax = axes[1]
        ax.imshow(rgb)
        ax.scatter(removed_px[:, 1], removed_px[:, 0], s=1, c="red", alpha=0.15,
                   linewidths=0, label="removed as robot points")
        ax.scatter(filtered_px[:, 1], filtered_px[:, 0], s=3, c="cyan", alpha=0.8,
                   linewidths=0, label="remaining points (sent to planner)")
        ax.scatter(
            *project_to_image(DEFAULT_TARGET[:3].reshape(1, 3), cam, height, width, max_points=1)[:, ::-1].T,
            s=160, c="lime", marker="*", edgecolors="black", label="memory target",
        )
        ax.set_title(f"After robot filter\nremaining={len(filtered_base)}")
        ax.legend(loc="upper right", fontsize=8, markerscale=3)
        ax.axis("off")

        fig.suptitle(
            f"init_state_index={args.init_state_index} robot-arm point removal check",
            fontsize=14,
        )
        fig.tight_layout()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"saved figure to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
