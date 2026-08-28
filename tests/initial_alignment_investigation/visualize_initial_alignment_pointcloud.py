#!/usr/bin/env python
"""Visualize the RGB-D point cloud used by CuroboPlanner initial alignment.

This is a standalone diagnostic script.  It does not modify any production code.
It reproduces the depth back-projection and surface sampling used by
``CuroboPlanner.plan`` for one episode and saves a point-cloud plot.
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
    """Convert points from cuRobo/robot-base frame back to LIBERO world frame."""
    base_pos = np.asarray(robot_base_pose[:3], dtype=np.float64)
    base_rot = Rotation.from_quat(np.asarray(robot_base_pose[3:], dtype=np.float64)).as_matrix()
    return (base_rot @ np.asarray(points_base, dtype=np.float64).T).T + base_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-state-index", type=int, default=1)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/initial_alignment_pointcloud.png")
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

        body_names = list(env.sim.model.body_names)
        anchor_names = {
            "bowl": "akita_black_bowl_1_main",
            "cabinet": "white_cabinet_1_main",
        }
        anchors = {}
        for label, body_name in anchor_names.items():
            if body_name in body_names:
                body_id = env.sim.model.body_name2id(body_name)
                anchors[label] = env.sim.data.body_xpos[body_id].copy()

        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        height, width = obs["agentview_image"].shape[:2]
        cam = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, cam, flip_images=True)

        joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        robot_base_pose = np.concatenate(
            [env.robots[0].base_pos, env.robots[0].base_ori]
        ).astype(np.float32)

        planner = CuroboPlanner(joint_execution=False)
        raw_world = planner._points_from_depth(depth, cam)
        print(f"raw_depth_points={len(raw_world)}")

        # Convert to the robot-base frame used inside CuroboPlanner.plan().
        base_pos = np.asarray(robot_base_pose[:3], dtype=np.float64)
        base_rot = Rotation.from_quat(
            np.asarray(robot_base_pose[3:], dtype=np.float64)
        ).as_matrix()
        raw_base = (base_rot.T @ (raw_world - base_pos).T).T

        # Compute the current robot collision spheres in cuRobo's base frame.
        import torch
        from curobo._src.geom.types import SceneCfg
        from curobo._src.state.state_joint import JointState

        motion_planner = planner._ensure_planner(SceneCfg())
        start = JointState.from_position(
            torch.as_tensor(joint_positions, device=planner.device, dtype=torch.float32).reshape(1, -1),
            joint_names=motion_planner.joint_names,
        )
        start_kin = motion_planner.compute_kinematics(start)
        robot_spheres = start_kin.robot_spheres.detach().cpu().numpy()

        filtered_base = filter_robot_points(
            raw_base, robot_spheres, planner.robot_padding
        )
        print(f"filtered_points={len(filtered_base)}")

        sample_target = base_rot.T @ (DEFAULT_TARGET[:3] - base_pos)
        selection = select_surface_points(
            filtered_base,
            sample_target,
            max_points=args.max_points,
            voxel_size=planner.voxel_size,
        )
        selected_world = to_world(selection.points, robot_base_pose)
        print(
            f"selected_obstacle_points={len(selection.points)} "
            f"(global={selection.global_count}, target={selection.target_count}, "
            f"mandatory={selection.mandatory_count})"
        )

        # ---- Plot ----
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Downsample raw points for plotting to avoid a wall of dots.
        rng = np.random.default_rng(0)
        raw_plot = raw_world
        if len(raw_plot) > 20000:
            idx = rng.choice(len(raw_plot), 20000, replace=False)
            raw_plot = raw_plot[idx]
        ax.scatter(
            raw_plot[:, 0], raw_plot[:, 1], raw_plot[:, 2],
            s=0.2, c="lightgray", alpha=0.25, label=f"raw depth points ({len(raw_world)})",
        )

        if len(selected_world):
            ax.scatter(
                selected_world[:, 0], selected_world[:, 1], selected_world[:, 2],
                s=4, c="blue", alpha=0.8, label=f"sampled obstacle points ({len(selected_world)})",
            )

        ax.scatter(
            DEFAULT_TARGET[0], DEFAULT_TARGET[1], DEFAULT_TARGET[2],
            s=120, c="green", marker="*", label="memory target",
        )
        for label, pos in anchors.items():
            ax.scatter(
                pos[0], pos[1], pos[2],
                s=80, c="black", marker="s", label=label,
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(
            f"init_state_index={args.init_state_index} point cloud before Curobo planning"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.view_init(elev=30, azim=-60)

        fig.tight_layout()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved point cloud plot to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
