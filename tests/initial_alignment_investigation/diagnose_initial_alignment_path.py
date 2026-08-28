#!/usr/bin/env python
"""Diagnose initial-alignment path vs actual EEF trajectory.

This is a standalone diagnostic script.  It does not modify any production code.
It reproduces one episode's initial alignment using the same building blocks as
the rollout path:

    CuroboPlanner -> waypoints -> WaypointPoseController -> OSC env.step()

and then plots the planned waypoint path against the actual executed EEF path.
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


def point_segment_distance(p, a, b):
    """Distance from point p to segment ab in 3D."""
    ab = b - a
    t = 0.0 if np.allclose(ab, 0) else np.dot(p - a, ab) / np.dot(ab, ab)
    t = float(np.clip(t, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def distance_to_polyline(p, waypoints):
    """Minimum distance from point p to a polyline defined by waypoints."""
    best = float("inf")
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        best = min(best, point_segment_distance(p, a[:3], b[:3]))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-state-index", type=int, default=1,
                        help="0-based init state index; episode 2 uses 1")
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--steps", type=int, default=48,
                        help="number of correction steps to execute")
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/diagnose_initial_alignment_path.png")
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

        # Object anchors for context in the plot.
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

        # Reproduce the first 10 dummy steps used by run_episode.
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

        planner = CuroboPlanner(joint_execution=False)
        plan = planner.plan(
            current_ee_states=current_ee,
            target_ee_states=DEFAULT_TARGET,
            depth=depth,
            camera_params=cam,
            joint_positions=joint_positions,
            robot_base_pose=robot_base_pose,
        )
        if plan is None or plan.controller is None:
            raise RuntimeError("CuroboPlanner returned no executable EE plan")

        controller = plan.controller
        waypoints = np.asarray(controller.waypoints, dtype=np.float64)
        print(f"init_state_index={args.init_state_index}")
        print(f"planned_correction_steps={plan.correction_steps}")
        print(f"num_waypoints={len(waypoints)}")
        print(f"last_waypoint_dpos_to_memory="
              f"{np.linalg.norm(waypoints[-1, :3] - DEFAULT_TARGET[:3]):.4f}")

        # Execute the same correction style as run_episode: step the controller,
        # feed the action to the OSC environment, and record the actual state.
        records = []
        current = current_ee.astype(np.float64)
        for step in range(args.steps):
            target_wp_idx = min(controller.index, len(waypoints) - 1)
            target_wp = waypoints[target_wp_idx]

            action6 = controller.step(current)
            action = np.zeros(7, dtype=np.float32)
            action[:6] = action6
            action[6] = 0.0  # gripper command used by initial alignment

            obs, _, _, _ = env.step(action)
            next_ee = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
                ]
            ).astype(np.float64)

            actual_delta = next_ee[:3] - current[:3]
            desired_dir = target_wp[:3] - current[:3]
            denom = (
                np.linalg.norm(actual_delta) * np.linalg.norm(desired_dir) + 1e-12
            )
            cos_dir = float(np.dot(actual_delta, desired_dir) / denom)

            rot_err = (
                Rotation.from_rotvec(next_ee[3:]).inv()
                * Rotation.from_rotvec(target_wp[3:])
            ).magnitude()

            records.append(
                {
                    "step": step,
                    "waypoint_index": controller.index,
                    "target_wp": target_wp.copy(),
                    "actual": current.copy(),
                    "action": np.asarray(action6, dtype=np.float64).copy(),
                    "next_actual": next_ee.copy(),
                    "dist_to_current_wp": float(
                        np.linalg.norm(current[:3] - target_wp[:3])
                    ),
                    "rot_err_to_current_wp": float(rot_err),
                    "dist_to_polyline": distance_to_polyline(current[:3], waypoints),
                    "cos_dir": cos_dir,
                }
            )
            current = next_ee

        actual_path = np.array([r["actual"] for r in records])
        actual_next = np.array([r["next_actual"] for r in records])
        dists_mem = np.linalg.norm(actual_next[:, :3] - DEFAULT_TARGET[:3], axis=1)

        print(f"after_{args.steps}_steps_dpos_to_memory={dists_mem[-1]:.4f}")
        print(f"controller_converged={controller.converged}")
        print(f"final_waypoint_index={controller.index}/{len(waypoints)}")

        # ----- Plot -----
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

        # Left: path comparison (top-down X-Y).
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(waypoints[:, 0], waypoints[:, 1], "-o", color="blue",
                linewidth=1.5, markersize=3, label="planned waypoints")
        ax.plot(actual_path[:, 0], actual_path[:, 1], "-x", color="red",
                linewidth=1.5, markersize=3, label="actual EEF path")
        ax.plot(DEFAULT_TARGET[0], DEFAULT_TARGET[1], "*", color="green",
                markersize=16, label="memory target")
        for label, pos in anchors.items():
            ax.plot(pos[0], pos[1], "s", color="black", markersize=8)
            ax.annotate(label, (pos[0], pos[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"init_state_index={args.init_state_index} after {args.steps} steps\n"
            f"dpos to memory = {dists_mem[-1]:.3f} m"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # Right: metrics vs step.
        ax2 = fig.add_subplot(gs[0, 1])
        steps = np.arange(args.steps)
        ax2.plot(steps, [r["dist_to_current_wp"] for r in records],
                 label="dist to current waypoint", color="blue")
        ax2.plot(steps, [r["dist_to_polyline"] for r in records],
                 label="dist to planned polyline", color="purple")
        ax2.plot(steps, [r["cos_dir"] for r in records],
                 label="cos_dir", color="orange")
        ax2.plot(steps, [r["rot_err_to_current_wp"] for r in records],
                 label="rot err to current waypoint", color="cyan")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("correction step")
        ax2.set_ylabel("metric")
        ax2.set_title("Per-step tracking metrics")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved plot to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
