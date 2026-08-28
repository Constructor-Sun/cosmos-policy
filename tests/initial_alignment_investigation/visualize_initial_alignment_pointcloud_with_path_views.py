#!/usr/bin/env python
"""Multi-view version of the labeled point cloud + initial-alignment trajectory.

This is a standalone diagnostic script.  It does not modify any production code.
It renders the same scene from several viewpoints:
  - upper-right (current default)
  - upper-left
  - top-down
  - front/side
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logging.disable(logging.CRITICAL)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

CATEGORY_COLORS = {
    "robot": "#d62728",
    "bowl": "#ff7f0e",
    "cabinet": "#2ca02c",
    "table": "#9467bd",
    "floor": "#8c564b",
    "other": "#7f7f7f",
}

VIEWS = [
    ("upper-right", 30, -60),
    ("upper-left", 30, 120),
    ("top-down", 89, -90),
    ("front", 5, 90),
]


def classify_geom(name: str | None) -> str:
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
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--max-points-per-category", type=int, default=15000)
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/initial_alignment_pointcloud_with_path_views.png")
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

        height, width = obs["agentview_image"].shape[:2]
        cam = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, cam, flip_images=True)

        # ---------- Labeled point cloud ----------
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
        categories = np.array([classify_geom(n) for n in names], dtype=object)
        counts = Counter(categories)

        # ---------- Plan and execute initial alignment ----------
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

        actual_path = [current_ee.astype(np.float64)]
        current = current_ee.astype(np.float64)
        for _ in range(args.steps):
            action6 = controller.step(current)
            action = np.zeros(7, dtype=np.float32)
            action[:6] = action6
            action[6] = 0.0
            obs, _, _, _ = env.step(action)
            current = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
                ]
            ).astype(np.float64)
            actual_path.append(current)
        actual_path = np.asarray(actual_path)

        print(f"point counts: {dict(counts)}")
        print(f"planned waypoints={len(waypoints)}, actual_path_steps={len(actual_path)}")
        print(f"after_{args.steps}_steps_dpos_to_memory="
              f"{np.linalg.norm(actual_path[-1, :3] - DEFAULT_TARGET[:3]):.4f}")

        # ---------- Downsample point cloud once for all views ----------
        rng = np.random.default_rng(0)
        plot_idx = []
        for cat in CATEGORY_COLORS:
            idx = np.flatnonzero(categories == cat)
            if len(idx) > args.max_points_per_category:
                idx = rng.choice(idx, args.max_points_per_category, replace=False)
            plot_idx.append(idx)
        plot_idx = np.concatenate(plot_idx) if plot_idx else np.array([], dtype=int)

        # ---------- Multi-view plot ----------
        fig = plt.figure(figsize=(18, 14))
        for panel, (view_name, elev, azim) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(2, 2, panel, projection="3d")

            for cat in CATEGORY_COLORS:
                mask = categories[plot_idx] == cat
                if not mask.any():
                    continue
                pts = raw_world[plot_idx][mask]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=0.4, c=CATEGORY_COLORS[cat], alpha=0.5,
                    label=f"{cat} ({counts.get(cat, 0)})",
                )

            ax.plot(
                waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                "-", color="blue", linewidth=2.2, alpha=0.9, label="planned waypoints",
            )
            ax.plot(
                actual_path[:, 0], actual_path[:, 1], actual_path[:, 2],
                "-", color="red", linewidth=2.2, alpha=0.9, label="actual EEF path",
            )
            ax.scatter(
                actual_path[0, 0], actual_path[0, 1], actual_path[0, 2],
                s=60, c="magenta", marker="o", label="start",
            )
            ax.scatter(
                DEFAULT_TARGET[0], DEFAULT_TARGET[1], DEFAULT_TARGET[2],
                s=140, c="black", marker="*", label="memory target",
            )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(view_name, fontsize=12)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc="upper right", fontsize=7)

        fig.suptitle(
            f"init_state_index={args.init_state_index} point cloud + initial-alignment path (multi-view)",
            fontsize=15,
        )
        fig.tight_layout()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved multi-view figure to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
