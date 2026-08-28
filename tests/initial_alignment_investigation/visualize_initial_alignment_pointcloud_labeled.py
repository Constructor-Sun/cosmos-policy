#!/usr/bin/env python
"""Visualize the Curobo initial-alignment point cloud colored by object.

This is a standalone diagnostic script.  It does not modify any production code.
It uses MuJoCo segmentation only as a visualization oracle to label which
object each depth point belongs to.
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
    parser.add_argument("--output", type=str,
                        default="tests/initial_alignment_investigation/initial_alignment_pointcloud_labeled.png")
    parser.add_argument("--max-points-per-category", type=int, default=20000)
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

        # Render segmentation in render space, then flip to canonical space.
        seg_render, _ = env.sim.render(
            height, width, camera_name="agentview", depth=True, segmentation=True
        )
        seg_canon = np.flipud(seg_render)  # (H, W, 2): (objtype, objid)

        # Back-project canonical depth to world, keeping the pixel for labels.
        rows, cols = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        pixels = np.stack([rows.ravel(), cols.ravel()], axis=-1)
        valid = (np.isfinite(depth) & (depth > 0.0)).ravel()
        pixels = pixels[valid]
        raw_world = pixel_to_world(pixels, depth, cam)

        seg_flat = seg_canon.reshape(-1, 2)[valid]
        objtypes = seg_flat[:, 0]
        objids = seg_flat[:, 1]

        names = np.empty(len(raw_world), dtype=object)
        for i in range(len(raw_world)):
            t = int(objtypes[i])
            oid = int(objids[i])
            if t == 5:
                name = env.sim.model.geom_id2name(oid)
            elif t == 6:
                name = env.sim.model.site_id2name(oid) if hasattr(env.sim.model, "site_id2name") else None
            else:
                name = None
            names[i] = name

        categories = np.array([classify_geom(n) for n in names], dtype=object)
        counts = Counter(categories)
        print("point category counts:")
        for cat, count in counts.most_common():
            print(f"  {cat}: {count}")

        # Downsample per category for plotting.
        rng = np.random.default_rng(0)
        plot_indices = []
        for cat in CATEGORY_COLORS:
            idx = np.flatnonzero(categories == cat)
            if len(idx) > args.max_points_per_category:
                idx = rng.choice(idx, args.max_points_per_category, replace=False)
            plot_indices.append(idx)
        plot_indices = np.concatenate(plot_indices) if plot_indices else np.array([], dtype=int)

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")

        for cat in CATEGORY_COLORS:
            mask = categories[plot_indices] == cat
            if not mask.any():
                continue
            pts = raw_world[plot_indices][mask]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=0.5, c=CATEGORY_COLORS[cat], alpha=0.7,
                label=f"{cat} ({counts.get(cat, 0)})",
            )

        ax.scatter(
            DEFAULT_TARGET[0], DEFAULT_TARGET[1], DEFAULT_TARGET[2],
            s=160, c="black", marker="*", label="memory target",
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(
            f"init_state_index={args.init_state_index} point cloud colored by object"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.view_init(elev=30, azim=-60)

        fig.tight_layout()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved labeled point cloud to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
