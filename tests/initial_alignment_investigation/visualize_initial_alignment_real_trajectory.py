#!/usr/bin/env python
"""Plot real initial-alignment trajectories from COSMOS_DEBUG_INIT_ALIGN logs.

This is a standalone diagnostic script.  It does not modify any production code.
It reads [INIT_ALIGN_DEBUG] JSON lines from a rollout eval log, reconstructs the
actual EEF trajectory, and draws it inside the labeled/cropped 3D point cloud.
"""

from __future__ import annotations

import argparse
import json
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


def parse_debug_records(log_path: Path, episode_index: int) -> list[dict]:
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        marker = "[INIT_ALIGN_DEBUG] "
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if int(data.get("episode", -1)) == episode_index:
            records.append(data)
    records.sort(key=lambda r: (int(r["t"]), int(r["correction_step_index"])))
    return records


def parse_full_plan_records(log_path: Path, episode_index: int) -> list[dict]:
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        marker = "[INIT_ALIGN_FULL_PLAN] "
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if int(data.get("episode", -1)) == episode_index:
            records.append(data)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True,
                        help="1-based episode number in the rollout video")
    parser.add_argument("--init-state-index", type=int, default=None)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--episodes-json", type=str,
                        default="scripts/experiments/libero10_robotinit_single/robotinit/robot_initial_states/episodes.json")
    parser.add_argument("--xmin", type=float, default=-0.70)
    parser.add_argument("--xmax", type=float, default=0.30)
    parser.add_argument("--ymin", type=float, default=-0.40)
    parser.add_argument("--ymax", type=float, default=0.60)
    parser.add_argument("--zmin", type=float, default=0.70)
    parser.add_argument("--zmax", type=float, default=1.30)
    parser.add_argument("--max-points-per-category", type=int, default=15000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    # Resolve init_state_index from episodes.json if not given.
    init_state_index = args.init_state_index
    if init_state_index is None:
        episodes = json.loads(Path(args.episodes_json).read_text())
        matches = [e for e in episodes if int(e["episode"]) == args.episode - 1]
        if matches:
            init_state_index = int(matches[0]["init_state_index"])
        else:
            raise RuntimeError(f"episode {args.episode} not found in {args.episodes_json}")
    print(f"episode={args.episode} init_state_index={init_state_index}")

    records = parse_debug_records(log_path, args.episode - 1)
    if not records:
        raise RuntimeError(f"no INIT_ALIGN_DEBUG records for episode {args.episode}")
    print(f"parsed {len(records)} debug records")

    # Reconstruct the actual EEF path from logged before/after poses.
    actual_path = [np.asarray(records[0]["eef_before"], dtype=np.float64)]
    for r in records:
        actual_path.append(np.asarray(r["eef_after"], dtype=np.float64))
    actual_path = np.asarray(actual_path)

    # Prefer the full Curobo plan from INIT_ALIGN_FULL_PLAN if available.
    full_plan_records = parse_full_plan_records(log_path, args.episode - 1)
    if full_plan_records:
        full_plan = full_plan_records[-1]
        waypoints = np.asarray(full_plan["waypoints"], dtype=np.float64)
        waypoint_label = "full planned waypoints"
        print(f"parsed {len(full_plan_records)} full-plan record(s)")
    else:
        # Fallback: reconstruct the partial waypoint path seen in the logs.
        waypoint_by_index = {}
        for r in records:
            idx = r.get("waypoint_index")
            wp = r.get("waypoint")
            if idx is not None and wp is not None:
                waypoint_by_index[int(idx)] = np.asarray(wp, dtype=np.float64)
        waypoints = (
            np.stack([waypoint_by_index[i] for i in sorted(waypoint_by_index)])
            if waypoint_by_index
            else np.zeros((0, 6), dtype=np.float64)
        )
        waypoint_label = "logged waypoints"

    print(f"reconstructed actual_path_points={len(actual_path)}")
    print(f"reconstructed waypoints={len(waypoints)} ({waypoint_label})")
    print(f"last_waypoint_dpos_to_memory="
          f"{np.linalg.norm(waypoints[-1, :3] - DEFAULT_TARGET[:3]):.4f}")
    print(f"last_eef_dpos_to_memory="
          f"{np.linalg.norm(actual_path[-1, :3] - DEFAULT_TARGET[:3]):.4f}")

    # Build labeled/cropped point cloud for this init state.
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
        obs = env.set_init_state(states[init_state_index])
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        height, width = obs["agentview_image"].shape[:2]
        cam = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, cam, flip_images=True)

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

        # Fixed workspace crop.
        lo = np.array([args.xmin, args.ymin, args.zmin], dtype=np.float64)
        hi = np.array([args.xmax, args.ymax, args.zmax], dtype=np.float64)
        inside = np.all((raw_world >= lo) & (raw_world <= hi), axis=1)
        raw_world = raw_world[inside]
        categories = categories[inside]
        counts = Counter(categories)
        print(f"cropped point counts: {dict(counts)}")

        # Downsample.
        rng = np.random.default_rng(0)
        plot_idx = []
        for cat in CATEGORY_COLORS:
            idx = np.flatnonzero(categories == cat)
            if len(idx) > args.max_points_per_category:
                idx = rng.choice(idx, args.max_points_per_category, replace=False)
            plot_idx.append(idx)
        plot_idx = np.concatenate(plot_idx) if plot_idx else np.array([], dtype=int)

        # Multi-view plot.
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

            if len(waypoints):
                ax.plot(
                    waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                    "-", color="blue", linewidth=2.2, alpha=0.9,
                    label=waypoint_label,
                )
            ax.plot(
                actual_path[:, 0], actual_path[:, 1], actual_path[:, 2],
                "-", color="red", linewidth=2.5, alpha=0.95,
                label="real EEF path (log)",
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
            ax.set_title(f"{view_name} (real trajectory)", fontsize=12)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc="upper right", fontsize=7)

        fig.suptitle(
            f"episode {args.episode} (init_state_index={init_state_index}) real initial-alignment path",
            fontsize=15,
        )
        fig.tight_layout()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved real-trajectory figure to {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
