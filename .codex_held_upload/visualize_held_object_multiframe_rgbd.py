#!/usr/bin/env python3
"""Visualize the multi-frame RGB-D held-object estimate.

This diagnostic uses only the memory RGB mask, RGB-D depth, and observed EEF
poses.  It deliberately does not use simulator segmentation.  Candidate
points from each frame are converted to the hand frame; only voxels observed
consistently across the capture window become the frozen attachment model.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")
logging.disable(logging.CRITICAL)

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import benchmark
from scipy.spatial.transform import Rotation

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.artifacts import PhaseTargetMemory
from memory_system.execute.planner.held_object.attachment import (
    HeldObjectAttachmentEstimator,
)
from memory_system.execute.planner.held_object.geometry_builder import (
    HeldObjectGeometryBuilder,
)
from memory_system.execute.planner.held_object.mask_matcher import MemoryTemplateMatcher
from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
from memory_system.geometry import camera_params as build_camera_params, pixel_to_world


BASE_TASK = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
DEMO_ID = "demo_1"
FRAMES = [88, 92, 96, 100, 104, 108, 110]
HDF5 = "LIBERO-Cosmos-Policy/success_only/libero_10_regen/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"
PHASE_TARGETS = "skill_memory_test/libero_10/phase_targets.pt"
OUTPUT_DIR = Path("tests/held_object")
HAND_OUTPUT = OUTPUT_DIR / "held_object_multiframe_rgbd_consensus_hand.png"
WORLD_OUTPUT = OUTPUT_DIR / "held_object_multiframe_rgbd_consensus_multiview.png"
CROP_LO = np.array([-0.70, -0.40, 0.70], dtype=np.float64)
CROP_HI = np.array([0.30, 0.60, 1.30], dtype=np.float64)
VIEWS = [
    ("front-left", 25, -60),
    ("front-right", 25, 30),
    ("top-ish", 60, -60),
    ("side", 10, -120),
]


def _hand_to_world(obs: dict) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(
        np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    ).as_matrix()
    pose[:3, 3] = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return pose


def _template() -> dict:
    memory = PhaseTargetMemory(PHASE_TARGETS)
    templates = [
        item
        for item in memory.templates
        if item["task_name"] == BASE_TASK
        and str(item.get("demo_id")) == DEMO_ID
        and item["skill"] == "Pick"
        and item["arguments"].get("item") == "white_yellow_mug_1"
    ]
    if not templates:
        raise RuntimeError("no Pick template found")
    return templates[0]


def _draw_spheres(ax, spheres, hand_to_world):
    if spheres is None or len(spheres) == 0:
        return
    rotation = hand_to_world[:3, :3]
    translation = hand_to_world[:3, 3]
    centers = (rotation @ spheres[:, :3].T).T + translation
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        s=28, c="black", marker="o", label=f"attachment centers ({len(spheres)})",
    )
    for center, radius in zip(centers, spheres[:, 3]):
        u, v = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 7j]
        xs = center[0] + radius * np.sin(v) * np.cos(u)
        ys = center[1] + radius * np.sin(v) * np.sin(u)
        zs = center[2] + radius * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color="black", alpha=0.18, linewidths=0.25)


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = next(
        item
        for item in (suite.get_task(i) for i in range(suite.n_tasks))
        if item.name == BASE_TASK
    )
    env, _ = get_libero_env(task, "cosmos", resolution=256, camera_depths=[True, False])
    try:
        with h5py.File(HDF5, "r") as file:
            states = file["data"][DEMO_ID]["states"][:]
        template = _template()
        matcher = MemoryTemplateMatcher()
        builder = HeldObjectGeometryBuilder(
            matcher, "white_yellow_mug_1", voxel_size=0.005
        )
        final_hand_to_world = None
        final_world = None
        final_cam = None
        for frame in FRAMES:
            env.reset()
            obs = env.set_init_state(states[frame])
            rgb = np.flipud(np.asarray(obs["agentview_image"]))
            height, width = rgb.shape[:2]
            cam = build_camera_params(env.sim, "agentview", height, width)
            depth = _make_main_depth(obs, cam, flip_images=True)
            hand_to_world = _hand_to_world(obs)
            matched = builder.update(rgb, depth, cam, template, hand_to_world)
            print(
                f"FRAME {frame}: matched={matched} "
                f"candidate_points={len(builder.frame_points[-1]) if matched else 0}"
            )
            final_hand_to_world = hand_to_world
            final_world = pixel_to_world(
                np.stack(np.nonzero(np.isfinite(depth) & (depth > 0.0)), axis=-1),
                depth,
                cam,
            )
            final_cam = cam

        estimator = HeldObjectAttachmentEstimator(
            "white_yellow_mug_1",
            voxel_size=0.005,
            min_frames=3,
            min_consensus_frames=2,
            consensus_ratio=0.5,
            min_hand_translation=0.005,
            min_hand_rotation=0.03,
            min_points=16,
        )
        estimate = builder.build_stable(estimator)
        print(
            "ESTIMATE",
            f"valid={estimate.valid}",
            f"reason={estimate.reason!r}",
            f"frames={estimate.frame_count}",
            f"stable_voxels={estimate.stable_voxels}",
            f"translation_span={estimate.hand_translation_span:.4f}",
            f"rotation_span={estimate.hand_rotation_span:.4f}",
            f"points={len(estimate.observation.points_hand)}",
        )

        all_hand = np.concatenate(builder.frame_points, axis=0) if builder.frame_points else np.empty((0, 3))
        stable_hand = np.asarray(estimate.observation.points_hand, dtype=np.float64).reshape(-1, 3)
        spheres = None
        if len(stable_hand):
            spheres = HeldObjectPlanner._build_spheres(
                stable_hand, max_slots=32, voxel_size=0.02, padding=0.005
            )

        # Hand-frame view: all RGB-D candidates versus the stable consensus.
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")
        if len(all_hand):
            idx = np.arange(len(all_hand))
            if len(idx) > 20000:
                idx = np.random.default_rng(0).choice(idx, 20000, replace=False)
            ax.scatter(
                all_hand[idx, 0], all_hand[idx, 1], all_hand[idx, 2],
                s=1.0, c="#bdbdbd", alpha=0.28, label=f"all candidates ({len(all_hand)})",
            )
        if len(stable_hand):
            ax.scatter(
                stable_hand[:, 0], stable_hand[:, 1], stable_hand[:, 2],
                s=4.0, c="#ff7f0e", alpha=0.9,
                label=f"stable hand-frame points ({len(stable_hand)})",
            )
        ax.set_title(
            "Multi-frame RGB-D attachment estimate in hand frame\n"
            f"valid={estimate.valid}, reason={estimate.reason}"
        )
        ax.set_xlabel("hand x (m)")
        ax.set_ylabel("hand y (m)")
        ax.set_zlabel("hand z (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.view_init(elev=25, azim=-60)
        fig.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(HAND_OUTPUT, dpi=150)
        print("SAVED", HAND_OUTPUT)
        plt.close(fig)

        # World-frame views at the last captured frame.
        final_world = np.asarray(final_world, dtype=np.float64)
        crop = np.all((final_world >= CROP_LO) & (final_world <= CROP_HI), axis=1)
        final_world = final_world[crop]
        stable_world = (
            final_hand_to_world[:3, :3] @ stable_hand.T
        ).T + final_hand_to_world[:3, 3] if len(stable_hand) else np.empty((0, 3))
        fig = plt.figure(figsize=(20, 16))
        for index, (name, elev, azim) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(2, 2, index, projection="3d")
            plot = final_world
            if len(plot) > 30000:
                plot = np.random.default_rng(0).choice(plot, 30000, axis=0)
            if len(plot):
                ax.scatter(plot[:, 0], plot[:, 1], plot[:, 2], s=0.6, c="#7f7f7f", alpha=0.38, label="RGB-D scene")
            if len(stable_world):
                ax.scatter(stable_world[:, 0], stable_world[:, 1], stable_world[:, 2], s=5.0, c="#ff7f0e", alpha=0.95, label="stable attached object")
            _draw_spheres(ax, spheres, final_hand_to_world)
            ax.set_title(f"frame 110: {name} (elev={elev}, azim={azim})")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            if index == 1:
                ax.legend(loc="upper right", fontsize=8)
            ax.view_init(elev=elev, azim=azim)
        fig.suptitle(
            "Multi-frame RGB-D held-object estimate in world frame\n"
            f"frames={FRAMES}, points={len(stable_hand)}, spheres={0 if spheres is None else len(spheres)}",
            fontsize=15,
        )
        fig.tight_layout()
        fig.savefig(WORLD_OUTPUT, dpi=150)
        print("SAVED", WORLD_OUTPUT)
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
