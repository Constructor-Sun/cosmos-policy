#!/usr/bin/env python3
"""Render one post-Pick Place-stage held-object case for every LIBERO-10 task."""
from __future__ import annotations

import csv
import importlib.util
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")
logging.disable(logging.CRITICAL)

import h5py
import matplotlib
import numpy as np
import torch
from libero.libero import benchmark
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
from memory_system.execute.planner.held_object.types import (
    HeldObjectObservation,
    HeldObjectPlannerInput,
)
from memory_system.execute.urdf_depth_filter import UrdfDepthFilterConfig
from memory_system.geometry import camera_params as build_camera_params, pixel_to_world


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).with_name("visualize_held_object_after_robot_removal.py")
SPEC = importlib.util.spec_from_file_location("held_object_gpu_urdf_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {BASE_SCRIPT}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

WarpUrdfDepthFilter = BASE.WarpUrdfDepthFilter

HDF5_DIR = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
OUTPUT_DIR = ROOT / "tests/held_object/libero10_gpu_urdf_no_oracle"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# One closed-gripper frame from the first post-Pick Place transport interval.
# KITCHEN_SCENE3 has an earlier stove-knob interaction, so its later interval is used.
CASES = [
    {
        "task": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "hdf5": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        "demo": "demo_1", "frame": 233, "slug": "01_kitchen_scene3_moka_pot",
    },
    {
        "task": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "hdf5": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
        "demo": "demo_1", "frame": 110, "slug": "02_kitchen_scene4_black_bowl",
    },
    {
        "task": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "hdf5": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
        "demo": "demo_1", "frame": 110, "slug": "03_kitchen_scene6_yellow_white_mug",
    },
    {
        "task": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "hdf5": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
        "demo": "demo_0", "frame": 389, "slug": "04_kitchen_scene8_moka_pot",
    },
    {
        "task": "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "hdf5": "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
        "demo": "demo_1", "frame": 82, "slug": "05_living_scene1_alphabet_soup",
    },
    {
        "task": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "hdf5": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
        "demo": "demo_1", "frame": 108, "slug": "06_living_scene2_soup_tomato",
    },
    {
        "task": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "hdf5": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
        "demo": "demo_1", "frame": 84, "slug": "07_living_scene2_cheese_butter",
    },
    {
        "task": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "hdf5": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
        "demo": "demo_10", "frame": 77, "slug": "08_living_scene5_white_mug",
    },
    {
        "task": "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "hdf5": "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
        "demo": "demo_0", "frame": 70, "slug": "09_living_scene6_white_mug",
    },
    {
        "task": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "hdf5": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
        "demo": "demo_0", "frame": 110, "slug": "10_study_scene1_book",
    },
]

VIEWS = [
    ("front-left", 25, -60),
    ("front-right", 25, 30),
    ("top-ish", 60, -60),
    ("side", 10, -120),
]
CROP_LO_FROM_EEF = np.array([-0.55, -0.55, -0.40], dtype=np.float64)
CROP_HI_FROM_EEF = np.array([0.55, 0.55, 0.35], dtype=np.float64)


def _to_base(points_world, robot_base_pose):
    base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
    rotation = Rotation.from_quat(base[3:]).as_matrix()
    return (rotation.T @ (points_world - base[:3]).T).T


def _urdf_joints(obs, expected):
    arm = np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(-1)
    fingers = np.asarray(obs.get("robot0_gripper_qpos", [0.04, 0.04]), dtype=np.float64).reshape(-1)
    values = np.concatenate([arm, np.clip(np.abs(fingers[:2]), 0.0, 0.04)])
    if len(values) != expected:
        raise ValueError(f"URDF expects {expected} joints, got {len(values)}")
    return values


def _robot_oracle(env, pixels, height, width):
    seg_render, _ = env.sim.render(
        height, width, camera_name="agentview", depth=True, segmentation=True
    )
    seg = np.flipud(seg_render)[pixels[:, 0], pixels[:, 1]]
    result = np.zeros(len(pixels), dtype=bool)
    for index, (objtype, objid) in enumerate(seg):
        if int(objtype) != 5:
            continue
        name = env.sim.model.geom_id2name(int(objid))
        if name and (
            "robot" in name.lower()
            or "gripper" in name.lower()
            or name.lower().startswith("panda")
            or "link" in name.lower()
        ):
            result[index] = True
    return result


def _extract_case(case, task, suite, planner, curobo, robot_filter, JointState):
    env, _ = get_libero_env(task, "cosmos", resolution=256, camera_depths=[True, False])
    try:
        with h5py.File(HDF5_DIR / case["hdf5"], "r") as handle:
            states = handle["data"][case["demo"]]["states"][:]
        obs = env.reset()
        obs = env.set_init_state(states[case["frame"]])
        height, width = obs["agentview_image"].shape[:2]
        camera = build_camera_params(env.sim, "agentview", height, width)
        depth = np.asarray(_make_main_depth(obs, camera, flip_images=True))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        robot_base_pose = np.concatenate(
            [env.robots[0].base_pos, env.robots[0].base_ori]
        ).astype(np.float64)

        filtered = robot_filter.filter(
            depth,
            camera_params=camera,
            joint_positions=_urdf_joints(obs, len(robot_filter.joint_names)),
            robot_base_pose=robot_base_pose,
        )
        urdf_depth = filtered.filtered_depth
        rows, cols = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        pixels_all = np.stack([rows.ravel(), cols.ravel()], axis=-1)
        valid = (np.isfinite(urdf_depth) & (urdf_depth > 0.0)).ravel()
        pixels = pixels_all[valid]
        world = pixel_to_world(pixels, urdf_depth, camera)

        state = JointState.from_position(
            torch.as_tensor(
                np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(1, -1),
                device=planner.device,
                dtype=torch.float32,
            ),
            joint_names=curobo.joint_names,
        )
        spheres = curobo.compute_kinematics(state).robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        world_base = _to_base(world, robot_base_pose)
        signed_distance = (
            np.linalg.norm(world_base[:, None, :] - spheres[None, :, :3], axis=2)
            - spheres[None, :, 3]
        )
        keep = ~(signed_distance <= planner.robot_padding).any(axis=1)
        remaining_world = world[keep]
        remaining_pixels = pixels[keep]

        # Extraction ends here. No simulator label has been read.
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        local_indices = np.flatnonzero(
            np.linalg.norm(remaining_world - eef, axis=1) <= 0.25
        )
        if len(local_indices) < 20:
            raise RuntimeError("fewer than 20 points in EEF ROI")
        labels = DBSCAN(eps=0.025, min_samples=10).fit_predict(
            remaining_world[local_indices]
        )
        best_indices = np.empty(0, dtype=np.int64)
        best_distance = float("inf")
        for label in np.unique(labels):
            if label == -1:
                continue
            indices = local_indices[labels == label]
            if len(indices) < 20:
                continue
            mean_distance = float(
                np.linalg.norm(remaining_world[indices] - eef, axis=1).mean()
            )
            if mean_distance < best_distance:
                best_distance = mean_distance
                best_indices = indices
        if len(best_indices) == 0:
            raise RuntimeError("no valid connected component near EEF")
        is_held = np.zeros(len(remaining_world), dtype=bool)
        is_held[best_indices] = True

        # Post-extraction oracle is used only to color unselected robot residuals.
        residual_oracle = _robot_oracle(env, remaining_pixels, height, width)
        is_residual = residual_oracle & ~is_held
        is_other = ~is_held & ~is_residual
        held_robot = int(np.count_nonzero(residual_oracle & is_held))

        instance = np.flipud(
            np.asarray(obs["agentview_segmentation_instance"])[..., 0]
        )
        held_instance_ids = instance[
            remaining_pixels[is_held, 0], remaining_pixels[is_held, 1]
        ]
        inverse_instances = {
            int(identifier): str(name)
            for name, identifier in env.instance_to_id.items()
        }
        identifiers, identifier_counts = np.unique(
            held_instance_ids, return_counts=True
        )
        order = np.argsort(-identifier_counts)
        held_instances = ";".join(
            f"{inverse_instances.get(int(identifiers[index]), str(int(identifiers[index])))}:{int(identifier_counts[index])}"
            for index in order[:5]
        )

        crop_lo = eef + CROP_LO_FROM_EEF
        crop_hi = eef + CROP_HI_FROM_EEF
        crop = np.all(
            (remaining_world >= crop_lo) & (remaining_world <= crop_hi), axis=1
        )
        plot_world = remaining_world[crop]
        plot_held = is_held[crop]
        plot_residual = is_residual[crop]
        plot_other = is_other[crop]

        counts = {
            "held": int(plot_held.sum()),
            "residual": int(plot_residual.sum()),
            "other": int(plot_other.sum()),
            "held_robot": held_robot,
            "urdf_removed": int(filtered.removed_pixel_count),
            "held_instances": held_instances,
        }

        rng = np.random.default_rng(0)
        held_points = plot_world[plot_held]
        residual_points = plot_world[plot_residual]
        other_points = plot_world[plot_other]

        def draw(axis, show_legend=False, title=None):
            for points, color, size, alpha, label in [
                (residual_points, "#d62728", 0.8, 0.9, f"robot_residual ({len(residual_points)})"),
                (held_points, "#ff7f0e", 0.8, 0.8, f"held_object ({len(held_points)})"),
                (other_points, "#7f7f7f", 0.6, 0.65, f"other ({len(other_points)})"),
            ]:
                if len(points) == 0:
                    continue
                indices = np.arange(len(points))
                if len(indices) > 20000:
                    indices = rng.choice(indices, 20000, replace=False)
                selected = points[indices]
                axis.scatter(
                    selected[:, 0], selected[:, 1], selected[:, 2],
                    s=size, c=color, alpha=alpha, label=label,
                )
            if title:
                axis.set_title(title, fontsize=10)
            if show_legend:
                axis.legend(loc="upper right", fontsize=8)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_zlabel("z (m)")

        figure = plt.figure(figsize=(20, 16))
        for view_index, (name, elevation, azimuth) in enumerate(VIEWS, start=1):
            axis = figure.add_subplot(2, 2, view_index, projection="3d")
            draw(
                axis,
                show_legend=view_index == 1,
                title=f"{name} (elev={elevation}, azim={azimuth})",
            )
            axis.view_init(elev=elevation, azim=azimuth)
        figure.suptitle(
            f"{case['task']}\n{case['demo']} frame={case['frame']} | GPU URDF + no-oracle DBSCAN",
            fontsize=14,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.96])
        output = OUTPUT_DIR / f"{case['slug']}_multiview.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        return output, counts, best_distance
    finally:
        env.close()


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    tasks = {
        suite.get_task(index).name: suite.get_task(index)
        for index in range(suite.n_tasks)
    }
    planner = HeldObjectPlanner(enable_urdf_robot_filter=True)
    curobo = planner._ensure_planner()
    from curobo._src.state.state_joint import JointState
    import curobo as curobo_package

    urdf_path = (
        Path(curobo_package.__file__).resolve().parent
        / "content/assets/robot/franka_description/franka_panda.urdf"
    )
    robot_filter = WarpUrdfDepthFilter(
        UrdfDepthFilterConfig(urdf_path=str(urdf_path)), device="cuda:0"
    )

    rows = []
    for case_index, case in enumerate(CASES, start=1):
        print(f"CASE {case_index}/10 START {case['task']}", flush=True)
        output, counts, mean_distance = _extract_case(
            case, tasks[case["task"]], suite, planner, curobo, robot_filter, JointState
        )
        row = {
            "task": case["task"], "demo": case["demo"], "frame": case["frame"],
            "output": str(output.relative_to(ROOT)), "mean_eef_distance": mean_distance,
            **counts,
        }
        rows.append(row)
        print(f"CASE {case_index}/10 SAVED {output} {counts}", flush=True)

    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("ALL_CASES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
