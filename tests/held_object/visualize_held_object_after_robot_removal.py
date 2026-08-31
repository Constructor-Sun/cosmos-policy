#!/usr/bin/env python
"""Show the point cloud after robot removal, and extract the held object by
connected component near the gripper (no memory matching / no instance id).

Pipeline:
1. Restore HDF5 frame 110 without dummy steps.
2. Apply HeldObjectPlanner's URDF depth filter (enable_urdf_robot_filter=True).
3. Apply cuRobo sphere filter to remove remaining robot points.
4. On the remaining depth image, find the connected component closest to the
   projected EEF; that component is treated as the held object.
5. After extraction is fixed, use simulator segmentation only to score it.
6. Plot the selected component (orange), unselected residual robot (red), and
   all other unselected points (gray).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")

import logging
logging.disable(logging.CRITICAL)

import cv2
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from libero.libero import benchmark
from scipy.spatial.transform import Rotation
import torch
import warp as wp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
from memory_system.execute.planner.held_object.types import (
    HeldObjectObservation,
    HeldObjectPlannerInput,
)
from memory_system.execute.surface_obstacles import filter_robot_points
from memory_system.execute.urdf_depth_filter import (
    UrdfDepthFilter,
    UrdfDepthFilterConfig,
)
from memory_system.geometry import (
    camera_params as build_camera_params,
    pixel_to_world,
    world_to_pixel,
)

BASE_TASK = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
DEMO_ID = "demo_1"
HELD_FRAME = 110
HDF5 = "LIBERO-Cosmos-Policy/success_only/libero_10_regen/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"
OUTPUT = "tests/held_object/held_object_gpu_urdf_no_oracle.png"
MULTIVIEW_OUTPUT = "tests/held_object/held_object_gpu_urdf_no_oracle_multiview.png"
CROP_LO = np.array([-0.70, -0.40, 0.70], dtype=np.float64)
CROP_HI = np.array([0.30, 0.60, 1.30], dtype=np.float64)

COLORS = {
    "robot_residual": "#d62728",
    "held_object": "#ff7f0e",
    "other": "#7f7f7f",
}

VIEWS = [
    ("front-left", 25, -60),
    ("front-right", 25, 30),
    ("top-ish", 60, -60),
    ("side", 10, -120),
]


@wp.kernel
def _raycast_urdf_depth(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_directions: wp.array(dtype=wp.vec3),
    camera_ray_z: wp.array(dtype=wp.float32),
    near: wp.float32,
    far: wp.float32,
    depth: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_directions[index], 1.0e6)
    if query.result:
        z = wp.float32(query.t) * camera_ray_z[index]
        if z >= near and z <= far:
            depth[index] = z


class WarpUrdfDepthFilter(UrdfDepthFilter):
    """URDF depth renderer backed by Warp CUDA ray queries."""

    def __init__(self, config: UrdfDepthFilterConfig, device: str = "cuda:0"):
        super().__init__(config)
        self.warp_device = device

    def render_robot_depth(
        self,
        *,
        camera_params,
        joint_positions,
        joint_names=None,
        robot_base_pose=None,
    ):
        mesh = self._posed_mesh(joint_positions, joint_names)
        directions_camera, camera_to_world = self._camera_rays(camera_params)
        world_to_base = np.linalg.inv(
            np.eye(4, dtype=np.float64)
            if robot_base_pose is None
            else _pose_from_xyz_quat(robot_base_pose)
        )
        camera_to_base = world_to_base @ camera_to_world
        ray_origin = camera_to_base[:3, 3].astype(np.float32)
        directions_base = (
            directions_camera @ camera_to_base[:3, :3].T
        ).astype(np.float32)

        device = self.warp_device
        points = wp.array(
            np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device
        )
        indices = wp.array(
            np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
            dtype=wp.int32,
            device=device,
        )
        warp_mesh = wp.Mesh(points=points, indices=indices)
        ray_directions = wp.array(directions_base, dtype=wp.vec3, device=device)
        camera_ray_z = wp.array(
            directions_camera[:, 2].astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        depth = wp.zeros(len(directions_base), dtype=wp.float32, device=device)
        wp.launch(
            _raycast_urdf_depth,
            dim=len(directions_base),
            inputs=[
                warp_mesh.id,
                wp.vec3(*ray_origin),
                ray_directions,
                camera_ray_z,
                float(camera_params.near),
                float(camera_params.far),
                depth,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        rendered = depth.numpy().reshape(
            int(camera_params.height), int(camera_params.width)
        )
        return np.flipud(rendered).copy()


def _pose_from_xyz_quat(pose):
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


def _to_base(points_world, robot_base_pose):
    base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
    rb = Rotation.from_quat(base[3:]).as_matrix()
    tb = base[:3]
    return (rb.T @ (points_world - tb).T).T


def _to_world(points_base, robot_base_pose):
    base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
    rb = Rotation.from_quat(base[3:]).as_matrix()
    tb = base[:3]
    return (rb @ points_base.T).T + tb


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = next(
        t for t in (suite.get_task(i) for i in range(suite.n_tasks))
        if t.name == BASE_TASK
    )
    env, _ = get_libero_env(task, "cosmos", resolution=256, camera_depths=[True, False])
    try:
        with h5py.File(HDF5, "r") as f:
            states = f["data"][DEMO_ID]["states"][:]

        env.reset()
        obs = env.set_init_state(states[HELD_FRAME])
        height, width = obs["agentview_image"].shape[:2]
        cam = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, cam, flip_images=True)
        robot_base_pose = np.concatenate(
            [env.robots[0].base_pos, env.robots[0].base_ori]
        ).astype(np.float64)

        # Dummy observation only used by _filter_urdf_depth.
        inp = HeldObjectPlannerInput(
            joint_positions=obs["robot0_joint_pos"],
            ee_states=np.zeros(6, dtype=np.float64),
            depth=depth,
            camera_params=cam,
            held_object=HeldObjectObservation(
                item="dummy", points_hand=np.zeros((3, 3), dtype=np.float32)
            ),
            ready_pose=np.zeros(6, dtype=np.float64),
            gripper_joint_positions=obs.get("robot0_gripper_qpos"),
            robot_base_pose=robot_base_pose,
        )

        # Use URDF depth removal first, then the cuRobo collision-sphere filter.
        planner = HeldObjectPlanner(enable_urdf_robot_filter=True)
        import curobo
        urdf_path = (
            Path(curobo.__file__).resolve().parent
            / "content/assets/robot/franka_description/franka_panda.urdf"
        )
        planner._urdf_filter = WarpUrdfDepthFilter(
            UrdfDepthFilterConfig(urdf_path=str(urdf_path)), device="cuda:0"
        )
        urdf_depth = planner._filter_urdf_depth(depth, inp)

        # Back-project after URDF removal.
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        pixels_all = np.stack([rows.ravel(), cols.ravel()], axis=-1)
        valid = (np.isfinite(urdf_depth) & (urdf_depth > 0.0)).ravel()
        pixels = pixels_all[valid]
        world_after_urdf = pixel_to_world(pixels, urdf_depth, cam)

        # Sphere filter for remaining robot points (same as CuroboPlanner path).
        robot_spheres = None
        from curobo._src.state.state_joint import JointState
        curobo = planner._ensure_planner()
        state = JointState.from_position(
            torch.as_tensor(
                np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(1, -1),
                device=planner.device,
                dtype=torch.float32,
            ),
            joint_names=curobo.joint_names,
        )
        kin0 = curobo.compute_kinematics(state)
        robot_spheres = kin0.robot_spheres.detach().cpu().numpy().reshape(-1, 4)

        world_base = _to_base(world_after_urdf, robot_base_pose)
        dist_robot = np.linalg.norm(
            world_base[:, None, :] - robot_spheres[None, :, :3], axis=2
        ) - robot_spheres[None, :, 3]
        inside_robot = (dist_robot <= planner.robot_padding).any(axis=1)
        removed_robot_world = world_after_urdf[inside_robot]
        remaining_world = world_after_urdf[~inside_robot]
        remaining_base = world_base[~inside_robot]
        # Map remaining points back to pixels for connected-component labeling.
        remaining_px = world_to_pixel(remaining_world, cam).astype(np.int64)
        inside_img = (
            (remaining_px[:, 0] >= 0) & (remaining_px[:, 0] < height) &
            (remaining_px[:, 1] >= 0) & (remaining_px[:, 1] < width)
        )
        remaining_px = remaining_px[inside_img]
        remaining_world = remaining_world[inside_img]
        remaining_base = remaining_base[inside_img]

        # Real-compatible extraction: cluster every point that remains after
        # GPU URDF + sphere filtering.  No simulator label enters this step.
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        local_radius = 0.25
        local_mask = (
            np.linalg.norm(remaining_world - eef_pos, axis=1) <= local_radius
        )
        local_idx = np.flatnonzero(local_mask)
        local_points = remaining_world[local_idx]

        clustering = DBSCAN(eps=0.025, min_samples=10).fit(local_points)
        labels = clustering.labels_

        held_orig_idx = []
        best_mean_dist = float("inf")
        cluster_info = []
        for label in np.unique(labels):
            if label == -1:
                continue
            idx = local_idx[labels == label]
            if len(idx) < 20:
                continue
            mean_dist = float(
                np.linalg.norm(remaining_world[idx] - eef_pos, axis=1).mean()
            )
            cluster_info.append((label, len(idx), mean_dist))
            if mean_dist < best_mean_dist:
                best_mean_dist = mean_dist
                held_orig_idx = idx
        print("3D_CLUSTERS_NEAR_EEF", cluster_info[:10], "total", len(cluster_info))
        print("HELD_CLUSTER", f"points={len(held_orig_idx)}", f"mean_dist_to_eef={best_mean_dist:.4f}")

        is_held = np.zeros(len(remaining_world), dtype=bool)
        if len(held_orig_idx):
            is_held[held_orig_idx] = True

        # Evaluation only, after the held component has already been selected.
        seg_render, _ = env.sim.render(
            height, width, camera_name="agentview", depth=True, segmentation=True
        )
        seg_canon = np.flipud(seg_render)
        seg_at_remaining = seg_canon[remaining_px[:, 0], remaining_px[:, 1]]
        residual_robot_oracle = np.zeros(len(remaining_world), dtype=bool)
        for i, (objtype, objid) in enumerate(seg_at_remaining):
            objtype = int(objtype)
            objid = int(objid)
            if objtype == 5:
                name = env.sim.model.geom_id2name(objid)
                if name and (
                    "robot" in name.lower()
                    or "gripper" in name.lower()
                    or name.lower().startswith("panda")
                    or "link" in name.lower()
                ):
                    residual_robot_oracle[i] = True

        instance = np.flipud(
            np.asarray(obs["agentview_segmentation_instance"])[..., 0]
        )
        target_oracle = (
            instance[remaining_px[:, 0], remaining_px[:, 1]]
            == env.instance_to_id["white_yellow_mug_1"]
        )
        held_target = int(np.count_nonzero(is_held & target_oracle))
        held_count = int(np.count_nonzero(is_held))
        target_count = int(np.count_nonzero(target_oracle))
        precision = held_target / held_count if held_count else 0.0
        recall = held_target / target_count if target_count else 0.0
        held_robot = int(np.count_nonzero(is_held & residual_robot_oracle))
        print(
            "POST_EXTRACTION_ORACLE_SCORE",
            f"precision={precision:.4f}",
            f"recall={recall:.4f}",
            f"held_robot_points={held_robot}",
        )

        # Keep categories disjoint for plotting.  Oracle robot labels never
        # change which points belong to the orange selected component.
        is_residual_robot = residual_robot_oracle & (~is_held)
        is_other = (~is_residual_robot) & (~is_held)

        # Crop to main-camera workspace.
        crop_mask = np.all((remaining_world >= CROP_LO) & (remaining_world <= CROP_HI), axis=1)
        remaining_world = remaining_world[crop_mask]
        is_residual_robot = is_residual_robot[crop_mask]
        is_held = is_held[crop_mask]
        is_other = is_other[crop_mask]

        residual_robot_world = remaining_world[is_residual_robot]

        counts = {
            "robot_residual": int(is_residual_robot.sum()),
            "held_object": int(is_held.sum()),
            "other": int(is_other.sum()),
        }
        print("AFTER_ROBOT_REMOVAL_COUNTS", counts)
        print("RAW_VALID", int(valid.sum()), "AFTER_URDF", len(world_after_urdf), "AFTER_SPHERE", len(remaining_world))

        # Plot.
        rng = np.random.default_rng(0)

        def draw_ax(ax, show_legend=False, title=None):
            # Residual robot points remaining after deletion.
            if len(residual_robot_world):
                idx = np.arange(len(residual_robot_world))
                if len(idx) > 20000:
                    idx = rng.choice(idx, 20000, replace=False)
                pts = residual_robot_world[idx]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=0.8, c=COLORS["robot_residual"], alpha=0.9,
                    label=f"robot_residual ({counts['robot_residual']})",
                )
            for cat, color in COLORS.items():
                if cat == "robot_residual":
                    continue
                idx = np.flatnonzero(
                    {"held_object": is_held, "other": is_other}[cat]
                )
                if len(idx) > 20000:
                    idx = rng.choice(idx, 20000, replace=False)
                if len(idx):
                    pts = remaining_world[idx]
                    ax.scatter(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        s=0.6, c=color, alpha=0.7,
                        label=f"{cat} ({counts[cat]})",
                    )
            if title:
                ax.set_title(title)
            if show_legend:
                ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")
        draw_ax(
            ax,
            show_legend=True,
            title=(
                f"After robot removal frame={HELD_FRAME}\n"
                f"red=residual robot, orange=held object (3D connected component), gray=other"
            ),
        )
        ax.view_init(elev=25, azim=-60)
        fig.tight_layout()
        Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=150)
        print("SAVED", OUTPUT)
        plt.close(fig)

        fig = plt.figure(figsize=(20, 16))
        for idx, (name, elev, azim) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")
            draw_ax(
                ax,
                show_legend=(idx == 1),
                title=f"view {idx}: {name} (elev={elev}, azim={azim})",
            )
            ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        Path(MULTIVIEW_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(MULTIVIEW_OUTPUT, dpi=150)
        print("SAVED", MULTIVIEW_OUTPUT)
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
