"""No-oracle held-object connected-component extractor.

This module extracts a held-object point cloud from RGB-D after Pick using the
same URDF robot-removal path as HeldObjectPlanner, then selects the connected
component closest to the end effector inside a spherical ROI.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
from memory_system.execute.planner.held_object.types import (
    HeldObjectObservation,
    HeldObjectPlannerInput,
)
from memory_system.execute.surface_obstacles import filter_robot_points, points_from_depth
from memory_system.types import CameraParams


class HeldObjectConnectedComponentExtractor:
    """Extract a hand-local held-object cloud without simulator segmentation.

    The output ``points_hand`` is expressed in the cuRobo tool frame, matching
    the convention used by ``HeldObjectPlanner`` and ``CuroboPlanner``.
    """

    def __init__(
        self,
        planner: HeldObjectPlanner | None = None,
        *,
        roi_radius: float = 0.25,
        dbscan_eps: float = 0.025,
        dbscan_min_samples: int = 10,
        min_cluster_points: int = 20,
        item: str = "held_object",
    ) -> None:
        self.planner = planner if planner is not None else HeldObjectPlanner()
        self.roi_radius = float(roi_radius)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.min_cluster_points = int(min_cluster_points)
        self.item = str(item)

    def extract(
        self,
        *,
        depth: np.ndarray,
        camera_params: CameraParams,
        joint_positions: np.ndarray,
        gripper_joint_positions: np.ndarray | None,
        robot_base_pose: np.ndarray | None,
        eef_pos: np.ndarray,
        item: str | None = None,
    ) -> HeldObjectObservation | None:
        if depth is None or camera_params is None or joint_positions is None or eef_pos is None:
            return None

        # Reuse the exact URDF depth-removal path used by HeldObjectPlanner.
        dummy_held = HeldObjectObservation(
            item="dummy",
            points_hand=np.zeros((3, 3), dtype=np.float32),
        )
        inp = HeldObjectPlannerInput(
            joint_positions=joint_positions,
            ee_states=np.zeros(6, dtype=np.float64),
            depth=depth,
            camera_params=camera_params,
            held_object=dummy_held,
            ready_pose=np.zeros(6, dtype=np.float64),
            gripper_joint_positions=gripper_joint_positions,
            robot_base_pose=robot_base_pose,
        )
        filtered_depth = self.planner._filter_urdf_depth(depth, inp)
        points_world = points_from_depth(filtered_depth, camera_params)
        if len(points_world) == 0:
            return None

        try:
            import torch
            from curobo._src.state.state_joint import JointState
        except Exception:
            return None

        curobo = self.planner._ensure_planner()
        if curobo is None:
            return None

        state = JointState.from_position(
            torch.as_tensor(
                np.asarray(joint_positions, dtype=np.float32).reshape(1, -1),
                device=self.planner.device,
                dtype=torch.float32,
            ),
            joint_names=curobo.joint_names,
        )
        kin = curobo.compute_kinematics(state)
        robot_spheres = kin.robot_spheres.detach().cpu().numpy().reshape(-1, 4)

        tool_frame = curobo.tool_frames[0]
        sk = kin.tool_poses.get_link_pose(tool_frame)
        p0 = sk.position.detach().cpu().numpy().reshape(3)
        q0 = sk.quaternion.detach().cpu().numpy().reshape(4)
        r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()

        r_world_base, t_world_base = self.planner._base_pose_parts(robot_base_pose)
        points_base = (r_world_base.T @ (points_world - t_world_base).T).T
        remaining_base = filter_robot_points(
            points_base, robot_spheres, self.planner.robot_padding
        )
        if len(remaining_base) == 0:
            return None
        remaining_world = (r_world_base @ remaining_base.T).T + t_world_base

        eef = np.asarray(eef_pos, dtype=np.float64).reshape(3)
        local_mask = np.linalg.norm(remaining_world - eef, axis=1) <= self.roi_radius
        local_idx = np.flatnonzero(local_mask)
        if len(local_idx) < self.dbscan_min_samples:
            return None

        local_points = remaining_world[local_idx]
        labels = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit_predict(local_points)

        best_indices = np.empty(0, dtype=np.int64)
        best_distance = float("inf")
        for label in np.unique(labels):
            if label == -1:
                continue
            indices = local_idx[labels == label]
            if len(indices) < self.min_cluster_points:
                continue
            mean_distance = float(
                np.linalg.norm(remaining_world[indices] - eef, axis=1).mean()
            )
            if mean_distance < best_distance:
                best_distance = mean_distance
                best_indices = indices

        if len(best_indices) == 0:
            return None

        selected_world = remaining_world[best_indices]
        selected_base = (r_world_base.T @ (selected_world - t_world_base).T).T
        points_hand = (r0.T @ (selected_base - p0).T).T
        if len(points_hand) < self.min_cluster_points:
            return None

        return HeldObjectObservation(
            item=item or self.item,
            points_hand=points_hand.astype(np.float32),
            confidence=1.0,
            source="gpu_urdf_connected_component",
        )


__all__ = ["HeldObjectConnectedComponentExtractor"]
