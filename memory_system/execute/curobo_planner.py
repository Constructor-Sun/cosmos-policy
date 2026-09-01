"""Collision-aware cuRobo planner for one-shot initial alignment.

This module is an optional backend for ``InitialAlignmentSelector``.  It builds
a point cloud, removes the current robot using its cuRobo collision spheres,
then loads the rest as cuboid obstacles and plans a collision-free trajectory to
the selected ready pose.  If anything is unavailable or planning fails, the
caller falls back to the original direct ``PoseController``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from memory_system.execute.curobo_trajectory import (
    JointTrajectoryPlan,
    WaypointPoseController,
    backoff_pose,
    find_minimum_feasible_backoff,
    retarget_tool_pose,
)
from memory_system.execute.surface_obstacles import (
    densify_joint_path,
    filter_robot_points,
    find_trajectory_conflict,
    points_from_depth,
    select_surface_points,
)
from memory_system.types import CameraParams

logger = logging.getLogger(__name__)

@dataclass
class PlanResult:
    """Minimal result consumed by the initial-alignment correction channel."""

    controller: Any
    correction_steps: int
    joint_trajectory: JointTrajectoryPlan | None = None
    target_ee_states: np.ndarray | None = None

class CuroboPlanner:
    """Plan a collision-free EE trajectory with cuRobo from RGB-D."""

    def __init__(
        self,
        robot: str = "franka.yml",
        voxel_size: float = 0.02,
        sphere_radius: float = 0.015,
        max_spheres: int = 512,
        max_waypoints: int = 48,
        robot_padding: float = 0.015,
        path_safety_margin: float = 0.005,
        max_surface_replans: int = 2,
        device: str = "cuda:0",
        joint_execution: bool = False,
        enable_urdf_robot_filter: bool = False,
        waypoint_step_budget: int = 96,
    ):
        self.robot = robot
        self.voxel_size = float(voxel_size)
        self.sphere_radius = float(sphere_radius)
        self.max_spheres = int(max_spheres)
        self.max_waypoints = int(max_waypoints)
        self.robot_padding = float(robot_padding)
        self.path_safety_margin = float(path_safety_margin)
        self.max_surface_replans = int(max_surface_replans)
        self.device = device
        self.joint_execution = bool(joint_execution)
        self.enable_urdf_robot_filter = bool(enable_urdf_robot_filter)
        self.waypoint_step_budget = int(waypoint_step_budget)
        if self.waypoint_step_budget <= 0:
            raise ValueError("waypoint_step_budget must be positive")
        self._planner = None
        self._urdf_filter = None
        self._urdf_filter_failed = False
        self.last_debug = None

    def _ensure_urdf_filter(self):
        if not self.enable_urdf_robot_filter or self._urdf_filter_failed:
            return None
        if self._urdf_filter is None:
            try:
                import curobo
                from pathlib import Path
                from memory_system.execute.urdf_depth_filter import (
                    UrdfDepthFilter, UrdfDepthFilterConfig,
                )
                urdf = (Path(curobo.__file__).resolve().parent /
                        "content/assets/robot/franka_description/franka_panda.urdf")
                self._urdf_filter = UrdfDepthFilter(
                    UrdfDepthFilterConfig(urdf_path=str(urdf))
                )
                logger.info("CuroboPlanner: enabled URDF robot filter: %s", urdf)
            except Exception as exc:
                self._urdf_filter_failed = True
                logger.warning("CuroboPlanner: URDF filter unavailable; using spheres: %s", exc)
        return self._urdf_filter

    def _ensure_planner(self, scene_cfg):
        import torch
        from curobo._src.geom.types import SceneCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        cfg = MotionPlannerCfg.create(
            robot=self.robot,
            scene_model=scene_cfg,
            collision_cache={"cuboid": self.max_spheres},
            device_cfg=DeviceCfg(device=torch.device(self.device), dtype=torch.float32),
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=False,
            max_goalset=1,
            position_tolerance=0.005,
            orientation_tolerance=0.02,
            optimizer_collision_activation_distance=0.01,
        )
        planner = MotionPlanner(cfg)
        planner.warmup(enable_graph=True, num_warmup_iterations=2)
        self._planner = planner
        return planner

    def _points_from_depth(self, depth: np.ndarray, cam: CameraParams) -> np.ndarray:
        return points_from_depth(depth, cam)

    def _filter_robot(self, points: np.ndarray, robot_spheres: np.ndarray) -> np.ndarray:
        return filter_robot_points(points, robot_spheres, self.robot_padding)

    def _sample_points(self, points: np.ndarray, target: np.ndarray) -> np.ndarray:
        return select_surface_points(
            points, target, self.max_spheres, self.voxel_size
        ).points

    def plan(
        self,
        *,
        current_ee_states: np.ndarray,
        target_ee_states: np.ndarray,
        depth: np.ndarray | None = None,
        camera_params: CameraParams | None = None,
        joint_positions: np.ndarray | None = None,
        gripper_joint_positions: np.ndarray | None = None,
        robot_base_pose: np.ndarray | None = None,
    ) -> PlanResult | None:
        if depth is None or camera_params is None or joint_positions is None:
            logger.warning("CuroboPlanner: missing depth/camera/joint positions; using direct PoseController")
            return None
        try:
            import torch
            from scipy.spatial.transform import Rotation
            from curobo._src.geom.types import Cuboid, SceneCfg
            from curobo._src.state.state_joint import JointState
            from curobo._src.types.pose import Pose
            from curobo._src.types.tool_pose import GoalToolPose
        except Exception as exc:
            logger.warning("CuroboPlanner: cuRobo import failed: %s", exc)
            return None
        depth_for_points = depth
        urdf_filter = self._ensure_urdf_filter()
        if urdf_filter is not None:
            try:
                arm = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
                fingers = np.asarray(
                    [0.04, 0.04] if gripper_joint_positions is None
                    else gripper_joint_positions,
                    dtype=np.float64,
                ).reshape(-1)
                urdf_joints = np.concatenate([arm, np.clip(np.abs(fingers[:2]), 0.0, 0.04)])
                if len(urdf_joints) != len(urdf_filter.joint_names):
                    raise ValueError(
                        f"URDF expects {len(urdf_filter.joint_names)} joints, got {len(urdf_joints)}"
                    )
                filtered = urdf_filter.filter(
                    depth,
                    camera_params=camera_params,
                    joint_positions=urdf_joints,
                    robot_base_pose=robot_base_pose,
                )
                depth_for_points = filtered.filtered_depth
                logger.info("CuroboPlanner: URDF removed %d depth pixels", filtered.removed_pixel_count)
            except Exception as exc:
                logger.warning("CuroboPlanner: URDF filtering failed; using spheres: %s", exc)
        points = self._points_from_depth(depth_for_points, camera_params)
        if len(points) == 0:
            logger.warning("CuroboPlanner: no valid depth points; using direct PoseController")
            return None
        try:
            planner = self._ensure_planner(SceneCfg())
        except Exception as exc:
            logger.warning("CuroboPlanner: planner init failed: %s", exc)
            return None
        try:
            start = JointState.from_position(
                torch.tensor(
                    np.asarray(joint_positions, dtype=np.float32).reshape(1, -1),
                    device=self.device,
                    dtype=torch.float32,
                ),
                joint_names=planner.joint_names,
            )
            tool_frames = planner.tool_frames
            start_kinematics = planner.compute_kinematics(start)
            sk = start_kinematics.tool_poses.get_link_pose(tool_frames[0])
            p0 = sk.position.detach().cpu().numpy().reshape(3)
            q0 = sk.quaternion.detach().cpu().numpy().reshape(4)
            R0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
            ce = np.asarray(current_ee_states, dtype=np.float64).reshape(6)
            Rw = R0 @ Rotation.from_rotvec(ce[3:]).as_matrix().T
            tw = p0 - Rw @ ce[:3]
            target = np.asarray(target_ee_states, dtype=np.float64).reshape(6)
            if robot_base_pose is not None:
                base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
                base_rotation = Rotation.from_quat(base[3:]).as_matrix()
                points = (base_rotation.T @ (points - base[:3]).T).T
                sample_target = base_rotation.T @ (target[:3] - base[:3])
            else:
                points = (Rw @ points.T).T + tw
                sample_target = Rw @ target[:3] + tw
            robot_spheres = start_kinematics.robot_spheres.detach().cpu().numpy()
            surface_points = self._filter_robot(points, robot_spheres)
            if robot_base_pose is not None:
                r_world_base = base_rotation
                t_world_base = base[:3]
            else:
                r_world_base = Rw.T
                t_world_base = -Rw.T @ tw
            self.last_debug = {
                "r_world_base": r_world_base,
                "t_world_base": t_world_base,
                "surface_points": surface_points,
                "target": target,
                "last_conflict_info": None,
                "nearest_obstacle_to_target": None,
            }
            if len(surface_points) == 0:
                logger.warning("CuroboPlanner: all depth points belong to the robot")
                return None
            width = min(self.voxel_size, 0.01)
            mandatory = np.zeros((0, 3), dtype=np.float64)

            def update_surface_world():
                selection = select_surface_points(
                    surface_points, sample_target, self.max_spheres,
                    self.voxel_size, mandatory_points=mandatory,
                )
                obstacles = [Cuboid(
                    name=f"obs_{i}",
                    pose=[*map(float, point), 1.0, 0.0, 0.0, 0.0],
                    dims=[width] * 3,
                ) for i, point in enumerate(selection.points)]
                planner.update_world(SceneCfg(cuboid=obstacles))
                logger.info(
                    "CuroboPlanner: surfaces=%d global=%d target=%d mandatory=%d",
                    len(surface_points), selection.global_count,
                    selection.target_count, selection.mandatory_count,
                )

            update_surface_world()

            def attempt(distance):
                nonlocal mandatory
                candidate = backoff_pose(target, distance)
                if self.joint_execution and robot_base_pose is not None:
                    tp, tr = retarget_tool_pose(ce, candidate, p0, R0, robot_base_pose)
                else:
                    tp = Rw @ candidate[:3] + tw
                    tr = Rotation.from_matrix(Rw @ Rotation.from_rotvec(candidate[3:]).as_matrix()).as_rotvec()
                quat = Rotation.from_rotvec(tr).as_quat()[[3, 0, 1, 2]]
                goal = Pose(position=torch.as_tensor(tp, device=self.device, dtype=torch.float32).reshape(1, 1, 3), quaternion=torch.as_tensor(quat, device=self.device, dtype=torch.float32).reshape(1, 1, 4))
                goal_tools = GoalToolPose.from_poses({tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames)
                for refinement in range(self.max_surface_replans + 1):
                    outcome = planner.plan_pose(
                        current_state=start, goal_tool_poses=goal_tools, max_attempts=3
                    )
                    if outcome is None or not bool(torch.as_tensor(outcome.success).any().item()):
                        logger.info("CuroboPlanner: backoff %.4fm is infeasible", distance)
                        return None
                    interp = outcome.get_interpolated_plan()
                    path = JointTrajectoryPlan.from_curobo(
                        interp, joint_names=planner.joint_names
                    ).position
                    dense = densify_joint_path(path)
                    state = JointState.from_position(
                        torch.as_tensor(dense, device=self.device, dtype=torch.float32),
                        joint_names=planner.joint_names,
                    )
                    spheres = planner.compute_kinematics(state).robot_spheres
                    conflict = find_trajectory_conflict(
                        surface_points, spheres.detach().cpu().numpy(),
                        self.path_safety_margin,
                    )
                    if conflict is None:
                        return candidate, outcome, interp
                    if self.last_debug is not None:
                        self.last_debug["last_conflict_info"] = {
                            "point": conflict.min_clearance_point,
                            "sphere_center": conflict.min_clearance_sphere_center,
                            "clearance": conflict.min_clearance,
                            "step": conflict.min_clearance_step,
                        }
                    logger.info(
                        "CuroboPlanner: dense surface conflict steps=%d..%d "
                        "clearance=%.4fm refinement=%d",
                        conflict.first_step, conflict.last_step,
                        conflict.min_clearance, refinement,
                    )
                    if refinement >= self.max_surface_replans:
                        return None
                    mandatory = np.concatenate(
                        [mandatory, surface_points[conflict.point_indices]], axis=0
                    )
                    update_surface_world()

            selected = find_minimum_feasible_backoff(attempt)
            if selected is None:
                logger.warning("CuroboPlanner: no feasible goal within 0.080m backoff")
                return None
            backoff, (target, result, interp) = selected
            if self.last_debug is not None:
                self.last_debug["target"] = target
            logger.info("CuroboPlanner: selected collision-free backoff=%.4fm", backoff)
            if self.joint_execution:
                trajectory = JointTrajectoryPlan.from_curobo(
                    interp, joint_names=planner.joint_names
                )
                return PlanResult(
                    controller=None,
                    correction_steps=64,
                    joint_trajectory=trajectory,
                    target_ee_states=target.astype(np.float32),
                )
            interp_pos = interp.position
            if interp_pos.ndim == 4:
                interp_pos = interp_pos.squeeze(1)
            interp_pos = interp_pos[..., :len(planner.joint_names)].contiguous()
            interp = JointState.from_position(interp_pos, joint_names=planner.joint_names)
            kin = planner.compute_kinematics(interp)
            tool_pose = kin.tool_poses.get_link_pose(tool_frames[0])
            pos = tool_pose.position.detach().cpu().numpy()
            quat_w = tool_pose.quaternion.detach().cpu().numpy()
            quat_xy = quat_w[:, [1, 2, 3, 0]]
            rotvec = Rotation.from_quat(quat_xy).as_rotvec()
            waypoints = np.concatenate([pos, rotvec], axis=-1).astype(np.float32)
            if len(waypoints) == 0:
                return None
            if len(waypoints) > self.max_waypoints:
                idx = np.linspace(0, len(waypoints) - 1, self.max_waypoints).astype(int)
                waypoints = waypoints[idx]

            # Convert waypoints from cuRobo's planning frame back to the
            # LIBERO/MuJoCo world frame used by the correction executor.
            # p_cu = Rw @ p_libero + tw  =>  p_libero = Rw^T @ (p_cu - tw)
            # R_cu = Rw @ R_libero       =>  R_libero = Rw^T @ R_cu
            pos_cu = waypoints[:, :3]
            rotvec_cu = waypoints[:, 3:]
            pos_libero = (Rw.T @ (pos_cu - tw).T).T
            rotmat_cu = Rotation.from_rotvec(rotvec_cu).as_matrix()
            rotmat_libero = Rw.T @ rotmat_cu
            rotvec_libero = Rotation.from_matrix(rotmat_libero).as_rotvec()
            waypoints = np.concatenate([pos_libero, rotvec_libero], axis=-1).astype(np.float32)

            # Waypoint count is path resolution, not the number of env steps.
            controller = WaypointPoseController(
                waypoints, max_steps=self.waypoint_step_budget
            )
            return PlanResult(
                controller=controller,
                correction_steps=self.waypoint_step_budget,
            )
        except Exception as exc:
            logger.warning("CuroboPlanner: planning failed: %s", exc)
            return None
