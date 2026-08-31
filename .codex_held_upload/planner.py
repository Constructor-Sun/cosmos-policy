"""Collision-aware planner for a robot carrying a held object."""
from __future__ import annotations
import logging
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from memory_system.execute.curobo_trajectory import (
    WaypointPoseController,
    backoff_pose,
    find_minimum_feasible_backoff,
)
from memory_system.execute.planner.held_object.types import (
    HeldObjectPlannerInput,
    HeldObjectPlannerResult,
)
from memory_system.execute.surface_obstacles import (
    densify_joint_path,
    filter_robot_points,
    find_trajectory_conflict,
    points_from_depth,
    select_surface_points,
)
from memory_system.geometry import world_to_pixel
logger = logging.getLogger(__name__)
class HeldObjectPlanner:
    """Plan a waypoint/OSC trajectory while carrying an object."""
    def __init__(
        self,
        robot: str = "franka.yml",
        voxel_size: float = 0.02,
        max_spheres: int = 512,
        max_waypoints: int = 48,
        robot_padding: float = 0.015,
        path_safety_margin: float = 0.005,
        device: str = "cuda:0",
        waypoint_step_budget: int = 96,
        attachment_slots: int = 32,
        attachment_padding: float = 0.005,
        enable_urdf_robot_filter: bool = True,
    ) -> None:
        self.robot = robot
        self.voxel_size = float(voxel_size)
        self.max_spheres = int(max_spheres)
        self.max_waypoints = int(max_waypoints)
        self.robot_padding = float(robot_padding)
        self.path_safety_margin = float(path_safety_margin)
        self.device = device
        self.waypoint_step_budget = int(waypoint_step_budget)
        self.attachment_slots = int(attachment_slots)
        self.attachment_padding = float(attachment_padding)
        self.enable_urdf_robot_filter = bool(enable_urdf_robot_filter)
        self._planner = None
        self._urdf_filter = None
        self._urdf_filter_failed = False
    def _ensure_urdf_filter(self):
        if not self.enable_urdf_robot_filter or self._urdf_filter_failed:
            return None
        if self._urdf_filter is None:
            try:
                import curobo
                from memory_system.execute.urdf_depth_filter import UrdfDepthFilter, UrdfDepthFilterConfig
                urdf = (Path(curobo.__file__).resolve().parent / "content/assets/robot/franka_description/franka_panda.urdf")
                self._urdf_filter = UrdfDepthFilter(UrdfDepthFilterConfig(urdf_path=str(urdf)))
                logger.info("HeldObjectPlanner: enabled URDF robot filter: %s", urdf)
            except Exception as exc:
                self._urdf_filter_failed = True
                logger.warning("HeldObjectPlanner: URDF filter unavailable; using spheres: %s", exc)
        return self._urdf_filter
    def _ensure_planner(self):
        if self._planner is not None:
            return self._planner
        import torch
        from curobo._src.geom.types import SceneCfg
        from curobo._src.motion.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_robot_configs_path
        robot_cfg = load_yaml(str(Path(get_robot_configs_path()) / self.robot))
        robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = self.attachment_slots
        cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=SceneCfg(),
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
    @staticmethod
    def _build_spheres(points_hand, max_slots, voxel_size, padding):
        """Build a conservative sphere cover of the observed object bounds.

        Spheres placed only at occupied point-cloud voxels cover the sampled
        points, but can leave gaps between sparse samples.  Covering the observed
        bounding box is deliberately conservative and also gives a useful model
        for removing the held object from the static depth cloud.
        """
        del voxel_size  # Kept in the helper signature for compatibility.
        points = np.asarray(points_hand, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        max_slots = max(1, int(max_slots))
        if len(points) == 0:
            return None

        margin = max(float(padding), 1e-4)
        lower = points.min(axis=0) - margin
        upper = points.max(axis=0) + margin
        extent = np.maximum(upper - lower, 1e-4)

        # Split the longest current cell until the slot budget is reached.
        grid = np.ones(3, dtype=np.int64)
        while True:
            candidates = []
            for axis in range(3):
                candidate = grid.copy()
                candidate[axis] += 1
                if int(np.prod(candidate)) <= max_slots:
                    candidates.append(candidate)
            if not candidates:
                break
            grid = min(
                candidates,
                key=lambda candidate: float(np.max(extent / candidate)),
            )

        cell = extent / grid.astype(np.float64)
        axes = [
            lower[axis]
            + (np.arange(grid[axis], dtype=np.float64) + 0.5) * cell[axis]
            for axis in range(3)
        ]
        centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        radius = float(0.5 * np.linalg.norm(cell) + margin)
        return np.concatenate([centers, np.full((len(centers), 1), radius)], axis=1)

    @staticmethod
    def _remove_attached_points(
        points_base,
        spheres_hand,
        hand_position_base,
        hand_rotation_base,
        padding,
    ):
        """Remove points occupied by the current attachment in 3-D."""
        points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
        spheres = np.asarray(spheres_hand, dtype=np.float64).reshape(-1, 4)
        if len(points) == 0 or len(spheres) == 0:
            return points
        rotation = np.asarray(hand_rotation_base, dtype=np.float64).reshape(3, 3)
        position = np.asarray(hand_position_base, dtype=np.float64).reshape(3)
        centers = (rotation @ spheres[:, :3].T).T + position
        clearance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        clearance -= spheres[None, :, 3]
        return points[np.all(clearance > float(padding), axis=1)]

    @staticmethod
    def _tool_mapping(current_ee, current_tool_position, current_tool_rotation):
        """Return the rigid map from external EEF coordinates to cuRobo tool."""
        current_ee = np.asarray(current_ee, dtype=np.float64).reshape(6)
        ee_rotation = Rotation.from_rotvec(current_ee[3:]).as_matrix()
        tool_rotation = np.asarray(current_tool_rotation, dtype=np.float64).reshape(3, 3)
        tool_position = np.asarray(current_tool_position, dtype=np.float64).reshape(3)
        # This is the same calibration used by CuroboPlanner's pose-waypoint
        # branch: p_tool = Rw @ p_ee + tw and R_tool = Rw @ R_ee.
        rotation = tool_rotation @ ee_rotation.T
        translation = tool_position - rotation @ current_ee[:3]
        return rotation, translation
    @staticmethod
    def _base_pose_parts(robot_base_pose):
        if robot_base_pose is None:
            return np.eye(3), np.zeros(3)
        base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
        return Rotation.from_quat(base[3:]).as_matrix(), base[:3]
    def _held_object_mask(self, points_hand, p0, r0, camera_params, robot_base_pose):
        r_world_base, t_world_base = self._base_pose_parts(robot_base_pose)
        hand_world = (r0 @ points_hand.T).T + p0
        points_world = (r_world_base @ hand_world.T).T + t_world_base
        pixels = world_to_pixel(points_world, camera_params)
        height, width = int(camera_params.height), int(camera_params.width)
        mask = np.zeros((height, width), dtype=np.uint8)
        rows = np.rint(pixels[:, 0]).astype(np.int64)
        cols = np.rint(pixels[:, 1]).astype(np.int64)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        mask[rows[valid], cols[valid]] = 1
        return cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8)) > 0
    def _target_tool_pose(self, current_ee, target_ee, p0, r0, robot_base_pose):
        del robot_base_pose
        rotation, translation = self._tool_mapping(current_ee, p0, r0)
        target = np.asarray(target_ee, dtype=np.float64).reshape(6)
        target_rotation = Rotation.from_rotvec(target[3:]).as_matrix()
        return (
            rotation @ target[:3] + translation,
            Rotation.from_matrix(rotation @ target_rotation).as_rotvec(),
        )
    def _filter_urdf_depth(self, depth, inp):
        urdf_filter = self._ensure_urdf_filter()
        if urdf_filter is None:
            return depth
        try:
            arm = np.asarray(inp.joint_positions, dtype=np.float64).reshape(-1)
            fingers = np.asarray(
                [0.04, 0.04] if inp.gripper_joint_positions is None else inp.gripper_joint_positions,
                dtype=np.float64,
            ).reshape(-1)
            urdf_joints = np.concatenate([arm, np.clip(np.abs(fingers[:2]), 0.0, 0.04)])
            if len(urdf_joints) != len(urdf_filter.joint_names):
                return depth
            result = urdf_filter.filter(
                depth,
                camera_params=inp.camera_params,
                joint_positions=urdf_joints,
                robot_base_pose=inp.robot_base_pose,
            )
            return result.filtered_depth
        except Exception:
            return depth
    def plan(self, inp: HeldObjectPlannerInput) -> HeldObjectPlannerResult | None:
        if inp.depth is None or inp.camera_params is None or inp.joint_positions is None:
            return None
        held_object = inp.held_object
        if held_object is None or not bool(getattr(held_object, "valid", True)):
            logger.warning(
                "HeldObjectPlanner: refusing invalid held-object observation: %s",
                getattr(held_object, "rejection_reason", "missing"),
            )
            return None
        try:
            import torch
            from curobo._src.geom.types import Cuboid, SceneCfg
            from curobo._src.state.state_joint import JointState
            from curobo._src.types.pose import Pose
            from curobo._src.types.tool_pose import GoalToolPose
        except Exception:
            return None
        try:
            planner = self._ensure_planner()
            start = JointState.from_position(
                torch.tensor(
                    np.asarray(inp.joint_positions, dtype=np.float32).reshape(1, -1),
                    device=self.device,
                    dtype=torch.float32,
                ),
                joint_names=planner.joint_names,
            )
            tool_frame = planner.tool_frames[0]
            kin0 = planner.compute_kinematics(start)
            sk = kin0.tool_poses.get_link_pose(tool_frame)
            p0 = sk.position.detach().cpu().numpy().reshape(3)
            q0 = sk.quaternion.detach().cpu().numpy().reshape(4)
            r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
            points_hand = np.asarray(inp.held_object.points_hand, dtype=np.float64).reshape(-1, 3)
            if len(points_hand) < 3:
                return None
            spheres_hand = self._build_spheres(
                points_hand, self.attachment_slots, self.voxel_size, self.attachment_padding
            )
            if spheres_hand is None:
                return None
            depth_for_points = np.asarray(inp.depth, dtype=np.float64).copy()
            if depth_for_points.ndim == 3:
                depth_for_points = depth_for_points[..., 0]
            depth_for_points = self._filter_urdf_depth(depth_for_points, inp)
            points = points_from_depth(depth_for_points, inp.camera_params)
            r_world_base, t_world_base = self._base_pose_parts(inp.robot_base_pose)
            points_base = (r_world_base.T @ (points - t_world_base).T).T
            points_base = self._remove_attached_points(
                points_base,
                spheres_hand,
                p0,
                r0,
                self.robot_padding,
            )
            surface_points = filter_robot_points(points_base, kin0.robot_spheres.detach().cpu().numpy(), self.robot_padding)
            if len(surface_points) == 0:
                return None
            attached = False
            try:
                planner.attachment_manager.update(
                    torch.as_tensor(spheres_hand, device=self.device, dtype=torch.float32),
                    start,
                    link_name="attached_object",
                )
                attached = True
                target = np.asarray(inp.ready_pose, dtype=np.float64).reshape(6)
                tp, tr = self._target_tool_pose(inp.ee_states, target, p0, r0, inp.robot_base_pose)
                quat = Rotation.from_rotvec(tr).as_quat()[[3, 0, 1, 2]]
                goal = Pose(
                    position=torch.as_tensor(tp, device=self.device, dtype=torch.float32).reshape(1, 1, 3),
                    quaternion=torch.as_tensor(quat, device=self.device, dtype=torch.float32).reshape(1, 1, 4),
                )
                width = min(self.voxel_size, 0.01)
                selection = select_surface_points(surface_points, tp, self.max_spheres, self.voxel_size)
                obstacles = [
                    Cuboid(
                        name=f"obs_{i}",
                        pose=[*map(float, point), 1.0, 0.0, 0.0, 0.0],
                        dims=[width] * 3,
                    )
                    for i, point in enumerate(selection.points)
                ]
                planner.update_world(SceneCfg(cuboid=obstacles))
                def attempt(distance):
                    candidate = backoff_pose(target, distance)
                    cp, cr = self._target_tool_pose(inp.ee_states, candidate, p0, r0, inp.robot_base_pose)
                    cquat = Rotation.from_rotvec(cr).as_quat()[[3, 0, 1, 2]]
                    cgoal = Pose(
                        position=torch.as_tensor(cp, device=self.device, dtype=torch.float32).reshape(1, 1, 3),
                        quaternion=torch.as_tensor(cquat, device=self.device, dtype=torch.float32).reshape(1, 1, 4),
                    )
                    outcome = planner.plan_pose(
                        current_state=start,
                        goal_tool_poses=GoalToolPose.from_poses(
                            {tool_frame: cgoal.unsqueeze(1)}, ordered_tool_frames=[tool_frame]
                        ),
                        max_attempts=5,
                    )
                    if outcome is None or not bool(torch.as_tensor(outcome.success).any().item()):
                        return None
                    interp = outcome.get_interpolated_plan()
                    path = interp.position
                    if path.ndim == 4:
                        path = path.squeeze(1)
                    path = path[..., : len(planner.joint_names)].contiguous().detach().cpu().numpy()
                    dense = densify_joint_path(path)
                    state = JointState.from_position(
                        torch.as_tensor(dense, device=self.device, dtype=torch.float32),
                        joint_names=planner.joint_names,
                    )
                    conflict = find_trajectory_conflict(
                        surface_points,
                        planner.compute_kinematics(state).robot_spheres.detach().cpu().numpy(),
                        self.path_safety_margin,
                    )
                    if conflict is None:
                        return candidate, outcome, interp
                    return None
                selected = find_minimum_feasible_backoff(attempt)
                if selected is None:
                    return None
                _, (target_used, _, interp) = selected
                interp_pos = interp.position
                if interp_pos.ndim == 4:
                    interp_pos = interp_pos.squeeze(1)
                interp_pos = interp_pos[..., : len(planner.joint_names)].contiguous()
                interp = JointState.from_position(interp_pos, joint_names=planner.joint_names)
                kin = planner.compute_kinematics(interp)
                tool = kin.tool_poses.get_link_pose(tool_frame)
                pos = tool.position.detach().cpu().numpy()
                quat_w = tool.quaternion.detach().cpu().numpy()
                rotvec = Rotation.from_quat(quat_w[:, [1, 2, 3, 0]]).as_rotvec()
                waypoints = np.concatenate([pos, rotvec], axis=-1).astype(np.float32)
                if len(waypoints) == 0:
                    return None
                if len(waypoints) > self.max_waypoints:
                    idx = np.linspace(0, len(waypoints) - 1, self.max_waypoints).astype(int)
                    waypoints = waypoints[idx]
                # Convert cuRobo panda_hand waypoints back to the external
                # robot0_eef frame consumed by WaypointPoseController.  This is
                # the inverse of the Rw/tw calibration used for the goal above.
                rw, tw = self._tool_mapping(inp.ee_states, p0, r0)
                eef_pos = (rw.T @ (waypoints[:, :3] - tw).T).T
                eef_rot = Rotation.from_matrix(
                    rw.T @ Rotation.from_rotvec(waypoints[:, 3:]).as_matrix()
                ).as_rotvec()
                waypoints = np.concatenate([eef_pos, eef_rot], axis=-1).astype(np.float32)
                controller = WaypointPoseController(waypoints, max_steps=self.waypoint_step_budget)
                return HeldObjectPlannerResult(
                    controller=controller,
                    correction_steps=self.waypoint_step_budget,
                    target_ee_states=target_used.astype(np.float32),
                    waypoints=waypoints,
                )
            finally:
                if attached:
                    planner.attachment_manager.detach("attached_object")
        except Exception:
            logger.exception("HeldObjectPlanner: planning failed")
            return None
