"""Motion primitives for lightweight ready-pose movement.

This module mirrors the motion-primitive layer of the original reference
approach: solve IK for a Cartesian target, execute a joint-space blocking
controller, and provide a convenient way to move through a small set of
waypoints.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.execute.curobo_trajectory import JointTrajectoryPlan
from memory_system.pointcloud_action.config import (
    READY_MOTION_CONTROL_DT,
    READY_MOTION_IK_SEEDS,
    READY_MOTION_JOINT_INTERP_STEPS,
    READY_MOTION_MAX_JOINT_STEP,
    READY_MOTION_MAX_MID_POINTS,
)
from memory_system.pointcloud_action.execute.mujoco_ik import solve_ee_ik
from memory_system.pointcloud_action.execute.ready_motion_planner import (
    ReadyMotionPlan,
)


class MotionPrimitives:
    """Thin IK + joint-space execution wrapper used by ready-pose motions."""

    def __init__(self, env, gripper_command: float = -1.0):
        self.env = env
        self.sim = env.env.sim
        self.gripper_command = float(gripper_command)

    def solve_ik(
        self,
        position: np.ndarray,
        rotvec: np.ndarray,
        prev_q: np.ndarray | None = None,
        num_seeds: int = READY_MOTION_IK_SEEDS,
    ) -> np.ndarray | None:
        """Solve IK, preferring a solution close to ``prev_q``.

        The first attempted seed is always the previous joint configuration.
        If that branch cannot reach the target, additional deterministic and
        randomized seeds are tried so that a different joint branch can be
        found instead of failing outright.
        """
        saved_qpos = self.sim.data.qpos.copy()
        target_pos = np.asarray(position, dtype=np.float64).reshape(3)
        target_rot = Rotation.from_rotvec(
            np.asarray(rotvec, dtype=np.float64).reshape(3)
        ).as_matrix()
        ranges = self._joint_ranges()

        prev = None if prev_q is None else np.asarray(prev_q, dtype=np.float64).reshape(7)
        seeds: list[np.ndarray] = []
        if prev is not None:
            seeds.append(prev.copy())
        else:
            seeds.append(np.zeros(7, dtype=np.float64))

        # A few deterministic, commonly useful seeds.
        seeds.append(np.array([0.0, -0.5, 0.0, -2.5, 0.0, 2.5, 0.5]))
        seeds.append(np.array([0.0, 0.0, 0.0, -1.5, 0.0, 2.0, 0.0]))

        rng = np.random.RandomState(12345)
        while len(seeds) < max(1, int(num_seeds)):
            seeds.append(rng.uniform(ranges[:, 0], ranges[:, 1]))

        best_q = None
        best_cost = float("inf")
        for seed in seeds:
            seed = np.clip(
                np.asarray(seed, dtype=np.float64).reshape(7),
                ranges[:, 0],
                ranges[:, 1],
            )
            q = solve_ee_ik(
                self.sim,
                target_pos,
                target_rot,
                init_q=seed,
            )
            if q is None:
                continue
            q = np.asarray(q, dtype=np.float64).reshape(7)
            if prev is None:
                best_q = q
                break
            cost = float(np.max(np.abs(q - prev)))
            if cost < best_cost:
                best_cost = cost
                best_q = q

        self.sim.data.qpos[:] = saved_qpos
        self.sim.forward()
        return best_q

    def _joint_ranges(self) -> np.ndarray:
        ranges = []
        for index in range(self.sim.model.njnt):
            name = self.sim.model.joint_names[index]
            if name.startswith("robot0_joint"):
                ranges.append(
                    (
                        float(self.sim.model.jnt_range[index, 0]),
                        float(self.sim.model.jnt_range[index, 1]),
                    )
                )
        return np.asarray(ranges, dtype=np.float64).reshape(-1, 2)

    def build_controller(
        self,
        plan: ReadyMotionPlan,
        obs: dict,
        max_steps: int = 200,
        time_scale: float = 2.0,
        joint_interp_steps: int = READY_MOTION_JOINT_INTERP_STEPS,
        dt: float = READY_MOTION_CONTROL_DT,
    ):
        """Build a joint-space controller for a Cartesian waypoint plan."""
        from cosmos_policy.experiments.robot.libero.libero_joint_control import (
            LiberoJointTrajectoryController,
        )

        q_current = np.asarray(
            obs["robot0_joint_pos"], dtype=np.float64
        ).reshape(7).copy()
        q_path = [q_current]
        previous_ee = plan.ee_waypoints[0]

        for waypoint in plan.ee_waypoints[1:]:
            q = self.solve_ik(waypoint[:3], waypoint[3:], prev_q=q_path[-1])
            if q is None:
                return None

            # If the IK solution jumps to a distant joint branch, refine the
            # segment with a few intermediate full-pose waypoints so that the
            # final joint path does not contain an unnaturally large jump.
            if np.max(np.abs(q - q_path[-1])) > READY_MOTION_MAX_JOINT_STEP:
                for t in np.linspace(
                    0.0, 1.0, READY_MOTION_MAX_MID_POINTS + 2
                )[1:-1]:
                    mid_ee = self._interpolate_ee(previous_ee, waypoint, t)
                    q_mid = self.solve_ik(
                        mid_ee[:3],
                        mid_ee[3:],
                        prev_q=q_path[-1],
                    )
                    if q_mid is None:
                        return None
                    q_path.append(q_mid)
                q = self.solve_ik(
                    waypoint[:3], waypoint[3:], prev_q=q_path[-1]
                )
                if q is None:
                    return None

            q_path.append(q)
            previous_ee = waypoint

        joint_plan = self._build_joint_plan(
            q_path,
            steps_per_segment=joint_interp_steps,
            dt=dt,
        )
        return LiberoJointTrajectoryController(
            self.env,
            joint_plan,
            plan.target_ee,
            gripper_command=self.gripper_command,
            max_steps=max_steps,
            time_scale=time_scale,
        )

    @staticmethod
    def _interpolate_ee(
        start: np.ndarray,
        end: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Linearly interpolate position and slerp orientation between EE poses."""
        start = np.asarray(start, dtype=np.float64).reshape(6)
        end = np.asarray(end, dtype=np.float64).reshape(6)
        pos = start[:3] * (1.0 - t) + end[:3] * t
        q1 = Rotation.from_rotvec(start[3:]).as_quat()
        q2 = Rotation.from_rotvec(end[3:]).as_quat()
        q = MotionPrimitives._slerp(q1, q2, t)
        rotvec = Rotation.from_quat(q).as_rotvec()
        return np.concatenate([pos, rotvec])

    @staticmethod
    def _slerp(
        q1: np.ndarray,
        q2: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Spherical linear interpolation between two quaternions."""
        q1 = np.asarray(q1, dtype=np.float64).reshape(4).copy()
        q2 = np.asarray(q2, dtype=np.float64).reshape(4).copy()
        dot = float(np.dot(q1, q2))
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))

        if dot > 0.9995:
            q = q1 + t * (q2 - q1)
        else:
            theta = float(np.arccos(dot))
            q = (
                np.sin((1.0 - t) * theta) * q1
                + np.sin(t * theta) * q2
            ) / np.sin(theta)

        norm = float(np.linalg.norm(q))
        if norm > 0.0:
            q = q / norm
        return q

    @staticmethod
    def _build_joint_plan(
        q_path: list[np.ndarray],
        steps_per_segment: int,
        dt: float,
    ) -> JointTrajectoryPlan:
        """Interpolate IK solutions into a JointTrajectoryPlan."""
        positions = []
        for start, end in zip(q_path[:-1], q_path[1:]):
            for index in range(steps_per_segment):
                ratio = index / steps_per_segment
                positions.append(start + (end - start) * ratio)
        positions.append(q_path[-1])

        position = np.asarray(positions, dtype=np.float64)
        return JointTrajectoryPlan(
            joint_names=tuple(f"robot0_joint{i}" for i in range(position.shape[1])),
            position=position,
            velocity=np.zeros_like(position),
            acceleration=np.zeros_like(position),
            dt=float(dt),
        )
