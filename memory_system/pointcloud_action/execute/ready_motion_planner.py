"""Lightweight ready-pose waypoint construction.

This module builds a small set of Cartesian waypoints for moving the gripper
from its current pose to a stored ready pose.  It is intentionally simple and
does not use a global collision planner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.pointcloud_action.config import (
    READY_MOTION_APPROACH_DISTANCE,
    READY_MOTION_LIFT_HEIGHT,
    READY_MOTION_MAX_WAYPOINTS,
)


@dataclass
class ReadyMotionPlan:
    """A small Cartesian waypoint plan for the ready-pose motion."""

    ee_waypoints: np.ndarray  # (N, 6) = [x, y, z, rotvec]
    target_ee: np.ndarray     # (6,)


class ReadyMotionPlanner:
    """Generate a fixed-template waypoint path from current EE to ready EE."""

    def __init__(
        self,
        current_ee: np.ndarray,
        ready_ee: np.ndarray,
        lift_height: float = READY_MOTION_LIFT_HEIGHT,
        approach_distance: float = READY_MOTION_APPROACH_DISTANCE,
        max_waypoints: int = READY_MOTION_MAX_WAYPOINTS,
    ):
        self.current_ee = np.asarray(current_ee, dtype=np.float64).reshape(6).copy()
        self.ready_ee = np.asarray(ready_ee, dtype=np.float64).reshape(6).copy()
        self.lift_height = float(lift_height)
        self.approach_distance = float(approach_distance)
        self.max_waypoints = int(max_waypoints)

    def plan(self) -> ReadyMotionPlan:
        waypoints = self._build_waypoints()
        return ReadyMotionPlan(
            ee_waypoints=np.asarray(waypoints, dtype=np.float64),
            target_ee=self.ready_ee,
        )

    def _build_waypoints(self) -> list[np.ndarray]:
        current = self.current_ee.copy()
        ready = self.ready_ee.copy()

        # 1. Lift from the current location before any large horizontal motion.
        lifted = current.copy()
        lifted[:3] += np.array([0.0, 0.0, self.lift_height])

        # 2. Move above/before the target at a safe height.
        high_z = max(current[2], ready[2]) + self.lift_height
        above = ready.copy()
        above[:3] = np.array([ready[0], ready[1], high_z])

        # 3. Approach pose, placed along the ready pose's local approach axis.
        approach = ready.copy()
        rotation = Rotation.from_rotvec(ready[3:])
        approach[:3] = ready[:3] + rotation.apply(
            np.array([0.0, 0.0, -self.approach_distance])
        )

        candidates = [current, lifted, above, approach, ready]

        # Keep the plan small and deterministic.
        if len(candidates) <= self.max_waypoints:
            return candidates

        indices = np.linspace(
            0, len(candidates) - 1, self.max_waypoints
        ).astype(int)
        return [candidates[int(i)] for i in indices]
