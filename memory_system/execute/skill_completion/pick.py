"""Rule-based 3D completion check for the Pick skill.

Pick completes when the gripper is non-empty closed, the target and end
effector accumulate enough shared upward motion while preserving their
relative 3D position from frame to frame, and those conditions remain true
for a configured number of consecutive frames.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.execute.skill_completion.base import TimedSkillCompletion

EMPTY_CLOSED_GAP = 0.003
DEFAULT_MIN_LIFT_DISTANCE = 0.02
DEFAULT_MAX_RIGID_ERROR = 0.01
DEFAULT_STABLE_FRAMES = 5


def _vector(value: Any, size: int) -> np.ndarray | None:
    """Return one finite vector of the requested size."""
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != size or not np.isfinite(vector).all():
        return None
    return vector


def _target_center(target_points: Any) -> np.ndarray | None:
    """Return the coordinate-wise median of finite target-cloud points."""
    if target_points is None:
        return None
    try:
        points = np.asarray(target_points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return None
    return np.median(finite, axis=0)


class PickCompletionChecker:
    """Stateful checker for one active Pick step.

    Args:
        empty_closed_gap: A closed gap at or below this value is an empty grasp.
        min_lift_distance: Minimum shared upward motion accumulated from the
            target center and EEF position.
        max_rigid_error: Maximum frame-to-frame drift of the target center
            expressed in the EEF frame.
        stable_frames: Number of consecutive matching frames required.
    """

    def __init__(
        self,
        *,
        empty_closed_gap: float = EMPTY_CLOSED_GAP,
        min_lift_distance: float = DEFAULT_MIN_LIFT_DISTANCE,
        max_rigid_error: float = DEFAULT_MAX_RIGID_ERROR,
        stable_frames: int = DEFAULT_STABLE_FRAMES,
    ) -> None:
        if empty_closed_gap < 0:
            raise ValueError("empty_closed_gap must be non-negative")
        if min_lift_distance <= 0:
            raise ValueError("min_lift_distance must be positive")
        if max_rigid_error < 0:
            raise ValueError("max_rigid_error must be non-negative")
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least one")
        self.empty_closed_gap = float(empty_closed_gap)
        self.min_lift_distance = float(min_lift_distance)
        self.max_rigid_error = float(max_rigid_error)
        self.stable_frames = int(stable_frames)
        self.reset()

    def reset(self) -> None:
        """Reset all state for a new Pick step."""
        self.completed = False
        self.last_gripper_gap: float | None = None
        self.last_object_dz: float | None = None
        self.last_eef_dz: float | None = None
        self.last_rigid_error: float | None = None
        self._reset_candidate()

    def _reset_candidate(self) -> None:
        self.confirmation_count = 0
        self.vertical_progress = 0.0
        self._previous_eef_pos: np.ndarray | None = None
        self._previous_object_center_world: np.ndarray | None = None
        self._previous_object_center_eef: np.ndarray | None = None

    def _nonempty_closed(
        self,
        gripper_closed: bool | None,
        gripper_qpos: Any,
    ) -> bool:
        qpos = _vector(gripper_qpos, 2)
        if qpos is None:
            self.last_gripper_gap = None
            return False
        self.last_gripper_gap = float(qpos[0] - qpos[1])
        return bool(gripper_closed) and self.last_gripper_gap > self.empty_closed_gap

    @staticmethod
    def _eef_rotation(eef_quat: Any) -> np.ndarray | None:
        quat = _vector(eef_quat, 4)
        if quat is None or np.linalg.norm(quat) == 0:
            return None
        try:
            return Rotation.from_quat(quat).as_matrix()
        except ValueError:
            return None

    def update(
        self,
        *,
        target_points: Any,
        eef_pos: Any,
        eef_quat: Any,
        gripper_closed: bool | None,
        gripper_qpos: Any,
    ) -> bool:
        """Consume one frame and return the latched Pick completion decision.

        ``eef_quat`` uses SciPy's ``(x, y, z, w)`` convention, matching the
        LIBERO ``robot0_eef_quat`` observation.
        """
        if self.completed:
            return True

        if not self._nonempty_closed(gripper_closed, gripper_qpos):
            self._reset_candidate()
            return False

        position = _vector(eef_pos, 3)
        rotation = self._eef_rotation(eef_quat)
        center_world = _target_center(target_points)
        if position is None or rotation is None or center_world is None:
            self._reset_candidate()
            return False

        center_eef = rotation.T @ (center_world - position)
        if self._previous_eef_pos is None:
            self._previous_eef_pos = position.copy()
            self._previous_object_center_world = center_world.copy()
            self._previous_object_center_eef = center_eef.copy()
            return False

        self.last_object_dz = float(
            center_world[2] - self._previous_object_center_world[2]
        )
        self.last_eef_dz = float(position[2] - self._previous_eef_pos[2])
        self.last_rigid_error = float(
            np.linalg.norm(center_eef - self._previous_object_center_eef)
        )
        rigid = self.last_rigid_error <= self.max_rigid_error
        if rigid:
            common_dz = min(self.last_object_dz, self.last_eef_dz)
            self.vertical_progress = max(0.0, self.vertical_progress + common_dz)
        else:
            self.vertical_progress = 0.0

        self._previous_eef_pos = position.copy()
        self._previous_object_center_world = center_world.copy()
        self._previous_object_center_eef = center_eef.copy()
        satisfied = rigid and self.vertical_progress >= self.min_lift_distance
        self.confirmation_count = self.confirmation_count + 1 if satisfied else 0
        self.completed = self.confirmation_count >= self.stable_frames
        return self.completed


class PickSkillCompletion(TimedSkillCompletion):
    """Pick rule plus the common VLA action-chunk execution budget."""

    def __init__(
        self,
        *,
        max_action_chunks: int = 7,
        empty_closed_gap: float = EMPTY_CLOSED_GAP,
        min_lift_distance: float = DEFAULT_MIN_LIFT_DISTANCE,
        max_rigid_error: float = DEFAULT_MAX_RIGID_ERROR,
        stable_frames: int = DEFAULT_STABLE_FRAMES,
    ) -> None:
        self.checker = PickCompletionChecker(
            empty_closed_gap=empty_closed_gap,
            min_lift_distance=min_lift_distance,
            max_rigid_error=max_rigid_error,
            stable_frames=stable_frames,
        )
        super().__init__(max_action_chunks=max_action_chunks)

    def _reset_rule(self) -> None:
        self.checker.reset()

    def _check_rule(self, **kwargs: Any) -> bool:
        return self.checker.update(**kwargs)
