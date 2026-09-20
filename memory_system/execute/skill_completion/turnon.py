"""Rule-based completion for the TurnOn skill.

Turning a stove on is a rotation of a knob, and the state that satisfies it is
a joint angle.  That angle is not observable outside the simulator, so this
rule does not try to read it.  What *is* observable is the robot's own wrist:
to drive a knob through some angle the end effector must sweep roughly the
same angle.

So the rule detects the ACTION ("the wrist was turned past a threshold"), not
the RESULT ("the knob actually moved").  The gap between those two is the same
one ReleaseSkillCompletion has -- a slip reads as a completed turn.  It is
measured and reported rather than hidden.

The baseline orientation is taken at the GRASP, not at the phase start.  The
approach motion itself rotates the wrist (up to ~28 deg in LIBERO-10 demos),
so anchoring at the phase start would leave only a thin margin.  Anchoring on
the first closed-gripper frame removes the approach entirely, and it makes
"never grasped the knob" a non-firing case for free -- which is the dominant
way this skill actually fails.

This rule cannot fire on a scene whose target is already on, because no
rotation happens.  That case is covered by the timeout backstop; it is the
reason the rule is a TimedSkillCompletion rather than a bare predicate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.execute.skill_completion.base import (
    DEFAULT_MAX_ACTION_CHUNKS,
    TimedSkillCompletion,
)

DEFAULT_MIN_ROTATION_DEG = 35.0


def _quat(value: Any) -> np.ndarray | None:
    """Return one finite unit quaternion, or None."""
    if value is None:
        return None
    try:
        quat = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if quat.size != 4 or not np.isfinite(quat).all():
        return None
    if np.linalg.norm(quat) == 0:
        return None
    return quat


class TurnOnCompletion(TimedSkillCompletion):
    """Net end-effector rotation accumulated since the phase started.

    Args:
        min_rotation_deg: Rotation from the grasp orientation that counts as
            having performed the turn.
    """

    def __init__(
        self,
        *,
        max_action_chunks: int = DEFAULT_MAX_ACTION_CHUNKS,
        min_rotation_deg: float = DEFAULT_MIN_ROTATION_DEG,
    ) -> None:
        if min_rotation_deg <= 0:
            raise ValueError("min_rotation_deg must be positive")
        self.min_rotation_deg = float(min_rotation_deg)
        super().__init__(max_action_chunks=max_action_chunks)

    def _reset_rule(self) -> None:
        self._start_quat: np.ndarray | None = None

    def _check_rule(
        self,
        *,
        eef_quat: Any = None,
        gripper_closed: bool | None = None,
        **kwargs: Any,
    ) -> bool:
        del kwargs  # Other observations are diagnostic only.
        quat = _quat(eef_quat)
        if quat is None:
            # Without an orientation there is nothing to measure; wait for the
            # timeout backstop rather than guessing.
            return False
        if not gripper_closed:
            # Open gripper: still approaching, or already released.  Re-arm so
            # that the measurement starts from the next grasp.
            self._start_quat = None
            return False
        if self._start_quat is None:
            self._start_quat = quat
            return False
        try:
            delta = Rotation.from_quat(self._start_quat).inv() * Rotation.from_quat(quat)
        except ValueError:
            return False
        return float(np.degrees(delta.magnitude())) >= self.min_rotation_deg
