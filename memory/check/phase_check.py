"""Rule-based phase-end checks for the VLA and alignment windows.

One ``PhaseChecker`` per rollout.  ``reset(phase)`` selects the rule and the
action-chunk budget for the current phase, ``observe`` consumes one executed
frame, and ``finish_action_chunk`` closes one action chunk.  A rule match
advances with ``reason='rule'``; otherwise the phase advances with
``reason='timeout'`` after the configured number of completed chunks -- timeout
is an execution boundary, not a semantic completion.

Rules are carried over from ``memory_system/execute/skill_completion``:
Pick (non-empty closed gripper, rigid target tracking, lift), PlaceIn/PlaceOn
(real gripper gap release), TurnOn (net wrist rotation since the grasp).
Open/Close have no rule and advance on the chunk budget alone.

This module only classifies.  It does not move the phase cursor, update the
held-object belief, or switch controllers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory.types import PhaseDecision, PhaseSpec

DEFAULT_CHUNK_BUDGET: dict[str, int] = {
    "Pick": 9,
    "PlaceOn": 9,
    "PlaceIn": 9,
    "Open": 3,
    "Close": 3,
    "TurnOn": 3,
}
# Skills with no rule: the chunk budget alone ends the phase.
_TIMEOUT_ONLY_SKILLS = {"Open", "Close"}


@dataclass(frozen=True)
class PhaseCheckConfig:
    """Calibrated values in one place; the defaults are the current ones."""

    chunk_budget: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_CHUNK_BUDGET)
    )
    default_chunks: int = 3
    # Pick
    empty_closed_gap: float = 0.003
    min_lift_distance: float = 0.02
    max_rigid_error: float = 0.01
    stable_frames: int = 5
    # PlaceIn / PlaceOn
    required_open_frames: int = 5
    open_gap_delta: float = 0.01
    # TurnOn
    min_rotation_deg: float = 30.0



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


def _target_geometry(target_points: Any) -> tuple[np.ndarray, float] | None:
    """Return the median center and lower-z quantile of finite target points."""
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
    return np.median(finite, axis=0), float(np.quantile(finite[:, 2], 0.1))


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


class _PickRule:
    """Non-empty closed gripper + rigid target tracking + lift, held N frames.

    Every observable condition must hold at once: the measured gripper gap is
    above ``empty_closed_gap``, the target center keeps its position in the EEF
    frame (frame-to-frame drift within ``max_rigid_error``), the target center
    and the EEF accumulate more than ``min_lift_distance`` of shared upward
    motion, and the target's own lower quantile has risen by the same distance.
    Anything else clears the candidate and starts over.
    """

    def __init__(self, config: PhaseCheckConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.vertical_progress = 0.0
        self.confirmation_count = 0
        self._initial_bottom_z: float | None = None
        self._previous_eef_pos: np.ndarray | None = None
        self._previous_object_center_world: np.ndarray | None = None
        self._previous_object_center_eef: np.ndarray | None = None

    def update(
        self,
        *,
        target_points: Any = None,
        eef_pos: Any = None,
        eef_quat: Any = None,
        gripper_closed: bool | None = None,
        gripper_qpos: Any = None,
    ) -> bool:
        config = self.config
        qpos = _vector(gripper_qpos, 2)
        if qpos is None:
            self.reset()
            return False
        if not gripper_closed or float(qpos[0] - qpos[1]) <= config.empty_closed_gap:
            self.reset()
            return False

        position = _vector(eef_pos, 3)
        rotation = _eef_rotation(eef_quat)
        geometry = _target_geometry(target_points)
        if position is None or rotation is None or geometry is None:
            self.reset()
            return False
        center_world, bottom_z = geometry
        if self._initial_bottom_z is None:
            self._initial_bottom_z = bottom_z

        center_eef = rotation.T @ (center_world - position)
        if self._previous_eef_pos is None:
            self._previous_eef_pos = position.copy()
            self._previous_object_center_world = center_world.copy()
            self._previous_object_center_eef = center_eef.copy()
            return False

        object_dz = float(center_world[2] - self._previous_object_center_world[2])
        eef_dz = float(position[2] - self._previous_eef_pos[2])
        rigid = bool(
            np.linalg.norm(center_eef - self._previous_object_center_eef)
            <= config.max_rigid_error
        )
        if rigid:
            self.vertical_progress = max(
                0.0, self.vertical_progress + min(object_dz, eef_dz)
            )
        else:
            self.vertical_progress = 0.0
        self._previous_eef_pos = position.copy()
        self._previous_object_center_world = center_world.copy()
        self._previous_object_center_eef = center_eef.copy()

        bottom_lift = bottom_z - self._initial_bottom_z
        satisfied = (
            rigid
            and self.vertical_progress >= config.min_lift_distance
            and bottom_lift >= config.min_lift_distance
        )
        self.confirmation_count = self.confirmation_count + 1 if satisfied else 0
        return self.confirmation_count >= config.stable_frames


def _eef_rotation(eef_quat: Any) -> np.ndarray | None:
    quat = _vector(eef_quat, 4)
    if quat is None or np.linalg.norm(quat) == 0:
        return None
    try:
        return Rotation.from_quat(quat).as_matrix()
    except ValueError:
        return None


class _ReleaseRule:
    """Release detection from the measured gripper qpos gap.

    When the episode already held the item after the preceding Pick the closed
    state is confirmed on entry; otherwise the first closed frame establishes
    the baseline.  Only consecutive frames whose gap exceeds the baseline by
    ``open_gap_delta`` complete the phase.  Without a measurable gap the rule
    stays silent rather than falling back to the commanded gripper action, and
    the chunk budget ends the phase instead.
    """

    def __init__(self, config: PhaseCheckConfig, *, closed_confirmed: bool) -> None:
        self.config = config
        self.closed_confirmed = bool(closed_confirmed)
        self.reset()

    def reset(self) -> None:
        self._open_frames = 0
        self._closed_ok = self.closed_confirmed
        self._baseline_gap: float | None = None

    def update(
        self,
        *,
        target_points: Any = None,
        eef_pos: Any = None,
        eef_quat: Any = None,
        gripper_closed: bool | None = None,
        gripper_qpos: Any = None,
    ) -> bool:
        del target_points, eef_pos, eef_quat  # Other observations are diagnostic only.
        if gripper_qpos is None:
            return False
        try:
            gap = float(gripper_qpos[0]) - float(gripper_qpos[1])
        except (TypeError, ValueError, IndexError):
            return False
        if not math.isfinite(gap):
            return False

        if not self._closed_ok:
            # First confirm the gripper is closed before establishing baseline.
            if gripper_closed:
                self._closed_ok = True
                self._baseline_gap = gap
                self._open_frames = 0
            return False

        if self._baseline_gap is None:
            self._baseline_gap = gap
            return False

        if gap <= self._baseline_gap + self.config.open_gap_delta:
            self._open_frames = 0
            return False

        self._open_frames += 1
        return self._open_frames >= self.config.required_open_frames


class _TurnOnRule:
    """Net wrist rotation since the grasp, thresholded.

    The knob angle is not observable outside the simulator, so this detects the
    action ("the wrist was turned past a threshold"), not the result ("the knob
    actually moved").  A slip reads as a completed turn; that gap is the same
    one the release rule has, and it is reported rather than hidden.

    The baseline orientation is taken at the GRASP, not at the phase start: the
    approach itself rotates the wrist (up to ~28 deg in LIBERO-10 demos), and
    anchoring on the first closed-gripper frame also makes "never grasped the
    knob" a non-firing case for free.  A target that is already on never fires,
    which is what the chunk budget is for.
    """

    def __init__(self, config: PhaseCheckConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._start_quat: np.ndarray | None = None

    def update(
        self,
        *,
        target_points: Any = None,
        eef_pos: Any = None,
        eef_quat: Any = None,
        gripper_closed: bool | None = None,
        gripper_qpos: Any = None,
    ) -> bool:
        del target_points, eef_pos, gripper_qpos  # Only the wrist is measured.
        quat = _quat(eef_quat)
        if quat is None:
            # Without an orientation there is nothing to measure; wait for the
            # chunk budget rather than guessing.
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
        return float(np.degrees(delta.magnitude())) >= self.config.min_rotation_deg


class PhaseChecker:
    """Classify one VLA or alignment window of one phase.

    ``reset`` is required before ``observe``/``finish_action_chunk``.  An
    unknown skill is rejected instead of silently falling back to the budget.
    """

    def __init__(self, config: PhaseCheckConfig | None = None) -> None:
        self.config = PhaseCheckConfig() if config is None else config
        self._rule: _PickRule | _ReleaseRule | _TurnOnRule | None = None
        self._ready = False
        self._max_chunks = self.config.default_chunks
        self.action_chunks = 0
        self.semantic_completed = False
        self.advance_reason: str | None = None

    def reset(self, phase: PhaseSpec, *, initially_holding: bool = False) -> None:
        """Select the rule and the chunk budget for ``phase``, clearing state."""
        skill = phase.skill
        if skill == "Pick":
            self._rule = _PickRule(self.config)
        elif skill in {"PlaceIn", "PlaceOn"}:
            self._rule = _ReleaseRule(self.config, closed_confirmed=initially_holding)
        elif skill == "TurnOn":
            self._rule = _TurnOnRule(self.config)
        elif skill in _TIMEOUT_ONLY_SKILLS:
            self._rule = None
        else:
            raise KeyError(f"Unknown skill completion: {skill!r}")

        self._max_chunks = self.config.chunk_budget.get(
            skill, self.config.default_chunks
        )
        self._ready = True
        self.action_chunks = 0
        self.semantic_completed = False
        self.advance_reason = None

    def observe(
        self,
        *,
        target_points: Any = None,
        eef_pos: Any = None,
        eef_quat: Any = None,
        gripper_closed: bool | None = None,
        gripper_qpos: Any = None,
    ) -> PhaseDecision:
        """Consume one executed frame and latch a rule match."""
        self._require_ready()
        if self.advance_reason is None and self._rule is not None:
            matched = self._rule.update(
                target_points=target_points,
                eef_pos=eef_pos,
                eef_quat=eef_quat,
                gripper_closed=gripper_closed,
                gripper_qpos=gripper_qpos,
            )
            if matched:
                self.semantic_completed = True
                self.advance_reason = "rule"
        return self.decision

    def finish_action_chunk(self) -> PhaseDecision:
        """Close one completed action chunk and apply the budget."""
        self._require_ready()
        self.action_chunks += 1
        if self.advance_reason is None and self.action_chunks >= self._max_chunks:
            self.advance_reason = "timeout"
        return self.decision

    @property
    def decision(self) -> PhaseDecision:
        reason = self.advance_reason or "running"
        return PhaseDecision(
            advance=reason != "running",
            semantic_completed=self.semantic_completed,
            reason=reason,
            action_chunks=self.action_chunks,
        )

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("reset(phase) must run before observe")
