"""Rule-based VLA closed->open completion for PlaceIn/PlaceOn.

Place completion uses the actual gripper qpos gap as the release signal:
  - if the episode session already holds an item from the preceding Pick,
    the closed state is considered confirmed at entry;
  - otherwise wait for one closed VLA frame before establishing a baseline;
  - only consecutive frames where the measured gap exceeds the baseline by
    ``open_gap_delta`` can advance the phase.
"""
from __future__ import annotations

import math
from typing import Any

from memory_system.execute.skill_completion.base import (
    DEFAULT_MAX_ACTION_CHUNKS,
    TimedSkillCompletion,
)

DEFAULT_OPEN_FRAMES = 16
DEFAULT_OPEN_GAP_DELTA = 0.01
PLACE_SKILLS = {"PlaceIn", "PlaceOn"}


class ReleaseSkillCompletion(TimedSkillCompletion):
    """Actual gripper qpos gap based release completion for Place skills.

    The VLA command is only used to confirm the initial closed state when the
    session does not already report a held item.  The common timeout fallback
    still applies because every skill completion in the current design uses
    timeout as a backstop.
    """

    def __init__(
        self,
        *,
        max_action_chunks: int = DEFAULT_MAX_ACTION_CHUNKS,
        required_open_frames: int = DEFAULT_OPEN_FRAMES,
        closed_confirmed: bool = False,
        open_gap_delta: float = DEFAULT_OPEN_GAP_DELTA,
    ) -> None:
        if required_open_frames < 1:
            raise ValueError("required_open_frames must be at least one")
        if open_gap_delta < 0:
            raise ValueError("open_gap_delta must be non-negative")
        self.required_open_frames = int(required_open_frames)
        self.closed_confirmed = bool(closed_confirmed)
        self.open_gap_delta = float(open_gap_delta)
        super().__init__(max_action_chunks=max_action_chunks)

    def _reset_rule(self) -> None:
        self._open_frames = 0
        self._closed_ok = self.closed_confirmed
        self._baseline_gap: float | None = None

    def _check_rule(
        self,
        *,
        gripper_closed: bool | None = None,
        gripper_qpos: Any = None,
        **kwargs: Any,
    ) -> bool:
        del kwargs  # Other observations are diagnostic only.
        if gripper_qpos is None:
            # Do not fall back to the command-only rule: if we cannot measure
            # the actual gripper gap, wait for the timeout backstop instead.
            return False
        try:
            q0 = float(gripper_qpos[0])
            q1 = float(gripper_qpos[1])
        except (TypeError, ValueError, IndexError):
            return False
        gap = q0 - q1
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

        if gap <= self._baseline_gap + self.open_gap_delta:
            self._open_frames = 0
            return False

        self._open_frames += 1
        return self._open_frames >= self.required_open_frames
