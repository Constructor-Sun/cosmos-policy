"""Place skill-completion verifier."""
from __future__ import annotations

from typing import Any

import numpy as np

from memory_system.execute.skill_completion._common import (
    COMPLETION_UNKNOWN,
    CONFIRMATIONS,
    PLACE_SKILLS,
    SKILL_COMPLETE,
    BaseCompletionVerifier,
)
from memory_system.types import CompletionResult


class PlaceCompletionVerifier(BaseCompletionVerifier):
    """PlaceOn/PlaceIn completion rules based on release transition and gates."""

    def update(
        self,
        current_vae: Any | None = None,
        *,
        gripper_closed: bool | None = None,
        gripper_xy: Any = None,
        target_bbox: Any = None,
        timestep: int | None = None,
        wrist_image: Any = None,
        gripper_qpos: Any = None,
        eef_pos: Any = None,
    ) -> CompletionResult:
        gxy = (
            np.asarray(gripper_xy, dtype=np.float32)
            if gripper_xy is not None else None
        )
        details: dict[str, Any] = {}
        if self.completed:
            reason = "latched_complete"
        else:
            if target_bbox is not None:
                self.target_bbox = tuple(float(v) for v in target_bbox)
            wrist_metrics = (
                self._update_wrist_metrics(wrist_image)
                if wrist_image is not None else None
            )
            specific = False
            if wrist_metrics is not None and self.skill in PLACE_SKILLS:
                released = self._prev_closed is True and gripper_closed is False
                met, reason, details = self._wrist_place_gate(
                    gripper_closed, wrist_metrics
                )
                if released:
                    self.confirmation_count = max(self.confirmation_count, CONFIRMATIONS)
                    reason = "released_wrist_object"
                    specific = True
                elif met and self.confirmation_count >= 1:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 0
            else:
                met, reason, details = self._gate_met(gripper_closed, gxy)
                if self.skill in PLACE_SKILLS:
                    released = self._prev_closed is True and gripper_closed is False
                    if released:
                        self.confirmation_count = max(self.confirmation_count, CONFIRMATIONS)
                        reason = "released_inside_region"
                        specific = True
                    elif met and self.confirmation_count >= 1:
                        self.confirmation_count += 1
                    else:
                        self.confirmation_count = 0
                elif met:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 0
            if wrist_metrics is not None:
                self.prev_stable_ratio = wrist_metrics["stable_ratio"]
            self._prev_closed = (
                None if gripper_closed is None else bool(gripper_closed)
            )
            self.completed = self.confirmation_count >= CONFIRMATIONS
            if self.completed:
                reason = "confirmed_complete"
            elif self.confirmation_count > 0 and not specific:
                reason = "completion_candidate"
        return self._finish(reason, timestep, details, gxy)
