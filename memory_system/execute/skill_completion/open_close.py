"""Open/Close skill-completion verifier."""
from __future__ import annotations

from typing import Any

import numpy as np

from memory_system.execute.skill_completion._common import (
    COMPLETION_UNKNOWN,
    CONFIRMATIONS,
    SKILL_COMPLETE,
    BaseCompletionVerifier,
)
from memory_system.types import CompletionResult


class OpenCloseCompletionVerifier(BaseCompletionVerifier):
    """Open/Close completion rules based on the demo anchor cluster."""

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
            met, reason, details = self._gate_met(gripper_closed, gxy)
            if met:
                self.confirmation_count += 1
            else:
                self.confirmation_count = 0
            self._prev_closed = (
                None if gripper_closed is None else bool(gripper_closed)
            )
            self.completed = self.confirmation_count >= CONFIRMATIONS
            if self.completed:
                reason = "confirmed_complete"
            elif self.confirmation_count > 0:
                reason = "completion_candidate"
        return self._finish(reason, timestep, details, gxy)
