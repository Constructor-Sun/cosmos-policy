"""Pick skill-completion verifier."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from memory_system.execute.skill_completion._common import (
    CENTER_JUMP_THRESHOLD,
    COMPLETION_UNKNOWN,
    CONFIRMATIONS,
    EMPTY_CLOSED_GAP,
    FLOW_INLIER_THRESHOLD,
    PICK_M_OF_M,
    PICK_N_OF_M,
    PICK_SKILLS,
    SKILL_COMPLETE,
    BaseCompletionVerifier,
)
from memory_system.types import CompletionResult


class PickCompletionVerifier(BaseCompletionVerifier):
    """Pick-specific wrist/flow/EEF completion rules."""

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
            if gripper_closed is True:
                if eef_pos is not None:
                    eef = np.asarray(eef_pos, dtype=np.float32).reshape(3)
                    if self._prev_closed is not True:
                        self.pick_close_xyz = eef.copy()
                        self.pick_moved = False
                        self.pick_last_center = None
                        self.pick_confirmations.clear()
                    elif not self.pick_moved and self.pick_close_xyz is not None:
                        if np.linalg.norm(eef - self.pick_close_xyz) >= 0.04:
                            self.pick_moved = True
            else:
                self.pick_close_xyz = None
                self.pick_moved = False
                self.pick_last_center = None
                self.pick_confirmations.clear()
            center = None
            if wrist_metrics is not None and self.skill in PICK_SKILLS:
                center = self._pick_center(wrist_image)
                jump = float("inf")
                if center is not None:
                    if self.pick_last_center is not None:
                        jump = float(np.hypot(
                            center[0] - self.pick_last_center[0],
                            center[1] - self.pick_last_center[1],
                        ))
                    self.pick_last_center = center
                gap = None
                if gripper_qpos is not None:
                    try:
                        gap = float(gripper_qpos[0]) - float(gripper_qpos[1])
                    except Exception:
                        gap = None
                met = bool(
                    gripper_closed is True
                    and center is not None
                    and self.pick_moved
                    and jump <= CENTER_JUMP_THRESHOLD
                    and self.flow_inlier_ratio >= FLOW_INLIER_THRESHOLD
                    and (gap is None or gap > EMPTY_CLOSED_GAP)
                    and (self.pick_close_xy2d is None
                         or self._inside_anchor_cluster(self.pick_close_xy2d))
                )
                details = dict(
                    wrist_metrics,
                    object_center=center,
                    pick_moved=self.pick_moved,
                    gap=gap,
                    flow_inlier_ratio=self.flow_inlier_ratio,
                    close_xy2d=(
                        self.pick_close_xy2d.tolist()
                        if self.pick_close_xy2d is not None else None
                    ),
                )
                reason = (
                    "object_center_stable_with_object" if met
                    else "gripper_not_closed" if gripper_closed is not True
                    else "gripper_empty_close" if gap is not None and gap <= EMPTY_CLOSED_GAP
                    else "object_center_unknown" if center is None
                    else "gripper_not_moved_since_close" if not self.pick_moved
                    else "gripper_outside_target_anchor" if (
                        self.pick_close_xy2d is not None
                        and not self._inside_anchor_cluster(self.pick_close_xy2d)
                    )
                    else "object_center_jump"
                )
                self.pick_confirmations.append(met)
                self.confirmation_count = sum(self.pick_confirmations)
            else:
                met, reason, details = self._gate_met(gripper_closed, gxy)
                if met:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 0
            if wrist_metrics is not None:
                self.prev_stable_ratio = wrist_metrics["stable_ratio"]
            self._prev_closed = (
                None if gripper_closed is None else bool(gripper_closed)
            )
            required_confirmations = (
                PICK_N_OF_M
                if wrist_metrics is not None and self.skill in PICK_SKILLS
                else CONFIRMATIONS
            )
            self.completed = self.confirmation_count >= required_confirmations
            if self.completed:
                reason = "confirmed_complete"
            elif self.confirmation_count > 0:
                reason = "completion_candidate"
        return self._finish(reason, timestep, details, gxy)
