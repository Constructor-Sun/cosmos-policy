"""Match memory object templates to current RGB frames."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from memory_system.artifacts import PhaseTargetMemory
from memory_system.execute.planner.held_object.types import MaskMatchResult


class MemoryTemplateMatcher:
    """Locate a memory crop in the current RGB and transfer its mask."""

    def __init__(self, phase_targets_path: str | Path | None = None) -> None:
        self.memory = (
            PhaseTargetMemory(phase_targets_path)
            if phase_targets_path is not None
            else None
        )

    def select_templates(
        self,
        task_name: str,
        demo_id: str | None,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
    ) -> list[dict[str, Any]]:
        if self.memory is None:
            return []
        templates = self.memory.select(
            task_name, int(planner_step_id), skill, arguments
        )
        if demo_id is not None:
            templates = [t for t in templates if str(t.get("demo_id")) == str(demo_id)]
        return templates

    def match(
        self,
        rgb: np.ndarray,
        template: dict[str, Any],
        threshold: float = 0.5,
    ) -> MaskMatchResult | None:
        current = np.asarray(rgb)
        crop_rgb = np.asarray(template["crop_rgb"])
        crop_mask = np.asarray(template.get("crop_mask"))
        if current.ndim != 3 or crop_rgb.ndim != 3:
            return None
        if crop_mask.ndim != 2:
            return None
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        if current_gray.shape[0] < crop_gray.shape[0] or current_gray.shape[1] < crop_gray.shape[1]:
            return None
        result = cv2.matchTemplate(current_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if float(max_val) < float(threshold):
            return None
        top, left = int(max_loc[1]), int(max_loc[0])
        height, width = crop_mask.shape
        mask = np.zeros(current.shape[:2], dtype=np.uint8)
        bottom = min(current.shape[0], top + height)
        right = min(current.shape[1], left + width)
        mask[top:bottom, left:right] = crop_mask[: bottom - top, : right - left]
        if int(mask.sum()) < 16:
            return None
        key = (
            str(template.get("task_name", "")),
            str(template.get("demo_id", "")),
            int(template.get("planner_step_id", -1)),
            str(template.get("skill", "")),
        )
        return MaskMatchResult(
            mask=mask > 0,
            confidence=float(max_val),
            template_key=key,
        )
