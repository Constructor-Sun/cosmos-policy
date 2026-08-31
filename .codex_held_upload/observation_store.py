"""Persist and load HeldObjectObservation between Pick and later phases.

The correct held-object flow is:

1. During Pick, build a hand-local ``HeldObjectObservation`` from memory-mask
   + depth while the object is still visible.
2. Save it with ``HeldObjectObservationStore``.
3. Later, ``HeldObjectPlanner`` loads this saved observation and uses it
   directly; it does NOT re-match the current RGB or re-extract a connected
   component near the gripper.
"""
from __future__ import annotations

from pathlib import Path

import torch

from memory_system.execute.planner.held_object.types import HeldObjectObservation


class HeldObjectObservationStore:
    """A simple file-backed store for hand-local held-object observations."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, task_name: str, demo_id: str, item: str) -> Path:
        safe_task = task_name.replace("/", "_").replace(" ", "_")
        safe_demo = str(demo_id).replace("/", "_").replace(" ", "_")
        safe_item = item.replace("/", "_").replace(" ", "_")
        return self.root / f"{safe_task}__{safe_demo}__{safe_item}.pt"

    def exists(self, task_name: str, demo_id: str, item: str) -> bool:
        return self.path_for(task_name, demo_id, item).exists()

    def save(
        self,
        observation: HeldObjectObservation,
        task_name: str,
        demo_id: str,
        item: str,
    ) -> Path:
        path = self.path_for(task_name, demo_id, item)
        payload = {
            "item": observation.item,
            "points_hand": observation.points_hand,
            "confidence": observation.confidence,
            "source": observation.source,
            "valid": observation.valid,
            "capture_frames": observation.capture_frames,
            "stable_fraction": observation.stable_fraction,
            "hand_translation_span": observation.hand_translation_span,
            "hand_rotation_span": observation.hand_rotation_span,
            "rejection_reason": observation.rejection_reason,
            "task_name": task_name,
            "demo_id": demo_id,
        }
        torch.save(payload, path)
        return path

    def load(self, task_name: str, demo_id: str, item: str) -> HeldObjectObservation:
        path = self.path_for(task_name, demo_id, item)
        if not path.exists():
            raise FileNotFoundError(f"HeldObjectObservation not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return HeldObjectObservation(
            item=str(payload["item"]),
            points_hand=payload["points_hand"],
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source", "memory_template_rgbd")),
            valid=bool(payload.get("valid", True)),
            capture_frames=int(payload.get("capture_frames", 1)),
            stable_fraction=float(payload.get("stable_fraction", 1.0)),
            hand_translation_span=float(payload.get("hand_translation_span", 0.0)),
            hand_rotation_span=float(payload.get("hand_rotation_span", 0.0)),
            rejection_reason=payload.get("rejection_reason"),
        )


__all__ = ["HeldObjectObservationStore"]
