"""Data contracts for held-object planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from memory_system.types import CameraParams


@dataclass(frozen=True)
class HeldObjectObservation:
    """Hand-local point cloud of the object currently held by the robot."""

    item: str
    points_hand: np.ndarray
    confidence: float = 1.0
    source: str = "memory_template_rgbd"
    valid: bool = True
    capture_frames: int = 1
    stable_fraction: float = 1.0
    hand_translation_span: float = 0.0
    hand_rotation_span: float = 0.0
    rejection_reason: str | None = None


@dataclass(frozen=True)
class MaskMatchResult:
    """A mask produced by matching a memory template to the current RGB."""

    mask: np.ndarray
    confidence: float
    source: str = "memory_template"
    template_key: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HeldObjectPlannerInput:
    """All inputs required by HeldObjectPlanner."""

    joint_positions: np.ndarray
    ee_states: np.ndarray
    depth: np.ndarray
    camera_params: CameraParams
    held_object: HeldObjectObservation
    ready_pose: np.ndarray
    gripper_joint_positions: np.ndarray | None = None
    robot_base_pose: np.ndarray | None = None


@dataclass(frozen=True)
class HeldObjectPlannerResult:
    """Successful waypoint/OSC plan from HeldObjectPlanner."""

    controller: Any
    correction_steps: int
    target_ee_states: np.ndarray
    waypoints: np.ndarray
