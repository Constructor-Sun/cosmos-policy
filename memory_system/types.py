"""Shared types for the memory_system package.

These types are shared by offline construction, Initial Alignment, and future
RGB-D skill checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MemoryKey:
    """Canonical key for selecting memory entries by task/step/skill/arguments.

    The ``arguments`` dict preserves the original LIBERO semantics:
    ``item`` is the manipulated object and ``target`` is the destination or
    other target object.
    """

    task_name: str
    planner_step_id: int
    skill: str
    arguments: dict[str, str] = field(default_factory=dict)

    def arguments_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted((str(key), str(value)) for key, value in self.arguments.items())
        )

    def as_tuple(self) -> tuple[str, int, str, tuple[tuple[str, str], ...]]:
        return (
            str(self.task_name),
            int(self.planner_step_id),
            str(self.skill),
            self.arguments_key(),
        )


@dataclass(frozen=True)
class SkillStep:
    planner_step_id: int
    skill: str
    arguments: dict[str, str] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    execution_mode: str = "default"


@dataclass(frozen=True)
class SkillPlan:
    task_name: str
    steps: tuple[SkillStep, ...] = ()


@dataclass(frozen=True)
class CameraParams:
    """Camera intrinsics/extrinsics and depth range for RGB-D geometry."""

    K: np.ndarray
    T_w2c: np.ndarray
    T_c2w: np.ndarray
    height: int
    width: int
    near: float = 0.0
    far: float = 0.0


@dataclass(frozen=True)
class VerifierObservation:
    """Legacy-named RGB-D observation payload reserved for future skill checks."""

    third_view_rgb: np.ndarray | None = None
    gripper_xy: tuple[float, float] | np.ndarray | None = None
    gripper_closed: bool | None = None
    wrist_image: np.ndarray | None = None
    eef_pos: np.ndarray | None = None
    eef_states: np.ndarray | None = None
    gripper_qpos: Any = None
    main_vae: Any = None
    wrist_vae: Any = None
    target_bbox: tuple[float, float, float, float] | None = None
    timestep: int | None = None
    main_depth: np.ndarray | None = None
    camera_params: CameraParams | None = None


@dataclass(frozen=True)
class TargetGeometry:
    center_xy: np.ndarray | None
    bbox_xyxy: np.ndarray | None
    # Optional 3D fields reserved for the future RGB-D/3D upgrade.
    center_xyz_world: np.ndarray | None = None
    points_xyz_world: np.ndarray | None = None
    depth_confidence: float = 0.0


@dataclass(frozen=True)
class RecoveryTarget:
    demo_ids: tuple[str, ...]
    target_ee_states: np.ndarray
    similarity: float
    frame: int
    z_lift: float = 0.0
