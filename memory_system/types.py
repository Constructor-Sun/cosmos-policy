"""Shared types for the memory_system package.

These types are shared by offline construction and online execution.  They are
kept intentionally small in Step 1 and extended as the migration proceeds.
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
class VerifierObservation:
    """Observation payload consumed by phase/feasible/completion verifiers."""

    third_view_rgb: np.ndarray | None = None
    gripper_xy: tuple[float, float] | np.ndarray | None = None
    gripper_closed: bool | None = None
    wrist_image: np.ndarray | None = None
    eef_pos: np.ndarray | None = None
    eef_states: np.ndarray | None = None
    main_vae: Any = None
    wrist_vae: Any = None
    target_bbox: tuple[float, float, float, float] | None = None
    timestep: int | None = None


@dataclass(frozen=True)
class TargetGeometry:
    center_xy: np.ndarray | None
    bbox_xyxy: np.ndarray | None
    # Optional 3D fields reserved for the future RGB-D/3D upgrade.
    center_xyz_world: np.ndarray | None = None
    points_xyz_world: np.ndarray | None = None
    depth_confidence: float = 0.0


@dataclass(frozen=True)
class PhaseResult:
    status: str
    confidence: float
    target_xy: tuple[float, float] | None
    progress_px: float | None
    match_count: int
    template_demo_id: str | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeasibleResult:
    status: str
    reason: str
    timestep: int | None
    entered_feasible: bool
    current_distance: float | None
    current_distance_px: float | None
    ready_distance: float | None
    ready_votes: int
    memory_count: int
    progress: float | None
    progress_px: float | None
    stall_count: int
    wrong_way_count: int
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResult:
    status: str
    reason: str
    timestep: int | None
    distance: float | None
    success_radius: float | None
    memory_count: int
    confirmation_count: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryRequest:
    reason: str
    task_name: str
    planner_step_id: int
    skill: str
    arguments: dict[str, str]
    current_ee_states: np.ndarray | None = None
    current_vae_main: Any = None
    current_vae_wrist: Any = None


@dataclass(frozen=True)
class RecoveryTarget:
    demo_ids: tuple[str, ...]
    target_ee_states: np.ndarray
    similarity: float
    frame: int
    z_lift: float = 0.0


@dataclass(frozen=True)
class RecoveryResult:
    target: RecoveryTarget
    correction_steps: int
    controller: Any = None
    correction_per_step: np.ndarray | None = None
