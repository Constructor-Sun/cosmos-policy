#!/usr/bin/env python3
"""Positive-only LIBERO feasible-region verifier.

The verifier is intentionally independent from the phase verifier.  Its caller
supplies neutral target geometry (target center, matched bounding box, and
gripper position), while this module owns its ready-distance memory and temporal
state.  No image or trajectory is treated as a negative example.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


FEASIBLE = "FEASIBLE"
NOT_FEASIBLE = "NOT_FEASIBLE"
FEASIBLE_UNKNOWN = "FEASIBLE_UNKNOWN"


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def _phase_key(
    task_name: str,
    planner_step_id: int,
    skill: str,
    arguments: dict[str, Any],
) -> tuple[str, int, str, tuple[tuple[str, str], ...]]:
    return (
        str(task_name),
        int(planner_step_id),
        str(skill),
        _arguments_key(arguments),
    )


def _as_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _bbox_diagonal(bbox_xyxy: Any) -> float:
    x0, y0, x1, y1 = _as_numpy(bbox_xyxy).reshape(4)
    width, height = float(x1 - x0), float(y1 - y0)
    diagonal = float(np.hypot(width, height))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError(f"Invalid target bounding box: {bbox_xyxy!r}")
    return diagonal


@dataclass(frozen=True)
class ReadyDistancePrototype:
    demo_id: str
    ready_frame: int
    distance_px: float
    normalized_distance: float


@dataclass(frozen=True)
class FeasibleRegionResult:
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


class ReadyDistanceMemory:
    """Index normalized gripper-target distances at fixed16 ready frames."""

    def __init__(self, phase_targets: str | Path, segments_manifest: str | Path):
        target_payload = torch.load(
            Path(phase_targets), map_location="cpu", weights_only=False
        )
        if target_payload.get("format") != "libero_phase_targets_v1":
            raise ValueError(
                f"Unsupported phase target format: {target_payload.get('format')!r}"
            )
        manifest = json.loads(Path(segments_manifest).read_text())
        ready_frames: dict[
            tuple[str, str, int, str, tuple[tuple[str, str], ...]], int
        ] = {}
        for record in manifest.get("records", []):
            if not record.get("valid"):
                continue
            for segment in record.get("segments", []):
                ready_frame = segment.get("ready_frame")
                if ready_frame is None:
                    continue
                key = (
                    str(record["task_name"]),
                    str(record["demo_id"]),
                    int(segment["planner_step_id"]),
                    str(segment["skill"]),
                    _arguments_key(segment.get("arguments", {})),
                )
                ready_frames[key] = int(ready_frame)

        per_phase: dict[
            tuple[str, int, str, tuple[tuple[str, str], ...]],
            dict[str, ReadyDistancePrototype],
        ] = {}
        self.templates = list(target_payload.get("templates", []))
        for template in self.templates:
            segment_key = (
                str(template["task_name"]),
                str(template["demo_id"]),
                int(template["planner_step_id"]),
                str(template["skill"]),
                _arguments_key(template.get("arguments", {})),
            )
            ready_frame = ready_frames.get(segment_key)
            if ready_frame is None or int(template["frame"]) != ready_frame:
                continue
            center = _as_numpy(template["target_center_xy"]).reshape(2)
            gripper = _as_numpy(template["gripper_xy"]).reshape(2)
            distance_px = float(np.linalg.norm(gripper - center))
            normalized = distance_px / _bbox_diagonal(template["bbox_xyxy"])
            phase_key = _phase_key(
                template["task_name"],
                template["planner_step_id"],
                template["skill"],
                template.get("arguments", {}),
            )
            per_phase.setdefault(phase_key, {})[str(template["demo_id"])] = (
                ReadyDistancePrototype(
                    demo_id=str(template["demo_id"]),
                    ready_frame=ready_frame,
                    distance_px=distance_px,
                    normalized_distance=normalized,
                )
            )
        self._per_phase = {
            key: tuple(sorted(values.values(), key=lambda item: item.demo_id))
            for key, values in per_phase.items()
        }

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> tuple[ReadyDistancePrototype, ...]:
        excluded = set(str(item) for item in exclude_demo_ids)
        exact_key = _phase_key(task_name, planner_step_id, skill, arguments)
        exact = tuple(
            item for item in self._per_phase.get(exact_key, ())
            if item.demo_id not in excluded
        )
        if exact:
            return exact

        # Match PhaseTargetMemory's fallback for manifests whose nominal step
        # numbers differ while the task, grounded skill, and target agree.
        arguments_key = _arguments_key(arguments)
        fallback = []
        for (task, _step, candidate_skill, candidate_args), prototypes in self._per_phase.items():
            if (
                task == task_name
                and candidate_skill == skill
                and candidate_args == arguments_key
            ):
                fallback.extend(
                    item for item in prototypes if item.demo_id not in excluded
                )
        by_demo = {item.demo_id: item for item in fallback}
        return tuple(sorted(by_demo.values(), key=lambda item: item.demo_id))


class LiberoFeasibleRegionVerifier:
    """Latch entry into the memory-defined ready radius and flag pre-entry drift."""

    def __init__(
        self,
        memory: ReadyDistanceMemory | str | Path,
        segments_manifest: str | Path | None = None,
        min_demo_votes: int = 2,
        progress_tolerance_px: float = 5.0,
        evidence_updates: int = 2,
    ):
        if isinstance(memory, ReadyDistanceMemory):
            self.memory = memory
        else:
            if segments_manifest is None:
                raise ValueError("segments_manifest is required with a phase-target path")
            self.memory = ReadyDistanceMemory(memory, segments_manifest)
        if min_demo_votes <= 0 or evidence_updates <= 0:
            raise ValueError("vote and update counts must be positive")
        if not np.isfinite(progress_tolerance_px) or progress_tolerance_px < 0:
            raise ValueError("progress_tolerance_px must be finite and non-negative")
        self.min_demo_votes = int(min_demo_votes)
        self.progress_tolerance_px = float(progress_tolerance_px)
        self.evidence_updates = int(evidence_updates)
        self.task_name: str | None = None
        self.planner_step_id: int | None = None
        self.skill = ""
        self.arguments: dict[str, str] = {}
        self.prototypes: tuple[ReadyDistancePrototype, ...] = ()
        self.previous_gripper: np.ndarray | None = None
        self.entered_feasible = False
        self.stall_count = 0
        self.wrong_way_count = 0
        self.history: list[FeasibleRegionResult] = []

    def reset(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        self.task_name = str(task_name)
        self.planner_step_id = int(planner_step_id)
        self.skill = str(skill)
        self.arguments = {str(key): str(value) for key, value in arguments.items()}
        self.prototypes = self.memory.select(
            self.task_name,
            self.planner_step_id,
            self.skill,
            self.arguments,
            exclude_demo_ids,
        )
        self.previous_gripper = None
        self.entered_feasible = False
        self.stall_count = 0
        self.wrong_way_count = 0

    def _result(self, **kwargs: Any) -> FeasibleRegionResult:
        result = FeasibleRegionResult(**kwargs)
        self.history.append(result)
        return result

    def update(
        self,
        target_xy: tuple[float, float] | np.ndarray | None,
        matched_bbox_xyxy: tuple[float, float, float, float] | np.ndarray | None,
        gripper_xy: tuple[float, float] | np.ndarray | None,
        *,
        confidence: float = 0.0,
        timestep: int | None = None,
    ) -> FeasibleRegionResult:
        if self.task_name is None:
            raise RuntimeError("reset() must be called before update()")
        common = {
            "timestep": timestep,
            "entered_feasible": self.entered_feasible,
            "current_distance": None,
            "current_distance_px": None,
            "ready_distance": None,
            "ready_votes": 0,
            "memory_count": len(self.prototypes),
            "progress": None,
            "progress_px": None,
            "stall_count": self.stall_count,
            "wrong_way_count": self.wrong_way_count,
            "confidence": float(confidence),
            "details": {},
        }
        if len(self.prototypes) < self.min_demo_votes:
            return self._result(
                status=FEASIBLE_UNKNOWN,
                reason="insufficient_ready_memory",
                **common,
            )
        if target_xy is None or matched_bbox_xyxy is None or gripper_xy is None:
            return self._result(
                status=FEASIBLE_UNKNOWN,
                reason="target_geometry_unknown",
                **common,
            )

        target = _as_numpy(target_xy).reshape(2)
        gripper = _as_numpy(gripper_xy).reshape(2)
        target_scale = _bbox_diagonal(matched_bbox_xyxy)
        current_distance_px = float(np.linalg.norm(gripper - target))
        current_distance = current_distance_px / target_scale
        ready_distances = sorted(
            (item.normalized_distance for item in self.prototypes), reverse=True
        )
        ready_distance = ready_distances[self.min_demo_votes - 1]
        ready_votes = sum(
            current_distance <= item.normalized_distance for item in self.prototypes
        )
        progress_px = None
        progress = None
        if self.previous_gripper is not None:
            previous_distance_px = float(np.linalg.norm(self.previous_gripper - target))
            progress_px = previous_distance_px - current_distance_px
            progress = progress_px / target_scale

        if ready_votes >= self.min_demo_votes:
            newly_entered = not self.entered_feasible
            self.entered_feasible = True
            self.stall_count = 0
            self.wrong_way_count = 0
            status, reason = FEASIBLE, (
                "entered_ready_radius" if newly_entered else "latched_feasible"
            )
        elif self.entered_feasible:
            status, reason = FEASIBLE, "latched_feasible"
        elif progress_px is None:
            status, reason = FEASIBLE_UNKNOWN, "warmup"
        elif progress_px < -self.progress_tolerance_px:
            self.wrong_way_count += 1
            self.stall_count = 0
            status = (
                NOT_FEASIBLE
                if self.wrong_way_count >= self.evidence_updates
                else FEASIBLE_UNKNOWN
            )
            reason = (
                "persistent_reversal"
                if status == NOT_FEASIBLE
                else "reversal_candidate"
            )
        elif abs(progress_px) <= self.progress_tolerance_px:
            self.stall_count += 1
            self.wrong_way_count = 0
            status = (
                NOT_FEASIBLE
                if self.stall_count >= self.evidence_updates
                else FEASIBLE_UNKNOWN
            )
            reason = (
                "persistent_stagnation"
                if status == NOT_FEASIBLE
                else "stagnation_candidate"
            )
        else:
            self.stall_count = 0
            self.wrong_way_count = 0
            status, reason = FEASIBLE_UNKNOWN, "approaching"

        self.previous_gripper = gripper.copy()
        return self._result(
            status=status,
            reason=reason,
            timestep=timestep,
            entered_feasible=self.entered_feasible,
            current_distance=current_distance,
            current_distance_px=current_distance_px,
            ready_distance=ready_distance,
            ready_votes=ready_votes,
            memory_count=len(self.prototypes),
            progress=progress,
            progress_px=progress_px,
            stall_count=self.stall_count,
            wrong_way_count=self.wrong_way_count,
            confidence=float(confidence),
            details={
                "target_scale_px": target_scale,
                "progress_tolerance": self.progress_tolerance_px / target_scale,
                "ready_demo_ids": [
                    item.demo_id
                    for item in self.prototypes
                    if current_distance <= item.normalized_distance
                ],
            },
        )
