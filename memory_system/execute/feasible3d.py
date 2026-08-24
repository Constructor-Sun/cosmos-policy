"""Feasible 3D verifier.

This module adds a 3D parallel path while keeping the existing 2D
``FeasibleVerifier`` untouched.  It reuses the 2D verifier only for wrist
feasible checks and for the shared status/latch semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from memory_system.artifacts import ReadyDistanceMemory
from memory_system.execute.feasible import (
    FEASIBLE,
    FEASIBLE_UNKNOWN,
    NOT_FEASIBLE,
    FeasibleVerifier,
)
from memory_system.types import FeasibleResult, VerifierObservation

READY3D_FORMAT = "libero_ready3d_targets_v1"


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


@dataclass(frozen=True)
class ReadyDistance3DPrototype:
    demo_id: str
    ready_frame: int
    task_name: str
    planner_step_id: int
    skill: str
    arguments_key: tuple[tuple[str, str], ...]
    target_xyz_world: np.ndarray
    eef_pos_ready: np.ndarray
    distance_m: float


class ReadyDistance3DMemory:
    """Minimal ready-distance memory built from a ready3d artifact."""

    def __init__(self, path: str | Path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != READY3D_FORMAT:
            raise ValueError(f"Unsupported {READY3D_FORMAT}: {payload.get('format')!r}")
        self.prototypes = [
            ReadyDistance3DPrototype(
                demo_id=str(item["demo_id"]),
                ready_frame=int(item["ready_frame"]),
                task_name=str(item["task_name"]),
                planner_step_id=int(item["planner_step_id"]),
                skill=str(item["skill"]),
                arguments_key=tuple(
                    (str(k), str(v)) for k, v in sorted(item["arguments"].items())
                ),
                target_xyz_world=np.asarray(item["target_xyz_world"], dtype=np.float64),
                eef_pos_ready=np.asarray(item["eef_pos_ready"], dtype=np.float64),
                distance_m=float(item["distance_m"]),
            )
            for item in payload.get("prototypes", [])
        ]

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> tuple[ReadyDistance3DPrototype, ...]:
        excluded = set(str(item) for item in exclude_demo_ids)
        expected = _arguments_key(arguments)
        exact = [
            p for p in self.prototypes
            if p.demo_id not in excluded
            and p.task_name == str(task_name)
            and p.planner_step_id == int(planner_step_id)
            and p.skill == str(skill)
            and p.arguments_key == expected
        ]
        if exact:
            return tuple(exact)
        fallback = [
            p for p in self.prototypes
            if p.demo_id not in excluded
            and p.task_name == str(task_name)
            and p.skill == str(skill)
            and p.arguments_key == expected
        ]
        by_demo = {p.demo_id: p for p in fallback}
        return tuple(sorted(by_demo.values(), key=lambda p: p.demo_id))


class Feasible3DVerifier(FeasibleVerifier):
    """3D ready-region verifier using metric distance to target surface point."""

    def __init__(
        self,
        ready3d_targets: str | Path,
        phase_targets: str | Path,
        segments_manifest: str | Path,
        wrist_feasible_targets: str | Path | None = None,
        min_demo_votes: int = 2,
        progress_tolerance_m: float = 0.005,
        evidence_updates: int = 2,
    ):
        memory = ReadyDistanceMemory(phase_targets, segments_manifest)
        super().__init__(
            memory,
            None,
            min_demo_votes=min_demo_votes,
            evidence_updates=evidence_updates,
            wrist_feasible_targets=wrist_feasible_targets,
        )
        self.ready3d = ReadyDistance3DMemory(ready3d_targets)
        self.progress_tolerance_m = float(progress_tolerance_m)
        self.previous_distance_m: float | None = None

    def reset(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        super().reset(task_name, planner_step_id, skill, arguments, exclude_demo_ids)
        self.prototypes = self.ready3d.select(
            task_name, planner_step_id, skill, arguments, exclude_demo_ids
        )
        self.previous_distance_m = None

    def update(
        self,
        observation: VerifierObservation,
        target_xyz_world: Any = None,
        confidence: float = 0.0,
    ) -> FeasibleResult:
        if self.task_name is None:
            raise RuntimeError("reset() must be called before update()")
        common = {
            "timestep": observation.timestep,
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
        }
        if len(self.prototypes) < self.min_demo_votes:
            return self._result(
                status=FEASIBLE_UNKNOWN, reason="insufficient_ready_memory", **common
            )
        if target_xyz_world is None or observation.eef_pos is None:
            return self._result(
                status=FEASIBLE_UNKNOWN, reason="target_geometry_unknown", **common
            )

        target = np.asarray(target_xyz_world, dtype=np.float64).reshape(3)
        eef = np.asarray(observation.eef_pos, dtype=np.float64).reshape(3)
        current_distance_m = float(np.linalg.norm(eef - target))
        ready_distances = sorted(
            (p.distance_m for p in self.prototypes), reverse=True
        )
        ready_distance_m = ready_distances[self.min_demo_votes - 1]
        ready_votes = sum(
            current_distance_m <= p.distance_m for p in self.prototypes
        )
        progress_m = None
        if self.previous_distance_m is not None:
            progress_m = self.previous_distance_m - current_distance_m
        self.previous_distance_m = current_distance_m

        wrist_details: dict[str, Any] = {}
        if ready_votes >= self.min_demo_votes:
            if self.wrist_templates:
                wrist_ready, wrist_details = self._wrist_ready(observation.wrist_image)
                if not wrist_ready:
                    status, reason = NOT_FEASIBLE, "wrist_not_ready"
                else:
                    newly_entered = not self.entered_feasible
                    self.entered_feasible = True
                    self.stall_count = 0
                    self.wrong_way_count = 0
                    status, reason = FEASIBLE, (
                        "entered_ready_radius" if newly_entered else "latched_feasible"
                    )
            else:
                newly_entered = not self.entered_feasible
                self.entered_feasible = True
                self.stall_count = 0
                self.wrong_way_count = 0
                status, reason = FEASIBLE, (
                    "entered_ready_radius" if newly_entered else "latched_feasible"
                )
        elif self.entered_feasible:
            status, reason = FEASIBLE, "latched_feasible"
        elif progress_m is None:
            status, reason = FEASIBLE_UNKNOWN, "warmup"
        elif progress_m < -self.progress_tolerance_m:
            self.wrong_way_count += 1
            self.stall_count = 0
            status = (
                NOT_FEASIBLE
                if self.wrong_way_count >= self.evidence_updates
                else FEASIBLE_UNKNOWN
            )
            reason = "persistent_reversal" if status == NOT_FEASIBLE else "reversal_candidate"
        elif abs(progress_m) <= self.progress_tolerance_m:
            self.stall_count += 1
            self.wrong_way_count = 0
            status = (
                NOT_FEASIBLE
                if self.stall_count >= self.evidence_updates
                else FEASIBLE_UNKNOWN
            )
            reason = "persistent_stagnation" if status == NOT_FEASIBLE else "stagnation_candidate"
        else:
            self.stall_count = 0
            self.wrong_way_count = 0
            status, reason = FEASIBLE_UNKNOWN, "approaching"

        return self._result(
            status=status,
            reason=reason,
            current_distance_m=current_distance_m,
            ready_distance_m=ready_distance_m,
            progress_m=progress_m,
            target_xyz_world=target,
            **common,
            details={
                **wrist_details,
                "target_scale_px": None,
                "ready_demo_ids": [
                    p.demo_id for p in self.prototypes
                    if current_distance_m <= p.distance_m
                ],
            },
        )
