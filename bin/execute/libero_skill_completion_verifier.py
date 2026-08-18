#!/usr/bin/env python3
"""Positive-only skill-completion verifier for LIBERO."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
try:
    from dtaidistance import dtw_ndim
except ImportError as exc:
    raise ImportError("Install temporal verifier dependency: pip install dtaidistance") from exc


SKILL_COMPLETE = "SKILL_COMPLETE"
COMPLETION_UNKNOWN = "COMPLETION_UNKNOWN"
WINDOW_SIZE = 5


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def _phase_key(task: str, step: int, skill: str, arguments: dict[str, Any]):
    return str(task), int(step), str(skill), _arguments_key(arguments)


def _delta_window(value: Any) -> np.ndarray | None:
    frames = torch.nan_to_num(torch.as_tensor(value, dtype=torch.float32))
    if frames.ndim < 2 or len(frames) < WINDOW_SIZE:
        return None
    frames = frames[-WINDOW_SIZE:].flatten(1)
    frames = frames / torch.linalg.vector_norm(frames, dim=1, keepdim=True).clamp_min(1e-12)
    return np.ascontiguousarray((frames[1:] - frames[:-1]).numpy(), dtype=np.double)


def _dtw(left: np.ndarray, right: np.ndarray) -> float:
    return float(dtw_ndim.distance_fast(left, right, use_pruning=True)) / sqrt(
        max(len(left), len(right))
    )


@dataclass(frozen=True)
class SuccessPrototype:
    demo_id: str
    positive: np.ndarray
    source: str


@dataclass(frozen=True)
class SkillCompletionResult:
    status: str
    reason: str
    timestep: int | None
    distance: float | None
    success_radius: float | None
    memory_count: int
    confirmation_count: int
    details: dict[str, Any] = field(default_factory=dict)


class SkillSuccessMemory:
    """Read only the success embeddings from skill-memory files."""

    def __init__(self, path: str | Path):
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.glob("skill_memory_*.pt"))
        self._per_phase: dict[tuple, list[SuccessPrototype]] = {}
        for file in files:
            payload = torch.load(file, map_location="cpu", weights_only=False)
            if payload.get("format") not in {"libero_skill_memory_v2", "libero_skill_memory_v3"}:
                continue
            for segment in payload.get("segments", []):
                sequence = segment.get("completion_vae_sequence")
                if not segment.get("completion_sequence_valid") or sequence is None:
                    continue
                sequence = torch.as_tensor(sequence)
                positive = _delta_window(sequence)
                if positive is None:
                    continue
                key = _phase_key(
                    payload["task_name"], segment["planner_step_id"],
                    segment["skill"], segment.get("arguments", {}),
                )
                self._per_phase.setdefault(key, []).append(SuccessPrototype(
                    str(payload["demo_id"]), positive,
                    str(segment.get("completion_sequence_source", "unknown")),
                ))

    def select(self, task: str, step: int, skill: str, arguments: dict[str, Any],
               exclude_demo_ids: Iterable[str] = ()) -> tuple[SuccessPrototype, ...]:
        excluded = {str(item) for item in exclude_demo_ids}
        exact = self._per_phase.get(_phase_key(task, step, skill, arguments), [])
        selected = [item for item in exact if item.demo_id not in excluded]
        if not selected:
            args_key = _arguments_key(arguments)
            selected = [item for (name, _step, kind, args), items in self._per_phase.items()
                        if name == task and kind == skill and args == args_key
                        for item in items if item.demo_id not in excluded]
        return tuple({item.demo_id: item for item in selected}.values())


class LiberoSkillCompletionVerifier:
    """Latch completion after two success-memory matches and a skill gate."""

    def __init__(self, memory: SkillSuccessMemory | str | Path):
        self.memory = memory if isinstance(memory, SkillSuccessMemory) else SkillSuccessMemory(memory)
        self.prototypes: tuple[SuccessPrototype, ...] = ()
        self.history: list[torch.Tensor] = []
        self.baseline_distances: list[float] = []
        self.test_threshold: float | None = None
        self.confirmation_count = 0
        self.completed = False

    def reset(self, task: str, step: int, skill: str, arguments: dict[str, Any],
              exclude_demo_ids: Iterable[str] = ()) -> None:
        self.prototypes = self.memory.select(task, step, skill, arguments, exclude_demo_ids)
        self.history = []
        self.baseline_distances = []
        self.test_threshold = None
        self.confirmation_count, self.completed = 0, False

    def _observe_distance(self, current_vae: Any | None):
        if current_vae is None:
            return None, None, "current_vae_unknown"
        self.history.append(torch.as_tensor(current_vae).cpu())
        self.history = self.history[-WINDOW_SIZE:]
        current = _delta_window(torch.stack(self.history))
        if current is None:
            return None, None, "temporal_warmup"
        candidates = [item for item in self.prototypes
                      if item.positive.shape[1] == current.shape[1]]
        if not candidates:
            return None, None, "vae_shape_mismatch"
        distance, nearest = min(
            ((_dtw(current, item.positive), item) for item in candidates),
            key=lambda pair: pair[0],
        )
        return distance, nearest, "distance_observed"

    def calibrate(self, current_vae: Any | None) -> None:
        distance, _nearest, _reason = self._observe_distance(current_vae)
        if distance is not None and np.isfinite(distance):
            self.baseline_distances.append(distance)

    def freeze_baseline(self) -> None:
        self.test_threshold = min(self.baseline_distances, default=None)

    def update(self, current_vae: Any | None, *, gripper_closed: bool | None = None,
               timestep: int | None = None, wrist_image: Any = None) -> SkillCompletionResult:
        distance, nearest = None, None
        if self.completed:
            reason = "latched_complete"
        elif not self.prototypes:
            reason = "insufficient_temporal_memory"
        else:
            distance, nearest, reason = self._observe_distance(current_vae)
            if distance is None:
                pass
            elif not np.isfinite(distance):
                reason = "dtw_nonfinite"
            elif self.test_threshold is None:
                self.test_threshold = distance
                reason = "baseline_initialized"
            elif distance >= self.test_threshold:
                reason = "above_test_threshold"
            else:
                self.confirmation_count += 1
                self.completed = self.confirmation_count >= 2
                reason = "confirmed_complete" if self.completed else "completion_candidate"
            if reason not in {"completion_candidate", "confirmed_complete"}:
                self.confirmation_count = 0
        return SkillCompletionResult(
            SKILL_COMPLETE if self.completed else COMPLETION_UNKNOWN, reason, timestep,
            distance, self.test_threshold, len(self.prototypes), self.confirmation_count,
            {"nearest_demo_id": nearest.demo_id if nearest else None,
             "success_embedding_source": nearest.source if nearest else None,
             "baseline_count": len(self.baseline_distances),
             "distance_ratio": (distance / self.test_threshold
                                if distance is not None and self.test_threshold else None),
             "window_size": WINDOW_SIZE},
        )
