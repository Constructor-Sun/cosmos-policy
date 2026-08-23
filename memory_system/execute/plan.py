"""Skill-plan loading and current/next phase monitoring.

This module is the memory_system replacement for
``bin/execute/libero_phase_monitor.py``.  The main execution monitor only
needs the plan loader; the comparative monitor is kept as an independent
strategy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from memory_system.artifacts import PhaseTargetMemory
from memory_system.execute.phase import PHASE_ERROR, PHASE_UNKNOWN, PhaseVerifier
from memory_system.types import PhaseResult, VerifierObservation


@dataclass(frozen=True)
class PhaseSpec:
    planner_step_id: int
    skill: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class PhaseMonitorResult:
    task_name: str
    timestep: int | None
    current_step_id: int
    next_step_id: int | None
    active_step_id: int
    current: PhaseResult
    next: PhaseResult | None
    switch_evidence: int
    switched: bool
    deviation_candidate: bool
    reason: str


def _argument_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def load_phase_plans(path: str | Path) -> dict[str, tuple[PhaseSpec, ...]]:
    """Load one validated, ordered planner sequence for every task."""
    payload = json.loads(Path(path).read_text())
    phases: dict[str, dict[int, PhaseSpec]] = {}
    for record in payload.get("records", []):
        if not record.get("valid"):
            continue
        task_name = str(record["task_name"])
        task_phases = phases.setdefault(task_name, {})
        for segment in record.get("segments", []):
            if segment.get("status") == "already_satisfied":
                continue
            step_id = int(segment["planner_step_id"])
            spec = PhaseSpec(
                step_id,
                str(segment["skill"]),
                {str(key): str(value) for key, value in segment.get("arguments", {}).items()},
            )
            existing = task_phases.get(step_id)
            if existing and (
                existing.skill != spec.skill
                or _argument_key(existing.arguments) != _argument_key(spec.arguments)
            ):
                raise ValueError(
                    f"Inconsistent phase {step_id} for {task_name}: {existing} != {spec}"
                )
            task_phases[step_id] = spec
    return {
        task_name: tuple(task_phases[index] for index in sorted(task_phases))
        for task_name, task_phases in phases.items()
    }


class PhaseMonitor:
    """Switch monotonically when motion favors the next target for N updates."""

    def __init__(
        self,
        phase_targets: str | Path | PhaseTargetMemory,
        segments_manifest: str | Path,
        switch_updates: int = 2,
        min_progress_px: float = 2.0,
        **verifier_kwargs: Any,
    ):
        if switch_updates <= 0:
            raise ValueError("switch_updates must be positive")
        if not np.isfinite(min_progress_px) or min_progress_px < 0:
            raise ValueError("min_progress_px must be finite and non-negative")
        self.memory = (
            phase_targets
            if isinstance(phase_targets, PhaseTargetMemory)
            else PhaseTargetMemory(phase_targets)
        )
        self.plans = load_phase_plans(segments_manifest)
        self.switch_updates = int(switch_updates)
        self.min_progress_px = float(min_progress_px)
        self.current_verifier = PhaseVerifier(self.memory, **verifier_kwargs)
        self.next_verifier = PhaseVerifier(self.memory, **verifier_kwargs)
        self.task_name: str | None = None
        self.phase_index = 0
        self.switch_evidence = 0
        self.exclude_demo_ids: tuple[str, ...] = ()
        self.history: list[PhaseMonitorResult] = []

    @property
    def current_phase(self) -> PhaseSpec:
        if self.task_name is None:
            raise RuntimeError("start_episode() must be called before accessing current_phase")
        return self.plans[self.task_name][self.phase_index]

    @property
    def next_phase(self) -> PhaseSpec | None:
        if self.task_name is None:
            raise RuntimeError("start_episode() must be called before accessing next_phase")
        plan = self.plans[self.task_name]
        return plan[self.phase_index + 1] if self.phase_index + 1 < len(plan) else None

    def _reset_verifier(self, verifier: PhaseVerifier, phase: PhaseSpec | None) -> None:
        if self.task_name is None:
            raise RuntimeError("No active task")
        if phase is None:
            verifier.reset(self.task_name, -1, "", {}, self.exclude_demo_ids)
        else:
            verifier.reset(
                self.task_name,
                phase.planner_step_id,
                phase.skill,
                phase.arguments,
                self.exclude_demo_ids,
            )

    def _reset_pair(self) -> None:
        self._reset_verifier(self.current_verifier, self.current_phase)
        self._reset_verifier(self.next_verifier, self.next_phase)

    def start_episode(
        self,
        task_name: str,
        exclude_demo_ids: Iterable[str] = (),
    ) -> tuple[PhaseSpec, ...]:
        if task_name not in self.plans:
            raise KeyError(f"No phase plan for task {task_name!r}")
        self.task_name = task_name
        self.phase_index = 0
        self.switch_evidence = 0
        self.exclude_demo_ids = tuple(exclude_demo_ids)
        self.history.clear()
        self._reset_pair()
        return self.plans[task_name]

    def observe(self, observation: VerifierObservation) -> PhaseMonitorResult:
        if self.task_name is None:
            raise RuntimeError("start_episode() must be called before observe()")
        current_phase, next_phase = self.current_phase, self.next_phase
        current = self.current_verifier.update(observation)
        following = (
            self.next_verifier.update(observation)
            if next_phase is not None
            else None
        )
        current_progress = current.progress_px
        next_progress = following.progress_px if following else None
        next_supported = (
            following is not None
            and following.status != PHASE_UNKNOWN
            and next_progress is not None
            and next_progress > self.min_progress_px
        )
        deviation = current.status == PHASE_ERROR and not next_supported
        switched, reason = False, "stay_current"
        if following is None:
            self.switch_evidence = 0
            reason = "final_phase"
        elif following.status == PHASE_UNKNOWN:
            reason = "unknown_next_target"
        elif next_progress is None:
            self.switch_evidence = 0
            reason = "warmup"
        elif next_supported and (
            current.status == PHASE_UNKNOWN
            or current_progress is None
            or next_progress > current_progress
        ):
            self.switch_evidence += 1
            reason = "moving_toward_next"
            if self.switch_evidence >= self.switch_updates:
                self.phase_index += 1
                self.switch_evidence = 0
                switched, reason = True, "switched_to_next"
                self._reset_pair()
        else:
            self.switch_evidence = 0
            reason = "deviation_candidate" if deviation else "stay_current"
        result = PhaseMonitorResult(
            task_name=self.task_name,
            timestep=observation.timestep,
            current_step_id=current_phase.planner_step_id,
            next_step_id=next_phase.planner_step_id if next_phase else None,
            active_step_id=self.current_phase.planner_step_id,
            current=current,
            next=following,
            switch_evidence=self.switch_evidence,
            switched=switched,
            deviation_candidate=deviation,
            reason=reason,
        )
        self.history.append(result)
        return result

    def finish_episode(self, success: bool | None = None) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "active_step_id": self.current_phase.planner_step_id if self.task_name else None,
            "observations": len(self.history),
            "switches": sum(item.switched for item in self.history),
            "deviation_candidates": sum(item.deviation_candidate for item in self.history),
            "success": success,
        }
