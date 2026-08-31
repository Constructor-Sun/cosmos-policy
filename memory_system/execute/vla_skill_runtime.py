"""Minimal VLA-only skill completion runtime.

This module owns only the memory phase cursor and completion lifecycle.  It
does not execute planner actions or mutate an action queue; the rollout caller
consumes ``SkillDecision.advance`` and performs those two integration steps.
"""
from __future__ import annotations

from typing import Any, Iterable

from memory_system.execute.plan import PhaseSpec
from memory_system.execute.skill_completion import (
    DEFAULT_MAX_ACTION_CHUNKS,
    PickSkillCompletion,
    SkillDecision,
    TimedSkillCompletion,
    TimeoutOnlySkillCompletion,
)

SKILL_MAX_ACTION_CHUNKS: dict[str, int] = {
    "Pick": 7,
    "PlaceOn": 3,
    "PlaceIn": 3,
    "Open": 3,
    "Close": 3,
    "TurnOn": 3,
}


def make_completion(phase: PhaseSpec) -> TimedSkillCompletion:
    """Construct the checker configured for one memory phase."""
    budget = SKILL_MAX_ACTION_CHUNKS.get(phase.skill, DEFAULT_MAX_ACTION_CHUNKS)
    if phase.skill == "Pick":
        return PickSkillCompletion(max_action_chunks=budget)
    return TimeoutOnlySkillCompletion(max_action_chunks=budget)


class VLASkillRuntime:
    """Track one exact memory sequence across VLA action chunks.

    ``begin_vla`` starts a completion window after any planner/alignment work
    has ended.  Planner code never calls ``observe_vla_frame`` or
    ``finish_action_chunk``.  When a rule or timeout advances the phase, this
    class deactivates the checker and leaves queue cleanup to the caller.
    """

    def __init__(
        self,
        phases: Iterable[PhaseSpec],
        *,
        task_name: str | None = None,
        demo_id: str | None = None,
        episode_id: int | str | None = None,
    ) -> None:
        self.phases = tuple(phases)
        if not self.phases:
            raise ValueError("VLASkillRuntime requires a non-empty memory sequence")
        self.task_name = None if task_name is None else str(task_name)
        self.demo_id = None if demo_id is None else str(demo_id)
        self.episode_id = episode_id
        self.phase_index = 0
        self._completion: TimedSkillCompletion | None = None
        self._active = False
        self._vla_start_frame: int | None = None
        self._summaries: list[dict[str, Any]] = []
        self._last_decision = SkillDecision(False, False, "running", 0)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def active_phase(self) -> PhaseSpec | None:
        if self.phase_index >= len(self.phases):
            return None
        return self.phases[self.phase_index]

    @property
    def completion(self) -> TimedSkillCompletion | None:
        return self._completion if self._active else None

    @property
    def exhausted(self) -> bool:
        return self.phase_index >= len(self.phases)

    @property
    def summaries(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(summary) for summary in self._summaries)

    def begin_vla(self, *, frame: int | None = None) -> PhaseSpec:
        """Activate the current phase and reset its completion state."""
        phase = self.active_phase
        if phase is None:
            raise RuntimeError("memory phase sequence is exhausted")
        if self._active:
            raise RuntimeError("current VLA phase is already active")
        self._completion = make_completion(phase)
        # Make the lifecycle contract explicit even though construction resets.
        self._completion.reset()
        self._active = True
        self._vla_start_frame = None if frame is None else int(frame)
        self._last_decision = self._completion.decision
        return phase

    def observe_vla_frame(
        self,
        *,
        target_points: Any = None,
        eef_pos: Any = None,
        eef_quat: Any = None,
        gripper_closed: bool | None = None,
        gripper_qpos: Any = None,
        frame: int | None = None,
    ) -> SkillDecision:
        """Observe one post-action VLA frame and advance on a rule match."""
        completion = self._require_active()
        decision = completion.observe_frame(
            target_points=target_points,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper_closed=gripper_closed,
            gripper_qpos=gripper_qpos,
        )
        self._last_decision = decision
        if decision.advance:
            self._advance(decision, frame)
        return decision

    def finish_action_chunk(self, *, frame: int | None = None) -> SkillDecision:
        """Close one naturally exhausted VLA action chunk."""
        completion = self._require_active()
        decision = completion.finish_action_chunk()
        self._last_decision = decision
        if decision.advance:
            self._advance(decision, frame)
        return decision

    def _require_active(self) -> TimedSkillCompletion:
        if not self._active or self._completion is None:
            raise RuntimeError("completion is inactive outside the VLA window")
        return self._completion

    def _advance(self, decision: SkillDecision, frame: int | None) -> None:
        phase = self.active_phase
        if phase is None:
            raise RuntimeError("cannot advance an exhausted memory sequence")
        self._summaries.append(
            {
                "episode_id": self.episode_id,
                "memory_demo_id": self.demo_id,
                "phase_index": self.phase_index,
                "planner_step_id": phase.planner_step_id,
                "skill": phase.skill,
                "arguments": dict(phase.arguments),
                "vla_start_frame": self._vla_start_frame,
                "vla_end_frame": None if frame is None else int(frame),
                "action_chunks": decision.action_chunks,
                "semantic_completed": bool(decision.semantic_completed),
                "advance_reason": decision.reason,
            }
        )
        self._active = False
        self._completion = None
        self._vla_start_frame = None
        self.phase_index += 1

    def finalize(self, task_success: bool) -> tuple[dict[str, Any], ...]:
        """Attach the final episode result to already completed phase summaries."""
        for summary in self._summaries:
            summary["task_success"] = bool(task_success)
        return self.summaries
