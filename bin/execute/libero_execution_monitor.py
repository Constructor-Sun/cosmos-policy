#!/usr/bin/env python3
"""Sequential, observation-only orchestration of the three LIBERO verifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    from .libero_feasible_region_verifier import FEASIBLE, NOT_FEASIBLE
    from .libero_phase_monitor import PhaseSpec
    from .libero_phase_verifier import PHASE_ERROR, PHASE_OK
    from .libero_skill_completion_verifier_geo import SKILL_COMPLETE
except ImportError:
    from libero_feasible_region_verifier import FEASIBLE, NOT_FEASIBLE  # type: ignore[no-redef]
    from libero_phase_monitor import PhaseSpec  # type: ignore[no-redef]
    from libero_phase_verifier import PHASE_ERROR, PHASE_OK  # type: ignore[no-redef]
    from libero_skill_completion_verifier_geo import SKILL_COMPLETE  # type: ignore[no-redef]


PHASE_CHECK = "PHASE_CHECK"
FEASIBLE_CHECK = "FEASIBLE_CHECK"
COMPLETION_CHECK = "COMPLETION_CHECK"
PLAN_COMPLETE = "PLAN_COMPLETE"

FEASIBLE_TIMEOUT_CHUNKS = 2
ACTION_CHUNK_STEPS = 16
PLACE_SKILLS = {"PlaceIn", "PlaceOn"}


@dataclass(frozen=True)
class ExecutionMonitorResult:
    task_name: str
    timestep: int | None
    observed_step_id: int
    active_step_id: int
    stage_before: str
    stage_after: str
    reason: str
    phase_result: Any | None = None
    feasible_result: Any | None = None
    completion_result: Any | None = None
    completed_step_id: int | None = None
    phase_advanced: bool = False
    plan_complete: bool = False
    should_intervene: bool = False
    intervention_reason: str | None = None


class LiberoExecutionMonitor:
    """Run exactly one verifier stage per observation and advance on completion."""

    def __init__(self, plans: dict[str, tuple[PhaseSpec, ...]], phase_verifier: Any,
                 feasible_verifier: Any, completion_verifier: Any):
        self.plans = plans
        self.phase_verifier = phase_verifier
        self.feasible_verifier = feasible_verifier
        self.completion_verifier = completion_verifier
        self.task_name: str | None = None
        self.phase_index = 0
        self.stage = PHASE_CHECK
        self.exclude_demo_ids: tuple[str, ...] = ()
        self.target_xy: tuple[float, float] | None = None
        self.target_bbox: tuple[float, float, float, float] | None = None
        self.target_confidence = 0.0
        self.feasible_enter_timestep: int | None = None
        self.intervention_reason: str | None = None

    @property
    def current_phase(self) -> PhaseSpec:
        if self.task_name is None:
            raise RuntimeError("start_episode() must be called first")
        return self.plans[self.task_name][self.phase_index]

    def _reset_phase(self) -> None:
        phase = self.current_phase
        arguments = phase.arguments
        for verifier in (self.phase_verifier, self.feasible_verifier,
                         self.completion_verifier):
            verifier.reset(self.task_name, phase.planner_step_id, phase.skill,
                           arguments, self.exclude_demo_ids)
        self.stage = PHASE_CHECK
        self.target_xy = self.target_bbox = None
        self.target_confidence = 0.0
        self.feasible_enter_timestep = None
        self.intervention_reason = None

    def retry_current_phase(self) -> None:
        """Reset the current phase after a recovery movement."""
        self._reset_phase()

    def mark_feasible_after_recovery(self, timestep: int | None = None) -> None:
        """Jump directly to completion after a feasible-region recovery."""
        self.feasible_verifier.entered_feasible = True
        self.feasible_verifier.stall_count = 0
        self.feasible_verifier.wrong_way_count = 0
        self.stage = COMPLETION_CHECK
        self.feasible_enter_timestep = timestep
        self.intervention_reason = None
        self.completion_verifier.freeze_baseline()

    def mark_wrong_grasp_recovery_finished(self, timestep: int | None = None) -> None:
        """After a wrong-object grasp recovery: keep the phase anchor, return
        to the feasible gate so the feasible verifier re-adjusts from scratch,
        and clear only the grasp-completion state (do NOT retry the phase)."""
        self.stage = PHASE_CHECK if self.target_xy is None else FEASIBLE_CHECK
        self.feasible_enter_timestep = timestep
        self.intervention_reason = None
        if hasattr(self.feasible_verifier, "entered_feasible"):
            self.feasible_verifier.entered_feasible = False
        self.feasible_verifier.stall_count = 0
        self.feasible_verifier.wrong_way_count = 0
        cv = self.completion_verifier
        cv.pick_close_xyz = None
        cv.pick_moved = False
        cv.pick_last_center = None
        cv.pick_confirmations.clear()

    def start_episode(self, task_name: str, exclude_demo_ids: Iterable[str] = ()):
        if task_name not in self.plans or not self.plans[task_name]:
            raise KeyError(f"No phase plan for task {task_name!r}")
        self.task_name, self.phase_index = str(task_name), 0
        self.exclude_demo_ids = tuple(str(item) for item in exclude_demo_ids)
        self._reset_phase()
        return self.plans[task_name]

    def observe_vae(self, current_vae: Any | None = None, *, gripper_closed: bool | None = None,
                    gripper_xy: Any = None, timestep: int | None = None,
                    wrist_image: Any = None, gripper_qpos: Any = None,
                    eef_pos: Any = None):
        """Consume one low-level observation; only completion can advance."""
        if self.task_name is None:
            raise RuntimeError("start_episode() must be called before observe_vae()")
        # Step-level wrong-object grasp guard (runs in every stage; the close
        # edge must be seen even while the monitor is still in PHASE/FEASIBLE).
        if hasattr(self.completion_verifier, "evaluate_close_edge"):
            self.completion_verifier.evaluate_close_edge(
                gripper_closed, gripper_xy, timestep
            )
            if self.completion_verifier.wrong_grasp_pending:
                self.completion_verifier.wrong_grasp_pending = False
                self.completion_verifier.wrong_grasp_count += 1
                self.intervention_reason = "wrong_grasp_close"
                return ExecutionMonitorResult(
                    self.task_name, timestep,
                    self.current_phase.planner_step_id,
                    self.current_phase.planner_step_id,
                    self.stage, self.stage, "wrong_grasp_close",
                    should_intervene=True,
                    intervention_reason="wrong_grasp_close",
                )
        if self.stage in {PHASE_CHECK, FEASIBLE_CHECK}:
            self.completion_verifier.calibrate(current_vae)
            if hasattr(self.completion_verifier, "observe_wrist"):
                self.completion_verifier.observe_wrist(wrist_image)

            # For Place skills, release can complete the phase even before
            # the feasible region has latched. Fall through to the shared
            # completion handling below; unknown results are not logged.
            if not (self.stage == FEASIBLE_CHECK and self.current_phase.skill in PLACE_SKILLS):
                return None
        elif self.stage != COMPLETION_CHECK:
            return None
        before, observed_step = self.stage, self.current_phase.planner_step_id
        result = self.completion_verifier.update(
            current_vae, gripper_closed=gripper_closed, gripper_xy=gripper_xy,
            target_bbox=self.target_bbox, timestep=timestep,
            wrist_image=wrist_image, gripper_qpos=gripper_qpos,
            eef_pos=eef_pos,
        )
        if result.status != SKILL_COMPLETE and before == FEASIBLE_CHECK:
            return None
        if (
            result.status != SKILL_COMPLETE
            and self.feasible_enter_timestep is not None
            and timestep is not None
            and timestep - self.feasible_enter_timestep
            >= FEASIBLE_TIMEOUT_CHUNKS * ACTION_CHUNK_STEPS
            and self.intervention_reason is None
        ):
            self.intervention_reason = "no_completion_within_two_chunks_after_feasible"
        completed_step, advanced, reason = None, False, result.reason
        if result.status == SKILL_COMPLETE:
            completed_step = observed_step
            if self.phase_index + 1 == len(self.plans[self.task_name]):
                self.stage, reason = PLAN_COMPLETE, "task_plan_complete"
            else:
                self.phase_index += 1
                self._reset_phase()
                advanced, reason = True, "skill_complete_next_phase"
        return ExecutionMonitorResult(
            self.task_name, timestep, observed_step, self.current_phase.planner_step_id,
            before, self.stage, reason, completion_result=result,
            completed_step_id=completed_step, phase_advanced=advanced,
            plan_complete=self.stage == PLAN_COMPLETE,
            should_intervene=self.intervention_reason is not None,
            intervention_reason=self.intervention_reason,
        )

    def observe(self, third_view_rgb: np.ndarray, gripper_xy: Any,
                current_vae: Any | None = None, *,
                gripper_closed: bool | None = None,
                timestep: int | None = None,
                wrist_image: Any = None,
                gripper_qpos: Any = None,
                eef_pos: Any = None) -> ExecutionMonitorResult:
        if self.task_name is None:
            raise RuntimeError("start_episode() must be called before observe()")
        before, observed_step = self.stage, self.current_phase.planner_step_id
        phase_result = feasible_result = completion_result = None
        completed_step, advanced = None, False

        if before in {PHASE_CHECK, FEASIBLE_CHECK} and hasattr(
            self.completion_verifier, "observe_wrist"
        ):
            self.completion_verifier.observe_wrist(wrist_image)

        # At chunk boundaries the release is often first observed. Route Place
        # skills through observe_vae as well so release-based completion can
        # fire before the feasible verifier turns leaving into a reversal.
        if before == FEASIBLE_CHECK and self.current_phase.skill in PLACE_SKILLS:
            step_result = self.observe_vae(
                current_vae, gripper_closed=gripper_closed,
                gripper_xy=gripper_xy, timestep=timestep,
                wrist_image=wrist_image, gripper_qpos=gripper_qpos,
                eef_pos=eef_pos,
            )
            if step_result is not None:
                return step_result

        if before == PHASE_CHECK:
            phase_result = self.phase_verifier.update(third_view_rgb, gripper_xy)
            details = phase_result.details
            bbox = details.get("matched_bbox_xyxy")
            if details.get("reason") == "no_templates":
                self.stage, reason = COMPLETION_CHECK, "no_phase_memory_try_completion"
                self.completion_verifier.freeze_baseline()
            elif (phase_result.status == PHASE_OK
                  and phase_result.progress_px is not None and phase_result.progress_px > 0
                  and phase_result.target_xy is not None and bbox is not None):
                self.target_xy = tuple(phase_result.target_xy)
                self.target_bbox = tuple(bbox)
                self.target_confidence = float(getattr(phase_result, "confidence", 0.0))
                enough_ready = (len(self.feasible_verifier.prototypes)
                                >= self.feasible_verifier.min_demo_votes)
                self.stage = FEASIBLE_CHECK if enough_ready else COMPLETION_CHECK
                reason = "phase_confirmed" if enough_ready else "no_ready_memory_try_completion"
                if self.stage == COMPLETION_CHECK:
                    self.completion_verifier.freeze_baseline()
            elif phase_result.status == PHASE_ERROR:
                self.intervention_reason = "phase_error"
                reason = "phase_error"
            else:
                reason = "waiting_for_phase_confirmation"
        elif before == FEASIBLE_CHECK:
            feasible_result = self.feasible_verifier.update(
                self.target_xy, self.target_bbox, gripper_xy,
                confidence=self.target_confidence, timestep=timestep,
                wrist_image=wrist_image,
            )
            if feasible_result.status == FEASIBLE:
                self.stage, reason = COMPLETION_CHECK, "entered_feasible_region"
                self.feasible_enter_timestep = timestep
                self.completion_verifier.freeze_baseline()
            elif feasible_result.status == NOT_FEASIBLE:
                self.intervention_reason = "feasible_error"
                reason = "feasible_error"
            else:
                reason = feasible_result.reason
        elif before == COMPLETION_CHECK:
            reason = "waiting_for_step_observation"
        else:
            reason = "task_plan_complete"

        return ExecutionMonitorResult(
            self.task_name, timestep, observed_step, self.current_phase.planner_step_id,
            before, self.stage, reason, phase_result, feasible_result, completion_result,
            completed_step, advanced, self.stage == PLAN_COMPLETE,
            self.intervention_reason is not None, self.intervention_reason,
        )
