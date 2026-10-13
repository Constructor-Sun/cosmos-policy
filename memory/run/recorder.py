"""Record one run's facts, and nothing else.

The recorder accumulates what the run tells it -- the steps that were executed
and the events that happened -- and hands back a ``Run`` that diagnosis, replay
and the training export can read.  It opens one span per phase and fills the
parts that are facts of the recording: when the phase ran, which decisions ended
it, and who held control.  Everything that would be a judgement (is the skill
done, what does the object look like, is the phase over) is somebody else's and
arrives as an event.

It never moves the cursor, changes the held-object belief, switches control,
retrieves, plans or maps poses.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from memory.types import CaseSpec, PhaseSpec, PhaseSpan, Run, RunEvent, RunStep

_HOLDING_FACT_GAP = 0.003  # measured closedness: above this the gripper holds something


class RunRecorder:
    """Accumulate one run."""

    def __init__(self, *, keep_observations: bool = False) -> None:
        self.keep_observations = bool(keep_observations)
        self._reset()

    # ---- lifecycle

    def begin(self, case: CaseSpec, phases: Sequence[PhaseSpec]) -> None:
        """Start a run: register identity and the phase plan."""
        self._reset()
        self._case = case
        self._phases = tuple(phases)

    def record_step(
        self,
        step: int,
        action: Any,
        source: str,
        observation: Mapping[str, Any] | None = None,
        *,
        chunk_start: bool = False,
        control_space: str = "cartesian",
        collected: bool = True,
        memory_frame: int | None = None,
        data_row: int | None = None,
    ) -> None:
        """Record one executed step.

        ``chunk_start`` marks the first step of a new action chunk, so the
        boundaries survive into the record.  ``collected`` is False for
        alignment frames, which stay in the record but not in the data stream.
        ``memory_frame``/``data_row`` build the index map when the caller knows
        them (memory frame <-> run step <-> data row).
        """
        self._steps.append(
            RunStep(
                step=int(step),
                action=np.asarray(action, dtype=np.float32).copy(),
                source=str(source),
                chunk_start=bool(chunk_start),
                control_space=str(control_space),
                collected=bool(collected),
            )
        )
        if observation is not None and self.keep_observations:
            self._observations.append((int(step), dict(observation)))
        # The span of the phase this step belongs to starts at its first step.
        phase = self._active_phase()
        if phase is not None:
            self._span_for(phase, int(step))
        gap = _gripper_gap(observation)
        if gap is not None:
            self._holding_facts.append((int(step), bool(gap > _HOLDING_FACT_GAP)))
        if memory_frame is not None and data_row is not None:
            self._frame_index.append((int(memory_frame), int(step), int(data_row)))

    def record_event(self, event: RunEvent) -> None:
        """Record one event and let it update the open phase span."""
        self._events.append(event)
        phase = event.phase if event.phase is not None else self._active_phase()
        if event.kind == "phase_boundary":
            self._close_span(int(event.step))
            self._phase_index += 1
            return
        if phase is None:
            return
        span = self._span_for(phase, int(event.step))
        if event.kind == "verdict":
            self._append_decision(span, event)
        if "holding" in event.detail:
            self._holding.append((int(event.step), bool(event.detail["holding"])))

    def finalize(self, success: bool) -> Run:
        """Close the run and produce it."""
        if self._steps:
            # A phase still open when the run ends ends at the last step.
            self._close_span(int(self._steps[-1].step))
        return Run(
            case=self._case,
            phases=self._phases,
            steps=tuple(self._steps),
            events=tuple(self._events),
            spans=tuple(self._spans),
            holding=tuple(self._holding),
            holding_facts=tuple(self._holding_facts),
            frame_index=tuple(self._frame_index),
            success=bool(success),
        )

    # ---- internals

    def _reset(self) -> None:
        self._case: CaseSpec | None = None
        self._phases: tuple[PhaseSpec, ...] = ()
        self._steps: list[RunStep] = []
        self._events: list[RunEvent] = []
        self._spans: list[PhaseSpan] = []
        self._span_index: dict[int, int] = {}
        self._phase_index = 0
        self._holding: list[tuple[int, bool]] = []
        self._holding_facts: list[tuple[int, bool]] = []
        self._frame_index: list[tuple[int, int, int]] = []
        self._observations: list[tuple[int, dict]] = []

    def _active_phase(self) -> PhaseSpec | None:
        if self._phase_index >= len(self._phases):
            return None
        return self._phases[self._phase_index]

    def _span_for(self, phase: PhaseSpec, step: int) -> PhaseSpan:
        key = phase.planner_step_id
        if key not in self._span_index:
            self._span_index[key] = len(self._spans)
            self._spans.append(
                PhaseSpan(
                    phase_index=self._phase_index,
                    planner_step_id=phase.planner_step_id,
                    skill=phase.skill,
                    arguments=dict(phase.arguments),
                    start_step=step,
                )
            )
        return self._spans[self._span_index[key]]

    def _close_span(self, end_step: int | None) -> None:
        phase = self._active_phase()
        if phase is None:
            return
        index = self._span_index.get(phase.planner_step_id)
        if index is None or end_step is None:
            return
        self._spans[index].end_step = int(end_step)

    def _append_decision(self, span: PhaseSpan, event: RunEvent) -> None:
        detail = event.detail
        if "reason" not in detail:
            return
        span.decisions.append(
            (
                int(event.step),
                _Decision(
                    advance=bool(detail.get("advance", True)),
                    semantic_completed=bool(detail.get("semantic_completed", False)),
                    reason=str(detail["reason"]),
                    action_chunks=int(detail.get("action_chunks", 0)),
                ),
            )
        )


class _Decision:
    """The decision fields a span stores, in the shape its as_dict expects."""

    __slots__ = ("advance", "semantic_completed", "reason", "action_chunks")

    def __init__(self, *, advance: bool, semantic_completed: bool, reason: str, action_chunks: int) -> None:
        self.advance = advance
        self.semantic_completed = semantic_completed
        self.reason = reason
        self.action_chunks = action_chunks


def _gripper_gap(observation: Mapping[str, Any] | None) -> float | None:
    """Measured gripper opening, or None when the frame does not report one."""
    if observation is None:
        return None
    value = observation.get("robot0_gripper_qpos")
    if value is None:
        return None
    try:
        qpos = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if qpos.size < 2:
        return None
    gap = float(qpos[0] - qpos[1])
    return gap if np.isfinite(gap) else None
