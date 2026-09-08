"""Observation-only phase, interaction and repair-point recording.

The recorder is deliberately separated from intervention. It consumes one
executed policy action and its post-action observation at a time, while the
caller remains responsible for policy queries and action queues. This keeps
the baseline trajectory unchanged and makes actions[:t_star] the exact replay
prefix used by Step 4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import json

import numpy as np

from memory_system.execute.plan import load_phase_sequences
from memory_system.execute.vla_skill_runtime import VLASkillRuntime

MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "skill_memory_test"
    / "libero_10"
    / "segments_ready_fixed16.json"
)

CUT_IN_STEPS = 16
FIRST_PHASE_MIN_CUT_IN_STEP = 10
CLOSE_CMD = 0.5
OPEN_CMD = -0.5

CONFIRMED = "confirmed"
RELEASED_ONLY = "released_only"
GRASP_FAILED = "grasp_failed"
NEVER_RELEASED = "never_released"
TIMEOUT_GAP = "timeout_gap"
PENDING = "pending"
OBJECT_MISMATCH = "object_mismatch"
OBJECT_UNKNOWN = "object_unknown"
UNLOCATED = "unlocatable"
CLOSE_RECORDED = "close_recorded"

_REPAIRABLE_FAILURES = {GRASP_FAILED, NEVER_RELEASED, OBJECT_MISMATCH}

# The interaction event is skill dependent.  A Place phase is evaluated at
# release, while Pick and the control skills are evaluated when the gripper
# first closes around the interacted object.
_CLOSE_INTERACTION_SKILLS = {"Pick", "TurnOn", "Open", "Close"}
_OPEN_INTERACTION_SKILLS = {"PlaceIn", "PlaceOn"}


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-safe values for records."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _normalise_name(value: Any) -> str:
    return str(value or "").lower().replace("_", "").replace("-", "").replace(" ", "")


def _expected_object(skill: str, arguments: Mapping[str, Any]) -> str | None:
    if skill == "Pick":
        return str(arguments.get("item")) if arguments.get("item") else None
    if skill in {"PlaceIn", "PlaceOn"}:
        return str(arguments.get("item")) if arguments.get("item") else None
    if skill in {"TurnOn", "Open", "Close"}:
        return str(arguments.get("target")) if arguments.get("target") else None
    return None


def interaction_direction(skill: str) -> str | None:
    """Return the gripper transition that represents this skill's interaction."""
    if skill in _OPEN_INTERACTION_SKILLS:
        return "1->0"
    if skill in _CLOSE_INTERACTION_SKILLS:
        return "0->1"
    return None


def object_name_matches(
    skill: str,
    arguments: Mapping[str, Any],
    actual_name: str,
    aliases: Iterable[str] = (),
) -> bool:
    """Match an observed instance against the memory interaction target."""
    expected = _expected_object(skill, arguments)
    if not expected:
        return False
    expected_names = {_normalise_name(expected)}
    expected_names.update(_normalise_name(item) for item in aliases)
    actual = _normalise_name(actual_name)
    if not actual:
        return False
    if actual in expected_names:
        return True
    return any(
        len(name) >= 4 and (name in actual or actual in name)
        for name in expected_names
    )


def _command_state(command: float, previous: int | None) -> int | None:
    if command >= CLOSE_CMD:
        return 1
    if command <= OPEN_CMD:
        return 0
    return previous


def _gripper_gap(obs: Mapping[str, Any]) -> float | None:
    value = obs.get("robot0_gripper_qpos")
    try:
        qpos = np.asarray(value, dtype=np.float64).reshape(-1)
        if qpos.size < 2:
            return None
        gap = float(qpos[0] - qpos[1])
        return gap if np.isfinite(gap) else None
    except (TypeError, ValueError):
        return None


@dataclass
class PhaseSpan:
    """One phase's execution span and event/interaction evidence."""

    phase_index: int
    planner_step_id: int
    skill: str
    arguments: dict
    start_step: int
    end_step: int | None = None
    holding_before: bool | None = None
    holding_after: bool | None = None
    first_flip_step: int | None = None
    flip_direction: str | None = None
    first_close_event_step: int | None = None
    first_open_event_step: int | None = None
    close_event_gap: float | None = None
    close_event_confirmed_step: int | None = None
    inherited_close_event_step: int | None = None
    object_status: str = "not_checked"
    object_match: bool | None = None
    actual_object: dict | None = None
    object_candidates: list[dict] = field(default_factory=list)
    memory_phase_index_at_event: int | None = None
    decisions: list = field(default_factory=list)

    def as_dict(self) -> dict:
        out = _json_value(asdict(self))
        out["decisions"] = [
            {
                "step": d_step,
                "advance": d.advance,
                "reason": d.reason,
                "semantic_completed": bool(d.semantic_completed),
                "action_chunks": int(d.action_chunks),
            }
            for d_step, d in self.decisions
        ]
        return out
@dataclass
class MemoryPhaseEvidence:
    """First-interaction evidence attached to an independent memory phase."""

    phase_index: int
    planner_step_id: int
    skill: str
    arguments: dict
    start_step: int | None = None
    end_step: int | None = None
    first_close_event_step: int | None = None
    first_open_event_step: int | None = None
    first_interaction_event_step: int | None = None
    interaction_direction: str | None = None
    close_event_gap: float | None = None
    close_event_confirmed_step: int | None = None
    object_status: str = "not_checked"
    object_match: bool | None = None
    actual_object: dict | None = None
    object_candidates: list[dict] = field(default_factory=list)
    runtime_phase_index_at_event: int | None = None
    runtime_phase: dict | None = None
    semantic_completed: bool | None = None
    completion_reason: str | None = None
    completion_step: int | None = None

    def as_dict(self) -> dict:
        return _json_value(asdict(self))


def load_task_sequence(
    task_name: str,
    manifest_path: Path = MANIFEST_PATH,
    demo_id: str | None = None,
) -> tuple[str, str, tuple]:
    """Bind one exact memory demo sequence without sorting planner ids."""
    sequences = load_phase_sequences(manifest_path)
    base = str(task_name)
    if base not in {key[0] for key in sequences}:
        matches = sorted(
            {key[0] for key in sequences if base.startswith(f"{key[0]}_")}
        )
        if len(matches) != 1:
            raise KeyError(f"no unique base task for {task_name!r}: {matches}")
        base = matches[0]
    available = sorted(key[1] for key in sequences if key[0] == base)
    if not available:
        raise KeyError(f"no memory demos for {base!r}")
    if demo_id is not None:
        selected = str(demo_id)
        if selected not in available:
            raise KeyError(f"demo {selected!r} is unavailable for {base!r}")
    else:
        signatures = {
            tuple(
                (phase.planner_step_id, phase.skill, tuple(sorted(phase.arguments.items())))
                for phase in sequences[(base, item)]
            )
            for item in available
        }
        if len(signatures) != 1:
            raise ValueError(
                f"task {base!r} has different demo phase orders; demo_id is required"
            )
        selected = available[0]
    return base, selected, sequences[(base, selected)]


class PhaseEventRecorder:
    """Drive VLASkillRuntime in observe-only mode over one episode."""

    def __init__(
        self,
        phases: tuple,
        *,
        task_name: str,
        demo_id: str | None = None,
        episode_id: str | None = None,
        object_query: Callable[[Mapping[str, Any], Any, Any], Any] | None = None,
    ) -> None:
        self._phases = tuple(phases)
        self._runtime = VLASkillRuntime(
            self._phases, task_name=task_name, demo_id=demo_id, episode_id=episode_id
        )
        self.spans: list[PhaseSpan] = []
        self._memory_evidence = [
            MemoryPhaseEvidence(
                phase_index=index,
                planner_step_id=int(phase.planner_step_id),
                skill=phase.skill,
                arguments=dict(phase.arguments),
            )
            for index, phase in enumerate(self._phases)
        ]
        self._memory_cursor = 0
        self._runtime_phase_starts: dict[int, int] = {}
        self._prev_gripper_state: int | None = None
        self._last_close_event_step: int | None = None
        self._pending_close_interaction: dict | None = None
        self._diagnosis_stopped = False
        self._candidate = None
        self._object_query = object_query
        self._memory_objects: set[str] = set()
        for phase in self._phases:
            name = _expected_object(phase.skill, phase.arguments)
            if name:
                self._memory_objects.add(_normalise_name(name))

    @property
    def active_phase(self):
        return self._runtime.active_phase

    def _begin_span(self, *, start_step: int) -> None:
        phase = self._runtime.active_phase
        if phase is None:
            return
        runtime_index = int(self._runtime.phase_index)
        self._runtime_phase_starts.setdefault(runtime_index, int(start_step))
        if (
            self._memory_cursor < len(self._memory_evidence)
            and self._memory_evidence[self._memory_cursor].start_step is None
        ):
            self._memory_evidence[self._memory_cursor].start_step = int(start_step)
        self.spans.append(
            PhaseSpan(
                phase_index=runtime_index,
                planner_step_id=int(phase.planner_step_id),
                skill=phase.skill,
                arguments=dict(phase.arguments),
                start_step=start_step,
                holding_before=(
                    None
                    if self._prev_gripper_state is None
                    else bool(self._prev_gripper_state)
                ),
                inherited_close_event_step=(
                    self._last_close_event_step
                    if self._prev_gripper_state == 1
                    else None
                ),
            )
        )

    def begin(
        self,
        *,
        step: int = 0,
        initially_holding: bool = False,
        initial_gripper_state: int | None = None,
    ) -> None:
        if initial_gripper_state is None and initially_holding:
            initial_gripper_state = 1
        self._prev_gripper_state = (
            None if initial_gripper_state is None else int(bool(initial_gripper_state))
        )
        if self._memory_evidence and self._memory_evidence[0].start_step is None:
            self._memory_evidence[0].start_step = int(step)
        self._runtime.begin_vla(frame=step, initially_holding=initially_holding)
        self._begin_span(start_step=step)

    @staticmethod
    def _valid_candidates(candidates: Any) -> list[dict]:
        """Keep only finite, well-formed candidates, nearest first."""
        if candidates is None:
            return []
        if isinstance(candidates, Mapping):
            candidates = candidates.get("candidates", [candidates])
        try:
            raw = list(candidates)
        except TypeError:
            raw = []
        valid = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("instance") or "")
            try:
                distance = float(item.get("distance"))
            except (TypeError, ValueError):
                continue
            if not name or not np.isfinite(distance):
                continue
            candidate = dict(item)
            candidate["name"] = name
            candidate["distance"] = distance
            valid.append(_json_value(candidate))
        return sorted(valid, key=lambda item: item["distance"])

    def _keep_memory_objects(self, candidates: list[dict]) -> list[dict]:
        """Scene distractors absent from the task's memory are not interactions."""
        if not self._memory_objects:
            return candidates
        kept = []
        for item in candidates:
            name = _normalise_name(item.get("name"))
            if name and any(
                name == known or name in known or known in name
                for known in self._memory_objects
            ):
                kept.append(item)
        return kept

    def _query_candidates(
        self, obs: Mapping[str, Any], eef_pos: Any, phase: Any,
        supplied: Any = None,
    ) -> Any:
        if supplied is not None:
            return supplied
        if self._object_query is None:
            return None
        return self._object_query(obs, eef_pos, phase)

    @staticmethod
    def _runtime_identity(span: PhaseSpan) -> dict:
        return {
            "phase_index": span.phase_index,
            "planner_step_id": span.planner_step_id,
            "skill": span.skill,
            "arguments": dict(span.arguments),
        }

    def _copy_evidence_to_span(self, evidence, runtime_span) -> None:
        runtime_span.memory_phase_index_at_event = evidence.phase_index
        runtime_span.object_status = evidence.object_status
        runtime_span.object_match = evidence.object_match
        runtime_span.actual_object = evidence.actual_object
        runtime_span.object_candidates = list(evidence.object_candidates)
        if evidence.close_event_confirmed_step is not None:
            runtime_span.close_event_confirmed_step = evidence.close_event_confirmed_step

    def _stop_with_candidate(self, status: str, reason: str,
                             evidence: MemoryPhaseEvidence,
                             event: int | None) -> None:
        """Freeze the first failing phase as the single repair candidate."""
        self._diagnosis_stopped = True
        start = evidence.start_step if evidence.start_step is not None else 0
        self._candidate = {
            "span": {
                "phase_index": evidence.phase_index,
                "memory_phase_index": evidence.phase_index,
                "planner_step_id": evidence.planner_step_id,
                "skill": evidence.skill,
                "arguments": dict(evidence.arguments),
                "start_step": int(start),
            },
            "status": status,
            "reason": reason,
            "event_step": event,
            "close_event_step": event,
        }

    def _record_memory_objects(
        self, evidence: MemoryPhaseEvidence, *, candidates: list[dict],
        step: int, runtime_span: PhaseSpan,
        event_gap: float | None, direction: str,
    ) -> None:
        """Record the first interaction for the current memory phase.

        The memory cursor chooses the expected phase.  Runtime phase identity
        is retained as provenance only; it never moves the cursor or selects a
        later memory phase.
        """
        if evidence.first_interaction_event_step is not None:
            return
        if evidence.start_step is None:
            evidence.start_step = self._runtime_phase_starts.get(
                runtime_span.phase_index, int(step)
            )
        evidence.first_interaction_event_step = int(step)
        evidence.interaction_direction = direction
        if direction == "0->1":
            evidence.first_close_event_step = int(step)
            evidence.close_event_gap = event_gap
        elif direction == "1->0":
            evidence.first_open_event_step = int(step)
        evidence.runtime_phase_index_at_event = runtime_span.phase_index
        evidence.runtime_phase = self._runtime_identity(runtime_span)
        evidence.object_candidates = list(candidates)
        if not candidates:
            evidence.object_status = OBJECT_UNKNOWN
            self._copy_evidence_to_span(evidence, runtime_span)
            return
        actual = candidates[0]
        evidence.actual_object = actual
        evidence.object_match = bool(
            object_name_matches(
                evidence.skill, evidence.arguments, actual["name"],
                actual.get("aliases", ()),
            )
        )
        evidence.object_status = (
            "match" if evidence.object_match else OBJECT_MISMATCH
        )
        evidence.close_event_confirmed_step = int(step)
        self._copy_evidence_to_span(evidence, runtime_span)
        if evidence.object_status == OBJECT_MISMATCH:
            self._stop_with_candidate(
                OBJECT_MISMATCH, "object_mismatch", evidence, int(step)
            )

    def _record_memory_completion(self, *, step: int, decision) -> None:
        """Advance memory order only after a semantic rule succeeds.

        Timeout advances the runtime queue, but it is the first failed
        attempt for diagnosis and therefore leaves the memory cursor in place.
        """
        if not self._memory_evidence:
            return
        index = min(self._memory_cursor, len(self._memory_evidence) - 1)
        evidence = self._memory_evidence[index]
        if evidence.completion_reason is not None:
            return
        evidence.semantic_completed = bool(decision.semantic_completed)
        evidence.completion_reason = str(decision.reason)
        evidence.completion_step = int(step)
        evidence.end_step = int(step)
        if decision.semantic_completed:
            self._memory_cursor = min(index + 1, len(self._memory_evidence))
        else:
            status = (
                GRASP_FAILED if evidence.skill == "Pick"
                else NEVER_RELEASED if evidence.skill in _OPEN_INTERACTION_SKILLS
                else TIMEOUT_GAP
            )
            self._stop_with_candidate(
                status, "execution_failure", evidence,
                evidence.first_interaction_event_step,
            )

    def _flush_close_interaction(self, runtime_span: PhaseSpan) -> None:
        pending = self._pending_close_interaction
        self._pending_close_interaction = None
        if pending is None or self._diagnosis_stopped:
            return
        index = pending["memory_phase_index"]
        if index >= len(self._memory_evidence):
            return
        evidence = self._memory_evidence[index]
        raw = self._query_candidates(
            pending["obs"], pending["eef_pos"], self._phases[index],
            pending["object_candidates"],
        )
        candidates = self._keep_memory_objects(self._valid_candidates(raw))
        self._record_memory_objects(
            evidence, candidates=candidates, step=pending["event_step"],
            runtime_span=runtime_span, event_gap=pending["best_gap"],
            direction="0->1",
        )
        evidence.close_event_confirmed_step = pending["best_step"]

    def observe(
        self, *, step: int, action, obs, target_points=None, object_candidates=None
    ):
        """Feed one executed policy action step (frame == real action step)."""
        if not self._runtime.active:
            return None
        cmd = float(np.asarray(action, dtype=np.float64).reshape(-1)[-1])
        span = self.spans[-1]
        state = _command_state(cmd, self._prev_gripper_state)
        if state is not None and self._prev_gripper_state is not None:
            if state != self._prev_gripper_state and span.first_flip_step is None:
                span.first_flip_step = int(step)
                span.flip_direction = f"{self._prev_gripper_state}->{state}"
            if state != self._prev_gripper_state and state == 1:
                if span.first_close_event_step is None:
                    span.first_close_event_step = int(step)
                    span.close_event_gap = _gripper_gap(obs)
                self._last_close_event_step = int(step)
                if not self._diagnosis_stopped and self._memory_cursor < len(self._memory_evidence):
                    evidence = self._memory_evidence[self._memory_cursor]
                    if interaction_direction(evidence.skill) == "0->1":
                        self._pending_close_interaction = {
                            "event_step": int(step),
                            "best_step": int(step),
                            "best_gap": _gripper_gap(obs),
                            "memory_phase_index": self._memory_cursor,
                            "obs": dict(obs),
                            "eef_pos": obs.get("robot0_eef_pos"),
                            "object_candidates": object_candidates,
                        }
            elif state == 1 and self._pending_close_interaction is not None:
                gap = _gripper_gap(obs)
                best = self._pending_close_interaction["best_gap"]
                if gap is not None and (best is None or gap < best):
                    self._pending_close_interaction.update(
                        best_gap=gap, best_step=int(step), obs=dict(obs),
                        eef_pos=obs.get("robot0_eef_pos"),
                        object_candidates=object_candidates,
                    )
            elif state != self._prev_gripper_state and state == 0:
                span.first_open_event_step = int(step)
                self._flush_close_interaction(span)
            if state != self._prev_gripper_state and not self._diagnosis_stopped:
                direction = f"{self._prev_gripper_state}->{state}"
                if self._memory_cursor < len(self._memory_evidence):
                    evidence = self._memory_evidence[self._memory_cursor]
                    expected_direction = interaction_direction(evidence.skill)
                    if direction == expected_direction and direction == "1->0":
                        raw_candidates = self._query_candidates(
                            obs,
                            obs.get("robot0_eef_pos"),
                            self._phases[self._memory_cursor],
                            object_candidates,
                        )
                        candidates = self._keep_memory_objects(
                            self._valid_candidates(raw_candidates)
                        )
                        self._record_memory_objects(
                            evidence, candidates=candidates, step=step,
                            runtime_span=span, event_gap=_gripper_gap(obs),
                            direction=direction,
                        )
        self._prev_gripper_state = state
        span.holding_after = None if state is None else bool(state)

        decision = self._runtime.observe_vla_frame(
            target_points=target_points,
            eef_pos=obs.get("robot0_eef_pos"),
            eef_quat=obs.get("robot0_eef_quat"),
            gripper_closed=bool(state) if state is not None else None,
            gripper_qpos=obs.get("robot0_gripper_qpos"),
            frame=step,
        )
        span.decisions.append((int(step), decision))
        if decision.advance:
            self._flush_close_interaction(span)
            self._record_memory_completion(step=step, decision=decision)
            self._advance(step)
        return decision

    def finish_chunk(self, *, step: int):
        """Close one naturally exhausted action chunk."""
        if not self._runtime.active:
            return None
        decision = self._runtime.finish_action_chunk(frame=step)
        self.spans[-1].decisions.append((int(step), decision))
        if decision.advance:
            self._flush_close_interaction(self.spans[-1])
            self._record_memory_completion(step=step, decision=decision)
            self._advance(step)
        return decision

    def _advance(self, step: int) -> None:
        span = self.spans[-1]
        span.end_step = int(step)
        span.holding_after = (
            None if self._prev_gripper_state is None else bool(self._prev_gripper_state)
        )
        if not self._runtime.exhausted:
            self._runtime.begin_vla(
                frame=step + 1, initially_holding=self._prev_gripper_state == 1
            )
            self._begin_span(start_step=step + 1)

    @property
    def pending(self) -> PhaseSpan | None:
        if self._runtime.exhausted or not self.spans:
            return None
        return self.spans[-1] if self.spans[-1].end_step is None else None

    def finalize(self, *, total_steps: int, success: bool) -> dict:
        if self.spans:
            self._flush_close_interaction(self.spans[-1])
        pending = self.pending
        return {
            "task_success": bool(success),
            "total_action_steps": int(total_steps),
            "gripper_cmd_convention": {
                "close": "action[-1] >= 0.5",
                "open": "action[-1] <= -0.5",
                "neutral": "otherwise; preserves previous state",
            },
            "recording_contract": {
                "action_index": "action[k] is the k-th executed pending action",
                "proprio_index": "proprio[k] is the state before action[k]",
                "replay": "reset, settle, then actions[:k]",
                "cut_in_window_steps": CUT_IN_STEPS,
            },
            "phases": [span.as_dict() for span in self.spans],
            "memory_phases": [evidence.as_dict() for evidence in self._memory_evidence],
            "diagnosis_stopped": bool(self._diagnosis_stopped),
            "candidate": self._candidate,
            "pending": None
            if pending is None
            else {
                "phase_index": pending.phase_index,
                "planner_step_id": pending.planner_step_id,
                "skill": pending.skill,
                "start_step": pending.start_step,
                "action_chunks_so_far": 0 if self._runtime.completion is None else int(self._runtime.completion.decision.action_chunks),
            },
        }


def classify_phase(span: dict) -> str:
    """Classify only what existing completion semantics establish."""
    if span.get("object_status") == OBJECT_MISMATCH:
        return OBJECT_MISMATCH
    if span.get("skill") == "Close":
        return CLOSE_RECORDED
    if span.get("end_step") is None:
        return PENDING
    decisions = span.get("decisions", [])
    if not decisions:
        return PENDING
    final = decisions[-1]
    if not final["advance"]:
        return PENDING
    if final["reason"] == "rule":
        return CONFIRMED if span["skill"] == "Pick" else RELEASED_ONLY
    if span["skill"] == "Pick":
        return GRASP_FAILED
    if span["skill"] in ("PlaceIn", "PlaceOn"):
        return NEVER_RELEASED
    return TIMEOUT_GAP


def _candidate(
    span: dict, status: str, *, repairable: bool, reason: str,
    runtime_phase: dict | None = None,
) -> dict:
    event = _interaction_event_step(span)
    diagnosis_index = span.get("memory_phase_index", span.get("phase_index"))
    identity = {
        "phase_index": diagnosis_index,
        "planner_step_id": span.get("planner_step_id"),
        "skill": span.get("skill"),
        "arguments": dict(span.get("arguments", {})),
    }
    return {
        "span": span,
        "status": status,
        "repairable": bool(repairable),
        "reason": reason,
        "diagnosis_phase": identity,
        "repair_phase": identity if repairable else None,
        "runtime_phase": runtime_phase or span.get("runtime_phase"),
        "close_event_step": event,
        "event_step": event,
    }


def _interaction_event_step(value: Mapping[str, Any]) -> int | None:
    """Get the first event for a memory phase, with legacy-record fallback."""
    direct = value.get("first_interaction_event_step")
    if direct is not None:
        return int(direct)
    skill = str(value.get("skill", ""))
    if interaction_direction(skill) == "1->0":
        opened = value.get("first_open_event_step")
        return None if opened is None else int(opened)
    closed = value.get("first_close_event_step")
    return None if closed is None else int(closed)


def _memory_diagnostic_span(record: dict, evidence: dict, span: dict | None = None) -> dict:
    """Merge memory-order evidence with the runtime span that saw its event."""
    runtime_index = evidence.get("runtime_phase_index_at_event")
    runtime_span = span or next(
        (
            item for item in record.get("phases", ())
            if item.get("phase_index") == runtime_index
        ),
        None,
    )
    if runtime_span is None:
        runtime_span = next(
            (
                item for item in record.get("phases", ())
                if item.get("phase_index") == evidence.get("phase_index")
            ),
            {},
        )
    merged = dict(runtime_span)
    for key in (
        "phase_index", "planner_step_id", "skill", "arguments", "start_step",
        "end_step", "first_close_event_step", "first_open_event_step",
        "first_interaction_event_step", "interaction_direction",
        "close_event_gap", "close_event_confirmed_step", "object_status",
        "object_match", "actual_object", "object_candidates", "object_source",
        "semantic_completed", "completion_reason", "completion_step",
    ):
        if key in evidence and evidence[key] is not None:
            merged[key] = evidence[key]
    merged["phase_index"] = evidence.get("phase_index", merged.get("phase_index"))
    merged["memory_phase_index"] = evidence.get("phase_index")
    merged["runtime_phase_index"] = runtime_index
    merged["runtime_phase"] = evidence.get("runtime_phase")
    return merged


def _memory_span(record: dict, evidence: dict) -> dict:
    runtime_index = evidence.get("runtime_phase_index_at_event")
    if runtime_index is not None:
        span = next(
            (item for item in record.get("phases", ())
             if item.get("phase_index") == runtime_index),
            None,
        )
        if span is not None:
            return span
    return next(
        (item for item in record.get("phases", ())
         if item.get("phase_index") == evidence.get("phase_index")),
        {},
    )


def _phase_attempt_status(skill: str, span: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    if evidence.get("semantic_completed") is True:
        return CONFIRMED if skill == "Pick" else (
            RELEASED_ONLY if skill in _OPEN_INTERACTION_SKILLS else CLOSE_RECORDED
        )
    if evidence.get("semantic_completed") is False:
        if skill == "Pick":
            return GRASP_FAILED
        if skill in _OPEN_INTERACTION_SKILLS:
            return NEVER_RELEASED
        return TIMEOUT_GAP
    return classify_phase(dict(span)) if span else PENDING


def compute_t_star(candidate: dict) -> tuple[int | None, int | None]:
    """Compute a repair prefix from the local close event."""
    if not candidate or not candidate.get("repairable", True):
        return None, candidate.get("close_event_step") if candidate else None
    span = candidate["span"]
    event = (
        candidate.get("event_step")
        or candidate.get("close_event_step")
        or _interaction_event_step(span)
    )
    if candidate.get("reason") == "object_mismatch":
        t_star = int(span["start_step"])
    else:
        if event is None:
            return None, None
        t_star = max(int(span["start_step"]), int(event) - CUT_IN_STEPS)
    # The simulator needs an initial ten-step adjustment window before the
    # first memory skill can be taken over.  Later skills keep their original
    # phase-start / close-window calculation.
    if (
        t_star == 0
        and int(span.get("memory_phase_index", span.get("phase_index", -1))) == 0
    ):
        t_star = FIRST_PHASE_MIN_CUT_IN_STEP
    return t_star, event


def save_record(record: dict, path: Path) -> None:
    path.write_text(json.dumps(record, indent=2, default=str))
