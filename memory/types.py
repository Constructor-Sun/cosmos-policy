"""Shared types for the memory package."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


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


# Copied from memory_system/execute/plan.py during migration; the old tree
# keeps its own copy until it retires.
@dataclass(frozen=True)
class PhaseSpec:
    planner_step_id: int
    skill: str
    arguments: dict[str, str]


# Copied from memory_system/tta/phase_record.py during migration; the old tree
# keeps its own copy until it retires.
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


@dataclass(frozen=True)
class PhaseDecision:
    """One check result: still running, rule match, or budget exhausted."""

    advance: bool
    semantic_completed: bool
    reason: str
    action_chunks: int


@dataclass(frozen=True)
class ReplaySegment:
    """The recorded segment one memory candidate replays."""

    ready_frame: int
    ee_pose_object_sequence: np.ndarray
    gripper_sequence: np.ndarray
    action_sequence_object_physical: np.ndarray


@dataclass(frozen=True)
class MemoryCandidate:
    """One selectable memory record and what the intervention needs from it."""

    record: dict
    match: str  # exact / same_type
    rank: float | None  # reserved; not populated yet
    target_ee: np.ndarray  # ready pose in the object frame
    replay: ReplaySegment


@dataclass(frozen=True)
class ExecutionResult:
    """One intervention event: which stage reported it and how it ended."""

    outcome: str  # arrived / finished / timeout / failed
    stage: str  # approach / replay
    steps_used: int
    final_ee: np.ndarray | None


# Copied from memory_system/types.py during migration; the old tree keeps its
# own copy until it retires.
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
class ActionSource:
    """Who emits this step's action: ``vla``, ``intervention`` or ``stop``."""

    kind: str


@dataclass(frozen=True)
class CaseSpec:
    """One case: the task, one of its initial states, and the reset seed."""

    suite: str
    task: str
    init_index: int
    seed: int = 0


@dataclass(frozen=True)
class RunStep:
    """One executed step: who emitted it, in what space, and its chunk boundary."""

    step: int
    action: np.ndarray
    source: str  # ActionSource.kind
    chunk_start: bool
    control_space: str  # cartesian / joint
    collected: bool  # enters the policy data stream (alignment frames do not)


@dataclass(frozen=True)
class RunEvent:
    """One recorded event; ``kind`` is one of five.

    phase_boundary, handover, intervention, verdict, terminal.  ``detail``
    carries whatever that kind has to say -- a decision's reason, the holding
    belief after a verdict, an intervention's outcome.
    """

    kind: str
    step: int
    phase: PhaseSpec | None = None
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    """One run's facts, ready for diagnosis, replay and training export."""

    case: CaseSpec
    phases: tuple[PhaseSpec, ...]
    steps: tuple[RunStep, ...]
    events: tuple[RunEvent, ...]
    spans: tuple[PhaseSpan, ...]
    holding: tuple[tuple[int, bool], ...]  # belief changes: (step, holding)
    holding_facts: tuple[tuple[int, bool], ...]  # measured: (step, gripper closed)
    frame_index: tuple[tuple[int, int, int], ...]  # (memory frame, run step, data row)
    success: bool
