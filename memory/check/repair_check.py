"""Repair-point selection from one failed episode's diagnosis record.

The record is the diagnosis artifact: memory-ordered phase evidence plus the
episode's own runtime spans.  ``RepairChecker.check`` reads that evidence and
answers the three questions the repair path needs -- which phase failed, how it
failed, and where the repair should cut in (t*) -- as one ``RepairRequest``.
With nothing diagnosed it falls back to the plan's first phase from the start.

Phase evidence is produced elsewhere (the offline recorder that replays the
episode); this module only classifies it.  It does not replay the prefix,
retrieve memory, plan or execute actions, and it does not decide whether later
phases get an intervention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memory.types import PhaseSpec

OBJECT_MISMATCH = "object_mismatch"
GRASP_FAILED = "grasp_failed"
NEVER_RELEASED = "never_released"
TIMEOUT_GAP = "timeout_gap"

# Skills whose interaction is a release; every other recorded skill interacts
# by closing.
_OPEN_INTERACTION_SKILLS = {"PlaceIn", "PlaceOn"}


@dataclass(frozen=True)
class RepairCheckConfig:
    """Cut-in calibration; the defaults are the current values."""

    cut_in_steps: int = 16
    first_phase_min_cut_in: int = 10


@dataclass(frozen=True)
class RepairRequest:
    """One repair point: which phase, how it failed, and where to cut in."""

    task: str
    failure: str | None  # None when nothing was diagnosed to repair
    phase: PhaseSpec
    t_star: int | None
    plan: tuple[PhaseSpec, ...]


def _interaction_event_step(evidence: Mapping[str, Any]) -> int | None:
    """First interaction step of a phase, with the recorded-direction fallback."""
    direct = evidence.get("first_interaction_event_step")
    if direct is not None:
        return int(direct)
    if str(evidence.get("skill", "")) in _OPEN_INTERACTION_SKILLS:
        opened = evidence.get("first_open_event_step")
        return None if opened is None else int(opened)
    closed = evidence.get("first_close_event_step")
    return None if closed is None else int(closed)


def _classify(evidence: Mapping[str, Any]) -> str | None:
    """Failure kind of one memory phase, or None when it is not a failure."""
    if evidence.get("object_status") == OBJECT_MISMATCH:
        return OBJECT_MISMATCH
    if evidence.get("completion_reason") is None:
        return None  # never finished: the episode ended inside this phase
    if evidence.get("semantic_completed"):
        return None  # the phase's own rule fired
    skill = str(evidence.get("skill", ""))
    if skill == "Pick":
        return GRASP_FAILED
    if skill in _OPEN_INTERACTION_SKILLS:
        return NEVER_RELEASED
    return TIMEOUT_GAP


def _spec(entry: Mapping[str, Any]) -> PhaseSpec:
    return PhaseSpec(
        int(entry["planner_step_id"]),
        str(entry["skill"]),
        {str(key): str(value) for key, value in (entry.get("arguments") or {}).items()},
    )


class RepairChecker:
    """Select the repair point of one failed episode."""

    def __init__(self, config: RepairCheckConfig | None = None) -> None:
        self.config = RepairCheckConfig() if config is None else config

    def check(self, record: Mapping[str, Any]) -> RepairRequest | None:
        """Return the repair request for ``record``, or None without a plan."""
        plan = tuple(
            _spec(item)
            for item in record.get("memory_phases", ())
            if item.get("skill")
        )
        if not plan:
            return None
        # A successful episode is not diagnosed at all: nothing to repair.
        evidence = None if record.get("task_success") else next(
            (
                item
                for item in record.get("memory_phases", ())
                if _classify(item) is not None
            ),
            None,
        )
        failure = None if evidence is None else _classify(evidence)
        t_star = None if evidence is None else self._t_star(evidence, failure)
        if t_star is None:
            # Nothing diagnosed (or no interaction to cut in on): restart the
            # plan's first phase, the documented fallback.  The fallback phase
            # is not the diagnosed one, so no failure kind is reported with it.
            return RepairRequest(
                task=str(record.get("task_name_perturbed", "")),
                failure=None,
                phase=plan[0],
                t_star=self.config.first_phase_min_cut_in,
                plan=plan,
            )
        return RepairRequest(
            task=str(record.get("task_name_perturbed", "")),
            failure=failure,
            phase=_spec(evidence),
            t_star=t_star,
            plan=plan,
        )

    def _t_star(self, evidence: Mapping[str, Any], failure: str) -> int | None:
        """Cut-in step for the failed phase."""
        start = int(evidence.get("start_step") or 0)
        if failure == OBJECT_MISMATCH:
            # The object chosen at the phase start is wrong: repair from there.
            t_star = start
        else:
            event = _interaction_event_step(evidence)
            if event is None:
                return None
            t_star = max(start, event - self.config.cut_in_steps)
        if t_star == 0 and int(evidence.get("phase_index", -1)) == 0:
            # The simulator needs a ten-step settle window before the first
            # memory skill can take over.
            t_star = self.config.first_phase_min_cut_in
        return t_star
