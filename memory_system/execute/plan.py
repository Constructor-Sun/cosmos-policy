"""Load the ordered skill plan used by Initial Alignment.

The former current/next Phase monitor lived in this module as well.  The
runtime verifier pipeline has been removed; only the validated plan contract
needed by Initial Alignment remains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhaseSpec:
    planner_step_id: int
    skill: str
    arguments: dict[str, str]


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


def load_phase_sequences(
    path: str | Path,
) -> dict[tuple[str, str], tuple[PhaseSpec, ...]]:
    """Load each valid memory demo in its original segment order.

    ``planner_step_id`` is an identity/retrieval key, not an execution-order
    key.  The memory record's segment list is the only ordering source used by
    active skill continuation.  Records are keyed by the exact task and demo
    identifiers so a sequence is never assembled by voting across demos.
    """
    payload = json.loads(Path(path).read_text())
    sequences: dict[tuple[str, str], tuple[PhaseSpec, ...]] = {}
    for record in payload.get("records", []):
        if not record.get("valid"):
            continue
        task_name = str(record["task_name"])
        demo_id = str(record["demo_id"])
        phases: list[PhaseSpec] = []
        for segment in record.get("segments", []):
            if segment.get("status") == "already_satisfied":
                continue
            phases.append(
                PhaseSpec(
                    int(segment["planner_step_id"]),
                    str(segment["skill"]),
                    {
                        str(key): str(value)
                        for key, value in segment.get("arguments", {}).items()
                    },
                )
            )
        key = (task_name, demo_id)
        sequence = tuple(phases)
        existing = sequences.get(key)
        if existing is not None and existing != sequence:
            raise ValueError(f"Inconsistent memory sequence for {task_name}/{demo_id}")
        sequences[key] = sequence
    return sequences
