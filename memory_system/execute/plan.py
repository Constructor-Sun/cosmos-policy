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


# Goal predicates -> skills.  ``In``/``On`` only state the final relation, so
# they expand into Pick + Place; the predicate arguments are the same scene
# object/site names the memory arguments use, which makes the parsed plan
# directly retrievable memory keys for any task (libero-plus or LIBERO-PRO).
_RELATION_SKILLS = {"in": "PlaceIn", "on": "PlaceOn"}
_STATE_SKILLS = {"turnon": "TurnOn", "turnoff": "TurnOff", "open": "Open", "close": "Close"}


def _gates_placement(site: str, targets: set[str]) -> bool:
    """True when a door-like state predicate acts on a placement destination.

    Goals name the same container in two ways: ``(Close microwave_1)`` pairs
    with ``(In white_yellow_mug_1 microwave_1_heating_region)``, while
    KITCHEN_SCENE4 writes the region name on both sides.  A placement target is
    a region/site *of* the body the predicate names, so match the body prefix as
    well as the name itself (the ``_`` keeps ``plate_1`` from matching
    ``plate_10_region``).
    """
    return any(target == site or target.startswith(f"{site}_") for target in targets)


def plan_from_goal_state(goal_state: Any) -> tuple[PhaseSpec, ...]:
    """Order a parsed BDDL ``goal_state`` conjunction into an executable plan.

    The goal is a conjunction of final-state assertions -- the env folds the
    same list with ``and`` in ``_check_success`` -- so it fixes *what* has to end
    up true, and on which scene objects, but not the order in which to get
    there.  Three rules order it, in decreasing strength:

    1. ``In``/``On`` expand into Pick + Place; placements keep BDDL order.
    2. ``TurnOn``/``TurnOff``/``Open`` are preconditions and run first.  For the
       stove that is a convention rather than a derivation (the burner is lit
       before the pot goes on it, as the LIBERO-10 demos of KITCHEN_SCENE3 do);
       for ``Open`` it follows from the container having to admit the object.
    3. A ``Close`` acting on a placement destination runs after that placement,
       because a container cannot receive an object while closed.

    Everything else keeps BDDL order.  Two things this cannot know: a predicate
    whose precondition the initial state already satisfies (KITCHEN_SCENE8
    starts with the stove lit, so its demos carry no TurnOn segment) has to be
    confirmed and skipped by the runtime, and independent placements the demos
    perform in another order are still emitted in goal order.
    """
    relations: list[tuple[str, str, str]] = []
    states: list[tuple[str, str]] = []
    for pred, *args in goal_state:
        pred = str(pred).lower()
        if pred in _RELATION_SKILLS:
            target = " ".join(str(arg) for arg in args[1:])
            relations.append((str(args[0]), target, _RELATION_SKILLS[pred]))
        elif pred in _STATE_SKILLS:
            states.append((str(args[0]), _STATE_SKILLS[pred]))
        else:
            raise ValueError(f"unsupported goal predicate {pred!r}")

    placed = {target for _item, target, _skill in relations}
    phases: list[PhaseSpec] = []
    for target, skill in states:
        if skill in {"TurnOn", "TurnOff", "Open"} or not _gates_placement(target, placed):
            phases.append(PhaseSpec(len(phases), skill, {"target": target}))
    for item, target, skill in relations:
        phases.append(PhaseSpec(len(phases), "Pick", {"item": item}))
        phases.append(PhaseSpec(len(phases), skill, {"item": item, "target": target}))
    for target, skill in states:
        if skill == "Close" and _gates_placement(target, placed):
            phases.append(PhaseSpec(len(phases), skill, {"target": target}))
    if not phases:
        raise ValueError("no goal predicates")
    return tuple(phases)


def plan_from_bddl(bddl_path: str | Path) -> tuple[PhaseSpec, ...]:
    """Read the ordered skill plan from the task's BDDL goal predicates.

    The order comes from ``plan_from_goal_state``: BDDL order is a conjunction,
    not a schedule (KITCHEN_SCENE4 writes its ``Close`` before the ``In`` it
    gates).
    """
    # Lazy import: whichever libero flavor the process already resolved
    # (LIBERO-plus or LIBERO-PRO) provides the same parser.
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    goal_state = robosuite_parse_problem(str(bddl_path))["goal_state"]
    try:
        return plan_from_goal_state(goal_state)
    except ValueError as exc:  # keep the offending file in the diagnostic
        raise ValueError(f"{exc} in {bddl_path}") from exc


def drop_presatisfied_turnon(
    phases: tuple[PhaseSpec, ...], env: Any
) -> tuple[PhaseSpec, ...]:
    """Runtime skip for TurnOn/TurnOff preconditions already satisfied.

    This is the check ``plan_from_goal_state``'s docstring delegates to the
    runtime: read the target's button joint at the current state and drop the
    phase when the predicate already holds (KITCHEN_SCENE8 starts with the
    stove lit).  Open/Close are exempt -- the instrumental open during a later
    Place invalidates their initial state.
    """
    kept: list[PhaseSpec] = []
    for phase in phases:
        if phase.skill in {"TurnOn", "TurnOff"}:
            try:
                sim = env.env.sim
                joint = sim.model.joint_name2id(
                    f"{phase.arguments['target']}_button"
                )
                qpos = float(sim.data.qpos[sim.model.jnt_qposadr[joint]])
            except (KeyError, ValueError):
                kept.append(phase)  # no button joint: keep conservatively
                continue
            lit = qpos >= 0.5  # Stove default_turnon_ranges lower bound
            if lit == (phase.skill == "TurnOn"):
                continue
        kept.append(phase)
    return tuple(kept)


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
