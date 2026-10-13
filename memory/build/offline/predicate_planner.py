"""Plan high-level LIBERO skills from BDDL predicates with Fast Downward.

This symbolic-only prototype does not start MuJoCo, inspect demonstrations, or
produce low-level actions. Example: ``python bin/memory/libero_predicate_planner.py
--suite libero_goal --task open_the_top_drawer_and_put_the_bowl_inside``.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS_ROOT = REPO_ROOT.parent / "LIBERO-plus"
if LIBERO_PLUS_ROOT.is_dir():
    sys.path.insert(0, str(LIBERO_PLUS_ROOT))

try:
    from libero.libero import get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
except ModuleNotFoundError as exc:
    raise SystemExit(
        "LIBERO/BDDL is unavailable. Activate the project's LIBERO environment "
        "or install cosmos_policy/experiments/robot/libero/libero_requirements.txt. "
        f"Original error: {exc}"
    ) from exc

try:
    from unified_planning.shortcuts import (
        BoolType,
        Fluent,
        InstantaneousAction,
        Not,
        Object,
        OneshotPlanner,
        Or,
        Problem,
        UserType,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        'Planning dependencies are unavailable. Install "unified-planning[fast-downward]". '
        f"Original error: {exc}"
    ) from exc


SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OPENABLE_HINTS = ("cabinet", "drawer", "microwave")
SUPPORTED_PREDICATES = {"on", "in", "open", "close", "turnon", "turnoff"}

def flatten(values: Iterable[Iterable[str]]) -> set[str]:
    return {item for group in values for item in group}

def normalized_states(states: Iterable[list[str]]) -> list[tuple[str, ...]]:
    return [tuple(str(token).lower() for token in state) for state in states]

def resolve_bddl(suite: str, task: str) -> Path:
    task_file = task if task.endswith(".bddl") else f"{task}.bddl"
    sibling_path = LIBERO_PLUS_ROOT / "libero" / "libero" / "bddl_files" / suite / task_file
    if sibling_path.is_file():
        return sibling_path
    installed_path = Path(get_libero_path("bddl_files")) / suite / task_file
    if installed_path.is_file():
        return installed_path
    raise FileNotFoundError(f"Cannot find BDDL for suite={suite!r}, task={task!r}")

def infer_entities(parsed: dict[str, Any], states: list[tuple[str, ...]]) -> set[str]:
    names = flatten(parsed["objects"].values())
    names |= flatten(parsed["fixtures"].values())
    names |= set(parsed["regions"].keys())
    for state in states:
        names.update(state[1:])
    return names

def infer_openables(parsed: dict[str, Any], states: list[tuple[str, ...]]) -> set[str]:
    fixture_type = {
        name: kind.lower()
        for kind, names in parsed["fixtures"].items()
        for name in names
    }
    openables = {
        state[1]
        for state in states
        if state[0] in {"open", "close"} and len(state) == 2
    }
    for region_name, region in parsed["regions"].items():
        target = str(region.get("target") or "").lower()
        kind = fixture_type.get(target, "")
        if any(hint in f"{target} {kind}" for hint in OPENABLE_HINTS):
            openables.add(region_name.lower())
    return openables

def add_actions(problem: Problem, entity: Any, fluents: dict[str, Any]) -> None:
    movable = fluents["movable"]
    place = fluents["place"]
    needs_open = fluents["needs_open"]
    pushable = fluents["pushable"]
    device = fluents["device"]
    hand_empty = fluents["hand_empty"]
    holding = fluents["holding"]
    on = fluents["on"]
    inside = fluents["inside"]
    opened = fluents["opened"]
    turned_on = fluents["turned_on"]

    def action(name: str, **parameters: Any) -> tuple[Any, dict[str, Any]]:
        result = InstantaneousAction(name, **parameters)
        return result, {key: result.parameter(key) for key in parameters}
    pick_on, p = action("pick_from_on", item=entity, source=entity)
    pick_on.add_precondition(movable(p["item"]))
    pick_on.add_precondition(on(p["item"], p["source"]))
    pick_on.add_precondition(hand_empty)
    pick_on.add_effect(on(p["item"], p["source"]), False)
    pick_on.add_effect(holding(p["item"]), True)
    pick_on.add_effect(hand_empty, False)
    pick_in, p = action("pick_from_in", item=entity, source=entity)
    pick_in.add_precondition(movable(p["item"]))
    pick_in.add_precondition(inside(p["item"], p["source"]))
    pick_in.add_precondition(Or(Not(needs_open(p["source"])), opened(p["source"])))
    pick_in.add_precondition(hand_empty)
    pick_in.add_effect(inside(p["item"], p["source"]), False)
    pick_in.add_effect(holding(p["item"]), True)
    pick_in.add_effect(hand_empty, False)
    place_on, p = action("place_on", item=entity, target=entity)
    place_on.add_precondition(movable(p["item"]))
    place_on.add_precondition(place(p["target"]))
    place_on.add_precondition(holding(p["item"]))
    place_on.add_effect(on(p["item"], p["target"]), True)
    place_on.add_effect(holding(p["item"]), False)
    place_on.add_effect(hand_empty, True)
    place_in, p = action("place_in", item=entity, target=entity)
    place_in.add_precondition(movable(p["item"]))
    place_in.add_precondition(place(p["target"]))
    place_in.add_precondition(Or(Not(needs_open(p["target"])), opened(p["target"])))
    place_in.add_precondition(holding(p["item"]))
    place_in.add_effect(inside(p["item"], p["target"]), True)
    place_in.add_effect(holding(p["item"]), False)
    place_in.add_effect(hand_empty, True)
    open_action, p = action("open", target=entity)
    open_action.add_precondition(needs_open(p["target"]))
    open_action.add_precondition(Not(opened(p["target"])))
    open_action.add_precondition(hand_empty)
    open_action.add_effect(opened(p["target"]), True)
    close_action, p = action("close", target=entity)
    close_action.add_precondition(needs_open(p["target"]))
    close_action.add_precondition(opened(p["target"]))
    close_action.add_precondition(hand_empty)
    close_action.add_effect(opened(p["target"]), False)
    turn_on, p = action("turn_on", target=entity)
    turn_on.add_precondition(device(p["target"]))
    turn_on.add_precondition(Not(turned_on(p["target"])))
    turn_on.add_effect(turned_on(p["target"]), True)
    turn_off, p = action("turn_off", target=entity)
    turn_off.add_precondition(device(p["target"]))
    turn_off.add_precondition(turned_on(p["target"]))
    turn_off.add_effect(turned_on(p["target"]), False)
    push, p = action("push_to", item=entity, source=entity, target=entity)
    push.add_precondition(pushable(p["item"]))
    push.add_precondition(on(p["item"], p["source"]))
    push.add_precondition(place(p["target"]))
    push.add_precondition(hand_empty)
    push.add_effect(on(p["item"], p["source"]), False)
    push.add_effect(on(p["item"], p["target"]), True)

    problem.add_actions(
        [pick_on, pick_in, place_on, place_in, open_action,
         close_action, turn_on, turn_off, push]
    )

def build_problem(parsed: dict[str, Any], task_name: str) -> tuple[Problem, list[tuple[str, ...]]]:
    initial = normalized_states(parsed["initial_state"])
    goals = normalized_states(parsed["goal_state"])
    states = initial + goals
    unsupported = sorted({state[0] for state in goals} - SUPPORTED_PREDICATES)
    if unsupported:
        raise ValueError(f"Unsupported LIBERO goal predicates: {unsupported}")
    entity = UserType("entity")
    problem = Problem(f"libero_{task_name}")
    fluents = {
        "movable": Fluent("movable", BoolType(), item=entity),
        "place": Fluent("place", BoolType(), target=entity),
        "needs_open": Fluent("needs_open", BoolType(), target=entity),
        "pushable": Fluent("pushable", BoolType(), item=entity),
        "device": Fluent("device", BoolType(), target=entity),
        "hand_empty": Fluent("hand_empty", BoolType()),
        "holding": Fluent("holding", BoolType(), item=entity),
        "on": Fluent("on", BoolType(), item=entity, target=entity),
        "inside": Fluent("inside", BoolType(), item=entity, target=entity),
        "opened": Fluent("opened", BoolType(), target=entity),
        "turned_on": Fluent("turned_on", BoolType(), target=entity),
    }
    for fluent in fluents.values():
        problem.add_fluent(fluent, default_initial_value=False)
    names = infer_entities(parsed, states)
    objects = {name: Object(name, entity) for name in sorted(names)}
    problem.add_objects(objects.values())
    movable_names = flatten(parsed["objects"].values())
    openables = infer_openables(parsed, states)
    devices = {
        state[1] for state in states
        if state[0] in {"turnon", "turnoff"} and len(state) == 2
    }
    pushables = set()
    if "push" in " ".join(parsed["language_instruction"]).lower():
        pushables = {state[1] for state in goals if state[0] == "on"}
    for name, obj in objects.items():
        problem.set_initial_value(fluents["place"](obj), True)
        if name in movable_names:
            problem.set_initial_value(fluents["movable"](obj), True)
        if name in openables:
            problem.set_initial_value(fluents["needs_open"](obj), True)
        if name in devices:
            problem.set_initial_value(fluents["device"](obj), True)
        if name in pushables:
            problem.set_initial_value(fluents["pushable"](obj), True)
    problem.set_initial_value(fluents["hand_empty"], True)
    def expression(state: tuple[str, ...]) -> Any:
        predicate, *args = state
        refs = [objects[arg] for arg in args]
        if predicate == "on":
            return fluents["on"](*refs)
        if predicate == "in":
            return fluents["inside"](*refs)
        if predicate == "open":
            return fluents["opened"](*refs)
        if predicate == "close":
            return Not(fluents["opened"](*refs))
        if predicate == "turnon":
            return fluents["turned_on"](*refs)
        if predicate == "turnoff":
            return Not(fluents["turned_on"](*refs))
        raise ValueError(f"Unsupported predicate: {predicate}")

    for state in initial:
        if state[0] in SUPPORTED_PREDICATES:
            value = state[0] not in {"close", "turnoff"}
            target = expression(state)
            if state[0] in {"close", "turnoff"}:
                target = target.arg(0)
            problem.set_initial_value(target, value)
    for state in goals:
        problem.add_goal(expression(state))
    add_actions(problem, entity, fluents)
    return problem, goals

def format_skill(action_instance: Any) -> str:
    name = action_instance.action.name
    args = [str(arg) for arg in action_instance.actual_parameters]
    if name.startswith("pick_"):
        return f"Pick({args[0]})"
    if name == "place_on":
        return f"PlaceOn({args[0]}, {args[1]})"
    if name.startswith("place_in"):
        return f"PlaceIn({args[0]}, {args[1]})"
    if name == "push_to":
        return f"PushTo({args[0]}, {args[2]})"
    display = {"open": "Open", "close": "Close", "turn_on": "TurnOn", "turn_off": "TurnOff"}
    return f"{display.get(name, name)}({', '.join(args)})"

