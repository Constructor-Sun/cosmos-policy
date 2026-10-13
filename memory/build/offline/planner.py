"""Build a state-free high-level skill skeleton for a LIBERO task.

Unlike ``libero_predicate_planner.py``, this script never reads the BDDL
initial state and does not use a simulator or a classical planner.  It expands
only task goals plus static fixture/region metadata into a nominal sequence for
a later memory-based executor to verify dynamically.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
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


SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUPPORTED_GOALS = {"in", "on", "open", "close", "turnon", "turnoff"}
OPENABLE_HINTS = ("cabinet", "drawer", "microwave")


@dataclass(frozen=True)
class SkillStep:
    """One state-free, object-grounded stage for the memory executor."""

    step_id: int
    skill: str
    arguments: dict[str, str]
    depends_on: tuple[int, ...]
    execution_mode: str
    state_assumption: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["depends_on"] = list(self.depends_on)
        return value


def normalized_goals(states: Iterable[list[str]]) -> list[tuple[str, ...]]:
    return [tuple(str(token).lower() for token in state) for state in states]


def resolve_bddl(suite: str, task: str) -> Path:
    task_file = task if task.endswith(".bddl") else f"{task}.bddl"
    sibling = LIBERO_PLUS_ROOT / "libero" / "libero" / "bddl_files" / suite / task_file
    if sibling.is_file():
        return sibling
    installed = Path(get_libero_path("bddl_files")) / suite / task_file
    if installed.is_file():
        return installed
    raise FileNotFoundError(f"Cannot find BDDL for suite={suite!r}, task={task!r}")


def infer_openables(parsed: dict[str, Any]) -> set[str]:
    """Infer affordances from static names/types, never from initial state."""
    fixture_types = {
        str(name).lower(): str(kind).lower()
        for kind, names in parsed["fixtures"].items()
        for name in names
    }
    fixtures = {
        name
        for name, kind in fixture_types.items()
        if any(hint in f"{name} {kind}" for hint in OPENABLE_HINTS)
    }
    openables = set(fixtures)
    for region_name, region in parsed["regions"].items():
        target = str(region.get("target") or "").lower()
        kind = fixture_types.get(target, "")
        if target in fixtures or any(
            hint in f"{region_name} {target} {kind}" for hint in OPENABLE_HINTS
        ):
            openables.add(str(region_name).lower())
    return openables


def build_skeleton(
    goals: list[tuple[str, ...]],
    instruction: str,
    openables: set[str],
) -> list[SkillStep]:
    """Expand desired relations into stages without consulting world state."""
    unsupported = sorted({goal[0] for goal in goals} - SUPPORTED_GOALS)
    if unsupported:
        raise ValueError(f"Unsupported LIBERO goal predicates: {unsupported}")

    steps: list[SkillStep] = []
    ensured_open: set[str] = set()
    deferred_close: list[str] = []
    push_task = "push" in instruction.lower()

    def add(skill: str, execution_mode: str = "execute", **arguments: str) -> None:
        step_id = len(steps) + 1
        depends_on = (step_id - 1,) if step_id > 1 else ()
        steps.append(
            SkillStep(
                step_id=step_id,
                skill=skill,
                arguments=arguments,
                depends_on=depends_on,
                execution_mode=execution_mode,
            )
        )

    def ensure_open(target: str) -> None:
        if target in openables and target not in ensured_open:
            add("Open", "ensure", target=target)
            ensured_open.add(target)

    for goal in goals:
        predicate, *args = goal
        if predicate == "close":
            deferred_close.append(args[0])
        elif predicate == "open":
            ensure_open(args[0])
            if args[0] not in ensured_open:
                add("Open", "ensure", target=args[0])
                ensured_open.add(args[0])
        elif predicate == "in":
            item, target = args
            ensure_open(target)
            add("Pick", item=item)
            add("PlaceIn", item=item, target=target)
        elif predicate == "on":
            item, target = args
            if push_task:
                add("PushTo", item=item, target=target)
            else:
                add("Pick", item=item)
                add("PlaceOn", item=item, target=target)
        elif predicate == "turnon":
            add("TurnOn", "ensure", target=args[0])
        elif predicate == "turnoff":
            add("TurnOff", "ensure", target=args[0])

    for target in dict.fromkeys(deferred_close):
        add("Close", "ensure", target=target)
    return steps


def format_step(step: SkillStep) -> str:
    args = ", ".join(step.arguments.values())
    ensure_names = {
        "Open": "EnsureOpen",
        "Close": "EnsureClosed",
        "TurnOn": "EnsureOn",
        "TurnOff": "EnsureOff",
    }
    skill = ensure_names.get(step.skill, step.skill)
    if step.execution_mode != "ensure":
        skill = step.skill
    return f"{skill}({args})"


