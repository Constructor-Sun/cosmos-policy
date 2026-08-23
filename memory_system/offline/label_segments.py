"""Label LIBERO demonstration skill boundaries with simulator predicates.

States are restored independently; one final action reconstructs the omitted terminal
state. The output aligns state-free planner steps with demonstration frame ranges.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
for path in (ROOT, LIBERO_PLUS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MUJOCO_GL", "egl")

from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env,
)
from memory_system.offline.planner import (  # noqa: E402
    build_skeleton,
    infer_openables,
    normalized_goals,
    resolve_bddl,
    robosuite_parse_problem,
)


PREDICATE_SKILLS = {
    "Open": "open",
    "Close": "close",
    "TurnOn": "turnon",
    "TurnOff": "turnoff",
    "PlaceIn": "in",
    "PlaceOn": "on",
    "PushTo": "on",
}


def restore(env: Any, state: np.ndarray) -> None:
    env.set_state(state)
    env.sim.forward()


def object_position(base_env: Any, name: str) -> np.ndarray:
    pos = base_env.object_states_dict[name].get_geom_state()["pos"]
    return np.asarray(pos, dtype=np.float64).copy()


def is_grasping(base_env: Any, item: str) -> bool:
    """Use robosuite's contact-based grasp check; do not add a new geometry model."""
    obj = base_env.get_object(item)
    checker = getattr(base_env, "_check_grasp", None)
    if checker is None:
        raise RuntimeError("This robosuite environment does not expose _check_grasp")
    gripper = base_env.robots[0].gripper
    try:
        return bool(checker(gripper, obj))
    except (AttributeError, TypeError):
        return bool(checker(gripper, obj.contact_geoms))


def step_succeeded(
    env: Any,
    step: Any,
    state: np.ndarray,
    pick_start_pos: np.ndarray | None,
    lift_threshold: float,
) -> bool:
    restore(env, state)
    base = env.env
    skill, args = step.skill, step.arguments
    if skill == "Pick":
        item = args["item"]
        displacement = object_position(base, item) - pick_start_pos
        contact_lift = is_grasping(base, item) and np.linalg.norm(displacement) >= lift_threshold
        return bool(contact_lift or displacement[2] >= lift_threshold)

    predicate = PREDICATE_SKILLS.get(skill)
    if predicate is None:
        raise ValueError(f"No success predicate mapping for skill {skill!r}")
    if skill in {"PlaceIn", "PlaceOn", "PushTo"}:
        state_expr = [predicate, args["item"], args["target"]]
    else:
        state_expr = [predicate, args["target"]]
    succeeded = bool(base._eval_predicate(state_expr))
    if not succeeded and skill in {"Open", "Close"}:
        target = args["target"]
        site = base.object_sites_dict.get(target)
        parent = getattr(base.object_states_dict[target], "parent_name", None)
        if parent and not getattr(site, "joints", ()):
            succeeded = bool(base._eval_predicate([predicate, parent]))
    return succeeded


def find_stable_success(
    env: Any,
    step: Any,
    states: np.ndarray,
    cursor: int,
    stable_frames: int,
    lift_threshold: float,
) -> tuple[int, int] | None:
    pick_start_pos = None
    if step.skill == "Pick":
        restore(env, states[cursor])
        pick_start_pos = object_position(env.env, step.arguments["item"])
    run = 0
    for frame in range(cursor, len(states)):
        if step_succeeded(env, step, states[frame], pick_start_pos, lift_threshold):
            run += 1
            if run >= stable_frames:
                success_start = frame - stable_frames + 1
                if step.skill in {"PlaceIn", "PlaceOn"}:
                    released_run = 0
                    for released in range(success_start, len(states)):
                        relation_holds = step_succeeded(
                            env, step, states[released], pick_start_pos, lift_threshold
                        )
                        if relation_holds and not is_grasping(
                            env.env, step.arguments["item"]
                        ):
                            released_run += 1
                            if released_run >= stable_frames:
                                return released - stable_frames + 1, released + 1
                        else:
                            released_run = 0
                return success_start, frame + 1
        else:
            run = 0
    return None


def add_terminal_state(
    env: Any,
    states: np.ndarray,
    actions: np.ndarray,
    stable_frames: int,
) -> np.ndarray:
    """Recover the post-final-action state omitted by the regeneration loop."""
    restore(env, states[-1])
    env.step(actions[-1].tolist())
    terminal = env.sim.get_state().flatten()
    tail = np.repeat(terminal[None, :], stable_frames, axis=0)
    return np.concatenate([states, tail], axis=0)


def observed_step_order(
    env: Any,
    states: np.ndarray,
    steps: list[Any],
    stable_frames: int,
    lift_threshold: float,
) -> list[Any]:
    """Order independent goal units by their observed completion time."""
    chunks, prefix = [], []
    index = 0
    while index < len(steps):
        step = steps[index]
        if step.skill == "Open":
            prefix.append(step)
            index += 1
        elif (
            step.skill == "Pick"
            and index + 1 < len(steps)
            and steps[index + 1].skill in {"PlaceIn", "PlaceOn"}
        ):
            chunks.append(prefix + steps[index : index + 2])
            prefix, index = [], index + 2
        else:
            chunks.append(prefix + [step])
            prefix, index = [], index + 1
    if prefix:
        chunks.append(prefix)

    def completion_key(chunk: list[Any]) -> tuple[int, int]:
        if chunk[-1].skill == "Close":
            return 1, len(states)
        found = find_stable_success(
            env, chunk[-1], states, 0, stable_frames, lift_threshold
        )
        return 0, found[0] if found else len(states)

    return [step for chunk in sorted(chunks, key=completion_key) for step in chunk]


def label_demo(
    env: Any,
    states: np.ndarray,
    steps: list[Any],
    stable_frames: int,
    lift_threshold: float,
) -> tuple[list[dict[str, Any]], str | None]:
    segments: list[dict[str, Any]] = []
    cursor = 0
    ordered_steps = observed_step_order(
        env, states, steps, stable_frames, lift_threshold
    )
    for step in ordered_steps:
        if cursor >= len(states):
            return segments, f"ran out of states before planner step {step.step_id}"
        found = find_stable_success(
            env, step, states, cursor, stable_frames, lift_threshold
        )
        if found is None:
            return segments, f"no stable success for step {step.step_id} ({step.skill})"
        success_start, success_end = found
        status = "already_satisfied" if success_start == cursor else "executed"
        segments.append(
            {
                "planner_step_id": step.step_id,
                "skill": step.skill,
                "arguments": step.arguments,
                "execution_mode": step.execution_mode,
                "status": status,
                "start": cursor,
                "end": success_start,
                "success_start": success_start,
                "success_end": success_end,
            }
        )
        cursor = success_start
    return segments, None




def label_manifest(
    input_dir: str | Path,
    output: str | Path,
    suite: str = "libero_10",
    task: str | None = None,
    task_config: str | Path | None = None,
    max_per_task: int = 10,
    stable_frames: int = 3,
    lift_threshold: float = 0.02,
) -> int:
    """Label LIBERO demo segments and write the segment manifest.

    This is the library equivalent of the original ``bin/memory`` CLI.  It
    preserves the same LIBERO/MuJoCo/HDF5 dependencies and writes the same
    manifest format.
    """
    if stable_frames <= 0 or lift_threshold <= 0:
        raise ValueError("stable-frames and lift-threshold must be positive")

    input_dir = Path(input_dir).resolve()
    output = Path(output).resolve()
    if task_config is None:
        task_config = ROOT / "configs/libero10_experiment_tasks.json"
    task_config = Path(task_config).resolve()

    files = sorted(input_dir.glob("*_demo.hdf5"))
    if task:
        files = [p for p in files if p.stem.removesuffix("_demo") == task]
    if not files:
        raise FileNotFoundError(f"No *_demo.hdf5 in {input_dir}")
    task_data = json.loads(task_config.read_text())
    if task_data["suite"] != suite:
        raise ValueError("task-config suite does not match --suite")
    tasks = {item["name"]: item for item in task_data["tasks"]}

    records: list[dict[str, Any]] = []
    for h5_path in files:
        task_name = h5_path.stem.removesuffix("_demo")
        if task_name not in tasks:
            print(f"WARNING: task is absent from config: {task_name}")
            continue
        try:
            bddl_path = resolve_bddl(suite, task_name)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            continue
        parsed = robosuite_parse_problem(str(bddl_path))
        language = tasks[task_name]["language"]
        task_spec = SimpleNamespace(
            language=language,
            problem_folder=suite,
            bddl_file=bddl_path.name,
        )
        env, language = get_libero_env(task_spec, "llava", resolution=64)
        goals = normalized_goals(parsed["goal_state"])
        steps = build_skeleton(goals, language, infer_openables(parsed))
        try:
            with h5py.File(h5_path, "r") as h5:
                demos = sorted(
                    h5["data"].keys(), key=lambda name: int(name.split("_")[1])
                )
                if max_per_task > 0:
                    demos = demos[:max_per_task]
                for demo_id in demos:
                    demo = h5["data"][demo_id]
                    recorded_states = demo["states"][:]
                    env.reset()
                    states = add_terminal_state(
                        env, recorded_states, demo["actions"][:], stable_frames
                    )
                    segments, error = label_demo(
                        env, states, steps, stable_frames, lift_threshold
                    )
                    records.append(
                        {
                            "task_name": task_name,
                            "demo_id": demo_id,
                            "num_states": len(recorded_states),
                            "terminal_state_replayed": True,
                            "valid": error is None,
                            "error": error,
                            "segments": segments,
                        }
                    )
                    state = "OK" if error is None else f"FAILED: {error}"
                    print(f"{task_name}/{demo_id}: {state}", flush=True)
        finally:
            env.close()

    manifest = {
        "version": 1,
        "suite": suite,
        "stable_frames": stable_frames,
        "lift_threshold": lift_threshold,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    valid = sum(record["valid"] for record in records)
    print(f"Wrote {valid}/{len(records)} valid demos to {output}")
    return 0 if valid else 1
