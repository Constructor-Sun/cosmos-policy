#!/usr/bin/env python
"""Re-label existing phase-check trajectories with the current rule config.

Replays the saved action streams of the census phase-check datasets through a
fresh PhaseEventRecorder driven by the *current* completion rules and
SKILL_MAX_ACTION_CHUNKS, then writes new phase_record.json files next to
symlinks to the original episode.h5. No policy inference. Comparison against
oracle timings stays in evaluate_phase_timing.py; this script only produces
the re-labelled records.
"""
from __future__ import annotations
import argparse, os, shutil, sys
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "LIBERO-plus")]
os.environ.setdefault("MUJOCO_GL", "egl")
# get_libero_env only attaches instance segmentation (required by
# SceneObjectQuery and PickTargetPointCloud) when the shadow flag is set.
os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from check_oracle_failures import DEFAULT_ROOTS, load_cases  # noqa: E402
from memory_system.execute.skill_completion import DEFAULT_MAX_ACTION_CHUNKS  # noqa: E402
from memory_system.execute.vla_skill_runtime import SKILL_MAX_ACTION_CHUNKS  # noqa: E402
from memory_system.tta.object_query import SceneObjectQuery  # noqa: E402
from memory_system.tta.phase_record import (  # noqa: E402
    PhaseEventRecorder, compute_t_star, load_task_sequence, save_record)

NUM_SETTLE = 10   # census settle window, identical to the original collection
RESOLUTION = 256  # rule-collection perception config (task_context uses 64)


def make_replay_context(task_name: str):
    """Create one task's env, init states, memory phases and object query.

    Deliberately not task_context(): that helper builds a 64-resolution env
    without depth, while the Pick rule needs the RGB-D point cloud the rules
    were originally collected with (256 + agentview depth). Changing the
    perception config would mix non-timeout differences into the comparison.
    """
    from memory_system.tta.variant_spec import resolve_variant
    spec = resolve_variant(task_name, suite="libero_10")
    suite = benchmark.get_benchmark_dict()[spec.suite](category_value=spec.category)
    tids = [i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name]
    if len(tids) != 1:
        raise RuntimeError(f"expected one task match for {task_name}, got {tids}")
    env, _ = get_libero_env(suite.get_task(tids[0]), "cosmos",
                            resolution=RESOLUTION, camera_depths=[True, False])
    init_states = suite.get_task_init_states(tids[0])
    _, _, phases = load_task_sequence(task_name)
    object_query = SceneObjectQuery(env, resolution=RESOLUTION)
    return env, init_states, phases, object_query


def replay_case(env, init_state, phases, object_query, actions, *,
                task_name: str, chunk_size: int = 16,
                source_proprio: np.ndarray | None = None) -> dict:
    """Replay one saved action stream and re-label it under current rules.

    Chunk boundaries stay on the original global grid (every ``chunk_size``
    actions): a phase that observe() advanced on a boundary step gets no
    extra finish_chunk, and the chunk count is never restarted after a
    switch. This replicates run_one_init()'s queue-drain behaviour without a
    policy: the action sequence is exactly the saved one.
    """
    from memory_system.execute.skill_completion.shadow import PickTargetPointCloud

    base, demo_id, _ = load_task_sequence(task_name)
    recorder = PhaseEventRecorder(
        phases, task_name=base, demo_id=demo_id, object_query=object_query)
    point_cloud = PickTargetPointCloud(env, RESOLUTION)

    def pick_points(phase, observation):
        if phase is None or phase.skill != "Pick":
            return None
        item = phase.arguments.get("item")
        if not item:
            return None
        try:
            return point_cloud.points(observation, item)
        except Exception:
            return None

    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_SETTLE):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    recorder.begin(step=0, initial_gripper_state=0)
    proprio_buf, success, executed = [], False, 0
    for step, action in enumerate(np.asarray(actions, dtype=np.float64)):
        phase = recorder.active_phase
        proprio_buf.append(np.concatenate([
            obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"],
        ]).astype(np.float64))
        obs, _, done, _ = env.step(action.tolist())
        decision = recorder.observe(step=step, action=action, obs=obs,
                                    target_points=pick_points(phase, obs))
        executed = step + 1
        if done:
            success = True
            break
        if not (decision is not None and decision.advance) and \
                (step + 1) % chunk_size == 0:
            recorder.finish_chunk(step=step)

    record = recorder.finalize(total_steps=executed, success=success)
    validation = {
        "chunk_size": int(chunk_size),
        "executed_steps": int(executed),
        "source_action_steps": int(len(actions)),
        "consumed_all_actions": bool(executed == len(actions)),
        "replay_success": bool(success),
    }
    if source_proprio is not None and len(proprio_buf):
        proprio = np.stack(proprio_buf)
        n = min(len(proprio), len(source_proprio))
        diff = np.abs(proprio[:n] - np.asarray(source_proprio[:n], dtype=np.float64))
        validation.update(
            proprio_compared_steps=int(n),
            proprio_max_abs_diff=float(diff.max()),
            proprio_mean_abs_diff=float(diff.mean()))
    record["replay_validation"] = validation
    return record


def save_case(output_dir: Path, source_dir: Path, source_record: dict,
              replay_record: dict, *, overwrite: bool = False) -> None:
    """Write the replayed record plus a symlink to the source episode.h5."""
    if (output_dir / "phase_record.json").exists() or \
            (output_dir / "episode.h5").exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite {output_dir}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists with unexpected content")
    record = dict(replay_record)
    for key in ("census_abs_init", "suite_task_name", "task_name_perturbed",
                "perturbation_category", "perturbation_condition", "bddl_file"):
        if source_record.get(key) is not None:
            record[key] = source_record[key]
    candidate = None if record["task_success"] else record.get("candidate")
    if candidate is not None:
        t_star, event_step = compute_t_star(candidate)
        candidate["t_star"] = t_star
        candidate["event_step"] = event_step
    record["rule_config"] = {
        "skill_max_action_chunks": dict(SKILL_MAX_ACTION_CHUNKS),
        "default_max_action_chunks": int(DEFAULT_MAX_ACTION_CHUNKS),
        "chunk_size": record["replay_validation"]["chunk_size"],
        "resolution": RESOLUTION,
        "source_case_dir": str(source_dir.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    episode = (source_dir / "episode.h5").resolve()
    link = output_dir / "episode.h5"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(episode)
    save_record(record, output_dir / "phase_record.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS,
                        help="input datasets with phase_record.json cases")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="output directory for the re-labelled records")
    parser.add_argument("--limit", type=int, default=0,
                        help="only replay the first N cases (0 = all)")
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="action chunk size of the original collection")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo cases whose output already exists")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for dataset, case_dir, record in load_cases(args.roots)[:args.limit or None]:
        grouped[record["suite_task_name"]].append((dataset, case_dir, record))

    ok = failed = 0
    for name, cases in grouped.items():
        env, init_states, phases, object_query = make_replay_context(name)
        try:
            for dataset, case_dir, record in cases:
                output_dir = (args.output_root / dataset /
                              case_dir.parent.name / case_dir.name)
                try:
                    with h5py.File(case_dir / "episode.h5") as h5:
                        actions = h5["actions"][:]
                        source_proprio = h5["proprio"][:]
                    replay = replay_case(
                        env, init_states[record["census_abs_init"]], phases,
                        object_query, actions, task_name=name,
                        chunk_size=args.chunk_size, source_proprio=source_proprio)
                    save_case(output_dir, case_dir, record, replay,
                              overwrite=args.overwrite)
                except Exception as exc:
                    failed += 1
                    print(f"{dataset}/{case_dir.name}: ERROR {exc}", flush=True)
                    continue
                ok += 1
                val = replay["replay_validation"]
                cand = replay.get("candidate")
                cand_txt = "none" if cand is None else (
                    f"{cand['span']['skill']}#{cand['span']['phase_index']} "
                    f"{cand['status']} t*={cand.get('t_star')}")
                print(f"{dataset}/{case_dir.name}: success "
                      f"{record['task_success']}->{val['replay_success']} "
                      f"proprio_dmax={val.get('proprio_max_abs_diff', float('nan')):.2e} "
                      f"| {cand_txt}", flush=True)
        finally:
            env.close()
    print(f"done: {ok} ok, {failed} failed -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
