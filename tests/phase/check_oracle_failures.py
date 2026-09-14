#!/usr/bin/env python
"""Replay the 68 robot-init and 37 background cases; label failed phases."""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "LIBERO-plus")]
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark  # noqa: E402
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402
from memory_system.offline.label_segments import label_demo, observed_step_order  # noqa: E402
from memory_system.offline.planner import (  # noqa: E402
    build_skeleton, infer_openables, normalized_goals)
from memory_system.tta.variant_spec import resolve_variant  # noqa: E402
DEFAULT_ROOTS = [ROOT / "experiments/tta/tta_phase_check_repro_68",
                 ROOT / "experiments/tta/tta_phase_check_bg5_correct"]
def load_cases(roots):
    cases = []
    for root in roots:
        for path in sorted(root.rglob("phase_record.json")):
            cases.append((root.name, path.parent, json.loads(path.read_text())))
    return cases
def task_context(name):
    spec = resolve_variant(name, suite="libero_10")
    suite = benchmark.get_benchmark_dict()[spec.suite](category_value=spec.category)
    tids = [i for i in range(suite.n_tasks) if suite.get_task(i).name == name]
    if len(tids) != 1:
        raise RuntimeError(f"expected one task match for {name}, got {tids}")
    tid, task = tids[0], suite.get_task(tids[0])
    env, language = get_libero_env(task, "cosmos", resolution=64)
    parsed = env.env.parsed_problem
    steps = build_skeleton(normalized_goals(parsed["goal_state"]), language,
                           infer_openables(parsed))
    return env, suite.get_task_init_states(tid), steps
def check_case(env, init_state, steps, case_dir, record):
    with h5py.File(case_dir / "episode.h5") as h5:
        actions = h5["actions"][:]
    env.reset()
    env.set_init_state(init_state)
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    states, replay_success = [env.sim.get_state().flatten()], False
    for action in actions:
        _, _, done, _ = env.step(action.tolist())
        replay_success |= bool(done)
        states.append(env.sim.get_state().flatten())
    states += [states[-1]] * 3
    states = np.asarray(states)
    ordered = observed_step_order(env, states, steps, 3, 0.02)
    segments, error = label_demo(env, states, ordered, 3, 0.02)
    failed = ordered[len(segments)] if len(segments) < len(ordered) else None
    return {
        "case": str(case_dir.relative_to(ROOT)),
        "recorded_success": bool(record["task_success"]),
        "replay_success": replay_success,
        "oracle_failed_phase": None if failed is None else {
            "phase_index": len(segments), "planner_step_id": failed.step_id,
            "skill": failed.skill, "arguments": failed.arguments},
        "completed_phases": [s["skill"] for s in segments],
        "oracle_error": error}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tests/phase/oracle_failures.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for dataset, case_dir, record in load_cases(args.roots)[:args.limit or None]:
        grouped[record["suite_task_name"]].append((dataset, case_dir, record))
    results = []
    for name, cases in grouped.items():
        env, init_states, steps = task_context(name)
        try:
            for dataset, case_dir, record in cases:
                row = check_case(env, init_states[record["census_abs_init"]],
                                 steps, case_dir, record)
                row["dataset"] = dataset
                results.append(row)
                failed = row["oracle_failed_phase"]
                print(f"{dataset}/{case_dir.name}: " + ("none" if failed is None else
                      f"{failed['skill']}#{failed['phase_index']}"), flush=True)
        finally:
            env.close()
    counts = Counter((r["dataset"], (r["oracle_failed_phase"] or {}).get("skill", "none"))
                     for r in results)
    payload = {"summary": {str(k): v for k, v in counts.items()}, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(results)} cases to {args.output}")
if __name__ == "__main__":
    main()
