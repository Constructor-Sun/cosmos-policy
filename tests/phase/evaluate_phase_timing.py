#!/usr/bin/env python
"""Compare recorded rule decisions with oracle confirmation times; no policy inference.

Times count executed actions: state[0] is initial, decision at action k is k+1.
Oracle terminal padding matches check_oracle_failures, but is flagged, not timed.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np
from check_oracle_failures import ROOT, DEFAULT_ROOTS, load_cases, task_context
from memory_system.offline.label_segments import label_demo, observed_step_order


def identity(skill, arguments):
    return skill, tuple(sorted(arguments.items()))


def evaluate(env, init, steps, directory, record):
    with h5py.File(directory / "episode.h5") as h5:
        actions = h5["actions"][:]
    env.reset()
    env.set_init_state(init)
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    states, success = [env.sim.get_state().flatten()], False
    for action in actions:
        _, _, done, _ = env.step(action.tolist())
        success |= bool(done)
        states.append(env.sim.get_state().flatten())
    states = np.asarray(states + [states[-1]] * 3)
    ordered = observed_step_order(env, states, steps, 3, 0.02)
    segments, error = label_demo(env, states, ordered, 3, 0.02)
    found = {identity(s["skill"], s["arguments"]): s for s in segments}
    failed = ordered[len(segments)] if len(segments) < len(ordered) else None
    failed_key = identity(failed.skill, failed.arguments) if failed else None
    candidate = (record.get("candidate") or {}).get("span", {})
    rows = []
    for phase in record["phases"]:
        key = identity(phase["skill"], phase["arguments"])
        decisions = [d for d in phase["decisions"] if d["advance"]]
        decision = decisions[-1] if decisions else None
        reason = decision["reason"] if decision else "pending"
        end = decision["step"] + 1 if decision else None
        segment = found.get(key)
        confirmed = segment["success_end"] - 1 if segment else None
        if segment is None:
            category = ("oracle_unreachable" if key != failed_key else
                        "false_success" if reason == "rule" else "oracle_uncompleted")
        elif confirmed > len(actions):
            category = "terminal_padding_only"
        elif confirmed <= phase["start_step"]:
            category = "completed_before_phase"
        elif end is None:
            category = "pending"
        elif reason == "timeout":
            category = "rule_missed" if confirmed <= end else "timeout_early"
        else:
            category = "success_supported" if confirmed <= end else "rule_early"
        rows.append(dict(phase_index=phase["phase_index"], skill=phase["skill"],
                         arguments=phase["arguments"], rule_start=phase["start_step"],
                         rule_end=end, rule_reason=reason, oracle_confirm=confirmed,
                         delta=None if confirmed is None or end is None else confirmed-end,
                         candidate=phase["phase_index"] == candidate.get("phase_index"),
                         category=category))
    return dict(case=str(directory), recorded_success=record["task_success"],
                replay_success=success, oracle_error=error, phases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/phase/phase_timing.json")
    args = parser.parse_args()
    grouped, results, counts = defaultdict(list), [], Counter()
    for dataset, directory, record in load_cases(args.roots)[:args.limit or None]:
        grouped[record["suite_task_name"]].append((dataset, directory, record))
    for name, cases in grouped.items():
        env, inits, steps = task_context(name)
        try:
            for dataset, directory, record in cases:
                row = evaluate(env, inits[record["census_abs_init"]], steps, directory, record)
                row["dataset"] = dataset
                results.append(row)
                counts.update((dataset, p["skill"], p["category"], p["candidate"])
                              for p in row["phases"])
                print(f"{directory}: {[(p['skill'], p['category']) for p in row['phases']]}", flush=True)
        finally:
            env.close()
    payload = dict(summary={str(k): v for k, v in counts.items()}, cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
