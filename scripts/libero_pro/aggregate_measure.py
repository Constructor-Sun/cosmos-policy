"""Aggregate per-episode measure json into rule-vs-truth verdicts.

Only phases that actually have an implemented rule are scored.  Phases whose
skill maps to TimeoutOnlySkillCompletion (Open/Close/TurnOn) have no semantic
rule, so "the rule did not fire" carries no information and they are reported
separately as truth-only.

Per scored phase, two independent booleans:

  truth_done   the simulator says the phase actually completed (truth_rise set)
  rule_fired   our rule reported completion

  truth_done & rule_fired          -> TP   (delta = rule_fired - truth_rise)
  truth_done & not rule_fired      -> FN   (phase stalls; timeout then desyncs)
  not truth_done & rule_fired      -> FP   (worst: advances on a failed phase)
  not truth_done & not rule_fired  -> TN   (correctly reports "not done")

Also reports where each episode got stuck: the first phase whose truth never
completed, judging over ALL phases (rule or not), since that is a property of
the episode, not of our rule.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path


def verdict(row):
    truth_done = row.get("truth_rise") is not None
    rule_fired = row.get("rule_fired") is not None
    if truth_done and rule_fired:
        return "TP"
    if truth_done and not rule_fired:
        return "FN"
    if not truth_done and rule_fired:
        return "FP"
    return "TN"


def blank():
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "deltas": [], "n": 0, "no_rule": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-glob",
                    default="experiments/liberopro/measure_swap_seed7/json/*.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(args.json_glob))
    if not files:
        raise SystemExit(f"no json matched {args.json_glob}")

    per_phase = collections.defaultdict(blank)
    stuck = collections.Counter()
    n_episodes = 0
    unresolved = 0

    for path in files:
        with open(path) as f:
            episodes = json.load(f)
        for ep in episodes:
            n_episodes += 1
            stuck_at = None
            for row in ep["phases"]:
                key = row["key"]
                slot = per_phase[key]
                slot["n"] += 1

                if row.get("item") and not row.get("item_instance"):
                    unresolved += 1

                # Stuck point is a property of the episode: first phase whose
                # truth never completed, regardless of whether a rule exists.
                if stuck_at is None and row.get("truth_rise") is None:
                    item_or_target = row.get("item") or row.get("target")
                    stuck_at = f"{row['skill']}:{item_or_target}"

                if not row.get("rule_active", False):
                    slot["no_rule"] += 1
                    continue

                slot[verdict(row)] += 1
                if row.get("delta") is not None:
                    slot["deltas"].append(row["delta"])

            stuck[stuck_at or "<all phases truly completed>"] += 1

    print(f"episodes: {n_episodes}   (unresolved instance names: {unresolved})")
    print("only phases with an implemented rule are scored; "
          "phases without one are marked (no rule)")
    print()
    print(f"{'phase':<52} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  delta")
    for key in sorted(per_phase, key=lambda k: (-per_phase[k]["n"], k)):
        s = per_phase[key]
        d = s["deltas"]
        dtxt = (f"n={len(d)} med={statistics.median(d):+.0f} "
                f"min={min(d):+d} max={max(d):+d}") if d else "-"
        mark = "  (no rule)" if s["no_rule"] else ""
        print(f"{key:<52} {s['n']:>4} {s['TP']:>4} {s['FP']:>4} "
              f"{s['FN']:>4} {s['TN']:>4}  {dtxt}{mark}")

    print()
    print("=== 卡在哪一步 (第一个 truth 未完成的阶段) ===")
    for name, count in stuck.most_common():
        print(f"  {count:>4}  {name}")

    if args.out:
        payload = {
            "n_episodes": n_episodes,
            "per_phase": dict(per_phase),
            "stuck_at": dict(stuck),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
