#!/usr/bin/env python
"""Evaluate an SFT checkpoint on the 132 baseline-success census cases.

The original robot-initial-state census contains 10 LIBERO-10 tasks and 20
initial states per task (absolute init indices 4..23).  failed_census_68.json
records the 68 baseline failures.  This script constructs the exact set
difference (200 - 68), writes its metadata into the work directory, and then
delegates evaluation to eval_sft_68.py so both evaluations use identical
rollout machinery.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EVALUATOR = REPO / "scripts/eval_sft_68.py"
DEFAULT_FAILURES = REPO / "memory_system/pointcloud_action/failed_census_68.json"
INIT_INDICES = range(4, 24)
EXPECTED_FAILURES = 68
EXPECTED_SUCCESSES = 132

# tag prefix, base task, robot_initial_states variant, language
TASKS = (
    (
        "KSCENE3",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        273,
        "turn on the stove and put the moka pot on it",
    ),
    (
        "KSCENE4",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        274,
        "put the black bowl in the bottom drawer of the cabinet and close it",
    ),
    (
        "KSCENE6",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        270,
        "put the yellow and white mug in the microwave and close it",
    ),
    (
        "KSCENE8",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        269,
        "put both moka pots on the stove",
    ),
    (
        "LRSCENE1",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        268,
        "put both the alphabet soup and the cream cheese box in the basket",
    ),
    (
        "LRSCENE2",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        271,
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
    (
        "LRSCENE2B",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        282,
        "put both the cream cheese box and the butter in the basket",
    ),
    (
        "LRSCENE5",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        265,
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
    ),
    (
        "LRSCENE6",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        267,
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    ),
    (
        "SSCENE1",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        276,
        "pick up the book and place it in the back compartment of the caddy",
    ),
)


def build_success_meta(failure_path: Path, seed: int) -> dict[str, dict]:
    failure_doc = json.loads(failure_path.read_text())
    specs_by_task = {task: (prefix, variant, language) for prefix, task, variant, language in TASKS}
    failures: set[tuple[str, int]] = set()

    for row in failure_doc["tasks"]:
        task = row["task"]
        if task not in specs_by_task:
            raise ValueError(f"failure census contains an unknown task: {task}")
        _, expected_variant, _ = specs_by_task[task]
        if row["variant_state"] != expected_variant:
            raise ValueError(
                f"variant mismatch for {task}: {row['variant_state']} != {expected_variant}"
            )
        indices = row["fail_init_indices_abs"]
        if len(indices) != row["n_fail"]:
            raise ValueError(f"n_fail mismatch for {task}")
        for init_idx in indices:
            if init_idx not in INIT_INDICES:
                raise ValueError(f"failure init index outside 4..23: {task} init{init_idx:03d}")
            key = (task, init_idx)
            if key in failures:
                raise ValueError(f"duplicate failure case: {task} init{init_idx:03d}")
            failures.add(key)

    if len(failures) != EXPECTED_FAILURES:
        raise ValueError(f"expected 68 unique failures, found {len(failures)}")

    meta: dict[str, dict] = {}
    for prefix, task, variant, language in TASKS:
        pert_task = f"{task}_view_0_0_100_0_0_initstate_{variant}"
        for init_idx in INIT_INDICES:
            if (task, init_idx) in failures:
                continue
            tag = f"{prefix}-init{init_idx:03d}"
            meta[tag] = {
                "task": task,
                "base_task": task,
                "language": language,
                "pert_name": "robot_initial_states",
                "pert_category": "Robot Initial States",
                "pert_task": pert_task,
                "suite_task_name": pert_task,
                "variant_state": variant,
                "init": f"init{init_idx:03d}",
                "census_abs_init": init_idx,
                "seed": seed,
                "baseline_success": True,
            }

    if len(meta) != EXPECTED_SUCCESSES:
        raise ValueError(f"expected 132 baseline-success cases, found {len(meta)}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3", help="GPUs to use, one chain each")
    ap.add_argument("--tags", default="", help="optional comma-separated subset of the 132 tags")
    ap.add_argument("--ckpt", required=True, help="SFT checkpoint (.pt or DCP directory)")
    ap.add_argument(
        "--work-dir",
        default=str(REPO / "scripts/experiments/tta_sft_eval_success_132"),
    )
    ap.add_argument("--failures", default=str(DEFAULT_FAILURES), help="68-case failure census JSON")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--list-only", action="store_true", help="validate and print cases without evaluation")
    args = ap.parse_args()

    meta = build_success_meta(Path(args.failures), args.seed)
    all_tags = sorted(meta)
    requested = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    unknown = sorted(set(requested) - set(meta))
    if unknown:
        raise ValueError(f"unknown/non-success tags: {', '.join(unknown)}")
    tags = requested or all_tags

    counts = Counter(row["task"] for row in meta.values())
    print(f"Validated census complement: 200 - 68 = {len(meta)} cases")
    for _, task, _, _ in TASKS:
        print(f"  {task}: {counts[task]}")
    print(f"Selected for this run: {len(tags)}")
    if args.list_only:
        print("\n".join(tags))
        return 0

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    meta_path = work_dir / "baseline_success_132_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--gpus",
        args.gpus,
        "--tags",
        ",".join(tags),
        "--ckpt",
        str(Path(args.ckpt).resolve()),
        "--work-dir",
        str(work_dir),
        "--meta",
        str(meta_path),
    ]
    returncode = subprocess.run(cmd, cwd=str(REPO), check=False).returncode
    if returncode:
        return returncode

    missing = []
    ambiguous = []
    summary_paths = {}
    for tag in tags:
        summaries = list((work_dir / "results" / tag).glob("*_summary.json"))
        if not summaries:
            missing.append(tag)
        elif len(summaries) != 1:
            ambiguous.append((tag, len(summaries)))
        else:
            summary_paths[tag] = summaries[0]
    print(f"Completed summaries: {len(tags) - len(missing) - len(ambiguous)}/{len(tags)}")
    if missing:
        print(f"Missing summaries ({len(missing)}): {', '.join(missing)}", file=sys.stderr)
    if ambiguous:
        details = ", ".join(f"{tag}={count}" for tag, count in ambiguous)
        print(f"Cases with multiple summaries: {details}", file=sys.stderr)
    if missing or ambiguous:
        return 1

    case_results = {}
    total_successes = 0
    total_trials = 0
    for tag, summary_path in summary_paths.items():
        summary = json.loads(summary_path.read_text())
        conditions = summary.get("conditions", [])
        if len(conditions) != 1:
            raise ValueError(f"expected one condition in {summary_path}, found {len(conditions)}")
        condition = conditions[0]
        successes = int(condition["successes"])
        num_trials = int(condition["num_trials"])
        total_successes += successes
        total_trials += num_trials
        case_results[tag] = {
            "successes": successes,
            "num_trials": num_trials,
            "success_rate": condition["success_rate"],
            "summary_path": str(summary_path),
        }

    aggregate = {
        "set": "baseline-success census complement (200 - 68)",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "selected_cases": len(tags),
        "successes": total_successes,
        "num_trials": total_trials,
        "success_rate": total_successes / total_trials if total_trials else 0.0,
        "cases": case_results,
    }
    aggregate_path = work_dir / "baseline_success_132_eval_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Baseline-success retention: {total_successes}/{total_trials} "
        f"({aggregate['success_rate']:.1%})"
    )
    print(f"Aggregate summary: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
