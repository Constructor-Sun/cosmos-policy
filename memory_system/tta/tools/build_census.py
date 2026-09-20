"""Build the TTA repair failure census from smoke-test episodes.json files.

run_libero_smoke_test.py records one episodes.json per task/perturbation with
one entry per evaluated case (variant name, init_state_index, success).  The
diagnosis stage (memory_system/tta/diagnose_failed.py) consumes a failure
census instead:

    {"tasks": [{"task", "task_name_perturbed", "n_fail",
                "variant_state", "fail_init_indices_abs"}]}

This tool converts one census file per perturbation category, so the
diagnose/repair stages can be run and measured per category.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def condition_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def variant_suffix(base_task: str, variant_task: str) -> str:
    if variant_task.startswith(base_task + "_"):
        return variant_task[len(base_task) + 1:]
    return variant_task


def collect_episode_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("episodes.json")))
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    if not files:
        raise FileNotFoundError("no episodes.json inputs found")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+",
        help="episodes.json files, or directories to search recursively",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO / "experiments/tta/tta_census/all7dims",
        help="output directory for <condition>_failures.json files",
    )
    args = parser.parse_args()

    # (category, base_task, variant_task) -> set of failed init indices
    failures: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    total = 0
    for path in collect_episode_files(args.inputs):
        episodes = json.loads(path.read_text())
        for episode in episodes:
            total += 1
            if episode.get("success", False):
                continue
            key = (
                str(episode["category"]),
                str(episode["base_task"]),
                str(episode["task_name"]),
            )
            failures[key].add(int(episode["init_state_index"]))

    per_category: dict[str, list[dict]] = defaultdict(list)
    for (category, base_task, variant_task), inits in sorted(failures.items()):
        slug = condition_slug(category)
        per_category[slug].append({
            "task": base_task,
            "task_name_perturbed": variant_task,
            "n_fail": len(inits),
            "variant_state": f"{slug}_{variant_suffix(base_task, variant_task)}",
            "fail_init_indices_abs": sorted(inits),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_fail_total = 0
    for slug, tasks in sorted(per_category.items()):
        out_path = args.out_dir / f"{slug}_failures.json"
        out_path.write_text(json.dumps(
            {"version": "episodes_json_v1", "tasks": tasks}, indent=1,
            ensure_ascii=False) + "\n")
        n_fail_total += sum(t["n_fail"] for t in tasks)
        print(f"{out_path.name}: {len(tasks)} failed variants, "
              f"{sum(t['n_fail'] for t in tasks)} failed cases")
    print(f"total: {total} evaluated cases, {n_fail_total} failures "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
