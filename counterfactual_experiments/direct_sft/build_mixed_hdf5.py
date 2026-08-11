#!/usr/bin/env python3
"""Build a balanced clean/camera/background/light LIBERO-10 HDF5 dataset.

The clean and camera domains use the same camera dataset episodes.  Clean
exports the observation streams, while camera exports the perturbed streams,
so those two domains remain trajectory-aligned.

The output can be passed directly to ``LIBERODataset`` because its loader
recursively discovers HDF5 files::

    OUTPUT/{train,val}/{clean,camera,background,light}/*.hdf5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py

from convert_lerobot_to_hdf5 import Episode, load_episodes, output_path, write_task_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT = DATASET_ROOT / "mixed_clean_camera_background_light_libero10_hdf5"

SOURCES = {
    "clean": ("paired_camera_render_libero10_500", "observation"),
    "camera": ("paired_camera_render_libero10_500", "perturbed"),
    "background": ("paired_libero_plus_background_libero10_500", "perturbed"),
    "light": ("paired_libero_plus_light_libero10_500", "perturbed"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-task", type=int, default=40)
    parser.add_argument("--val-per-task", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--require-all-tasks",
        action="store_true",
        help="Fail unless all 10 tasks meet both split quotas.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_balanced(
    episodes: list[Episode], domain: str, quotas: dict[str, int], tasks: list[str]
) -> dict[tuple[str, str], list[Episode]]:
    grouped: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.split, episode.task)].append(episode)

    selected = {}
    for split, quota in quotas.items():
        if quota < 1:
            raise ValueError(f"--{split}-per-task must be positive")
        for task in tasks:
            candidates = sorted(grouped[(split, task)], key=lambda item: item.sample_id)
            if len(candidates) < quota:
                raise ValueError(
                    f"{domain}/{split}/{task}: requested {quota} episodes, "
                    f"but only {len(candidates)} are available"
                )
            selected[(split, task)] = candidates[:quota]
    return selected


def eligible_tasks(episodes: list[Episode], quotas: dict[str, int]) -> set[str]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for episode in episodes:
        counts[(episode.split, episode.task)] += 1
    tasks = {episode.task for episode in episodes}
    return {
        task
        for task in tasks
        if all(counts[(split, task)] >= quota for split, quota in quotas.items())
    }


def annotate_file(path: Path, domain: str, image_source: str) -> None:
    with h5py.File(path, "r+") as handle:
        handle.attrs["domain"] = domain
        handle.attrs["image_source"] = image_source
        for demo in handle["data"].values():
            demo.attrs["domain"] = domain


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    quotas = {"train": args.train_per_task, "val": args.val_per_task}
    summary = {
        "format": "cosmos_policy_libero_hdf5",
        "domains": {},
        "quotas_per_task": quotas,
    }

    loaded = {}
    eligible_by_domain = {}
    for domain, (source_name, image_source) in SOURCES.items():
        source_dir = dataset_root / source_name
        episodes = load_episodes(source_dir, image_source, "all")
        loaded[domain] = (source_dir, image_source, episodes)
        eligible_by_domain[domain] = eligible_tasks(episodes, quotas)

    tasks = sorted(set.intersection(*eligible_by_domain.values()))
    all_tasks = set.union(*eligible_by_domain.values())
    excluded = sorted(all_tasks - set(tasks))
    if args.require_all_tasks and len(tasks) != 10:
        details = {domain: sorted(all_tasks - domain_tasks) for domain, domain_tasks in eligible_by_domain.items()}
        raise ValueError(f"only {len(tasks)} tasks meet all quotas; missing by domain: {details}")
    if not tasks:
        raise ValueError("no task meets the requested quotas in every domain")
    print(f"Using {len(tasks)} shared tasks; excluded {len(excluded)}: {excluded}")
    summary["tasks"] = tasks
    summary["excluded_tasks"] = excluded

    for domain, (source_dir, image_source, episodes) in loaded.items():
        selected = select_balanced(episodes, domain, quotas, tasks)
        domain_summary = {"source": str(source_dir), "image_source": image_source, "splits": {}}

        for (split, task), task_episodes in sorted(selected.items()):
            filename = output_path(Path("."), split, task).name
            path = output_dir / split / domain / filename
            frames = sum(episode.expected_frames for episode in task_episodes)
            print(f"{domain:10s} {split:5s} {len(task_episodes):2d} episodes {frames:6d} frames -> {path}")
            if not args.dry_run:
                write_task_file(path, task_episodes, args.jpeg_quality, args.overwrite)
                annotate_file(path, domain, image_source)

            split_summary = domain_summary["splits"].setdefault(split, {"episodes": 0, "frames": 0})
            split_summary["episodes"] += len(task_episodes)
            split_summary["frames"] += frames
        summary["domains"][domain] = domain_summary

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
