#!/usr/bin/env python3
"""Merge completed LIBERO-plus shards into one consistent LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch


VIDEO_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
    "perturbed.images.front",
    "perturbed.images.wrist",
)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"missing parquet column: {name}")
    return table.set_column(index, name, values)


def update_stat_constant(stat: dict[str, Any], value: int) -> None:
    stat["min"] = [value]
    stat["max"] = [value]
    stat["mean"] = [float(value)]
    stat["std"] = [0.0]


def merge(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    shard_dirs = [source / f"shard-{idx:03d}-of-008" for idx in args.shards]
    missing = [path for path in shard_dirs if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete or missing shards: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_rows: list[tuple[pathlib.Path, dict[str, Any]]] = []
    stats_lookup: dict[tuple[pathlib.Path, int], dict[str, Any]] = {}
    task_rows: dict[int, dict[str, Any]] = {}
    selections: dict[str, Any] = {}
    catalog_sha = None
    condition = None

    for shard in shard_dirs:
        rows = read_jsonl(shard / "manifest.jsonl")
        source_rows.extend((shard, row) for row in rows)
        for stat in read_jsonl(shard / "meta/episodes_stats.jsonl"):
            stats_lookup[(shard, int(stat["episode_index"]))] = stat
        for task in read_jsonl(shard / "meta/tasks.jsonl"):
            task_rows[int(task["task_index"])] = task
        selection = read_json(shard / "variant_selection.json")
        catalog_sha = catalog_sha or selection["catalog_sha256"]
        condition = condition or selection["condition"]
        if catalog_sha != selection["catalog_sha256"] or condition != selection["condition"]:
            raise ValueError(f"incompatible variant selection in {shard}")
        selections.update(selection["tasks"])

    source_rows.sort(
        key=lambda item: (
            0 if item[1]["split"] == "train" else 1,
            int(item[1]["task_index"]),
            int(item[1]["episode_index"]),
        )
    )

    manifest: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    global_frame = 0

    for new_episode, (shard, old_row) in enumerate(source_rows):
        old_episode = int(old_row["episode_index"])
        old_parquet = shard / old_row["lerobot_parquet_path"]
        table = pq.read_table(old_parquet)
        frame_count = table.num_rows
        table = replace_column(
            table,
            "episode_index",
            pa.array([new_episode] * frame_count, type=pa.int64()),
        )
        table = replace_column(
            table,
            "index",
            pa.array(range(global_frame, global_frame + frame_count), type=pa.int64()),
        )
        new_parquet_rel = pathlib.Path("data/chunk-000") / f"episode_{new_episode:06d}.parquet"
        new_parquet = output / new_parquet_rel
        new_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_parquet)

        new_video_paths: dict[str, str] = {}
        for key in VIDEO_KEYS:
            old_video_rel = pathlib.Path(old_row["rgb_video_paths"][key])
            new_video_rel = pathlib.Path("videos/chunk-000") / key / f"episode_{new_episode:06d}.mp4"
            new_video = output / new_video_rel
            new_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shard / old_video_rel, new_video)
            new_video_paths[key] = str(new_video_rel)

        old_sample = shard / old_row["path"]
        new_sample = output / old_row["path"]
        new_sample.parent.mkdir(parents=True, exist_ok=True)
        payload = torch.load(old_sample, map_location="cpu", weights_only=False)
        payload["meta"]["episode_index"] = new_episode
        payload["meta"]["lerobot_parquet_path"] = str(new_parquet_rel)
        payload["meta"]["rgb_video_paths"] = new_video_paths
        torch.save(payload, new_sample)
        del payload

        row = dict(old_row)
        row["episode_index"] = new_episode
        row["lerobot_parquet_path"] = str(new_parquet_rel)
        row["rgb_video_paths"] = new_video_paths
        manifest.append(row)
        episodes.append(
            {
                "episode_index": new_episode,
                "tasks": [row["clean_instruction"]],
                "length": frame_count,
            }
        )

        stat = stats_lookup[(shard, old_episode)]
        stat["episode_index"] = new_episode
        update_stat_constant(stat["stats"]["episode_index"], new_episode)
        index_stat = stat["stats"]["index"]
        index_stat["min"] = [global_frame]
        index_stat["max"] = [global_frame + frame_count - 1]
        index_stat["mean"] = [global_frame + (frame_count - 1) / 2.0]
        episode_stats.append(stat)
        global_frame += frame_count

        if (new_episode + 1) % 25 == 0 or new_episode + 1 == len(source_rows):
            print(f"merged {new_episode + 1}/{len(source_rows)} episodes", flush=True)

    train_count = sum(row["split"] == "train" for row in manifest)
    val_count = len(manifest) - train_count
    template_info = read_json(shard_dirs[0] / "meta/info.json")
    template_info.update(
        {
            "total_episodes": len(manifest),
            "total_frames": global_frame,
            "total_tasks": len(task_rows),
            "total_videos": len(manifest) * len(VIDEO_KEYS),
            "total_chunks": 1,
            "splits": {"train": f"0:{train_count}", "val": f"{train_count}:{len(manifest)}"},
        }
    )

    write_jsonl(output / "manifest.jsonl", manifest)
    write_jsonl(output / "meta/episodes.jsonl", episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", episode_stats)
    write_jsonl(output / "meta/tasks.jsonl", [task_rows[key] for key in sorted(task_rows)])
    write_json(output / "meta/info.json", template_info)
    write_json(
        output / "variant_selection.json",
        {"catalog_sha256": catalog_sha, "condition": condition, "tasks": selections},
    )
    write_json(
        output / "summary.json",
        {
            "dataset": "paired_libero_plus_variant_merged",
            "condition": condition,
            "source": str(source),
            "source_shards": args.shards,
            "num_samples": len(manifest),
            "num_frames": global_frame,
            "num_tasks": len(task_rows),
            "split_counts": {"train": train_count, "val": val_count},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", default=list(range(7)))
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
