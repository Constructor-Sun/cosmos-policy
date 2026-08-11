#!/usr/bin/env python3
"""Convert the paired LIBERO-Plus LeRobot dataset to Cosmos Policy HDF5.

The output schema is consumed directly by ``cosmos_policy.datasets.LIBERODataset``.
By default, perturbed camera frames are paired with the shared clean-policy
actions and proprioception stored in the LeRobot parquet files.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "dataset" / "paired_libero_plus_light_libero10_500"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "paired_libero_plus_light_libero10_500_hdf5"


@dataclass(frozen=True)
class Episode:
    shard_root: Path
    split: str
    task: str
    instruction: str
    sample_id: str
    parquet_path: Path
    front_video_path: Path
    wrist_video_path: Path
    expected_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--image-source",
        choices=("perturbed", "observation"),
        default="perturbed",
        help="Camera stream to export (default: perturbed).",
    )
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_episodes(input_dir: Path, image_source: str, split: str) -> list[Episode]:
    manifests = sorted(input_dir.glob("shard-*-of-*/manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no shard manifests found under {input_dir}")

    episodes = []
    seen_ids = set()
    for manifest in manifests:
        shard_root = manifest.parent
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                item_split = item["split"]
                if split != "all" and item_split != split:
                    continue

                sample_id = item["sample_id"]
                unique_id = (shard_root.name, sample_id)
                if unique_id in seen_ids:
                    raise ValueError(f"duplicate sample in {manifest}:{line_number}: {sample_id}")
                seen_ids.add(unique_id)

                videos = item["rgb_video_paths"]
                front_key = f"{image_source}.images.front"
                wrist_key = f"{image_source}.images.wrist"
                episode = Episode(
                    shard_root=shard_root,
                    split=item_split,
                    task=item["base_task"],
                    instruction=item.get("perturbed_instruction")
                    or item.get("instruction")
                    or item["clean_instruction"],
                    sample_id=sample_id,
                    parquet_path=shard_root / item["lerobot_parquet_path"],
                    front_video_path=shard_root / videos[front_key],
                    wrist_video_path=shard_root / videos[wrist_key],
                    expected_frames=int(item["num_frames"]),
                )
                for path in (episode.parquet_path, episode.front_video_path, episode.wrist_video_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"missing file referenced by {manifest}: {path}")
                episodes.append(episode)
    return episodes


def read_numeric_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["observation.state", "action"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"invalid state/action shapes in {path}: {states.shape}, {actions.shape}")
    return states, actions


def encode_video_as_jpeg(path: Path, quality: int) -> list[np.ndarray]:
    encoded = []
    for frame in iio.imiter(path, plugin="pyav"):
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"expected RGB frames in {path}, got {frame.shape}")
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
        encoded.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
    if not encoded:
        raise ValueError(f"video has no frames: {path}")
    return encoded


def write_jpeg_dataset(group: h5py.Group, name: str, frames: list[np.ndarray]) -> None:
    dataset = group.create_dataset(name, shape=(len(frames),), dtype=h5py.vlen_dtype(np.uint8))
    for index, frame in enumerate(frames):
        dataset[index] = frame


def output_path(output_dir: Path, split: str, task: str) -> Path:
    # LIBERODataset strips the final "_demo.hdf5" and derives the instruction
    # from all words following the SCENE token in this filename.
    return output_dir / "libero_10" / split / f"{task}_demo.hdf5"


def write_task_file(path: Path, episodes: list[Episode], quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite to replace it): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with h5py.File(temporary, "w") as handle:
            data_group = handle.create_group("data")
            data_group.attrs["num_demos"] = len(episodes)
            for demo_index, episode in enumerate(sorted(episodes, key=lambda item: item.sample_id)):
                states, actions = read_numeric_episode(episode.parquet_path)
                front = encode_video_as_jpeg(episode.front_video_path, quality)
                wrist = encode_video_as_jpeg(episode.wrist_video_path, quality)
                lengths = (len(states), len(actions), len(front), len(wrist))
                if len(set(lengths)) != 1 or lengths[0] != episode.expected_frames:
                    raise ValueError(
                        f"frame count mismatch for {episode.sample_id}: "
                        f"state/action/front/wrist={lengths}, manifest={episode.expected_frames}"
                    )

                demo = data_group.create_group(f"demo_{demo_index}")
                demo.attrs["num_samples"] = lengths[0]
                demo.attrs["sample_id"] = episode.sample_id
                demo.attrs["instruction"] = episode.instruction
                demo.create_dataset("actions", data=actions, dtype=np.float32)
                demo.create_dataset("robot_states", data=states, dtype=np.float32)
                obs = demo.create_group("obs")
                write_jpeg_dataset(obs, "agentview_rgb_jpeg", front)
                write_jpeg_dataset(obs, "eye_in_hand_rgb_jpeg", wrist)
            data_group.attrs["total"] = sum(ep.expected_frames for ep in episodes)

        os.replace(temporary, path)
        validate_file(path, len(episodes))
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def validate_file(path: Path, expected_demos: int) -> None:
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"], key=lambda name: int(name.split("_")[1]))
        if len(demos) != expected_demos:
            raise ValueError(f"validation failed for {path}: expected {expected_demos} demos")
        for name in demos:
            demo = handle[f"data/{name}"]
            required = ("actions", "robot_states", "obs/agentview_rgb_jpeg", "obs/eye_in_hand_rgb_jpeg")
            if any(key not in demo for key in required):
                raise ValueError(f"validation failed for {path}: incomplete {name}")
            lengths = tuple(len(demo[key]) for key in required)
            if len(set(lengths)) != 1:
                raise ValueError(f"validation failed for {path}/{name}: lengths={lengths}")


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    episodes = load_episodes(args.input_dir.resolve(), args.image_source, args.split)
    grouped: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.split, episode.task)].append(episode)

    total_frames = sum(episode.expected_frames for episode in episodes)
    print(f"Found {len(episodes)} episodes, {total_frames} frames, {len(grouped)} output files")
    for (split, task), task_episodes in sorted(grouped.items()):
        path = output_path(args.output_dir.resolve(), split, task)
        frames = sum(episode.expected_frames for episode in task_episodes)
        print(f"{split:5s} {len(task_episodes):3d} episodes {frames:6d} frames -> {path}")
        if not args.dry_run:
            write_task_file(path, task_episodes, args.jpeg_quality, args.overwrite)


if __name__ == "__main__":
    main()
