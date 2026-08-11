#!/usr/bin/env python3
"""Pack episode files into memory-mappable train/validation latent shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dataset import EXPECTED_FRAME_SHAPE, read_manifest, split_rows_by_policy_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=["dataset/paired_libero_plus_background_libero10_500"],
    )
    parser.add_argument(
        "--output-dir", default="dataset/paired_libero_plus_background_libero10_500/packed_latents"
    )
    parser.add_argument("--frames-per-shard", type=int, default=8192)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def load_pair(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(row["path"], map_location="cpu", weights_only=False)
    clean = payload["latent"]["clean"]["vae_video"]
    perturbed = payload["latent"]["perturbed"]["vae_video"]
    if clean.shape != perturbed.shape or tuple(clean.shape[1:]) != EXPECTED_FRAME_SHAPE:
        raise ValueError(f"invalid paired latent shape: {row['path']}")
    if not torch.isfinite(clean).all() or not torch.isfinite(perturbed).all():
        raise ValueError(f"non-finite latent: {row['path']}")
    return perturbed.contiguous(), clean.contiguous()


def save_shard(
    output_dir: Path,
    split: str,
    shard_index: int,
    perturbed: list[torch.Tensor],
    clean: list[torch.Tensor],
) -> dict[str, Any]:
    path = output_dir / f"{split}-{shard_index:04d}.pt"
    perturbed_tensor = torch.cat(perturbed, dim=0)
    clean_tensor = torch.cat(clean, dim=0)
    torch.save({"perturbed": perturbed_tensor, "clean": clean_tensor}, path)
    print(f"saved {path.name}: {clean_tensor.shape[0]} frames", flush=True)
    return {"path": path.name, "frames": clean_tensor.shape[0]}


def pack_split(
    output_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
    frames_per_shard: int,
) -> list[dict[str, Any]]:
    records, perturbed_parts, clean_parts = [], [], []
    buffered, shard_index = 0, 0
    for episode_index, row in enumerate(rows, start=1):
        perturbed, clean = load_pair(row)
        offset = 0
        while offset < clean.shape[0]:
            take = min(frames_per_shard - buffered, clean.shape[0] - offset)
            perturbed_parts.append(perturbed[offset : offset + take])
            clean_parts.append(clean[offset : offset + take])
            offset += take
            buffered += take
            if buffered == frames_per_shard:
                records.append(
                    save_shard(output_dir, split, shard_index, perturbed_parts, clean_parts)
                )
                perturbed_parts, clean_parts = [], []
                buffered, shard_index = 0, shard_index + 1
        if episode_index % 25 == 0 or episode_index == len(rows):
            print(f"packed {split} episodes {episode_index}/{len(rows)}", flush=True)
    if buffered:
        records.append(save_shard(output_dir, split, shard_index, perturbed_parts, clean_parts))
    return records


def main() -> None:
    args = parse_args()
    if args.frames_per_shard < 1:
        raise ValueError("frames_per_shard must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    if index_path.exists():
        raise FileExistsError(f"packed dataset already exists: {index_path}")
    rows = []
    for data_dir in args.data_dir:
        _root, source_rows = read_manifest(data_dir)
        rows.extend(source_rows)
    train_rows, val_rows = split_rows_by_policy_seed(
        rows, val_fraction=args.val_fraction, seed=args.split_seed
    )
    index = {
        "sources": [str(Path(path).expanduser().resolve()) for path in args.data_dir],
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "frame_shape": EXPECTED_FRAME_SHAPE,
        "train_policy_seeds": sorted({int(row["policy_seed"]) for row in train_rows}),
        "val_policy_seeds": sorted({int(row["policy_seed"]) for row in val_rows}),
        "splits": {},
    }
    index["splits"]["train"] = pack_split(
        output_dir, "train", train_rows, args.frames_per_shard
    )
    index["splits"]["val"] = pack_split(output_dir, "val", val_rows, args.frames_per_shard)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
