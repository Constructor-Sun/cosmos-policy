"""Frame-level loader for cached paired Cosmos video latents."""

from __future__ import annotations

import bisect
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


EXPECTED_FRAME_SHAPE = (16, 2, 28, 28)


def read_manifest(data_dir: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(data_dir).expanduser().resolve()
    manifests = [root / "manifest.jsonl"]
    if not manifests[0].exists():
        manifests = sorted(root.glob("shard-*/manifest.jsonl"))
    if not manifests:
        sample_paths = sorted(root.glob("shard-*/samples/**/*.pt"))
        if not sample_paths:
            raise FileNotFoundError(
                f"no manifest or shard sample files found under {root}"
            )
        rows = []
        for index, path in enumerate(sample_paths, start=1):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            row = dict(payload["meta"])
            row["path"] = str(path.resolve())
            row["num_frames"] = int(payload["latent"]["clean"]["vae_video"].shape[0])
            rows.append(row)
            if index % 25 == 0 or index == len(sample_paths):
                print(f"indexed samples {index}/{len(sample_paths)} under {root}", flush=True)
        return root, rows
    rows = []
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["path"] = str((manifest.parent / row["path"]).resolve())
                rows.append(row)
    if not rows:
        raise ValueError(f"all manifests are empty under {root}")
    return root, rows


def split_rows_by_policy_seed(
    rows: list[dict[str, Any]], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if "policy_seed" not in row:
            raise KeyError("every manifest row must contain policy_seed")
        by_seed.setdefault(int(row["policy_seed"]), []).append(row)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        raise ValueError("at least two distinct policy seeds are required")
    random.Random(seed).shuffle(seeds)
    n_val = max(1, min(len(seeds) - 1, round(len(seeds) * val_fraction)))
    val_seeds = set(seeds[:n_val])
    train_rows, val_rows = [], []
    for policy_seed, group in by_seed.items():
        (val_rows if policy_seed in val_seeds else train_rows).extend(group)
    return train_rows, val_rows


class PairedLatentDataset(Dataset):
    def __init__(
        self, root: str | Path, rows: list[dict[str, Any]], cache_size: int = 2
    ) -> None:
        self.root = Path(root)
        self.rows = rows
        if not self.rows:
            raise ValueError("dataset rows cannot be empty")
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.ends: list[int] = []
        total = 0
        for row in self.rows:
            frames = int(row.get("num_frames", 0))
            if frames <= 0:
                frames = self._load_file(row)[0].shape[0]
            total += frames
            self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1]

    def _load_file(self, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.root / row["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        try:
            clean = payload["latent"]["clean"]["vae_video"]
            perturbed = payload["latent"]["perturbed"]["vae_video"]
        except KeyError as exc:
            raise KeyError(f"missing paired vae_video latent in {path}") from exc
        if clean.shape != perturbed.shape or tuple(clean.shape[1:]) != EXPECTED_FRAME_SHAPE:
            raise ValueError(
                f"invalid latent shapes in {path}: clean={tuple(clean.shape)}, "
                f"perturbed={tuple(perturbed.shape)}"
            )
        if not clean.is_floating_point() or not perturbed.is_floating_point():
            raise TypeError(f"latents must be floating point: {path}")
        if not torch.isfinite(clean).all() or not torch.isfinite(perturbed).all():
            raise ValueError(f"non-finite latent in {path}")
        return perturbed.contiguous(), clean.contiguous()

    def _episode(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        pair = self._load_file(self.rows[index])
        if self.cache_size:
            self.cache[index] = pair
            self.cache.move_to_end(index)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return pair

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.ends, index)
        start = 0 if episode_index == 0 else self.ends[episode_index - 1]
        perturbed, clean = self._episode(episode_index)
        local_index = index - start
        if local_index >= clean.shape[0]:
            raise IndexError(f"manifest num_frames exceeds tensor length for row {episode_index}")
        return {
            "perturbed": perturbed[local_index],
            "clean": clean[local_index],
        }


class PackedLatentDataset(Dataset):
    """Memory-map contiguous latent shards prepared by ``prepare_data.py``."""

    def __init__(self, packed_dir: str | Path, split: str) -> None:
        self.root = Path(packed_dir).expanduser().resolve()
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"packed dataset not found: {index_path}; run prepare_data.py first"
            )
        self.metadata = json.loads(index_path.read_text(encoding="utf-8"))
        records = self.metadata["splits"].get(split, [])
        if not records:
            raise ValueError(f"packed dataset has no {split!r} shards")
        self.shards: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.ends: list[int] = []
        total = 0
        for record in records:
            payload = torch.load(
                self.root / record["path"], map_location="cpu", mmap=True, weights_only=True
            )
            perturbed, clean = payload["perturbed"], payload["clean"]
            if perturbed.shape != clean.shape or tuple(clean.shape[1:]) != EXPECTED_FRAME_SHAPE:
                raise ValueError(f"invalid packed shard: {record['path']}")
            self.shards.append((perturbed, clean))
            total += clean.shape[0]
            self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        perturbed, clean = self.shards[shard_index]
        local = index - start
        return {"perturbed": perturbed[local], "clean": clean[local]}
