"""Paired preference dataset for memory-guided TTA (Diffusion-DPO, v1).

One item = ONE pair = one chosen chunk + one rejected chunk (K=1, decided
2026-09-09): tensors are stacked with chosen first, so a DataLoader batch has
shape [B, 2, ...] and the model sees both sides of a pair in a SINGLE forward
with ONE shared sigma/epsilon draw.

Objective scope: with K=1 this is a single-chunk preference objective, not a
trajectory-level sum (docs/TTA_TRAINING_IMPLEMENTATION.md §目标公式).

Chunk choice: deterministic per (pair, epoch) via set_epoch(), so the custom
training loop can vary chunks across epochs while staying reproducible.

Strict labels: the h5 `success` attr is ground truth — chosen must have
succeeded, rejected must have failed (doc §9).
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_common import compute_monte_carlo_returns
from cosmos_policy.datasets.dataset_utils import decode_single_jpeg_frame, rescale_episode_data
from cosmos_policy.datasets.libero_dataset import build_action_chunk_sample

CHOSEN, REJECTED = 0, 1


def load_pair_manifest(path: str | Path) -> list[dict]:
    with open(path) as f:
        pairs = json.load(f)["pairs"]
    if not pairs:
        raise ValueError(f"manifest {path} has no pairs")
    for p in pairs:
        for key in ("pair_id", "chosen_path", "rejected_path"):
            value = p.get(key)
            if not value:
                raise ValueError(
                    f"manifest {path}: pair {p.get('pair_id', '?')} has invalid {key}={value!r} "
                    "(incomplete pairs must be dropped at manifest build time)"
                )
        for key in ("chosen_path", "rejected_path"):
            if not Path(p[key]).exists():
                raise FileNotFoundError(f"manifest {path}: pair {p['pair_id']} {key} missing: {p[key]}")
    return pairs


@lru_cache(maxsize=4)
def _load_episode(
    path: str,
    actions_min: bytes,
    actions_max: bytes,
    proprio_min: bytes,
    proprio_max: bytes,
    normalize_actions: bool,
    normalize_proprio: bool,
    gamma: float,
) -> dict:
    """Load + decode + normalize one episode (lru-cached to bound host memory)."""
    with h5py.File(path, "r") as f:
        if "primary_images_jpeg" in f:
            images = [decode_single_jpeg_frame(b) for b in f["primary_images_jpeg"][:]]
            wrist_images = [decode_single_jpeg_frame(b) for b in f["wrist_images_jpeg"][:]]
        elif "primary_images" in f:
            images, wrist_images = f["primary_images"][:], f["wrist_images"][:]
        else:
            raise KeyError(f"no images in episode: {path}")
        actions = f["actions"][:].astype(np.float32)
        proprio = f["proprio"][:].astype(np.float32)
        success = bool(f.attrs.get("success", False))
        command = str(f.attrs.get("task_description", ""))
    stats = {
        "actions_min": np.frombuffer(actions_min, dtype=np.float32),
        "actions_max": np.frombuffer(actions_max, dtype=np.float32),
        "proprio_min": np.frombuffer(proprio_min, dtype=np.float32),
        "proprio_max": np.frombuffer(proprio_max, dtype=np.float32),
    }
    if normalize_actions:
        actions = rescale_episode_data({"actions": actions}, stats, "actions")
    if normalize_proprio:
        proprio = rescale_episode_data({"proprio": proprio}, stats, "proprio")
    num_steps = len(actions)
    return dict(
        images=images,
        wrist_images=wrist_images,
        proprio=proprio,
        actions=actions,
        command=command,
        num_steps=num_steps,
        success=success,
        returns=compute_monte_carlo_returns(num_steps, terminal_reward=1.0 if success else 0.0, gamma=gamma),
    )


class PreferenceDataset(Dataset):
    """__getitem__(pair_idx) -> one pair item: per-field tensors stacked [2, ...]
    (chosen first), plus bookkeeping scalars. Default DataLoader collate then
    produces [B, 2, ...] batches."""

    def __init__(
        self,
        manifest_path: str,
        t5_text_embeddings_path: str,
        dataset_stats_path: str,
        chunk_size: int = 16,
        gamma: float = 0.99,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        final_image_size: int = 224,
        num_duplicates_per_image: int = 4,
        strict_labels: bool = True,
        seed: int = 0,
    ):
        import pickle

        self.pairs = load_pair_manifest(manifest_path)
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.num_duplicates_per_image = num_duplicates_per_image
        self.gamma = gamma
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.strict_labels = strict_labels
        self.seed = seed
        self.epoch = 0
        with open(t5_text_embeddings_path, "rb") as f:
            self.t5_text_embeddings = pickle.load(f)
        with open(dataset_stats_path) as f:
            stats = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
        self._stats_bytes = tuple(stats[k].tobytes() for k in ("actions_min", "actions_max", "proprio_min", "proprio_max"))

        # Validate pairs once: chunk counts and (optionally) h5 label ground truth.
        self._valid: list[dict] = []
        for pair in self.pairs:
            entry = {}
            for key in ("chosen", "rejected"):
                ep = self._episode(pair[f"{key}_path"])
                if self.strict_labels:
                    if key == "chosen" and not ep["success"]:
                        raise ValueError(f"{pair['pair_id']}: chosen did not succeed ({pair['chosen_path']})")
                    if key == "rejected" and ep["success"]:
                        raise ValueError(f"{pair['pair_id']}: rejected succeeded ({pair['rejected_path']})")
                    label = pair.get(f"{key}_success")
                    if label is not None and bool(label) != ep["success"]:
                        raise ValueError(f"{pair['pair_id']}: manifest label contradicts h5 attr for {key}")
                n = ep["num_steps"]
                entry[key] = list(range(0, max(0, (n - self.chunk_size) // self.chunk_size + 1)))
            if not entry["chosen"] or not entry["rejected"]:
                raise ValueError(f"{pair['pair_id']}: episode shorter than one chunk")
            self._valid.append(entry)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def _episode(self, path: str) -> dict:
        a_min, a_max, p_min, p_max = self._stats_bytes
        return _load_episode(path, a_min, a_max, p_min, p_max, self.normalize_actions, self.normalize_proprio, self.gamma)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        # ONE random ratio per pair per epoch, mapped onto BOTH trajectories:
        # independent draws would compare unrelated phases and turn the K=1
        # preference label into noise (review finding, 2026-09-09).
        rng = random.Random(self.seed + 100_000 * self.epoch + idx)
        ratio = rng.random()
        sides = []
        for side, key in ((CHOSEN, "chosen"), (REJECTED, "rejected")):
            ep = self._episode(pair[f"{key}_path"])
            starts = self._valid[idx][key]
            start = starts[int(ratio * (len(starts) - 1))] * self.chunk_size
            sample = build_action_chunk_sample(
                ep,
                start,
                self.chunk_size,
                self.t5_text_embeddings,
                use_proprio=True,
                use_wrist_images=True,
                use_third_person_images=True,
                return_value_function_returns=True,
                num_duplicates_per_image=self.num_duplicates_per_image,
                final_image_size=self.final_image_size,
                normalize_images=False,
                use_image_aug=False,  # v1 freezes augmentation randomness
                stronger_image_aug=False,
                decompress_jpeg=False,  # frames already decoded by _load_episode
                returns=ep["returns"],
                rollout_data_mask=0,  # policy-mode forward
                rollout_data_success_mask=int(ep["success"]),
                sample_key=f"{pair['pair_id']}_{key}_start{start}",
                extra_fields={
                    "pair_id": pair["pair_id"],
                    "side": side,
                    "chunk_start": start,
                },
            )
            sides.append(sample)
        return _stack_pair(sides[0], sides[1])


def _stack_pair(chosen: dict, rejected: dict) -> dict:
    """Stack the two side samples into one pair item (leading dim 2)."""
    out = {}
    for key in chosen:
        a, b = chosen[key], rejected[key]
        if isinstance(a, torch.Tensor):
            out[key] = torch.stack([a, b], dim=0)
        elif isinstance(a, np.ndarray):
            out[key] = np.stack([a, b], axis=0)
        else:
            out[key] = [a, b]
    return out
