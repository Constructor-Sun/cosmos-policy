#!/usr/bin/env python3
"""Compare Cosmos Policy clean-branch actions against LIBERO-Plus expert actions.

For each Phase-1 episode, runs the clean branch denoising trajectory,
extracts the final action chunk, and computes MSE against the corresponding
expert demonstration action from libero_plus_10.zip.

Usage:
  cd /data1/liu/exp/counterfactual/external/cosmos-policy
  .venv/bin/python bin/compare_clean_vs_expert.py \
    --summary experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json \
    --expert-actions-zip /data1/liu/exp/counterfactual/libero_plus_10.zip \
    --output-dir experiments/clean_vs_expert_comparison
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import site
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DETERMINISTIC", "True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", str(ROOT.parent / "LIBERO-plus")))
for item in (str(LIBERO_PLUS), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
import torch

from cosmos_policy._src.imaginaire.functional.multi_step import is_multi_step_fn_supported
from cosmos_policy._src.imaginaire.functional.runge_kutta import is_runge_kutta_fn_supported
from cosmos_policy._src.imaginaire.modules.res_sampler import (
    SamplerConfig,
    SolverConfig,
    SolverTimestampConfig,
    differential_equation_solver,
    get_rev_ts,
)
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE,
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    extract_action_chunk_from_latent_sequence,
    get_model,
    get_t5_embedding_from_cache,
    load_dataset_stats,
    prepare_images_for_model,
    rescale_proprio,
)
from cosmos_policy.utils.utils import duplicate_array, set_seed_everywhere

CHECKPOINT_ROOT = ROOT.parent.parent / "checkpoints"
POLICY_DIR = CHECKPOINT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
EPS = 1e-12


# ---------------------------------------------------------------------------
# LeRobot expert-action reader (same logic as run_denoise_hessian_cosmos.py)
# ---------------------------------------------------------------------------

def normalize_actions_np(actions: np.ndarray, dataset_stats: dict[str, Any]) -> np.ndarray:
    actions_min = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
    actions_max = np.asarray(dataset_stats["actions_max"], dtype=np.float32)
    return (2.0 * ((actions.astype(np.float32) - actions_min) / (actions_max - actions_min)) - 1.0).astype(np.float32)


def unnormalize_actions_np(actions: np.ndarray, dataset_stats: dict[str, Any]) -> np.ndarray:
    actions_min = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
    actions_max = np.asarray(dataset_stats["actions_max"], dtype=np.float32)
    return (0.5 * (actions.astype(np.float32) + 1.0) * (actions_max - actions_min) + actions_min).astype(np.float32)


class ExpertActionStore:
    """Read same-task LeRobot expert action chunks from a zip archive."""

    def __init__(self, zip_path: pathlib.Path):
        if not zip_path.exists():
            raise FileNotFoundError(f"Expert action zip not found: {zip_path}")
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(zip_path)
        self.prefix = self._find_prefix()
        self.info = json.loads(self.zip_file.read(self.prefix + "meta/info.json").decode("utf-8"))
        self.tasks = [json.loads(line) for line in self.zip_file.read(self.prefix + "meta/tasks.jsonl").decode("utf-8").splitlines()]
        self.episodes = [
            json.loads(line) for line in self.zip_file.read(self.prefix + "meta/episodes.jsonl").decode("utf-8").splitlines()
        ]
        self.task_by_text = {item["task"]: item for item in self.tasks}
        self.task_by_index = {int(item["task_index"]): item for item in self.tasks}
        self.episodes_by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in self.episodes:
            if not episode.get("tasks"):
                continue
            self.episodes_by_task.setdefault(episode["tasks"][0], []).append(episode)
        for episodes in self.episodes_by_task.values():
            episodes.sort(key=lambda item: int(item["episode_index"]))
        self._actions_cache: dict[int, np.ndarray] = {}

    def _find_prefix(self) -> str:
        for name in self.zip_file.namelist():
            if name.endswith("meta/info.json"):
                return name[: -len("meta/info.json")]
        raise RuntimeError(f"Could not find meta/info.json inside {self.zip_path}")

    def read_actions(self, episode_index: int) -> np.ndarray:
        if episode_index in self._actions_cache:
            return self._actions_cache[episode_index]
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading LeRobot parquet requires pyarrow.") from exc

        chunks_size = int(self.info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size
        rel = self.info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        path = self.prefix + rel.format(episode_chunk=episode_chunk, episode_index=episode_index)
        table = pq.read_table(self.zip_file.read(path), columns=["action"])
        action_col = table.column("action").combine_chunks()
        actions = action_col.values.to_numpy(zero_copy_only=False).reshape(len(action_col), ACTION_DIM).astype(np.float32)
        self._actions_cache[episode_index] = actions
        return actions

    def get_demo_episodes_for_task(self, task_text: str) -> list[dict[str, Any]]:
        return self.episodes_by_task.get(task_text, [])


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------

def build_inference_data_batch(
    cfg: SimpleNamespace,
    model: torch.nn.Module,
    dataset_stats: dict[str, Any],
    obs: dict[str, Any],
    task_label_or_embedding: Any,
    device: torch.device,
    batch_size: int = 1,
) -> dict[str, Any]:
    if cfg.suite != "libero":
        raise ValueError(f"LIBERO only, got suite={cfg.suite}")

    if isinstance(task_label_or_embedding, str):
        text_embedding = get_t5_embedding_from_cache(task_label_or_embedding)
    elif isinstance(task_label_or_embedding, np.ndarray):
        text_embedding = torch.tensor(task_label_or_embedding, dtype=torch.bfloat16, device=device)
    elif torch.is_tensor(task_label_or_embedding):
        text_embedding = task_label_or_embedding.to(device=device, dtype=torch.bfloat16)
    else:
        raise TypeError(f"Unsupported task label/embedding type: {type(task_label_or_embedding)!r}")

    all_camera_images = [obs["wrist_image"], obs["primary_image"]]
    wrist_image_idx, image_idx = 0, 1
    all_camera_images = prepare_images_for_model(all_camera_images, cfg)

    proprio = None
    if cfg.use_proprio:
        proprio = obs["proprio"]
        if cfg.normalize_proprio:
            proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0)

    primary_image = all_camera_images[image_idx]
    wrist_image = all_camera_images[wrist_image_idx]
    blank_image = np.zeros_like(primary_image)
    blank_image_duplicated = duplicate_array(blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    wrist_image_duplicated = duplicate_array(wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_image_duplicated = duplicate_array(primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)

    image_sequence = []
    image_sequence.append(np.expand_dims(np.zeros_like(blank_image), axis=0))
    image_sequence.append(blank_image_duplicated.copy())
    current_proprio_latent_idx = 2
    image_sequence.append(wrist_image_duplicated.copy())
    current_wrist_image_latent_idx = 3
    image_sequence.append(primary_image_duplicated.copy())
    current_image_latent_idx = 4
    image_sequence.append(blank_image_duplicated.copy())
    action_latent_idx = 5
    image_sequence.append(blank_image_duplicated.copy())
    future_proprio_latent_idx = 6
    image_sequence.append(wrist_image_duplicated.copy())
    future_wrist_image_latent_idx = 7
    image_sequence.append(primary_image_duplicated.copy())
    future_image_latent_idx = 8
    image_sequence.append(blank_image_duplicated.copy())
    value_latent_idx = 9

    raw_image_sequence = np.concatenate(image_sequence, axis=0)
    raw_image_sequence = np.expand_dims(raw_image_sequence, axis=0)
    raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))
    raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))
    raw_image_sequence_t = torch.from_numpy(raw_image_sequence).to(device=device, dtype=torch.uint8)

    proprio_tensor = None
    if cfg.use_proprio:
        proprio_tensor = torch.from_numpy(proprio).reshape(batch_size, -1).to(device=device, dtype=torch.bfloat16)

    def idx_tensor(value: int) -> torch.Tensor:
        return torch.tensor([value] * batch_size, dtype=torch.int64, device=device)

    return {
        "dataset_name": "video_data",
        "video": raw_image_sequence_t,
        "t5_text_embeddings": text_embedding.repeat(batch_size, 1, 1).to(device=device, dtype=torch.bfloat16),
        "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros((batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16, device=device),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio_tensor,
        "current_proprio_latent_idx": idx_tensor(current_proprio_latent_idx),
        "current_wrist_image_latent_idx": idx_tensor(current_wrist_image_latent_idx),
        "current_wrist_image2_latent_idx": idx_tensor(-1),
        "current_image_latent_idx": idx_tensor(current_image_latent_idx),
        "current_image2_latent_idx": idx_tensor(-1),
        "action_latent_idx": idx_tensor(action_latent_idx),
        "future_proprio_latent_idx": idx_tensor(future_proprio_latent_idx),
        "future_wrist_image_latent_idx": idx_tensor(future_wrist_image_latent_idx),
        "future_wrist_image2_latent_idx": idx_tensor(-1),
        "future_image_latent_idx": idx_tensor(future_image_latent_idx),
        "future_image2_latent_idx": idx_tensor(-1),
        "value_latent_idx": idx_tensor(value_latent_idx),
    }


def state_shape_from_batch(model, data_batch):
    _, _, frames, height, width = data_batch["video"].shape
    return (
        model.config.state_ch,
        model.tokenizer.get_latent_num_frames(frames),
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    )


def initial_noise(model, data_batch, seed, sigma_max, device):
    batch_size = int(data_batch["video"].shape[0])
    shape = (batch_size,) + state_shape_from_batch(model, data_batch)
    return misc.arch_invariant_rand(shape, torch.float32, device, seed) * sigma_max


def make_solver_cfg(solver_option: str) -> SolverConfig:
    is_multistep = is_multi_step_fn_supported(solver_option)
    is_rk = is_runge_kutta_fn_supported(solver_option)
    if not (is_multistep or is_rk):
        raise ValueError(f"Unsupported solver option: {solver_option}")
    return SolverConfig(
        s_churn=0, s_t_max=float("inf"), s_t_min=0, s_noise=1,
        is_multi=is_multistep, rk=solver_option, multistep=solver_option,
    )


def extract_action_np(sample: torch.Tensor, data_batch: dict[str, Any], chunk_size: int) -> np.ndarray:
    action_idx = data_batch["action_latent_idx"].to(device=sample.device)
    action = extract_action_chunk_from_latent_sequence(
        sample, action_shape=(chunk_size, ACTION_DIM), action_indices=action_idx,
    )
    return action.detach().float().cpu().numpy()


def run_clean_branch(
    model, data_batch, x_sigma_max, args, device,
) -> np.ndarray:
    """Run clean-branch denoising and return the final action chunk (normalized)."""
    x0_fn, condition_target = model.get_x0_fn_from_batch(
        data_batch, guidance=args.guidance, is_negative_prompt=False,
        return_orig_clean_latent_frames=True,
    )

    solver_steps = args.num_denoising_steps - 1 if args.num_denoising_steps > 1 else 1
    sigmas_l = get_rev_ts(args.sigma_min, args.sigma_max, solver_steps, args.rho).to(device)
    solver_cfg = make_solver_cfg(args.solver_option)
    timestamps_cfg = SolverTimestampConfig(nfe=solver_steps, t_min=args.sigma_min, t_max=args.sigma_max, order=args.rho)
    sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True)

    in_dtype = x_sigma_max.dtype
    call_counter = [0]

    def recorded_x0_fn(x_state, sigma_b):
        call_counter[0] += 1
        return x0_fn(x_state.to(in_dtype), sigma_b.to(in_dtype)).to(torch.float64)

    with torch.no_grad():
        if args.num_denoising_steps > 1:
            denoised = differential_equation_solver(
                recorded_x0_fn, sigmas_l, sampler_cfg.solver, callback_fns=None,
            )(x_sigma_max.to(torch.float64))
            ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
            final_sample = recorded_x0_fn(denoised, sigmas_l[-1] * ones)
        else:
            ones = torch.ones(x_sigma_max.size(0), device=x_sigma_max.device, dtype=torch.float64)
            final_sample = recorded_x0_fn(x_sigma_max.to(torch.float64), sigmas_l[0] * ones)

    action = extract_action_np(final_sample, data_batch, args.chunk_size)
    return action  # normalized action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    # --- load phase helpers ---
    from run_phase2_angular_cosmos import (
        discover_pairs, first_observation, load_extra_t5, make_cfg, patch_checkpoint_db,
    )
    patch_checkpoint_db(pathlib.Path(args.policy_dir))

    summary_path = pathlib.Path(args.summary)
    summary, pairs = discover_pairs(summary_path, set(), {"preserved", "flipped"})
    print(f"Loaded {len(pairs)} pairs from {summary_path}")

    # deduplicate episodes (same episode appears across perturb conditions)
    seen_episodes = {}
    unique_pairs = []
    for p in pairs:
        ep = int(p["clean"]["episode"])
        if ep not in seen_episodes:
            seen_episodes[ep] = p
            unique_pairs.append(p)
    print(f"Unique episodes: {len(unique_pairs)}")

    if args.max_episodes:
        unique_pairs = unique_pairs[: args.max_episodes]

    # --- load model ---
    cfg = make_cfg(args)
    cfg.num_denoising_steps_action = args.num_denoising_steps
    args.chunk_size = cfg.chunk_size
    set_seed_everywhere(args.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    stats = load_dataset_stats(cfg.dataset_stats_path)

    model, _ = get_model(cfg)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # --- load expert store ---
    expert_store = ExpertActionStore(pathlib.Path(args.expert_actions_zip))
    task_text = args.task or summary.get("base_task", "")
    demos = expert_store.get_demo_episodes_for_task(task_text)
    print(f"Expert demos for task '{task_text}': {len(demos)} episodes")

    # --- compare clean vs expert ---
    results = []
    for p in unique_pairs:
        clean_ep = p["clean"]
        episode = int(clean_ep["episode"])
        instruction = clean_ep["language"]

        # get expert action for this episode (demo_ordinal = episode + offset)
        demo_ordinal = episode + args.expert_demo_offset
        if demo_ordinal < 0 or demo_ordinal >= len(demos):
            print(f"  ep={episode}: demo_ordinal={demo_ordinal} out of range [0, {len(demos)}), skipping")
            continue

        demo = demos[demo_ordinal]
        demo_ep_index = int(demo["episode_index"])
        expert_actions_raw = expert_store.read_actions(demo_ep_index)
        expert_actions_norm = normalize_actions_np(expert_actions_raw, stats)

        # take the first chunk (starting at frame 0)
        chunk_size = args.chunk_size
        if args.expert_start_frame + chunk_size > len(expert_actions_norm):
            # pad with last frame if needed
            available = expert_actions_norm[args.expert_start_frame:]
            padding = np.tile(expert_actions_norm[-1], (chunk_size - len(available), 1))
            expert_chunk = np.concatenate([available, padding], axis=0)
        else:
            expert_chunk = expert_actions_norm[args.expert_start_frame:args.expert_start_frame + chunk_size]

        # --- run clean branch ---
        obs = first_observation(clean_ep, args)
        data_batch = build_inference_data_batch(cfg, model, stats, obs, instruction, device)
        noise_seed = args.seed * 100000 + episode
        x_sigma = initial_noise(model, data_batch, noise_seed, args.sigma_max, device)
        clean_action = run_clean_branch(model, data_batch, x_sigma, args, device)

        mse = float(np.mean((clean_action - expert_chunk) ** 2))
        # also compare per-action-dimension
        mse_per_dim = np.mean((clean_action - expert_chunk) ** 2, axis=0).tolist()
        l2_clean = float(np.linalg.norm(clean_action))
        l2_expert = float(np.linalg.norm(expert_chunk))
        cosine_sim = float(np.dot(clean_action.ravel(), expert_chunk.ravel()) / (np.linalg.norm(clean_action) * np.linalg.norm(expert_chunk) + 1e-12))

        results.append({
            "episode": episode,
            "demo_ordinal": demo_ordinal,
            "demo_episode": demo_ep_index,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mse_per_dim": mse_per_dim,
            "l2_clean": l2_clean,
            "l2_expert": l2_expert,
            "cosine_similarity": cosine_sim,
            "chunk_size": chunk_size,
            "action_dim": ACTION_DIM,
        })
        print(f"  ep={episode:02d} demo_ep={demo_ep_index:06d}  MSE={mse:.6f}  RMSE={math.sqrt(mse):.4f}  cos_sim={cosine_sim:.4f}")

    # --- summary statistics ---
    mses = [r["mse"] for r in results]
    cos_sims = [r["cosine_similarity"] for r in results]
    print(f"\n=== Summary over {len(results)} episodes ===")
    print(f"MSE:   mean={np.mean(mses):.6f}  median={np.median(mses):.6f}  std={np.std(mses):.6f}")
    print(f"RMSE:  mean={np.mean([math.sqrt(m) for m in mses]):.4f}  median={math.sqrt(np.median(mses)):.4f}")
    print(f"CosSim: mean={np.mean(cos_sims):.4f}  median={np.median(cos_sims):.4f}  std={np.std(cos_sims):.4f}")

    # --- save ---
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "clean_vs_expert.json").open("w") as f:
        json.dump({
            "task": task_text,
            "num_episodes": len(results),
            "chunk_size": args.chunk_size,
            "expert_start_frame": args.expert_start_frame,
            "seed": args.seed,
            "num_denoising_steps": args.num_denoising_steps,
            "summary_mse_mean": float(np.mean(mses)),
            "summary_mse_median": float(np.median(mses)),
            "summary_cosine_sim_mean": float(np.mean(cos_sims)),
            "results": results,
        }, f, indent=2)
    print(f"Saved {out / 'clean_vs_expert.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    default_summary = str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json")
    parser.add_argument("--summary", default=default_summary)
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/clean_vs_expert_comparison"))
    parser.add_argument("--policy-dir", default=str(POLICY_DIR))
    parser.add_argument("--t5-extra-embeddings", default=str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    parser.add_argument("--expert-actions-zip", default=str(ROOT.parent / "libero_plus_10.zip"))
    parser.add_argument("--task", default="")
    parser.add_argument("--expert-demo-offset", type=int, default=0)
    parser.add_argument("--expert-start-frame", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-denoising-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--solver-option", default="2ab")
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-resolution", type=int, default=256)
    return parser


if __name__ == "__main__":
    main()
