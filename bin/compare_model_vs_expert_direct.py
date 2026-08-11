#!/usr/bin/env python3
"""Direct comparison: model clean action vs expert action from the SAME demo.

Uses the first observation from each LIBERO-Plus expert demo as input,
runs one clean-branch denoising trajectory, extracts the action chunk,
and compares it against the demo's own expert action.

This avoids the Phase-1 episode-to-demo mapping problem entirely.
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
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import subprocess

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
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
import pyarrow.parquet as pq
import torch

from cosmos_policy._src.imaginaire.functional.multi_step import is_multi_step_fn_supported
from cosmos_policy._src.imaginaire.functional.runge_kutta import is_runge_kutta_fn_supported
from cosmos_policy._src.imaginaire.modules.res_sampler import (
    SamplerConfig, SolverConfig, SolverTimestampConfig,
    differential_equation_solver, get_rev_ts,
)
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE, COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    extract_action_chunk_from_latent_sequence,
    get_model, get_t5_embedding_from_cache,
    load_dataset_stats, prepare_images_for_model, rescale_proprio,
)
from cosmos_policy.utils.utils import duplicate_array, set_seed_everywhere

CHECKPOINT_ROOT = ROOT.parent.parent / "checkpoints"
POLICY_DIR = CHECKPOINT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
EPS = 1e-12


# ---------------------------------------------------------------------------
# LeRobot reader (filesystem-based, with video frame extraction)
# ---------------------------------------------------------------------------
class ExpertDemoReader:
    def __init__(self, root_dir: pathlib.Path):
        self.root = root_dir
        # Find the prefix inside the extracted directory
        meta = list(root_dir.rglob("meta/info.json"))
        if not meta:
            raise RuntimeError(f"meta/info.json not found under {root_dir}")
        self.prefix = meta[0].parent.parent  # e.g. root/.cache/.../libero_plus_10/
        self.info = json.loads((self.prefix / "meta/info.json").read_text())
        self.tasks = [json.loads(l) for l in (self.prefix / "meta/tasks.jsonl").read_text().splitlines()]
        self.episodes = [json.loads(l) for l in (self.prefix / "meta/episodes.jsonl").read_text().splitlines()]
        self.episodes_by_task: dict[str, list] = defaultdict(list)
        for ep in self.episodes:
            for t in ep.get("tasks", []):
                self.episodes_by_task[t].append(ep)
        for eps in self.episodes_by_task.values():
            eps.sort(key=lambda x: int(x["episode_index"]))

    def read_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        cs = int(self.info.get("chunks_size", 1000))
        chunk_id = episode_index // cs
        parquet_path = self.prefix / f"data/chunk-{chunk_id:03d}/episode_{episode_index:06d}.parquet"
        table = pq.read_table(str(parquet_path))
        out = {}
        for col_name in table.column_names:
            col_data = table.column(col_name)
            # Returns a ChunkedArray; convert to numpy via to_pylist for nested types
            if hasattr(col_data.type, 'value_type'):
                # fixed_size_list or list type: use to_pylist + np.array
                out[col_name] = np.array(col_data.combine_chunks().to_pylist(), dtype=np.float32)
            else:
                # scalar types
                out[col_name] = col_data.combine_chunks().to_numpy(zero_copy_only=False)
        return out

    def read_first_frame(self, episode_index: int, camera: str) -> np.ndarray:
        """Read the first frame from a video file via ffmpeg (AV1 support).
        camera: 'observation.images.front' or 'observation.images.wrist'
        """
        cs = int(self.info.get("chunks_size", 1000))
        chunk_id = episode_index // cs
        video_path = self.prefix / f"videos/chunk-{chunk_id:03d}/{camera}/episode_{episode_index:06d}.mp4"
        # Get expected dimensions from info.json
        cam_info = self.info["features"][camera]
        h, w = cam_info["shape"][0], cam_info["shape"][1]
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            raise RuntimeError(f"Cannot read first frame from {video_path}: {proc.stderr.decode()[-300:]}")
        raw = np.frombuffer(proc.stdout, dtype=np.uint8)
        return raw.reshape(h, w, 3)


def normalize_actions_np(actions: np.ndarray, stats: dict) -> np.ndarray:
    lo = np.asarray(stats["actions_min"], dtype=np.float32)
    hi = np.asarray(stats["actions_max"], dtype=np.float32)
    return (2.0 * (actions.astype(np.float32) - lo) / (hi - lo) - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
def build_data_batch(model, obs, task_embedding, cfg, stats, device):
    if isinstance(task_embedding, str):
        text_embedding = get_t5_embedding_from_cache(task_embedding)
    elif isinstance(task_embedding, np.ndarray):
        text_embedding = torch.tensor(task_embedding, dtype=torch.bfloat16, device=device)
    else:
        text_embedding = task_embedding.to(device=device, dtype=torch.bfloat16)

    images = [obs["wrist_image"], obs["primary_image"]]
    images = prepare_images_for_model(images, cfg)

    proprio = None
    if cfg.use_proprio:
        pr = obs["proprio"].astype(np.float32)
        if cfg.normalize_proprio:
            pr = rescale_proprio(pr, stats, non_negative_only=False, scale_multiplier=1.0)
        proprio = torch.from_numpy(pr).reshape(1, -1).to(device=device, dtype=torch.bfloat16)

    primary = images[1]
    wrist = images[0]
    blank = np.zeros_like(primary)
    bd = lambda x: duplicate_array(x.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    wd, pd, bld = bd(wrist), bd(primary), bd(blank)

    seq = [
        np.expand_dims(np.zeros_like(blank), 0),  # 0: empty
        bld,                                          # 1: proprio latent
        wd,                                           # 2: wrist latent
        pd,                                           # 3: image latent
        bld,                                          # 4: action latent
        bld,                                          # 5: future proprio
        wd,                                           # 6: future wrist
        pd,                                           # 7: future image
        bld,                                          # 8: value latent
    ]
    raw = np.concatenate(seq, 0)[None, ...]  # [1, 9*T, H, W, C]
    raw = np.tile(raw, (1, 1, 1, 1, 1))
    raw = np.transpose(raw, (0, 4, 1, 2, 3))  # [1, C, 9*T, H, W]
    raw_t = torch.from_numpy(raw).to(device=device, dtype=torch.uint8)

    def idx(v): return torch.tensor([v], dtype=torch.int64, device=device)

    B = raw_t.shape[0]
    return {
        "dataset_name": "video_data",
        "video": raw_t,
        "t5_text_embeddings": text_embedding.repeat(B, 1, 1).to(device=device, dtype=torch.bfloat16),
        "fps": torch.tensor([16] * B, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros((B, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16, device=device),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio,
        "current_proprio_latent_idx": idx(1),
        "current_wrist_image_latent_idx": idx(2),
        "current_wrist_image2_latent_idx": idx(-1),
        "current_image_latent_idx": idx(3),
        "current_image2_latent_idx": idx(-1),
        "action_latent_idx": idx(4),
        "future_proprio_latent_idx": idx(5),
        "future_wrist_image_latent_idx": idx(6),
        "future_wrist_image2_latent_idx": idx(-1),
        "future_image_latent_idx": idx(7),
        "future_image2_latent_idx": idx(-1),
        "value_latent_idx": idx(8),
    }


def state_shape(model, batch):
    _, _, frames, h, w = batch["video"].shape
    return (model.config.state_ch, model.tokenizer.get_latent_num_frames(frames),
            h // model.tokenizer.spatial_compression_factor,
            w // model.tokenizer.spatial_compression_factor)


def init_noise(model, batch, seed, sigma_max, device):
    shape = (batch["video"].shape[0],) + state_shape(model, batch)
    return misc.arch_invariant_rand(shape, torch.float32, device, seed) * sigma_max


def make_solver(solver_option):
    if not (is_multi_step_fn_supported(solver_option) or is_runge_kutta_fn_supported(solver_option)):
        raise ValueError(f"Unknown solver: {solver_option}")
    m = is_multi_step_fn_supported(solver_option)
    return SolverConfig(s_churn=0, s_t_max=float("inf"), s_t_min=0, s_noise=1,
                        is_multi=m, rk=solver_option, multistep=solver_option)


def extract_clean_action(model, batch, x_sigma, args, device) -> np.ndarray:
    x0_fn, _ = model.get_x0_fn_from_batch(batch, guidance=args.guidance, is_negative_prompt=False,
                                           return_orig_clean_latent_frames=True)

    steps = args.num_denoising_steps - 1 if args.num_denoising_steps > 1 else 1
    sig = get_rev_ts(args.sigma_min, args.sigma_max, steps, args.rho).to(device)
    solver = make_solver(args.solver_option)
    ts = SolverTimestampConfig(nfe=steps, t_min=args.sigma_min, t_max=args.sigma_max, order=args.rho)
    sc = SamplerConfig(solver=solver, timestamps=ts, sample_clean=True)
    dtype_in = x_sigma.dtype

    def fn(x, s):
        return x0_fn(x.to(dtype_in), s.to(dtype_in)).to(torch.float64)

    with torch.no_grad():
        if args.num_denoising_steps > 1:
            denoised = differential_equation_solver(fn, sig, sc.solver, callback_fns=None)(x_sigma.to(torch.float64))
            ones = torch.ones(denoised.size(0), device=device, dtype=torch.float64)
            final = fn(denoised, sig[-1] * ones)
        else:
            ones = torch.ones(x_sigma.size(0), device=device, dtype=torch.float64)
            final = fn(x_sigma.to(torch.float64), sig[0] * ones)

    idx = batch["action_latent_idx"].to(device)
    action = extract_action_chunk_from_latent_sequence(
        final, action_shape=(args.chunk_size, ACTION_DIM), action_indices=idx)
    return action.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    # Load model
    from run_phase2_angular_cosmos import make_cfg, patch_checkpoint_db, load_extra_t5
    args.policy_dir = pathlib.Path(args.policy_dir)
    patch_checkpoint_db(args.policy_dir)
    cfg = make_cfg(args)
    cfg.num_denoising_steps_action = args.num_denoising_steps
    args.chunk_size = cfg.chunk_size
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    stats = load_dataset_stats(cfg.dataset_stats_path)

    model, _ = get_model(cfg)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded, chunk_size={args.chunk_size}")

    # Load expert demos
    reader = ExpertDemoReader(pathlib.Path(args.expert_dir))
    demos = reader.episodes_by_task.get(args.task)
    if not demos:
        raise RuntimeError(f"Task {args.task!r} not found")
    print(f"Task: {args.task}, {len(demos)} demos")

    # Compare first N demos
    N = min(args.num_demos, len(demos))
    results = []

    for i in range(N):
        demo = demos[i]
        ep_idx = int(demo["episode_index"])
        data = reader.read_episode(ep_idx)  # action + state arrays

        # Read first video frames for both cameras
        primary = reader.read_first_frame(ep_idx, "observation.images.front")
        wrist = reader.read_first_frame(ep_idx, "observation.images.wrist")
        state = data["observation.state"][0]
        task_text = demo["tasks"][0]

        obs = {
            "primary_image": primary,
            "wrist_image": wrist,
            "proprio": state,
        }
        batch = build_data_batch(model, obs, task_text, cfg, stats, device)
        noise_seed = args.seed * 100000 + i  # same convention as run_denoise_hessian
        x_sigma = init_noise(model, batch, noise_seed, args.sigma_max, device)
        clean_action = extract_clean_action(model, batch, x_sigma, args, device)  # normalized

        # Expert action (normalized)
        raw_actions = data["action"].reshape(-1, ACTION_DIM).astype(np.float32)
        if args.expert_start_frame + args.chunk_size > len(raw_actions):
            available = raw_actions[args.expert_start_frame:]
            pad = np.tile(raw_actions[-1], (args.chunk_size - len(available), 1))
            expert_raw = np.concatenate([available, pad], axis=0)
        else:
            expert_raw = raw_actions[args.expert_start_frame:args.expert_start_frame + args.chunk_size]
        expert_norm = normalize_actions_np(expert_raw, stats)

        mse = float(np.mean((clean_action - expert_norm) ** 2))
        cos_sim = float(np.dot(clean_action.ravel(), expert_norm.ravel()) /
                        (np.linalg.norm(clean_action) * np.linalg.norm(expert_norm) + 1e-12))
        mse_per_dim = np.mean((clean_action - expert_norm) ** 2, axis=0).tolist()

        results.append({
            "demo_index": i,
            "episode_index": ep_idx,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "cosine_similarity": cos_sim,
            "l2_clean": float(np.linalg.norm(clean_action)),
            "l2_expert": float(np.linalg.norm(expert_norm)),
            "mse_per_dim": mse_per_dim,
        })
        print(f"[{i+1}/{N}] ep={ep_idx:06d}  MSE={mse:.6f}  RMSE={math.sqrt(mse):.4f}  cos_sim={cos_sim:.4f}  |clean|={np.linalg.norm(clean_action):.3f}  |expert|={np.linalg.norm(expert_norm):.3f}")

    # Summary
    mses = [r["mse"] for r in results]
    coss = [r["cosine_similarity"] for r in results]
    print(f"\n=== Summary ({len(results)} demos) ===")
    print(f"MSE:        mean={np.mean(mses):.6f}  median={np.median(mses):.6f}  std={np.std(mses):.6f}")
    print(f"RMSE:       mean={np.mean([math.sqrt(m) for m in mses]):.4f}  median={math.sqrt(np.median(mses)):.4f}")
    print(f"CosSim:     mean={np.mean(coss):.4f}  median={np.median(coss):.4f}  min={np.min(coss):.4f}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model_vs_expert_direct.json").open("w") as f:
        json.dump({
            "task": args.task, "num_demos": N, "chunk_size": args.chunk_size,
            "seed": args.seed, "num_denoising_steps": args.num_denoising_steps,
            "mse_mean": float(np.mean(mses)), "mse_median": float(np.median(mses)),
            "cosine_sim_mean": float(np.mean(coss)),
            "results": results,
        }, f, indent=2)
    print(f"Saved {out_dir / 'model_vs_expert_direct.json'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir", default=str(POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default=str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--expert-dir", default=str(ROOT / "dataset/libero_plus_10"))
    p.add_argument("--task", default="put the black bowl in the bottom drawer of the cabinet and close it")
    p.add_argument("--output-dir", default=str(ROOT / "experiments/model_vs_expert_direct"))
    p.add_argument("--num-demos", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-denoising-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--solver-option", default="2ab")
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--expert-start-frame", type=int, default=0)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--env-resolution", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    main()
