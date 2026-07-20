#!/usr/bin/env python3
"""
Phase 5 — Jacobian via autograd: compute ∂a/∂h directly through the DiT.

Key: instead of perturbation-based finite differences or PLS on observed data,
we compute the exact Jacobian of the action output w.r.t. the VAE latent input
through a single DiT denoising step.

Method:
  1. Load model
  2. Prepare data batch (VAE-encode images → latent_state)
  3. Set requires_grad on the video portion of latent_state
  4. Build condition (condition.gt_frames = latent_state)
  5. Create noisy xt at a chosen sigma
  6. Run model.denoise(xt, sigma, condition) with gradient tracking
  7. Extract action from x0_pred
  8. Compute J = ∂a/∂(video_latent) via autograd
  9. SVD(J) → action-sensitive subspace

The Jacobian is computed at the VAE latent input level — before any DiT processing.
No inference_mode, no finite differences, no perturbations.
"""

from __future__ import annotations

import argparse, json, math, os, pathlib, pickle, sys
from types import SimpleNamespace
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
# Don't set PYOPENGL_PLATFORM — let MuJoCo handle it
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DETERMINISTIC", "True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    prepare_images_for_model, duplicate_array, rescale_proprio,
)
from cosmos_policy.utils.utils import set_seed_everywhere
# Constants from cosmos_utils
COSMOS_IMAGE_SIZE = 256
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4

CKPT_ROOT = ROOT.parent.parent / "checkpoints"
POLICY_DIR = CKPT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
BASE_MODEL_DIR = CKPT_ROOT / "Cosmos-Predict2-2B-Video2World"
EPS = 1e-8


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def patch_checkpoint(policy_dir: pathlib.Path):
    from cosmos_policy._src.imaginaire.utils import checkpoint_db
    orig = checkpoint_db.get_checkpoint_path
    base_pfx = "hf://nvidia/Cosmos-Predict2-2B-Video2World/"
    aloha_uri = "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
    def patched(uri):
        uri = str(uri).rstrip("/")
        if uri.startswith(base_pfx):
            return str(BASE_MODEL_DIR / uri[len(base_pfx):])
        if uri == aloha_uri:
            return str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        return orig(uri)
    checkpoint_db.get_checkpoint_path = patched


def make_cfg(args) -> SimpleNamespace:
    return SimpleNamespace(
        suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(args.policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1,
        use_proprio=True, flip_images=True,
        use_variance_scale=False, use_jpeg_compression=True,
        num_denoising_steps_action=args.num_denoising_steps,
        unnormalize_actions=True, normalize_proprio=True,
        dataset_stats_path=str(args.policy_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(args.policy_dir / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
    )


# ---------------------------------------------------------------------------
# Data preparation (following get_action logic)
# ---------------------------------------------------------------------------

def prepare_data_batch(cfg, model, obs, instruction, batch_size=1):
    """Build the data_batch that get_action feeds to the model."""
    # Preprocess images
    wrist_img = obs["wrist_image"]
    primary_img = obs["primary_image"]
    all_camera_images = [wrist_img, primary_img]
    all_camera_images = prepare_images_for_model(all_camera_images, cfg)
    IMAGE_IDX, WRIST_IMAGE_IDX = 1, 0

    # Build image sequence
    blank = np.zeros_like(all_camera_images[IMAGE_IDX])
    wrist_dup = duplicate_array(all_camera_images[WRIST_IMAGE_IDX], COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_dup = duplicate_array(all_camera_images[IMAGE_IDX], COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    blank_dup = duplicate_array(blank.copy(), COSMOS_TEMPORAL_COMPRESSION_FACTOR)

    # LIBERO latent indices (state_t=9):
    # 0=blank(1 frame), 1=proprio(4), 2=wrist(4), 3=primary(4), 4=action(4),
    # 5=future_proprio(4), 6=future_wrist(4), 7=future_primary(4), 8=value(4)
    # Total: 1 + 8*4 = 33 frames → VAE compresses to 9 latent frames
    image_sequence = [
        np.expand_dims(np.zeros_like(blank), axis=0),    # 0: 1 frame
        blank_dup,                                         # 1: 4 frames
        wrist_dup,                                         # 2: 4 frames
        primary_dup,                                       # 3: 4 frames
        blank_dup,                                         # 4: 4 frames
        blank_dup,                                         # 5: 4 frames
        wrist_dup.copy(),                                  # 6: 4 frames
        primary_dup.copy(),                                # 7: 4 frames
        blank_dup,                                         # 8: 4 frames
    ]

    raw_image_sequence = np.concatenate(image_sequence, axis=0)  # (T, H, W, C)
    raw_image_sequence = np.expand_dims(raw_image_sequence, 0)   # (1, T, H, W, C)
    raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))
    raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))  # (B, C, T, H, W)
    raw_image_sequence = torch.from_numpy(raw_image_sequence).to(dtype=torch.uint8)

    # T5 embedding
    text_embedding = cosmos_utils.get_t5_embedding_from_cache(instruction)
    text_embedding = text_embedding.repeat(batch_size, 1, 1)

    data_batch = {
        "dataset_name": "video_data",
        "video": raw_image_sequence,
        "t5_text_embeddings": text_embedding,
        "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16),
        "padding_mask": torch.zeros((batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": None,  # simplified for now
        "current_proprio_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64),
        "current_wrist_image_latent_idx": torch.tensor([2] * batch_size, dtype=torch.int64),
        "current_image_latent_idx": torch.tensor([3] * batch_size, dtype=torch.int64),
        "action_latent_idx": torch.tensor([4] * batch_size, dtype=torch.int64),
        "future_image_latent_idx": torch.tensor([7] * batch_size, dtype=torch.int64),
        "future_wrist_image_latent_idx": torch.tensor([6] * batch_size, dtype=torch.int64),
        "future_proprio_latent_idx": torch.tensor([5] * batch_size, dtype=torch.int64),
        "value_latent_idx": torch.tensor([8] * batch_size, dtype=torch.int64),
        "current_wrist_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64),
        "current_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64),
        "future_wrist_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64),
        "future_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64),
    }
    return data_batch


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian_at_denoise_step(
    model, data_batch, sigma_val: float = 80.0, device="cuda"
):
    """Compute ∂a/∂h for the video conditioning latents through one denoising step.

    Args:
        model: CosmosPolicyVideo2WorldModel
        data_batch: prepared data batch (on CPU)
        sigma_val: noise level for the denoising step (default 80 = first step)
        device: "cuda" or "cpu"

    Returns:
        J: Jacobian matrix [action_dim, video_latent_dim] as numpy array
        singular_values: SVD singular values
        right_singular_vectors: V^T from SVD [rank, video_latent_dim]
    """
    # Move data to device
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            data_batch[k] = v.to(device)

    # Get VAE-encoded latent and condition
    raw_state, latent_state, condition = model.get_data_and_condition(data_batch)

    # latent_state shape: [B, C, T, H, W] = [1, 16, 9, 28, 28]
    print(f"  latent_state shape: {latent_state.shape}")

    # Set requires_grad on video frames only (indices 2, 3 for wrist + primary)
    # We need to keep requires_grad on the full tensor but zero-out non-video gradients
    latent_state.requires_grad_(True)

    # Build noisy xt at given sigma
    B = latent_state.shape[0]
    sigma = torch.full((B, 1), sigma_val, device=device, dtype=torch.float32)
    sigma_B_1_T_1_1 = rearrange(sigma, "b t -> b 1 t 1 1")

    # Generate random noise
    noise = torch.randn_like(latent_state)
    xt = latent_state + noise * sigma_B_1_T_1_1

    # Run denoise
    denoise_output = model.denoise(xt, sigma.squeeze(-1), condition)
    x0_pred = denoise_output.x0  # [B, C, T, H, W]

    print(f"  x0_pred shape: {x0_pred.shape}")

    # Extract action from x0_pred (index 4, flatten to 112)
    action_latent = x0_pred[:, :, [4], :, :]  # [B, 16, 1, 28, 28]
    action_flat = action_latent.reshape(B, -1)  # [B, 12544]
    action = action_flat[:, :112]  # [B, 112] — first 112 values are the action

    print(f"  action shape: {action.shape} (from latent slot 4, first 112 elements)")

    # Compute Jacobian: ∂action/∂latent_state for video frames (indices 2,3)
    # J has shape [112, 2*16*28*28] = [112, 25088] (2 video frames × C × H × W)
    #
    # The model uses activation checkpointing which only allows one backward per
    # forward pass. So we run a fresh forward+backward for each action dimension.
    # 112 passes × ~3s each ≈ 5-6 minutes.
    action_dim = 112
    video_latent_dim = 2 * 16 * 28 * 28  # 2 frames × C × H × W
    J = torch.zeros(action_dim, video_latent_dim, device=device)

    for i in range(action_dim):
        if i % 20 == 0:
            print(f"  Jacobian row {i}/{action_dim}...")

        # Fresh forward: get data+condition, set requires_grad on video latents
        raw_i, latent_i, condition_i = model.get_data_and_condition(data_batch)
        latent_i = latent_i.detach().requires_grad_(True)
        condition_i.gt_frames = latent_i.to(**model.tensor_kwargs)

        sigma_t = torch.full((B, 1), sigma_val, device=device, dtype=torch.float32)
        sigma_r = rearrange(sigma_t, "b t -> b 1 t 1 1")
        noise_i = torch.randn_like(latent_i)
        xt_i = latent_i + noise_i * sigma_r

        denoise_i = model.denoise(xt_i, sigma_t.squeeze(-1), condition_i)
        x0_i = denoise_i.x0
        action_i = x0_i[:, :, [4], :, :].reshape(B, -1)[:, :112]

        # Backward for action element i
        grad_out = torch.zeros_like(action_i)
        grad_out[0, i] = 1.0
        action_i.backward(gradient=grad_out)

        # Extract gradient for video frames (indices 2, 3)
        grad_video = latent_i.grad[0, :, [2, 3], :, :]  # [16, 2, 28, 28]
        J[i, :] = grad_video.reshape(-1).detach()

    print(f"  Jacobian computed: {J.shape}")

    J_np = J.cpu().float().numpy()

    # SVD
    # J is [112, 25088], we need top right singular vectors
    # Compute J @ J^T (112 × 112) for efficiency
    M = J_np @ J_np.T  # [112, 112]
    eigenvals, U = np.linalg.eigh(M)
    eigenvals = eigenvals[::-1]
    U = U[:, ::-1]

    # Keep non-zero singular values
    S = np.sqrt(np.maximum(eigenvals, 0))
    mask = S > S.max() * 1e-6
    r = int(mask.sum())
    S = S[:r]
    U = U[:, :r]

    # Right singular vectors: V = J^T @ U @ diag(1/S)
    Vt = (J_np.T @ U @ np.diag(1.0 / (S + 1e-12))).T  # [r, 25088]

    print(f"  Jacobian effective rank: {r}")
    print(f"  Top-5 singular values: {S[:min(5, r)].round(6)}")

    return J_np, S, Vt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir", default=str(POLICY_DIR))
    p.add_argument("--output-dir", default=str(ROOT / "experiments/phase5_jacobian_autograd"))
    p.add_argument("--sigma", type=float, default=80.0,
                   help="Sigma value for denoising step (80=first, 1-20=mid, 0.002=last)")
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    args = p.parse_args()

    args.policy_dir = pathlib.Path(args.policy_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    patch_checkpoint(args.policy_dir)
    with open(str(args.policy_dir / "libero_t5_embeddings.pkl"), "rb") as f:
        cosmos_utils.t5_text_embeddings_cache.update(pickle.load(f))

    cfg = make_cfg(args)
    set_seed_everywhere(args.seed)
    stats = cosmos_utils.load_dataset_stats(cfg.dataset_stats_path)

    print("Loading model...")
    model, _ = cosmos_utils.get_model(cfg)
    model.eval()
    print(f"Model loaded: {len(model.net.blocks)} blocks")

    # Get observation from saved images (avoids LIBERO env / MagickWand dependency)
    obs_dir = ROOT / "experiments/phase5_jacobian_autograd/clean_obs"
    primary = np.load(obs_dir / "primary.npy")
    wrist = np.load(obs_dir / "wrist.npy")
    proprio = np.load(obs_dir / "proprio.npy")
    obs_dict = {"primary_image": primary, "wrist_image": wrist, "proprio": proprio}
    print(f"Loaded saved observation: primary={primary.shape}, wrist={wrist.shape}")

    # Prepare data
    data_batch = prepare_data_batch(
        cfg, model, obs_dict,
        "put the black bowl in the bottom drawer of the cabinet and close it"
    )

    # Compute Jacobian
    print(f"\n=== Computing Jacobian at sigma={args.sigma} ===")
    J, S, Vt = compute_jacobian_at_denoise_step(
        model, data_batch, sigma_val=args.sigma, device="cuda"
    )

    # Save
    np.save(out_dir / "jacobian.npy", J)
    np.save(out_dir / "singular_values.npy", S)
    np.save(out_dir / "right_singular_vectors.npy", Vt)

    summary = {
        "sigma": args.sigma,
        "action_dim": J.shape[0],
        "video_latent_dim": J.shape[1],
        "effective_rank": len(S),
        "singular_values": S.tolist(),
        "top5_singular_values": S[:min(5, len(S))].tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
