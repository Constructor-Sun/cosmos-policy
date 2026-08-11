#!/usr/bin/env python3
"""
Phase 9 — Gradient Alignment: measure alignment between perturbation shift δh
and loss gradient ∇_h L(h_pert) along the denoising trajectory.

Core idea
---------
Instead of computing SVD(∂a/∂h) to get a k-dimensional "action-sensitive subspace"
(as in phases 5–7), we directly compute the gradient of the action-matching loss

    L(h) = MSE( extract_action(pred_x0(h)),  a_clean )

w.r.t. hidden states at each site along each denoising call.  The gradient
∇_h L is a single vector that encodes the local steepest-ascent direction of
action error — it answers "in which direction should I nudge the latent to
maximally increase action deviation from the clean-success policy?"

We then measure

    alignment = cos(δh,  ∇_h L(h_pert))
    directional_derivative = δhᵀ ∇_h L(h_pert) / ‖δh‖

where δh = h_pert − h_clean is the perturbation-induced latent shift.

This approach fixes the fundamental issue in phases 5–7: those phases compute
SVD(J) where J = ∂a/∂h, which measures sensitivity of the *action output* (not
the *action error*).  The SVD treats all action dimensions equally, whereas the
loss gradient naturally weights each action dimension by its actual deviation
(a_i − a_clean,i).

What this script produces
-------------------------
For every (condition, episode, denoising_step, site):
  - δh  (perturbation shift)
  - ∇_h L  at both clean and perturbed hidden states
  - alignment  = cos(δh, ∇L)   ←  MAIN METRIC
  - directional derivative
  - gradient norm, loss values, and gradient-gradient alignment

Outputs:
  gradient_alignment_detail.csv   — per-pair × call × site rows
  gradient_alignment_summary.json — aggregated by condition × group

Usage
-----
  cd /data3/liu/exp/counterfactual/external/cosmos-policy

  .venv/bin/python bin/phase9_gradient_alignment.py \
    --output-dir experiments/phase9_gradient_alignment/kitchen_scene4_seed7
"""

from __future__ import annotations

import argparse, csv, gc, json, math, os, pathlib, pickle, site, sys, time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy-matplotlib")
os.environ.setdefault("DETERMINISTIC", "True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(
    os.environ.get("LIBERO_PLUS_PATH", str(ROOT.parent / "LIBERO-plus"))
)
for item in (str(LIBERO_PLUS), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)


def preload_wand_imagemagick() -> None:
    lib_dir = pathlib.Path(sys.prefix) / "lib"
    if (lib_dir / "libMagickWand-7.Q16HDRI.so").exists():
        os.environ.setdefault("MAGICK_HOME", sys.prefix)
        os.environ.setdefault("WAND_MAGICK_LIBRARY_SUFFIX", "-7.Q16HDRI")
    try:
        from wand.api import library as _wand_library  # noqa: F401
    except ImportError:
        return


preload_wand_imagemagick()

import numpy as np
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
    get_model, get_t5_embedding_from_cache, load_dataset_stats,
    prepare_images_for_model, rescale_proprio,
)
from cosmos_policy.utils.utils import duplicate_array, set_seed_everywhere

CHECKPOINT_ROOT = ROOT.parent.parent / "checkpoints"
POLICY_DIR = CHECKPOINT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
BASE_MODEL_DIR = CHECKPOINT_ROOT / "Cosmos-Predict2-2B-Video2World"
EPS = 1e-12

# ── CSV field definitions ──────────────────────────────────────────────────

DETAIL_FIELDS = [
    "condition", "episode", "group",
    "denoise_call_index", "denoise_num_calls", "sigma_clean", "sigma_pert",
    "site", "layer",
    "norm_delta_h",
    "norm_grad_clean", "norm_grad_pert",
    "alignment_clean",                   # cos(δh,  ∇L_clean(h_clean))
    "alignment_pert",                    # cos(δh,  ∇L_pert(h_pert))   ← MAIN
    "directional_deriv_clean",           # δhᵀ ∇L_clean / ‖δh‖
    "directional_deriv_pert",            # δhᵀ ∇L_pert  / ‖δh‖        ← MAIN
    "loss_clean", "loss_pert",
    "grad_norm_ratio",                   # ‖∇L_pert‖ / ‖∇L_clean‖
    "grad_grad_alignment",               # cos(∇L_clean,  ∇L_pert)
    "clean_task_name", "pert_task_name", "instruction_mode",
]


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CapturedCall:
    call_index: int
    sigma: float
    x_in: torch.Tensor                              # [1, C, T, H, W]
    hidden: dict[int, torch.Tensor]                 # layer → tensor


@dataclass
class BranchResult:
    final_sample: torch.Tensor
    condition_target: torch.Tensor
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    captures: dict[int, CapturedCall]
    data_batch: dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# Model loading  (adapted from run_denoise_hessian_cosmos.py)
# ══════════════════════════════════════════════════════════════════════════════

def patch_checkpoint(policy_dir: pathlib.Path) -> None:
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


def load_phase2_helpers():
    from run_phase2_angular_cosmos import (
        discover_pairs, first_observation, load_extra_t5,
        make_cfg, patch_checkpoint_db,
    )
    return discover_pairs, first_observation, load_extra_t5, make_cfg, patch_checkpoint_db


# ══════════════════════════════════════════════════════════════════════════════
# Layer / site / call selection
# ══════════════════════════════════════════════════════════════════════════════

def parse_layers(spec: list[str], n_layers: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in spec:
        if item in {"vae", "input"}:
            continue
        if item in {"0", "block0", "layer0"}:
            out["layer0"] = 0
        elif item in {"mid", "middle"}:
            out["mid"] = n_layers // 2
        elif item in {"last", "final"}:
            out["last"] = n_layers - 1
        else:
            layer = int(item)
            out[f"layer{layer}"] = layer
    return {name: layer for name, layer in out.items() if 0 <= layer < n_layers}


def parse_sites(spec: list[str], layer_map: dict[str, int]) -> list[str]:
    if spec == ["all"]:
        return ["vae", *layer_map.keys()]
    sites = []
    for item in spec:
        if item in {"vae", "input"}:
            sites.append("vae")
        elif item in {"0", "block0", "layer0"}:
            sites.append("layer0")
        elif item in {"mid", "middle"}:
            sites.append("mid")
        elif item in {"last", "final"}:
            sites.append("last")
        elif item.startswith("layer"):
            sites.append(item)
        else:
            sites.append(f"layer{int(item)}")
    out = []
    for site in sites:
        if site == "vae" or site in layer_map:
            out.append(site)
    return sorted(set(out), key=lambda x: (-1 if x == "vae" else layer_map[x]))


def select_call_indices(num_calls: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted({idx for idx in explicit if 0 <= idx < num_calls})
    if num_calls <= 1:
        return [0]
    return sorted({
        int(math.floor((num_calls - 1) * q / 4.0 + 0.5)) for q in range(5)
    })


# ══════════════════════════════════════════════════════════════════════════════
# Data batch building  (from run_denoise_hessian_cosmos.py)
# ══════════════════════════════════════════════════════════════════════════════

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
        raise ValueError(f"This script implements LIBERO only, got suite={cfg.suite}")

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
            proprio = rescale_proprio(
                proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0,
            )

    primary_image = all_camera_images[image_idx]
    wrist_image = all_camera_images[wrist_image_idx]
    blank_image = np.zeros_like(primary_image)
    blank_image_duplicated = duplicate_array(
        blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    )
    wrist_image_duplicated = duplicate_array(
        wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    )
    primary_image_duplicated = duplicate_array(
        primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    )

    image_sequence = []
    cur = 0
    image_sequence.append(np.expand_dims(np.zeros_like(blank_image), axis=0))  # 0
    cur += 1
    image_sequence.append(blank_image_duplicated.copy())                       # 1
    current_proprio_latent_idx = cur; cur += 1
    image_sequence.append(wrist_image_duplicated.copy())                       # 2
    current_wrist_image_latent_idx = cur; cur += 1
    current_wrist_image2_latent_idx = -1
    image_sequence.append(primary_image_duplicated.copy())                     # 3
    current_image_latent_idx = cur; cur += 1
    current_image2_latent_idx = -1
    image_sequence.append(blank_image_duplicated.copy())                       # 4
    action_latent_idx = cur; cur += 1
    image_sequence.append(blank_image_duplicated.copy())                       # 5
    future_proprio_latent_idx = cur; cur += 1
    image_sequence.append(wrist_image_duplicated.copy())                       # 6
    future_wrist_image_latent_idx = cur; cur += 1
    future_wrist_image2_latent_idx = -1
    image_sequence.append(primary_image_duplicated.copy())                     # 7
    future_image_latent_idx = cur; cur += 1
    future_image2_latent_idx = -1
    image_sequence.append(blank_image_duplicated.copy())                       # 8
    value_latent_idx = cur

    raw_image_sequence = np.concatenate(image_sequence, axis=0)
    raw_image_sequence = np.expand_dims(raw_image_sequence, axis=0)
    raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))
    raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))
    raw_image_sequence_t = torch.from_numpy(raw_image_sequence).to(
        device=device, dtype=torch.uint8,
    )

    proprio_tensor = None
    if cfg.use_proprio:
        proprio_tensor = torch.from_numpy(proprio).reshape(
            batch_size, -1,
        ).to(device=device, dtype=torch.bfloat16)

    def idx_tensor(value: int) -> torch.Tensor:
        return torch.tensor([value] * batch_size, dtype=torch.int64, device=device)

    return {
        "dataset_name": "video_data",
        "video": raw_image_sequence_t,
        "t5_text_embeddings": text_embedding.repeat(batch_size, 1, 1).to(
            device=device, dtype=torch.bfloat16,
        ),
        "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(
            (batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE),
            dtype=torch.bfloat16, device=device,
        ),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio_tensor,
        "current_proprio_latent_idx": idx_tensor(current_proprio_latent_idx),
        "current_wrist_image_latent_idx": idx_tensor(current_wrist_image_latent_idx),
        "current_wrist_image2_latent_idx": idx_tensor(current_wrist_image2_latent_idx),
        "current_image_latent_idx": idx_tensor(current_image_latent_idx),
        "current_image2_latent_idx": idx_tensor(current_image2_latent_idx),
        "action_latent_idx": idx_tensor(action_latent_idx),
        "future_proprio_latent_idx": idx_tensor(future_proprio_latent_idx),
        "future_wrist_image_latent_idx": idx_tensor(future_wrist_image_latent_idx),
        "future_wrist_image2_latent_idx": idx_tensor(future_wrist_image2_latent_idx),
        "future_image_latent_idx": idx_tensor(future_image_latent_idx),
        "future_image2_latent_idx": idx_tensor(future_image2_latent_idx),
        "value_latent_idx": idx_tensor(value_latent_idx),
    }


def state_shape_from_batch(
    model: torch.nn.Module, data_batch: dict[str, Any],
) -> tuple[int, int, int, int]:
    _, _, frames, height, width = data_batch["video"].shape
    return (
        model.config.state_ch,
        model.tokenizer.get_latent_num_frames(frames),
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    )


def initial_noise(
    model: torch.nn.Module, data_batch: dict[str, Any],
    seed: int, sigma_max: float, device: torch.device,
) -> torch.Tensor:
    batch_size = int(data_batch["video"].shape[0])
    shape = (batch_size,) + state_shape_from_batch(model, data_batch)
    return misc.arch_invariant_rand(shape, torch.float32, device, seed) * sigma_max


# ══════════════════════════════════════════════════════════════════════════════
# Solver
# ══════════════════════════════════════════════════════════════════════════════

def make_solver_cfg(solver_option: str) -> SolverConfig:
    is_multistep = is_multi_step_fn_supported(solver_option)
    is_rk = is_runge_kutta_fn_supported(solver_option)
    if not (is_multistep or is_rk):
        raise ValueError(f"Unsupported solver option: {solver_option}")
    return SolverConfig(
        s_churn=0, s_t_max=float("inf"), s_t_min=0, s_noise=1,
        is_multi=is_multistep, rk=solver_option, multistep=solver_option,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory capture  (from run_denoise_hessian_cosmos.py)
# ══════════════════════════════════════════════════════════════════════════════

class DenoiseCapture:
    def __init__(
        self, model: torch.nn.Module,
        selected_calls: set[int], layers: dict[str, int],
    ):
        self.selected_calls = selected_calls
        self.layers = set(layers.values())
        self.current_call_index = -1
        self.current_sigma = float("nan")
        self.captures: dict[int, CapturedCall] = {}
        self.handles = []
        for layer in sorted(self.layers):
            self.handles.append(
                model.net.blocks[layer].register_forward_hook(self._hook(layer))
            )

    def start_call(
        self, call_index: int, sigma: torch.Tensor, x_in: torch.Tensor,
    ) -> None:
        self.current_call_index = call_index
        sigma_value = float(sigma.reshape(-1)[0].detach().cpu())
        self.current_sigma = sigma_value
        if call_index in self.selected_calls:
            self.captures[call_index] = CapturedCall(
                call_index=call_index, sigma=sigma_value,
                x_in=x_in.detach().float().cpu(), hidden={},
            )

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self.current_call_index not in self.selected_calls:
                return
            x = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(x):
                return
            self.captures[self.current_call_index].hidden[layer] = (
                x.detach().float().cpu()
            )
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def freeze_model_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def run_branch_trajectory(
    model: torch.nn.Module,
    data_batch: dict[str, Any],
    x_sigma_max: torch.Tensor,
    selected_calls: set[int],
    layers: dict[str, int],
    args,
) -> BranchResult:
    x0_fn, condition_target = model.get_x0_fn_from_batch(
        data_batch, guidance=args.guidance,
        is_negative_prompt=False, return_orig_clean_latent_frames=True,
    )
    in_dtype = x_sigma_max.dtype
    solver_steps = args.num_denoising_steps - 1 if args.num_denoising_steps > 1 else 1
    sigmas_l = get_rev_ts(
        args.sigma_min, args.sigma_max, solver_steps, args.rho,
    ).to(x_sigma_max.device)
    solver_cfg = make_solver_cfg(args.solver_option)
    timestamps_cfg = SolverTimestampConfig(
        nfe=solver_steps, t_min=args.sigma_min, t_max=args.sigma_max, order=args.rho,
    )
    sampler_cfg = SamplerConfig(
        solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True,
    )

    capture = DenoiseCapture(model, selected_calls, layers)
    call_index = {"value": -1}

    def recorded_x0_fn(x_state: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
        call_index["value"] += 1
        capture.start_call(call_index["value"], sigma_b, x_state)
        return x0_fn(x_state.to(in_dtype), sigma_b.to(in_dtype)).to(torch.float64)

    try:
        with torch.no_grad():
            if args.num_denoising_steps > 1:
                denoised = differential_equation_solver(
                    recorded_x0_fn, sigmas_l, sampler_cfg.solver, callback_fns=None,
                )(x_sigma_max.to(torch.float64))
                ones = torch.ones(
                    denoised.size(0), device=denoised.device, dtype=denoised.dtype,
                )
                final_sample = recorded_x0_fn(denoised, sigmas_l[-1] * ones)
            else:
                ones = torch.ones(
                    x_sigma_max.size(0), device=x_sigma_max.device, dtype=torch.float64,
                )
                final_sample = recorded_x0_fn(
                    x_sigma_max.to(torch.float64), sigmas_l[0] * ones,
                )
    finally:
        capture.close()

    return BranchResult(
        final_sample=final_sample.detach().float().cpu(),
        condition_target=condition_target.detach().float().cpu(),
        x0_fn=x0_fn, captures=capture.captures, data_batch=data_batch,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Action extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def action_tensor_from_latent(
    sample: torch.Tensor, data_batch: dict[str, Any], chunk_size: int,
) -> torch.Tensor:
    action_idx = data_batch["action_latent_idx"].to(device=sample.device)
    return extract_action_chunk_from_latent_sequence(
        sample, action_shape=(chunk_size, ACTION_DIM), action_indices=action_idx,
    ).to(torch.float32)


def action_from_latent_np(
    sample: torch.Tensor, data_batch: dict[str, Any], chunk_size: int,
) -> np.ndarray:
    return action_tensor_from_latent(sample, data_batch, chunk_size).detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Loss construction  (simplified — clean_final_action target only)
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def replace_block_output(block: torch.nn.Module, replacement: torch.Tensor):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (replacement, *output[1:])
        return replacement
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def make_loss_fn(
    model: torch.nn.Module,
    branch: BranchResult,
    capture: CapturedCall,
    site: str,
    layer: int | None,
    target_action: torch.Tensor,          # [1, chunk_size, ACTION_DIM]
    chunk_size: int,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """Build a loss function L(h) = MSE(action_from_pred_x0(h), target_action).

    Returns (loss_fn, x0) where loss_fn takes a tensor and returns a scalar.
    """
    sigma = torch.full(
        (capture.x_in.shape[0],), capture.sigma, device=device, dtype=torch.float32,
    )
    x_in = capture.x_in.to(device=device, dtype=torch.float32)
    target = target_action.to(device=device, dtype=torch.float32)

    def action_mse(pred_x0: torch.Tensor) -> torch.Tensor:
        pred_action = action_tensor_from_latent(pred_x0, branch.data_batch, chunk_size)
        return ((pred_action - target) ** 2).mean()

    if site == "vae":
        x0_tensor = x_in

        def fn(x_var: torch.Tensor) -> torch.Tensor:
            pred = branch.x0_fn(x_var, sigma).float()
            return action_mse(pred)

        return fn, x0_tensor

    if layer is None:
        raise ValueError(f"Layer site {site} has no layer")
    if layer not in capture.hidden:
        raise RuntimeError(
            f"Missing hidden capture for layer {layer} at call {capture.call_index}"
        )
    h0_tensor = capture.hidden[layer].to(device=device, dtype=torch.float32)

    def fn(h_var: torch.Tensor) -> torch.Tensor:
        replacement = h_var.to(dtype=next(model.parameters()).dtype)
        with replace_block_output(model.net.blocks[layer], replacement):
            pred = branch.x0_fn(x_in.detach(), sigma).float()
        return action_mse(pred)

    return fn, h0_tensor


# ══════════════════════════════════════════════════════════════════════════════
# Gradient computation  (NEW — the core of this phase)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gradient(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Compute ∇_x loss_fn(x) at x = x0.

    Returns (gradient, loss_value).  Gradient is detached and on CPU.
    """
    x = x0.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    loss = loss_fn(x)
    grad = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)[0]
    grad_cpu = grad.detach().cpu()
    loss_val = float(loss.detach().cpu())
    del x, loss, grad
    return grad_cpu, loss_val


def compute_alignment(
    delta_h: torch.Tensor,          # flattened or any shape
    grad_pert: torch.Tensor,        # same shape as delta_h
    grad_clean: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute alignment metrics between shift δh and loss gradients.

    Args:
        delta_h: h_pert − h_clean
        grad_pert: ∇_h L at the perturbed hidden state
        grad_clean: ∇_h L at the clean hidden state (optional)

    Returns dict with alignment_pert, directional_deriv_pert, etc.
    """
    dh_flat = delta_h.reshape(-1).float()
    gp_flat = grad_pert.reshape(-1).float()

    norm_dh = float(torch.linalg.vector_norm(dh_flat))
    norm_gp = float(torch.linalg.vector_norm(gp_flat))

    result: dict[str, float] = {
        "norm_delta_h": norm_dh,
        "norm_grad_pert": norm_gp,
    }

    if norm_dh > EPS and norm_gp > EPS:
        dot_pert = float(torch.dot(dh_flat, gp_flat))
        result["alignment_pert"] = dot_pert / (norm_dh * norm_gp)
        result["directional_deriv_pert"] = dot_pert / norm_dh
    else:
        result["alignment_pert"] = float("nan")
        result["directional_deriv_pert"] = float("nan")

    if grad_clean is not None:
        gc_flat = grad_clean.reshape(-1).float()
        norm_gc = float(torch.linalg.vector_norm(gc_flat))
        result["norm_grad_clean"] = norm_gc

        if norm_dh > EPS and norm_gc > EPS:
            dot_clean = float(torch.dot(dh_flat, gc_flat))
            result["alignment_clean"] = dot_clean / (norm_dh * norm_gc)
            result["directional_deriv_clean"] = dot_clean / norm_dh
        else:
            result["alignment_clean"] = float("nan")
            result["directional_deriv_clean"] = float("nan")

        if norm_gc > EPS and norm_gp > EPS:
            result["grad_grad_alignment"] = float(
                torch.dot(gc_flat, gp_flat) / (norm_gc * norm_gp)
            )
        else:
            result["grad_grad_alignment"] = float("nan")

        result["grad_norm_ratio"] = (
            norm_gp / norm_gc if norm_gc > EPS else float("nan")
        )
    else:
        result["norm_grad_clean"] = float("nan")
        result["alignment_clean"] = float("nan")
        result["directional_deriv_clean"] = float("nan")
        result["grad_grad_alignment"] = float("nan")
        result["grad_norm_ratio"] = float("nan")

    return result


def compute_delta_h(
    capture_clean: CapturedCall,
    capture_pert: CapturedCall,
    site: str,
    layer: int | None,
) -> torch.Tensor:
    """Extract δh = h_pert − h_clean for a given site."""
    if site == "vae":
        h_clean = capture_clean.x_in
        h_pert = capture_pert.x_in
    else:
        if layer is None:
            raise ValueError(f"Layer required for site {site}")
        h_clean = capture_clean.hidden[layer]
        h_pert = capture_pert.hidden[layer]
    return (h_pert - h_clean).float()


# ══════════════════════════════════════════════════════════════════════════════
# CSVs and summaries
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(path: pathlib.Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    arr = np.array(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {"n": len(values), "mean": float("nan"), "std": float("nan")}
    return {
        "n": len(values),
        "n_finite": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = build_parser().parse_args()

    discover_pairs, first_observation, load_extra_t5, make_cfg, patch_checkpoint_db = \
        load_phase2_helpers()

    start = time.time()
    args.policy_dir = pathlib.Path(args.policy_dir)
    patch_checkpoint_db(args.policy_dir)
    patch_checkpoint(args.policy_dir)

    # ── Discover pairs ──────────────────────────────────────────────────
    summary_path = pathlib.Path(args.summary)
    summary, pairs = discover_pairs(
        summary_path, set(args.conditions or []), set(args.groups),
    )
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Discovered {len(pairs)} Phase-1 pairs from {summary_path}", flush=True)
    if not pairs:
        print("No pairs found — exiting.")
        return

    # ── Model ───────────────────────────────────────────────────────────
    cfg = make_cfg(args)
    cfg.num_denoising_steps_action = args.num_denoising_steps
    chunk_size = cfg.chunk_size
    set_seed_everywhere(args.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    stats = load_dataset_stats(cfg.dataset_stats_path)

    model, _ = get_model(cfg)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    freeze_model_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model frozen; trainable_params={trainable}", flush=True)

    layer_map = parse_layers(args.layers, len(model.net.blocks))
    sites = parse_sites(args.sites, layer_map)
    selected_calls = select_call_indices(args.num_denoising_steps, args.step_indices)
    print(f"Sites: {sites}", flush=True)
    print(f"Layer map: {layer_map}", flush=True)
    print(f"Selected denoise calls: {selected_calls} / num_calls={args.num_denoising_steps}", flush=True)

    # ── Per-pair loop ───────────────────────────────────────────────────
    out_root = pathlib.Path(args.output_dir)
    detail_rows: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(pairs, 1):
        clean_ep = pair["clean"]
        pert_ep = pair["pert"]
        condition = pair["condition"]
        group = pair["group"]
        episode = int(pert_ep["episode"])
        print(f"\n[{pair_idx}/{len(pairs)}] {condition}/ep{episode:02d} {group}", flush=True)

        # Observations
        clean_obs = first_observation(clean_ep, args)
        pert_obs = (
            clean_obs if condition == "language_instructions"
            else first_observation(pert_ep, args)
        )
        clean_instr = clean_ep["language"]
        pert_instr = (
            pert_ep["language"]
            if pert_ep.get("instruction_mode") == "strict"
            else clean_instr
        )

        # Data batches
        clean_batch = build_inference_data_batch(
            cfg, model, stats, clean_obs, clean_instr, device,
        )
        pert_batch = build_inference_data_batch(
            cfg, model, stats, pert_obs, pert_instr, device,
        )

        # Initial noise
        noise_seed = (
            args.seed * 100000 + episode if args.shared_noise
            else args.seed * 100000 + pair_idx
        )
        x_sigma_clean = initial_noise(
            model, clean_batch, noise_seed, args.sigma_max, device,
        )
        x_sigma_pert = (
            x_sigma_clean.clone() if args.shared_noise
            else initial_noise(model, pert_batch, noise_seed + 17, args.sigma_max, device)
        )

        # ── Run trajectories ──────────────────────────────────────────
        t0_branch = time.time()
        branch_clean = run_branch_trajectory(
            model, clean_batch, x_sigma_clean, set(selected_calls), layer_map, args,
        )
        branch_pert = run_branch_trajectory(
            model, pert_batch, x_sigma_pert, set(selected_calls), layer_map, args,
        )
        print(f"  Trajectories: {time.time() - t0_branch:.1f}s", flush=True)

        # ── Target action ──────────────────────────────────────────────
        clean_action_np = action_from_latent_np(
            branch_clean.final_sample.to(device), clean_batch, chunk_size,
        )
        pert_action_np = action_from_latent_np(
            branch_pert.final_sample.to(device), pert_batch, chunk_size,
        )
        a_clean = torch.from_numpy(clean_action_np.astype(np.float32))
        action_error = float(
            np.linalg.norm(pert_action_np.reshape(-1) - clean_action_np.reshape(-1))
            / (np.linalg.norm(clean_action_np.reshape(-1)) + EPS)
        )

        pair_records.append({
            "condition": condition, "episode": episode, "group": group,
            "action_error_from_final_latent": action_error,
            "clean_success": bool(clean_ep["success"]),
            "pert_success": bool(pert_ep["success"]),
            "pert_action_mse_to_clean_final": float(
                np.mean((pert_action_np - clean_action_np) ** 2)
            ),
        })

        # ── Per-call × per-site gradient alignment ────────────────────
        for call_idx in selected_calls:
            cap_clean = branch_clean.captures.get(call_idx)
            cap_pert = branch_pert.captures.get(call_idx)
            if cap_clean is None or cap_pert is None:
                print(f"  Missing capture call={call_idx}; skipping", flush=True)
                continue

            for site in sites:
                layer = None if site == "vae" else layer_map[site]
                t0_site = time.time()

                try:
                    # 1. Compute δh
                    delta_h = compute_delta_h(cap_clean, cap_pert, site, layer)

                    # 2. Build loss functions
                    loss_fn_clean, h0_clean = make_loss_fn(
                        model, branch_clean, cap_clean, site, layer,
                        a_clean, chunk_size, device,
                    )
                    loss_fn_pert, h0_pert = make_loss_fn(
                        model, branch_pert, cap_pert, site, layer,
                        a_clean, chunk_size, device,
                    )

                    # 3. Compute gradients
                    grad_clean, loss_clean = compute_gradient(loss_fn_clean, h0_clean, device)
                    grad_pert, loss_pert = compute_gradient(loss_fn_pert, h0_pert, device)

                    # 4. Compute alignment
                    al = compute_alignment(delta_h, grad_pert, grad_clean)

                    # 5. Record
                    detail_rows.append({
                        "condition": condition,
                        "episode": episode,
                        "group": group,
                        "denoise_call_index": call_idx,
                        "denoise_num_calls": args.num_denoising_steps,
                        "sigma_clean": cap_clean.sigma,
                        "sigma_pert": cap_pert.sigma,
                        "site": site,
                        "layer": "" if layer is None else layer,
                        "norm_delta_h": al["norm_delta_h"],
                        "norm_grad_clean": al.get("norm_grad_clean", float("nan")),
                        "norm_grad_pert": al["norm_grad_pert"],
                        "alignment_clean": al.get("alignment_clean", float("nan")),
                        "alignment_pert": al["alignment_pert"],
                        "directional_deriv_clean": al.get("directional_deriv_clean", float("nan")),
                        "directional_deriv_pert": al["directional_deriv_pert"],
                        "loss_clean": loss_clean,
                        "loss_pert": loss_pert,
                        "grad_norm_ratio": al.get("grad_norm_ratio", float("nan")),
                        "grad_grad_alignment": al.get("grad_grad_alignment", float("nan")),
                        "clean_task_name": clean_ep["task_name"],
                        "pert_task_name": pert_ep["task_name"],
                        "instruction_mode": pert_ep.get("instruction_mode", "task"),
                    })

                    dt = time.time() - t0_site
                    if args.verbose:
                        print(
                            f"    call={call_idx} site={site} "
                            f"align_pert={al['alignment_pert']:.4f}  "
                            f"norm_δh={al['norm_delta_h']:.2e}  "
                            f"({dt:.1f}s)",
                            flush=True,
                        )

                except Exception as exc:
                    print(f"    ERROR call={call_idx} site={site}: {exc}", flush=True)
                    continue

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        # Write incrementally
        write_csv(out_root / "gradient_alignment_detail.csv", DETAIL_FIELDS, detail_rows)

    # ── Aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Gradient Alignment Summary")
    print(f"{'='*70}")

    # Per-condition × group
    for metric in ["alignment_pert", "directional_deriv_pert",
                   "alignment_clean", "grad_grad_alignment", "grad_norm_ratio"]:
        print(f"\n── {metric} ──")
        print(f"  {'Site':10s} {'Call':>5s} {'Condition':28s} {'Group':10s} "
              f"{'N':>4s} {'Mean':>10s} {'Std':>10s}")
        print(f"  {'-'*80}")

        for site in sites:
            for call_idx in selected_calls:
                rows_site_call = [
                    r for r in detail_rows
                    if r["site"] == site and r["denoise_call_index"] == call_idx
                ]
                for cond in sorted({r["condition"] for r in rows_site_call}):
                    for grp in ["flipped", "preserved"]:
                        vals = [
                            r[metric] for r in rows_site_call
                            if r["condition"] == cond and r["group"] == grp
                            and not math.isnan(r[metric])
                        ]
                        if not vals:
                            continue
                        print(
                            f"  {site:10s} {call_idx:>5d} {cond:28s} {grp:10s} "
                            f"{len(vals):>4d} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}"
                        )

    # ── Save final outputs ──────────────────────────────────────────────────
    write_csv(out_root / "gradient_alignment_detail.csv", DETAIL_FIELDS, detail_rows)

    # Build structured summary
    summary_out: dict[str, Any] = {
        "summary": str(summary_path),
        "suite": summary.get("suite"),
        "base_task": summary.get("base_task"),
        "num_pairs": len(pairs),
        "num_denoising_steps": args.num_denoising_steps,
        "selected_calls": selected_calls,
        "sites": sites,
        "layer_map": layer_map,
        "shared_noise": bool(args.shared_noise),
        "elapsed_s": time.time() - start,
        "pairs": pair_records,
        "by_site_call": {},
    }

    for site in sites:
        summary_out["by_site_call"][site] = {}
        for call_idx in selected_calls:
            rows_sc = [
                r for r in detail_rows
                if r["site"] == site and r["denoise_call_index"] == call_idx
            ]
            entry: dict[str, Any] = {}
            for cond in sorted({r["condition"] for r in rows_sc}):
                for grp in ["flipped", "preserved"]:
                    key = f"{cond}/{grp}"
                    vals_ap = [
                        r["alignment_pert"] for r in rows_sc
                        if r["condition"] == cond and r["group"] == grp
                        and not math.isnan(r["alignment_pert"])
                    ]
                    vals_dd = [
                        r["directional_deriv_pert"] for r in rows_sc
                        if r["condition"] == cond and r["group"] == grp
                        and not math.isnan(r["directional_deriv_pert"])
                    ]
                    entry[key] = {
                        "alignment_pert": safe_mean_std(vals_ap),
                        "directional_deriv_pert": safe_mean_std(vals_dd),
                    }
            summary_out["by_site_call"][site][str(call_idx)] = entry

    (out_root / "gradient_alignment_summary.json").write_text(
        json.dumps(summary_out, indent=2, default=str), encoding="utf-8",
    )

    print(f"\nDone in {time.time() - start:.0f}s")
    print(f"Detail CSV:  {out_root / 'gradient_alignment_detail.csv'}")
    print(f"Summary JSON: {out_root / 'gradient_alignment_summary.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    default_summary = (
        ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    default_output = (
        ROOT / "experiments/phase9_gradient_alignment/kitchen_scene4_seed7"
    )

    parser.add_argument("--summary", default=str(default_summary))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--policy-dir", default=str(POLICY_DIR))
    parser.add_argument("--t5-extra-embeddings",
        default=str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-seed", type=int, default=0)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")

    # Denoising trajectory
    parser.add_argument("--num-denoising-steps", type=int, default=50)
    parser.add_argument("--step-indices", nargs="*", type=int, default=None)
    parser.add_argument("--layers", nargs="+", default=["0", "mid", "last"])
    parser.add_argument("--sites", nargs="+", default=["vae", "0", "mid", "last"])
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--solver-option", default="2ab")
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--shared-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    main()
