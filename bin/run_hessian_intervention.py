#!/usr/bin/env python3
"""Recover final actions from the L27 action latent with gradient or GN-CG.

The experiment is deliberately narrow:
  * flipped clean/perturbed pairs only;
  * final denoising call (49 by default);
  * last DiT block output, action slot, action-readout grid rows only;
  * equal hidden-space budgets for normalized gradient and Gauss-Newton CG;
  * matrix-free J'Jv: central-difference Jv plus one fresh reverse-mode VJP.

This is an oracle recoverability test: the clean final action defines the loss,
and the clean hidden state defines only the common intervention budget.
"""
from __future__ import annotations

import argparse, csv, gc, json, os, pathlib, runpy, site, sys, time
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
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
if site.getusersitepackages() in sys.path:
    sys.path.remove(site.getusersitepackages())

_lib = pathlib.Path(sys.prefix) / "lib"
if (_lib / "libMagickWand-7.Q16HDRI.so").exists():
    os.environ.setdefault("MAGICK_HOME", sys.prefix)
    os.environ.setdefault("WAND_MAGICK_LIBRARY_SUFFIX", "-7.Q16HDRI")
try:
    from wand.api import library as _wand_library  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import get_model, load_dataset_stats
from cosmos_policy.utils.utils import set_seed_everywhere

_base = runpy.run_path(str(ROOT / "bin" / "run_denoise_hessian_cosmos.py"), run_name="__hessian_base__")
action_from_latent = _base["action_from_latent"]
action_tensor_from_latent = _base["action_tensor_from_latent"]
build_inference_data_batch = _base["build_inference_data_batch"]
freeze_model_parameters = _base["freeze_model_parameters"]
initial_noise = _base["initial_noise"]
load_phase2_helpers = _base["load_phase2_helpers"]
replace_block_output = _base["replace_block_output"]
run_branch_trajectory = _base["run_branch_trajectory"]

EPS = 1e-12
ACTION_READOUT_ROWS = 2
POLICY_DIR = ROOT.parent.parent / "checkpoints" / "Cosmos-Policy-LIBERO-Predict2-2B"
FIELDS = [
    "condition", "episode", "group", "call", "layer", "action_slot",
    "radius_fraction", "radius", "clean_pert_hidden_distance",
    "mse_before", "mse_gradient", "mse_gauss_newton",
    "recovery_gradient", "recovery_gauss_newton", "gauss_newton_minus_gradient",
    "grad_norm", "gradient_step_norm", "gauss_newton_step_norm", "gauss_newton_raw_norm",
    "gauss_newton_predicted_decrease", "cg_iterations", "cg_gvp_calls",
    "cg_residual_ratio", "cg_termination", "elapsed_s",
]


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def conjugate_gradient(
    grad: torch.Tensor,
    gvp: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int,
    relative_tolerance: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Approximately solve Gp=-g; caller normalizes the resulting direction."""
    p = torch.zeros_like(grad)
    residual = grad.clone()  # q'(p) = g + Hp; p starts at zero.
    direction = -residual
    residual_sq = dot(residual, residual)
    grad_norm = float(torch.sqrt(residual_sq).cpu())
    hvp_calls = 0

    if grad_norm <= EPS:
        return p, {
            "iterations": 0, "hvp_calls": 0, "residual_ratio": 0.0,
            "termination": "zero_gradient",
        }

    for iteration in range(1, max_iter + 1):
        h_direction = gvp(direction)
        hvp_calls += 1
        curvature = float(dot(direction, h_direction).cpu())
        if not np.isfinite(curvature):
            raise RuntimeError("Non-finite curvature from Gauss-Newton vector product")
        if curvature <= 0.0:
            return p, {
                "iterations": iteration, "hvp_calls": hvp_calls,
                "residual_ratio": float(torch.linalg.vector_norm(residual).cpu()) / grad_norm,
                "termination": "nonpositive_curvature",
            }

        alpha = float(residual_sq.cpu()) / curvature
        candidate = p + alpha * direction
        p = candidate
        next_residual = residual + alpha * h_direction
        next_sq = dot(next_residual, next_residual)
        residual_ratio = float(torch.sqrt(next_sq).cpu()) / grad_norm
        if residual_ratio <= relative_tolerance:
            return p, {
                "iterations": iteration, "hvp_calls": hvp_calls,
                "residual_ratio": residual_ratio, "termination": "converged",
            }
        beta = float((next_sq / residual_sq).cpu())
        direction = -next_residual + beta * direction
        residual, residual_sq = next_residual, next_sq

    return p, {
        "iterations": max_iter, "hvp_calls": hvp_calls,
        "residual_ratio": float(torch.linalg.vector_norm(residual).cpu()) / grad_norm,
        "termination": "max_iter",
    }


def build_last_action_problem(model, branch, capture, layer, chunk_size, device):
    """Return the action function over the active portion of the L27 action slot."""
    if layer not in capture.hidden:
        raise RuntimeError(f"Missing layer {layer} capture at call {capture.call_index}")
    full_hidden = capture.hidden[layer].to(device=device, dtype=next(model.parameters()).dtype)
    if full_hidden.ndim != 5 or full_hidden.shape[2] < ACTION_READOUT_ROWS:
        raise RuntimeError(f"Unexpected L27 hidden shape: {tuple(full_hidden.shape)}")

    indices = branch.data_batch["action_latent_idx"].detach().reshape(-1).cpu().tolist()
    if len(set(int(x) for x in indices)) != 1:
        raise RuntimeError(f"Expected one shared action slot, got {indices}")
    action_slot = int(indices[0])
    sigma = torch.full(
        (capture.x_in.shape[0],), capture.sigma, device=device, dtype=torch.float32,
    )
    x_in = capture.x_in.to(device=device, dtype=torch.float32)
    base_before = full_hidden[:, :action_slot]
    base_after = full_hidden[:, action_slot + 1:]
    inactive_rows = full_hidden[:, action_slot, ACTION_READOUT_ROWS:]

    def prediction(active_action: torch.Tensor) -> torch.Tensor:
        slot = torch.cat(
            [active_action.to(full_hidden.dtype), inactive_rows], dim=1,
        ).unsqueeze(1)
        replacement = torch.cat([base_before, slot, base_after], dim=1)
        with replace_block_output(model.net.blocks[layer], replacement):
            return branch.x0_fn(x_in, sigma).float()

    def action_fn(active_action: torch.Tensor) -> torch.Tensor:
        return action_tensor_from_latent(
            prediction(active_action), branch.data_batch, chunk_size,
        )

    active = full_hidden[:, action_slot, :ACTION_READOUT_ROWS].detach().float()
    return action_fn, active, action_slot


def mse(action: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((action.astype(np.float64) - target.astype(np.float64)) ** 2))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)


def run_pair(model, clean_branch, pert_branch, clean_capture, pert_capture, args, device):
    started = time.time()
    layer = len(model.net.blocks) - 1
    clean_action_np = action_from_latent(
        clean_branch.final_sample.to(device), clean_branch.data_batch, args.chunk_size,
    ).astype(np.float32)
    target = torch.from_numpy(clean_action_np)
    action_fn, h_pert, action_slot = build_last_action_problem(
        model, pert_branch, pert_capture, layer, args.chunk_size, device,
    )
    _, h_clean, clean_slot = build_last_action_problem(
        model, clean_branch, clean_capture, layer, args.chunk_size, device,
    )
    if clean_slot != action_slot:
        raise RuntimeError(f"Clean/perturbed action slots differ: {clean_slot} vs {action_slot}")

    hidden_distance = float(torch.linalg.vector_norm(h_clean - h_pert).cpu())
    radius = args.radius_fraction * hidden_distance
    if radius <= EPS:
        raise RuntimeError("Clean/perturbed last-layer action latents are identical")

    variable = h_pert.detach().clone().requires_grad_(True)
    action_graph = action_fn(variable)
    target_device = target.to(device=device, dtype=action_graph.dtype)
    loss_before = ((action_graph - target_device) ** 2).mean()
    grad = torch.autograd.grad(loss_before, variable)[0].detach()
    grad_norm = float(torch.linalg.vector_norm(grad).cpu())
    if grad_norm <= EPS:
        raise RuntimeError("Action loss gradient is zero at the perturbed action latent")

    gn_scale = 2.0 / action_graph.numel()
    fd_step = args.gn_fd_relative_step * float(torch.linalg.vector_norm(h_pert).cpu())

    def gauss_newton_vp(vector: torch.Tensor) -> torch.Tensor:
        vector_norm = float(torch.linalg.vector_norm(vector).cpu())
        if vector_norm <= EPS:
            return torch.zeros_like(vector)
        direction = vector / vector_norm
        with torch.no_grad():
            plus = action_fn(h_pert + fd_step * direction)
            minus = action_fn(h_pert - fd_step * direction)
            jv = (plus - minus) * (vector_norm / (2.0 * fd_step + EPS))
        probe = h_pert.detach().clone().requires_grad_(True)
        output = action_fn(probe)
        return torch.autograd.grad(
            output, probe, grad_outputs=gn_scale * jv,
        )[0].detach()

    p_gradient = -radius * grad / (grad_norm + EPS)
    p_gn_raw, cg = conjugate_gradient(
        grad, gauss_newton_vp, args.cg_max_iter, args.cg_tolerance,
    )
    gn_raw_norm = float(torch.linalg.vector_norm(p_gn_raw).cpu())
    if gn_raw_norm <= EPS:
        raise RuntimeError(f"Gauss-Newton CG returned a zero direction: {cg}")
    p_gn = radius * p_gn_raw / gn_raw_norm
    gp = gauss_newton_vp(p_gn)
    predicted_decrease = -float((dot(grad, p_gn) + 0.5 * dot(p_gn, gp)).cpu())

    with torch.no_grad():
        action_before = action_fn(h_pert).detach().float().cpu().numpy()
        action_gradient = action_fn(h_pert + p_gradient).detach().float().cpu().numpy()
        action_gn = action_fn(h_pert + p_gn).detach().float().cpu().numpy()
    mse_before = mse(action_before, clean_action_np)
    mse_gradient = mse(action_gradient, clean_action_np)
    mse_gn = mse(action_gn, clean_action_np)
    recovery_gradient = 1.0 - mse_gradient / (mse_before + EPS)
    recovery_gn = 1.0 - mse_gn / (mse_before + EPS)

    return {
        "call": args.call_index, "layer": layer, "action_slot": action_slot,
        "radius_fraction": args.radius_fraction, "radius": radius,
        "clean_pert_hidden_distance": hidden_distance,
        "mse_before": mse_before, "mse_gradient": mse_gradient, "mse_gauss_newton": mse_gn,
        "recovery_gradient": recovery_gradient, "recovery_gauss_newton": recovery_gn,
        "gauss_newton_minus_gradient": recovery_gn - recovery_gradient,
        "grad_norm": grad_norm,
        "gradient_step_norm": float(torch.linalg.vector_norm(p_gradient).cpu()),
        "gauss_newton_step_norm": float(torch.linalg.vector_norm(p_gn).cpu()),
        "gauss_newton_raw_norm": gn_raw_norm,
        "gauss_newton_predicted_decrease": predicted_decrease,
        "cg_iterations": cg["iterations"], "cg_gvp_calls": int(cg["hvp_calls"]) + 1,
        "cg_residual_ratio": cg["residual_ratio"], "cg_termination": cg["termination"],
        "elapsed_s": time.time() - started,
    }


def main() -> None:
    args = parser().parse_args()
    discover_pairs, first_observation, load_extra_t5, make_cfg, patch_checkpoint_db = load_phase2_helpers()
    args.policy_dir = pathlib.Path(args.policy_dir)
    patch_checkpoint_db(args.policy_dir)
    _, pairs = discover_pairs(pathlib.Path(args.summary), set(args.conditions or []), {"flipped"})
    if args.episodes:
        pairs = [p for p in pairs if int(p["pert"]["episode"]) in set(args.episodes)]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise RuntimeError("No flipped pairs matched the requested filters")

    cfg = make_cfg(args)
    args.chunk_size = cfg.chunk_size
    set_seed_everywhere(args.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)
    device = torch.device(args.device)
    model = model.to(device).eval()
    freeze_model_parameters(model)
    layer = len(model.net.blocks) - 1
    print(
        f"pairs={len(pairs)} layer={layer} call={args.call_index} "
        f"radius_fraction={args.radius_fraction} cg_max_iter={args.cg_max_iter}", flush=True,
    )

    rows: list[dict] = []
    output = pathlib.Path(args.output_dir) / "action_recovery.csv"
    for pair_index, pair in enumerate(pairs, 1):
        clean_ep, pert_ep = pair["clean"], pair["pert"]
        condition, episode = pair["condition"], int(pert_ep["episode"])
        print(f"[{pair_index}/{len(pairs)}] {condition}/ep{episode:02d}", flush=True)
        clean_obs = first_observation(clean_ep, args)
        pert_obs = clean_obs if condition == "language_instructions" else first_observation(pert_ep, args)
        clean_instruction = clean_ep["language"]
        pert_instruction = pert_ep["language"] if pert_ep.get("instruction_mode") == "strict" else clean_instruction
        clean_batch = build_inference_data_batch(cfg, model, stats, clean_obs, clean_instruction, device)
        pert_batch = build_inference_data_batch(cfg, model, stats, pert_obs, pert_instruction, device)
        noise = initial_noise(model, clean_batch, args.seed * 100000 + episode, args.sigma_max, device)
        layer_map = {"last": layer}
        clean_branch = run_branch_trajectory(model, clean_batch, noise, {args.call_index}, layer_map, args)
        pert_branch = run_branch_trajectory(model, pert_batch, noise.clone(), {args.call_index}, layer_map, args)
        clean_capture = clean_branch.captures.get(args.call_index)
        pert_capture = pert_branch.captures.get(args.call_index)
        if clean_capture is None or pert_capture is None:
            raise RuntimeError(f"Denoise call {args.call_index} was not captured")
        row = run_pair(model, clean_branch, pert_branch, clean_capture, pert_capture, args, device)
        row.update(condition=condition, episode=episode, group="flipped")
        rows.append(row)
        write_csv(output, rows)
        print(
            f"  recovery gradient={row['recovery_gradient']:+.4f} "
            f"gauss_newton={row['recovery_gauss_newton']:+.4f} "
            f"CG={row['cg_termination']}/{row['cg_iterations']}", flush=True,
        )
        del clean_branch, pert_branch
        torch.cuda.empty_cache()
        gc.collect()

    grad_values = [r["recovery_gradient"] for r in rows]
    gn_values = [r["recovery_gauss_newton"] for r in rows]
    summary = {
        "num_pairs": len(rows), "site": "last_action_readout", "layer": layer,
        "call": args.call_index, "radius_fraction": args.radius_fraction,
        "gn_fd_relative_step": args.gn_fd_relative_step,
        "median_gradient_recovery": float(np.median(grad_values)),
        "median_gauss_newton_recovery": float(np.median(gn_values)),
        "gradient_positive_rate": float(np.mean(np.asarray(grad_values) > 0)),
        "gauss_newton_positive_rate": float(np.mean(np.asarray(gn_values) > 0)),
        "median_gauss_newton_minus_gradient": float(np.median(np.asarray(gn_values) - grad_values)),
    }
    summary_path = pathlib.Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    default_summary = ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default=str(default_summary))
    p.add_argument("--output-dir", default=str(ROOT / "experiments/gn_action_recovery_last"))
    p.add_argument("--policy-dir", default=str(POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default=str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--episodes", nargs="*", type=int, default=None)
    p.add_argument("--max-pairs", type=int, default=10)
    p.add_argument("--radius-fraction", type=float, default=0.1)
    p.add_argument("--cg-max-iter", type=int, default=10)
    p.add_argument("--cg-tolerance", type=float, default=1e-2)
    p.add_argument("--gn-fd-relative-step", type=float, default=1e-2)
    p.add_argument("--call-index", type=int, default=49)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-denoising-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--solver-option", default="2ab")
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    main()
