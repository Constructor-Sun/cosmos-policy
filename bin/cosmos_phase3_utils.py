"""Utility functions for Cosmos Phase 3 mean-shift recovery."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Optional

import numpy as np
import torch

EPS = 1e-8


def alpha_key(alpha: float) -> str:
    return f"{float(alpha):g}"


def alpha_file_key(alpha: float) -> str:
    return alpha_key(alpha).replace("-", "m").replace(".", "p")


def unique_alphas(alphas: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[str] = set()
    for alpha in [0.0, *alphas]:
        key = alpha_key(alpha)
        if key not in seen:
            out.append(float(alpha))
            seen.add(key)
    return out


def group_pairs_by_condition(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(pair["condition"], []).append(pair)
    for items in grouped.values():
        items.sort(key=lambda p: int(p["pert"]["episode"]))
    return dict(sorted(grouped.items()))


def action_rel_error(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    return float(np.linalg.norm(aa - bb) / (np.linalg.norm(bb) + EPS))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a - b).detach().float().reshape(-1)).cpu())


def tensor_norm(a: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.detach().float().reshape(-1)).cpu())


def tensor_cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(denom.cpu()) <= EPS:
        return None
    value = float(torch.dot(aa, bb).div(denom).cpu())
    return max(-1.0, min(1.0, value))


def signed_projection(delta: torch.Tensor, perturbation_shift: torch.Tensor) -> Optional[float]:
    d = delta.detach().float().reshape(-1)
    s = perturbation_shift.detach().float().reshape(-1)
    denom = torch.dot(s, s)
    if float(denom.cpu()) <= EPS:
        return None
    return float(torch.dot(d, s).div(denom).cpu())


def recovery_metrics(
    *,
    action_np: np.ndarray,
    clean_action: np.ndarray,
    pert_action: np.ndarray,
    hidden_after: torch.Tensor,
    clean_h: torch.Tensor,
    pert_h: torch.Tensor,
) -> dict[str, Any]:
    """Compute action and hidden recovery for one intervention result."""

    base_error = action_rel_error(pert_action, clean_action)
    corr_error = action_rel_error(action_np, clean_action)
    base_mse = mse(pert_action, clean_action)
    corr_mse = mse(action_np, clean_action)

    hidden_base = tensor_l2(pert_h, clean_h)
    hidden_corr = tensor_l2(hidden_after, clean_h)
    perturbation_shift = pert_h - clean_h
    intervention_delta = hidden_after - pert_h

    return {
        "action_error_to_clean": corr_error,
        "action_error_to_pert": action_rel_error(action_np, pert_action),
        "baseline_action_error": base_error,
        "recovery_vs_pert": None if base_error <= EPS else 1.0 - corr_error / base_error,
        "action_mse_to_clean": corr_mse,
        "action_mse_to_pert": mse(action_np, pert_action),
        "baseline_action_mse": base_mse,
        "mse_recovery_vs_pert": None if base_mse <= EPS else 1.0 - corr_mse / base_mse,
        "hidden_l2_to_clean": hidden_corr,
        "hidden_l2_to_pert": tensor_l2(hidden_after, pert_h),
        "baseline_hidden_l2": hidden_base,
        "hidden_recovery_vs_pert": None if hidden_base <= EPS else 1.0 - hidden_corr / hidden_base,
        "hidden_norm_clean": tensor_norm(clean_h),
        "hidden_norm_pert": tensor_norm(pert_h),
        "hidden_norm_after": tensor_norm(hidden_after),
        "hidden_cos_to_clean": tensor_cos(hidden_after, clean_h),
        "hidden_cos_to_pert": tensor_cos(hidden_after, pert_h),
        "intervention_projection_on_perturbation_shift": signed_projection(
            intervention_delta, perturbation_shift
        ),
    }


def direction_consistency(
    directions: list[torch.Tensor],
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Assess alignment of direction vectors in one perturbation group."""

    def cos_flat(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
        denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        if float(denom.cpu()) <= EPS:
            return None
        value = float(torch.dot(a, b).div(denom).cpu())
        return max(-1.0, min(1.0, value))

    def float_stats(values: list[Optional[float]]) -> dict[str, Optional[float]]:
        vals = [v for v in values if v is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        t = torch.tensor(vals, dtype=torch.float32)
        return {
            "mean": float(t.mean().cpu()),
            "std": float(t.std(unbiased=True).cpu()) if len(vals) > 1 else 0.0,
            "min": float(t.min().cpu()),
            "max": float(t.max().cpu()),
        }

    flats: list[torch.Tensor] = []
    units: list[torch.Tensor] = []
    norms: list[float] = []
    kept_labels: list[str] = []
    for idx, direction in enumerate(directions):
        flat = direction.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(flat)
        norm_f = float(norm.cpu())
        if norm_f <= EPS:
            continue
        flats.append(flat)
        units.append(flat / norm)
        norms.append(norm_f)
        kept_labels.append(labels[idx] if labels is not None else str(idx))

    out: dict[str, Any] = {
        "n": len(directions),
        "nonzero_n": len(units),
        "mean_resultant_length": None,
        "mean_direction_norm": None,
        "mean_unit_pairwise_cos": None,
        "cos_to_global_mean_direction": float_stats([]),
        "cos_to_leave_one_out_mean_direction": float_stats([]),
        "per_sample_alignment": [],
    }
    if not units:
        return out

    flat_stack = torch.stack(flats, dim=0)
    unit_stack = torch.stack(units, dim=0)
    global_mean = flat_stack.mean(dim=0)
    cos_global = [cos_flat(flat, global_mean) for flat in flats]
    cos_loo: list[Optional[float]]

    out["mean_resultant_length"] = float(torch.linalg.vector_norm(unit_stack.mean(dim=0)).cpu())
    out["mean_direction_norm"] = float(sum(norms) / len(norms))
    out["cos_to_global_mean_direction"] = float_stats(cos_global)

    if len(units) > 1:
        idx = torch.triu_indices(len(units), len(units), offset=1)
        cos_mat = unit_stack @ unit_stack.T
        out["mean_unit_pairwise_cos"] = float(cos_mat[idx[0], idx[1]].mean().cpu())
        total = flat_stack.sum(dim=0)
        cos_loo = [cos_flat(flat, (total - flat) / (len(flats) - 1)) for flat in flats]
        out["cos_to_leave_one_out_mean_direction"] = float_stats(cos_loo)
    else:
        cos_loo = [None for _ in flats]

    out["per_sample_alignment"] = [
        {
            "label": label,
            "direction_norm": norm,
            "cos_to_global_mean_direction": cg,
            "cos_to_leave_one_out_mean_direction": cloo,
        }
        for label, norm, cg, cloo in zip(kept_labels, norms, cos_global, cos_loo)
    ]
    return out


def build_aggregates(
    groups: dict[str, dict[str, Any]],
    layers: list[int],
    alphas: list[float],
) -> dict[str, Any]:
    """Aggregate per-sample Phase 3 metrics by perturbation/layer/alpha."""

    out: dict[str, Any] = {}
    for condition, metrics in groups.items():
        cond_out: dict[str, Any] = {"condition": condition, "layers": {}}
        for layer in layers:
            layer_out: dict[str, Any] = {}
            for alpha in alphas:
                key = alpha_key(alpha)
                rec_vals: list[float] = []
                mse_rec_vals: list[float] = []
                hidden_rec_vals: list[float] = []
                target_hidden_rec_vals: list[float] = []
                video_hidden_rec_vals: list[float] = []
                action_hidden_rec_vals: list[float] = []
                action_err_vals: list[float] = []
                for ep_data in metrics.get("results", {}).values():
                    alpha_data = (
                        ep_data.get("layers", {})
                        .get(str(layer), {})
                        .get("per_alpha", {})
                        .get(key, {})
                    )
                    rec = alpha_data.get("recovery_vs_pert")
                    mse_rec = alpha_data.get("mse_recovery_vs_pert")
                    hidden_rec = alpha_data.get("hidden_recovery_vs_pert")
                    target_hidden_rec = alpha_data.get("target_hidden_recovery_vs_pert")
                    if target_hidden_rec is None:
                        target_hidden_rec = hidden_rec
                    video_hidden_rec = alpha_data.get("video_hidden_recovery_vs_pert")
                    action_hidden_rec = alpha_data.get("action_hidden_recovery_vs_pert")
                    action_err = alpha_data.get("action_error_to_clean")
                    if rec is not None:
                        rec_vals.append(float(rec))
                    if mse_rec is not None:
                        mse_rec_vals.append(float(mse_rec))
                    if hidden_rec is not None:
                        hidden_rec_vals.append(float(hidden_rec))
                    if target_hidden_rec is not None:
                        target_hidden_rec_vals.append(float(target_hidden_rec))
                    if video_hidden_rec is not None:
                        video_hidden_rec_vals.append(float(video_hidden_rec))
                    if action_hidden_rec is not None:
                        action_hidden_rec_vals.append(float(action_hidden_rec))
                    if action_err is not None:
                        action_err_vals.append(float(action_err))
                positives = [v for v in rec_vals if v > 0]
                layer_out[key] = {
                    "n": len(rec_vals),
                    "mean_action_recovery": mean(rec_vals) if rec_vals else None,
                    "median_action_recovery": median(rec_vals) if rec_vals else None,
                    "positive_recovery_rate": (
                        len(positives) / len(rec_vals) if rec_vals else None
                    ),
                    "mean_mse_recovery": mean(mse_rec_vals) if mse_rec_vals else None,
                    "median_mse_recovery": median(mse_rec_vals) if mse_rec_vals else None,
                    "mean_hidden_recovery": mean(hidden_rec_vals) if hidden_rec_vals else None,
                    "median_hidden_recovery": median(hidden_rec_vals) if hidden_rec_vals else None,
                    "mean_target_hidden_recovery": (
                        mean(target_hidden_rec_vals) if target_hidden_rec_vals else None
                    ),
                    "median_target_hidden_recovery": (
                        median(target_hidden_rec_vals) if target_hidden_rec_vals else None
                    ),
                    "mean_video_hidden_recovery": (
                        mean(video_hidden_rec_vals) if video_hidden_rec_vals else None
                    ),
                    "median_video_hidden_recovery": (
                        median(video_hidden_rec_vals) if video_hidden_rec_vals else None
                    ),
                    "mean_action_hidden_recovery": (
                        mean(action_hidden_rec_vals) if action_hidden_rec_vals else None
                    ),
                    "median_action_hidden_recovery": (
                        median(action_hidden_rec_vals) if action_hidden_rec_vals else None
                    ),
                    "mean_action_error_to_clean": mean(action_err_vals) if action_err_vals else None,
                }
            cond_out["layers"][str(layer)] = layer_out
        out[condition] = cond_out
    return out
