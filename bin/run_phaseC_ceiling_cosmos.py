#!/usr/bin/env python3
"""Phase C: Ceiling analysis for action-slot subspace correction.

Two-phase design:
  Phase A: capture clean/pert hidden states for all flipped pairs (once)
  Phase B: fit PCA, then run recovery with multiple direction types:
    mean_shift    – global mean delta (Phase 3 baseline)
    oracle_k{N}   – delta projected onto top-N PCA of own perturb
    oracle_full   – unprojected delta (absolute ceiling)
    cross_k{N}    – delta projected onto another perturb's PCA
    merged_k{N}   – delta projected onto all-perturb merged PCA
    mean_component – delta projected onto mean direction only
    within_k8     – top-8 projection minus mean component
    orthogonal_k8  – complement of top-8 projection

Key question: does per-sample oracle projection (k directions) beat
the global mean-shift (1 direction)?  The gap is the ceiling for
linear subspace correction.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import site
import sys
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy-matplotlib")
os.environ.setdefault("DETERMINISTIC", "True")

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
import torch

import run_phase2_angular_cosmos as phase2
from cosmos_layer_shift import (
    CosmosLayerCaptureIntervener,
    layer_shift_context,
    token_mean_shift_direction,
)
from cosmos_phase3_utils import (
    alpha_key,
    build_aggregates,
    group_pairs_by_condition,
    recovery_metrics,
    unique_alphas,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PCA utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().reshape(-1).cpu().numpy().astype(np.float64)


def _unflatten(flat: np.ndarray, template: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(flat.astype(np.float32)).reshape(template.shape)


def fit_pca(deltas: list[torch.Tensor]) -> dict[str, Any]:
    """Fit PCA on flattened delta tensors (clean_h - pert_h).

    Returns dict with mean, components, evr, r90.
    """
    stacked = np.stack([_flatten(d) for d in deltas], axis=0)
    mean_flat = stacked.mean(axis=0)
    centered = stacked - mean_flat
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    evr = (s ** 2) / (np.sum(s ** 2) + 1e-16)
    r90 = int(np.sum(np.cumsum(evr) < 0.90) + 1)
    return {
        "mean": mean_flat,
        "components": vt,       # [K, D]
        "singular_values": s,   # [K]
        "evr": evr,             # [K]
        "r90": r90,
    }


def project_onto(delta: torch.Tensor, pca: dict[str, Any], k: int) -> torch.Tensor:
    """Project delta onto top-k PCA subspace."""
    flat = _flatten(delta)
    centered = flat - pca["mean"]
    comps = pca["components"][:k]
    coeffs = centered @ comps.T
    proj = pca["mean"] + comps.T @ coeffs
    return _unflatten(proj, delta)


def project_mean_only(delta: torch.Tensor, pca: dict[str, Any]) -> torch.Tensor:
    """Project delta onto the mean direction only."""
    flat = _flatten(delta)
    m = pca["mean"]
    msq = float(np.dot(m, m))
    if msq < 1e-16:
        return torch.zeros_like(delta)
    coeff = float(np.dot(flat, m)) / msq
    return _unflatten(coeff * m, delta)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference helpers (reused from Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

def _instruction_pair(clean: dict, pert: dict) -> tuple[str, str]:
    ci = clean["language"]
    pi = pert["language"] if pert.get("instruction_mode") == "strict" else ci
    return ci, pi


def _pair_observations(pair: dict, args) -> tuple[dict, dict]:
    clean_obs = phase2.first_observation(pair["clean"], args)
    if pair["condition"] == "language_instructions":
        pert_obs = clean_obs
    else:
        pert_obs = phase2.first_observation(pair["pert"], args)
    return clean_obs, pert_obs


def _forward(cfg, model, stats, obs, instruction, seed, layers, args,
             target_indices, intervene_layer=None, direction=None, alpha=0.0):
    intervener = CosmosLayerCaptureIntervener(
        target_indices=target_indices,
        condition_pass_only=args.condition_pass_only,
    )
    intervener.reset(
        capture_layers=set(layers),
        intervene_layer=intervene_layer,
        direction=direction,
        alpha=alpha,
    )
    hook_layers = sorted(set(layers) | ({intervene_layer} if intervene_layer is not None else set()))
    with layer_shift_context(model, intervener, hook_layers):
        phase2.set_seed_everywhere(args.reset_seed)
        out = phase2.get_action(
            cfg, model, stats, obs, instruction, seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    return (np.asarray(out["actions"], dtype=np.float32),
            intervener.captured, intervener.after)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Capture
# ═══════════════════════════════════════════════════════════════════════════════

def capture_all(
    by_condition: dict[str, list[dict]],
    cfg, model, stats, layers: list[int], args,
    target_indices: list[int],
    out_root: pathlib.Path,
) -> dict[str, dict[int, dict]]:
    """Capture clean/pert hidden states for all flipped pairs.

    Returns: {condition: {ep: capture_dict}}
    Captures kept in memory (observation dicts are not serializable).
    """
    all_captures: dict[str, dict[int, dict]] = {}
    for condition, pairs in by_condition.items():
        cond_dir = out_root / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        captures: dict[int, dict] = {}
        print(f"\n[Capture] {condition} n={len(pairs)}", flush=True)
        for idx, pair in enumerate(pairs, 1):
            ep = int(pair["pert"]["episode"])
            print(f"  [{idx}/{len(pairs)}] ep{ep:02d}", flush=True)
            try:
                clean, pert = pair["clean"], pair["pert"]
                clean_obs, pert_obs = _pair_observations(pair, args)
                clean_instr, pert_instr = _instruction_pair(clean, pert)

                clean_action, clean_h, _ = _forward(
                    cfg, model, stats, clean_obs, clean_instr,
                    args.seed, layers, args, target_indices)
                pert_action, pert_h, _ = _forward(
                    cfg, model, stats, pert_obs, pert_instr,
                    args.seed, layers, args, target_indices)

                direction = token_mean_shift_direction(
                    clean_h[layers[0]], pert_h[layers[0]])

                cap = {
                    "condition": condition, "episode": ep,
                    "clean_action": clean_action, "pert_action": pert_action,
                    "clean_h": clean_h[layers[0]], "pert_h": pert_h[layers[0]],
                    "delta": direction,
                    "pert_obs": pert_obs,           # kept in memory
                    "pert_instr": pert_instr,        # kept in memory
                }
                captures[ep] = cap

                # Save key artifacts to disk (for inspection / reuse)
                ep_dir = cond_dir / f"ep{ep:02d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                np.save(ep_dir / "action_clean.npy", clean_action)
                np.save(ep_dir / "action_pert.npy", pert_action)
                torch.save(clean_h[layers[0]], ep_dir / "hidden_clean.pt")
                torch.save(pert_h[layers[0]], ep_dir / "hidden_pert.pt")
                torch.save(direction, ep_dir / "delta.pt")

            except Exception as exc:
                print(f"    skip ep{ep:02d}: {exc}", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not captures:
            raise RuntimeError(f"no captures for {condition}")
        all_captures[condition] = captures
    return all_captures


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: PCA fitting + ceiling recovery
# ═══════════════════════════════════════════════════════════════════════════════

def build_pca_registry(
    all_captures: dict[str, dict[int, dict]],
    oracle_ks: list[int],
) -> dict[str, dict[str, Any]]:
    """Fit per-perturb PCA and build direction registry.

    Returns: {cond: {"pca": ..., "mean_shift": tensor, "deltas": list[tensor]}}
    """
    registry = {}
    for cond, captures in all_captures.items():
        deltas = [cap["delta"] for _, cap in sorted(captures.items())]
        pca = fit_pca(deltas)
        mean_shift = torch.stack(deltas, dim=0).mean(dim=0)
        registry[cond] = {
            "pca": pca,
            "mean_shift": mean_shift,
            "deltas": {ep: cap["delta"] for ep, cap in captures.items()},
        }
        print(f"  {cond}: r90={pca['r90']}  ev_top1={pca['evr'][0]:.3f}  "
              f"ev_top4={sum(pca['evr'][:4]):.3f}  ev_top8={sum(pca['evr'][:min(8, len(pca['evr']))]):.3f}",
              flush=True)
    return registry


def build_merged_pca(registry: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Build merged PCA from all perturbations' deltas.

    Pools per-perturb components and re-orthogonalizes via SVD.
    """
    all_deltas = []
    for cond_data in registry.values():
        all_deltas.extend(cond_data["deltas"].values())
    if not all_deltas:
        return None
    pca = fit_pca(all_deltas)
    print(f"  merged: r90={pca['r90']}  ev_top4={sum(pca['evr'][:4]):.3f}  "
          f"ev_top8={sum(pca['evr'][:min(8, len(pca['evr']))]):.3f}", flush=True)
    return pca


def build_direction_plan(
    registry: dict[str, dict[str, Any]],
    merged_pca: dict[str, Any] | None,
    oracle_ks: list[int],
    no_cross: bool = False,
    no_merged: bool = False,
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """For each condition and each sample, pre-compute all direction types.

    Returns: {cond: {ep: {dir_type: tensor}}}
    """
    plan: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    all_conditions = sorted(registry.keys())

    for cond in all_conditions:
        cond_data = registry[cond]
        pca = cond_data["pca"]
        mean_shift = cond_data["mean_shift"]
        plan[cond] = {}

        for ep, delta in cond_data["deltas"].items():
            dirs: dict[str, torch.Tensor] = {}

            # mean_shift (global, same for all samples in this condition)
            dirs["mean_shift"] = mean_shift.clone()

            # oracle_k{N} and oracle_full
            for k in oracle_ks:
                dirs[f"oracle_k{k}"] = project_onto(delta, pca, k)
            dirs["oracle_full"] = delta.clone()

            # decomposition
            dirs["mean_component"] = project_mean_only(delta, pca)
            k8 = min(8, len(pca["components"]))
            proj_k8 = project_onto(delta, pca, k8)
            dirs["within_k8"] = _unflatten(
                _flatten(proj_k8) - _flatten(dirs["mean_component"]), delta)
            dirs["orthogonal_k8"] = _unflatten(
                _flatten(delta) - _flatten(proj_k8), delta)

            # cross-perturb
            if not no_cross:
                for other_cond in all_conditions:
                    if other_cond == cond:
                        continue
                    other_pca = registry[other_cond]["pca"]
                    for k in oracle_ks:
                        dirs[f"cross_{other_cond}_k{k}"] = project_onto(
                            delta, other_pca, k)

            # merged
            if not no_merged and merged_pca is not None:
                for k in oracle_ks:
                    dirs[f"merged_k{k}"] = project_onto(delta, merged_pca, k)

            plan[cond][ep] = dirs

    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C: Recovery (model inference with interventions)
# ═══════════════════════════════════════════════════════════════════════════════

def run_recovery(
    direction_plan: dict[str, dict[int, dict[str, torch.Tensor]]],
    all_captures: dict[str, dict[int, dict]],
    cfg, model, stats, layer: int, args,
    target_indices: list[int],
    out_root: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    """Run intervention recovery for all direction types."""

    test_alphas = [0.5, 1.0]
    all_metrics: dict[str, dict[str, Any]] = {}

    for condition in sorted(all_captures.keys()):
        captures = all_captures[condition]
        plan = direction_plan[condition]
        cond_dir = out_root / condition
        metrics: dict[str, Any] = {"results": {}}

        some_ep = next(iter(plan))
        direction_order = list(plan[some_ep].keys())
        metrics["direction_order"] = direction_order
        metrics["alphas"] = [alpha_key(a) for a in [0.0] + test_alphas]

        print(f"\n[Recovery] {condition} n={len(captures)}  "
              f"directions={len(direction_order)}", flush=True)

        for ep in sorted(captures):
            cap = captures[ep]
            ep_metrics: dict[str, Any] = {"episode": ep, "direction_types": {}}

            for dir_type in direction_order:
                direction = plan[ep].get(dir_type)
                if direction is None:
                    continue
                dt_metrics: dict[str, Any] = {"per_alpha": {}}

                for alpha_val in [0.0] + test_alphas:
                    key = alpha_key(alpha_val)
                    if alpha_val == 0.0:
                        action_np = cap["pert_action"]
                        hidden_after = cap["pert_h"]
                    else:
                        action_np, _, after = _forward(
                            cfg, model, stats,
                            cap["pert_obs"], cap["pert_instr"],
                            args.seed, [layer], args, target_indices,
                            intervene_layer=layer, direction=direction,
                            alpha=alpha_val,
                        )
                        hidden_after = after[layer]

                    rec = recovery_metrics(
                        action_np=action_np,
                        clean_action=cap["clean_action"],
                        pert_action=cap["pert_action"],
                        hidden_after=hidden_after,
                        clean_h=cap["clean_h"],
                        pert_h=cap["pert_h"],
                    )
                    rec["alpha"] = alpha_val
                    dt_metrics["per_alpha"][key] = rec
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                ep_metrics["direction_types"][dir_type] = dt_metrics

            metrics["results"][str(ep)] = ep_metrics

        (cond_dir / "recovery_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
        all_metrics[condition] = metrics

    return all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate & reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate(all_metrics: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """Build per-condition, per-direction recovery aggregates."""
    out: dict[str, dict] = {}
    for condition, metrics in all_metrics.items():
        cond_agg: dict[str, Any] = {}
        direction_order = metrics.get("direction_order", [])
        for dir_type in direction_order:
            for alpha_key_str in ["0.5", "1"]:  # alpha_key(0.5), alpha_key(1.0)
                vals = []
                for ep_data in metrics.get("results", {}).values():
                    dt = ep_data.get("direction_types", {}).get(dir_type, {})
                    pa = dt.get("per_alpha", {}).get(alpha_key_str, {})
                    rec = pa.get("recovery_vs_pert")
                    if rec is not None:
                        vals.append(float(rec))
                if not vals:
                    continue
                label = f"{dir_type}_a{alpha_key_str}"
                positives = [v for v in vals if v > 0]
                cond_agg[label] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "pos_rate": len(positives) / len(vals),
                }
        out[condition] = cond_agg
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    default_summary = phase2.ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    default_output = phase2.ROOT / "experiments/phaseC_ceiling_cosmos/kitchen_scene4_seed7_action_slot_last"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default=str(default_summary))
    p.add_argument("--output-dir", default=str(default_output))
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument("--t5-extra-embeddings",
                   default=str(phase2.ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--layers", nargs="+", default=["last"])
    p.add_argument("--oracle-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--no-cross", action="store_true")
    p.add_argument("--no-merged", action="store_true")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--condition-pass-only", action=argparse.BooleanOptionalAction, default=True)
    return p


def main():
    args = build_parser().parse_args()
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)

    summary_path = pathlib.Path(args.summary)
    summary, pairs = phase2.discover_pairs(
        summary_path, set(args.conditions or []), {"flipped"})
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    by_condition = group_pairs_by_condition(pairs)
    print(f"Flipped pairs: {len(pairs)}", flush=True)
    for c, items in by_condition.items():
        print(f"  {c}: {len(items)}", flush=True)
    if not pairs:
        print("No flipped pairs; exiting.", flush=True)
        return

    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    layers = phase2.parse_layers(args.layers, len(model.net.blocks))
    _, action_index = phase2.latent_indices_for_libero()
    target_indices = [int(action_index)]
    layer = layers[0]
    print(f"Layer: {layer}  Action slot: {action_index}  Ks: {args.oracle_ks}", flush=True)

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Phase A: Capture ──
    print("\n" + "=" * 60)
    print("Phase A: Capture")
    print("=" * 60)
    all_captures = capture_all(
        by_condition, cfg, model, stats, layers, args,
        target_indices, out_root)

    # ── Phase B: PCA + direction plan ──
    print("\n" + "=" * 60)
    print("Phase B: PCA fitting + direction plan")
    print("=" * 60)
    registry = build_pca_registry(all_captures, args.oracle_ks)
    merged_pca = None if args.no_merged else build_merged_pca(registry)
    if merged_pca is None:
        print("  (no merged PCA)", flush=True)

    direction_plan = build_direction_plan(
        registry, merged_pca, args.oracle_ks,
        no_cross=args.no_cross, no_merged=args.no_merged)
    total_dirs = sum(
        len(eps) for cond_eps in direction_plan.values()
        for eps in cond_eps.values())
    print(f"  Total per-sample directions pre-computed: {total_dirs}", flush=True)

    # ── Phase C: Recovery ──
    print("\n" + "=" * 60)
    print("Phase C: Recovery interventions")
    print("=" * 60)
    all_metrics = run_recovery(
        direction_plan, all_captures,
        cfg, model, stats, layer, args, target_indices, out_root)

    # ── Aggregate & save ──
    aggregates = _aggregate(all_metrics)

    # Determine common direction order
    first_cond = next(iter(all_metrics.values()))
    direction_order = first_cond.get("direction_order", [])

    overall = {
        "suite": summary["suite"],
        "base_task": summary["base_task"],
        "summary": str(summary_path),
        "output_dir": str(out_root),
        "layer": layer,
        "target_indices": target_indices,
        "oracle_ks": args.oracle_ks,
        "direction_order": direction_order,
        "num_flipped_pairs": len(pairs),
        "elapsed_s": time.time() - t0,
        "aggregates": aggregates,
    }
    (out_root / "summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_root / 'summary.json'}", flush=True)

    # ── Print key comparison ──
    print("\n" + "=" * 80)
    print("Phase C Ceiling: mean_shift vs oracle (alpha=1.0)")
    print("=" * 80)
    cols = ["mean_shift_a1", "oracle_k1_a1", "oracle_k4_a1", "oracle_k8_a1", "oracle_full_a1"]
    header = f"{'condition':25s}"
    for c in cols:
        header += f" {c:>10s}"
    print(header)
    print("-" * len(header))
    for condition in sorted(aggregates):
        agg = aggregates[condition]
        row = f"{condition:25s}"
        for c in cols:
            v = agg.get(c, {}).get("mean", float("nan"))
            row += f" {v:+.3f}   "
        print(row)

    # Cross-perturb summary
    cross_cols = [d for d in direction_order if d.startswith("cross_") and "_a1" not in d]
    if cross_cols:
        print("\n--- Cross-perturb oracle (k=8, alpha=1.0) ---")
        for condition in sorted(aggregates):
            agg = aggregates[condition]
            row = f"{condition:25s}"
            for other in sorted(registry):
                if other == condition:
                    continue
                label = f"cross_{other}_k8_a1"
                v = agg.get(label, {}).get("mean", float("nan"))
                row += f" {other[:8]:>8s}:{v:+.3f}"
            print(row)

    # Decomposition summary
    decomp_cols = ["mean_shift_a1", "mean_component_a1", "within_k8_a1", "orthogonal_k8_a1"]
    print("\n--- Decomposition: mean / within-subspace / orthogonal (alpha=1.0) ---")
    header2 = f"{'condition':25s}"
    for c in decomp_cols:
        header2 += f" {c:>16s}"
    print(header2)
    print("-" * len(header2))
    for condition in sorted(aggregates):
        agg = aggregates[condition]
        row = f"{condition:25s}"
        for c in decomp_cols:
            v = agg.get(c, {}).get("mean", float("nan"))
            row += f" {v:+.3f}         "
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    main()
