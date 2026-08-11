#!/usr/bin/env python3
"""Oracle true-delta intervention for Phase 8 latent shift correction.

Injects the ground-truth ``delta_z = z_clean - z_pert`` at a target DiT block's
action slot and measures the resulting action recovery.  This reveals whether the
injection position itself (last layer + action-only) imposes a hard ceiling, or
whether the corrector model simply fails to predict the right direction.

Key diagnostic logic (see Section 9.1 of ShiftLearning.md):

    oracle action recovery ≈ 1
      → injection position is fine; problem is corrector direction prediction
    oracle action recovery << 1
      → last/action-only has a hard ceiling; need earlier layers or joint
        video correction before improving the corrector
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any

import numpy as np
import torch

import run_phase2_angular_cosmos as phase2
from cosmos_layer_shift import CosmosLayerCaptureIntervener, layer_shift_context
from phase8_correction_lib import (
    DynamicCorrectionContext,
    EPS,
    action_mse,
    action_rel_error,
    append_jsonl,
    hidden_recovery,
    load_corrector,
    target_model_indices,
    write_json,
)


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def forward_capture(
    *,
    cfg,
    model,
    stats,
    obs: dict[str, np.ndarray],
    instruction: str,
    seed: int,
    layer: int,
    target: str,
    args,
) -> tuple[np.ndarray, torch.Tensor]:
    """Run a single forward pass and capture hidden state at *layer*."""
    hook_mode = getattr(args, "hook_mode", "pre")
    intervener = CosmosLayerCaptureIntervener(
        target_indices=target_model_indices(target),
        condition_pass_only=args.condition_pass_only,
    )
    intervener.reset(capture_layers={layer})
    with layer_shift_context(model, intervener, [layer], hook_mode=hook_mode):
        phase2.set_seed_everywhere(args.reset_seed)
        out = phase2.get_action(
            cfg,
            model,
            stats,
            obs,
            instruction,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    if layer not in intervener.captured:
        raise RuntimeError(f"layer {layer} was not captured")
    return np.asarray(out["actions"], dtype=np.float32), intervener.captured[layer].cpu()


def forward_oracle(
    *,
    cfg,
    model,
    stats,
    obs: dict[str, np.ndarray],
    instruction: str,
    seed: int,
    layer: int,
    target: str,
    true_delta: torch.Tensor,
    args,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Run perturbed forward with *true_delta* injected at *layer* action slot.

    In ``"pre"`` mode the delta is injected at block entry, before self-attention.
    In ``"forward"`` mode the delta is injected at block exit, after all attention
    and MLP — only the per-position ``final_layer`` remains, so the correction is
    preserved to the output.

    Returns
    -------
    action : np.ndarray
        Decoded action after oracle correction.
    before : torch.Tensor
        The perturbed hidden state at the injection point (before injection).
    after : torch.Tensor
        The hidden state after oracle injection (≈ z_clean at this position).
    """
    hook_mode = getattr(args, "hook_mode", "pre")
    intervener = CosmosLayerCaptureIntervener(
        target_indices=target_model_indices(target),
        condition_pass_only=args.condition_pass_only,
    )
    # capture AND intervene at the same layer — capture runs first in the hook,
    # so *before* records the pre-intervention (perturbed) hidden state.
    intervener.reset(
        capture_layers={layer},
        intervene_layer=layer,
        direction=true_delta,
        alpha=1.0,
    )
    with layer_shift_context(model, intervener, [layer], hook_mode=hook_mode):
        phase2.set_seed_everywhere(args.reset_seed)
        out = phase2.get_action(
            cfg,
            model,
            stats,
            obs,
            instruction,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    if layer not in intervener.after:
        raise RuntimeError(f"oracle intervention at layer {layer} did not run")
    return (
        np.asarray(out["actions"], dtype=np.float32),
        intervener.captured[layer].cpu(),  # z_pert before injection
        intervener.after[layer].cpu(),      # z_pert + true_delta
    )


def forward_corrected(
    *,
    cfg,
    model,
    stats,
    corrector,
    obs: dict[str, np.ndarray],
    instruction: str,
    seed: int,
    layer: int,
    target: str,
    target_rms: float,
    args,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Run perturbed forward with the learned corrector (for comparison)."""
    with DynamicCorrectionContext(
        model,
        corrector,
        layer=layer,
        target=target,
        target_rms=target_rms,
        alpha=args.alpha,
        condition_pass_only=args.condition_pass_only,
    ) as ctx:
        phase2.set_seed_everywhere(args.reset_seed)
        out = phase2.get_action(
            cfg,
            model,
            stats,
            obs,
            instruction,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    if ctx.before is None or ctx.after is None:
        raise RuntimeError(f"correction at layer {layer} did not run")
    return np.asarray(out["actions"], dtype=np.float32), ctx.before, ctx.after


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    *,
    clean_action: np.ndarray,
    pert_action: np.ndarray,
    candidate_action: np.ndarray,
    clean_h: torch.Tensor,
    pert_h: torch.Tensor,
    candidate_h: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    """Return per-pair metrics for one correction method."""
    base_rel = action_rel_error(pert_action, clean_action)
    cand_rel = action_rel_error(candidate_action, clean_action)
    base_mse = action_mse(pert_action, clean_action)
    cand_mse = action_mse(candidate_action, clean_action)
    out: dict[str, Any] = {
        f"{label}_action_rel_error": cand_rel,
        f"{label}_action_recovery_vs_pert": 1.0 - cand_rel / (base_rel + EPS),
        f"{label}_action_mse": cand_mse,
        f"{label}_action_mse_recovery": 1.0 - cand_mse / (base_mse + EPS),
    }
    hr = hidden_recovery(clean_h, pert_h, candidate_h)
    out[f"{label}_hidden_recovery_vs_pert"] = hr["hidden_recovery_vs_pert"]
    out[f"{label}_baseline_hidden_mse"] = hr["baseline_hidden_mse"]
    out[f"{label}_corrected_hidden_mse"] = hr["corrected_hidden_mse"]

    # Delta direction diagnostics
    true_delta = clean_h.detach().float() - pert_h.detach().float()
    effective_delta = candidate_h.detach().float() - pert_h.detach().float()
    out.update(_delta_metrics(true_delta, effective_delta, prefix=f"{label}_delta"))
    return out


def _delta_metrics(
    true_delta: torch.Tensor,
    pred_delta: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    true_flat = true_delta.reshape(-1)
    pred_flat = pred_delta.reshape(-1)
    t_norm = torch.linalg.vector_norm(true_flat)
    p_norm = torch.linalg.vector_norm(pred_flat)
    dot = torch.dot(true_flat, pred_flat)
    cosine = float(torch.clamp(dot / (t_norm * p_norm + EPS), -1.0, 1.0))
    optimal_alpha = float(dot / (p_norm.square() + EPS))
    oracle_error = torch.mean((optimal_alpha * pred_flat - true_flat) ** 2)
    baseline_error = torch.mean(true_flat ** 2)
    return {
        f"{prefix}_cosine": cosine,
        f"{prefix}_angle_deg": float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine))))),
        f"{prefix}_norm_ratio": float(p_norm / (t_norm + EPS)),
        f"{prefix}_optimal_alpha": optimal_alpha,
        f"{prefix}_oracle_scaled_recovery": float(1.0 - oracle_error / (baseline_error + EPS)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if key in row]
    return float(sum(vals) / len(vals)) if vals else None


def group_stats(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in ("preserved", "flipped", "unknown", "all"):
        group_rows = rows if group == "all" else [r for r in rows if r.get("group") == group]
        vals = [float(r[key]) for r in group_rows if key in r]
        out[group] = {
            "n": len(vals),
            "mean": float(sum(vals) / len(vals)) if vals else None,
            "num_positive": sum(v > 0.0 for v in vals),
            "frac_positive": sum(v > 0.0 for v in vals) / len(vals) if vals else None,
        }
    return out


def print_group_table(stats: dict[str, dict[str, Any]], label: str) -> None:
    print(f"\n{label} by group:", flush=True)
    for group in ("preserved", "flipped", "all"):
        row = stats[group]
        mean_s = "nan" if row["mean"] is None else f"{float(row['mean']):.6f}"
        frac_s = "nan" if row["frac_positive"] is None else f"{float(row['frac_positive']):.3f}"
        print(f"  {group}: n={row['n']} mean={mean_s} "
              f"num_positive={row['num_positive']} frac_positive={frac_s}", flush=True)


def print_delta_report(rows: list[dict[str, Any]], prefix: str, label: str) -> None:
    print(f"\n{label} delta diagnostics:", flush=True)
    if not rows:
        print("  no data", flush=True)
        return
    cosine = mean_metric(rows, f"{prefix}_delta_cosine")
    angle = mean_metric(rows, f"{prefix}_delta_angle_deg")
    ratio = mean_metric(rows, f"{prefix}_delta_norm_ratio")
    alpha = mean_metric(rows, f"{prefix}_delta_optimal_alpha")
    oracle_rec = mean_metric(rows, f"{prefix}_delta_oracle_scaled_recovery")
    print(f"  cosine={cosine:.6f}  angle_deg={angle:.3f}  norm_ratio={ratio:.6f}  "
          f"optimal_alpha={alpha:.6f}  oracle_scaled_recovery={oracle_rec:.6f}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    default_summary = str(
        phase2.ROOT
        / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case"
        / "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
          "__all_conditions__20pair_combined_summary.json"
    )
    default_output = str(
        phase2.ROOT / "experiments/phase8_oracle_intervention/camera_viewpoints"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default=default_summary)
    p.add_argument("--output-dir", default=default_output)
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument(
        "--t5-extra-embeddings",
        default=str(
            phase2.ROOT
            / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"
        ),
    )
    p.add_argument("--conditions", nargs="*", default=["camera_viewpoints"])
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--max-pairs", type=int, default=0)

    # Injection target
    p.add_argument("--target-layer", default="last", help="layer spec: 'last' or integer")
    p.add_argument("--target", default="action", choices=["action", "video", "video_action"])
    p.add_argument("--alpha", type=float, default=1.0,
                   help="scaling for the *corrector* path (oracle always uses 1.0)")
    p.add_argument("--hook-mode", default="pre", choices=["pre", "forward"],
                   help="'pre' = block entry (register_forward_pre_hook); "
                        "'forward' = block exit (register_forward_hook)")

    # Corrector (optional — for side-by-side comparison; only meaningful with --hook-mode pre)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to a learned corrector .pt; if given, also run corrector for comparison",
    )

    # Runtime
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--condition-pass-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fail-fast", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    # --- Load corrector (optional) ---
    corrector = None
    target_cfg: dict[str, Any] = {}
    target_rms: float = 1.0
    if args.checkpoint is not None:
        if args.hook_mode == "forward":
            print("WARNING: --checkpoint with --hook-mode forward is not meaningful — "
                  "corrector was trained on pre_hook data.  Skipping corrector comparison.",
                  flush=True)
        else:
            corrector, ckpt = load_corrector(args.checkpoint, device)
            target_cfg = ckpt["target"]
            target_rms = float(target_cfg["target_rms"])
            print(f"Loaded corrector: {args.checkpoint}", flush=True)
    if corrector is None and args.checkpoint is None:
        print("No --checkpoint given; running oracle-only (no corrector comparison).",
              flush=True)

    # --- Resolve target layer ---
    # We need the model to be loaded to resolve "last".  Do a two-pass:
    # load model, resolve layer, then continue.
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)

    summary_path = pathlib.Path(args.summary)
    summary, pairs = phase2.discover_pairs(
        summary_path, set(args.conditions or []), set(args.groups)
    )
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"Discovered {len(pairs)} eval pairs from {summary_path}", flush=True)
    if not pairs:
        print("No pairs to evaluate.", flush=True)
        return

    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)

    # Resolve target layer
    n_blocks = len(model.net.blocks)
    if args.target_layer == "last":
        layer = n_blocks - 1
    else:
        layer = int(args.target_layer)
    if layer < 0 or layer >= n_blocks:
        raise ValueError(f"target_layer {layer} out of range [0, {n_blocks - 1}]")
    target = args.target
    print(f"Target: layer={layer} ({layer}/{n_blocks})  target={target}  "
          f"hook_mode={args.hook_mode}", flush=True)

    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    start = time.time()

    for idx, pair in enumerate(pairs, 1):
        cond = pair["condition"]
        ep = int(pair["pert"]["episode"])
        grp = pair["group"]
        print(f"\n[{idx}/{len(pairs)}] {cond}/ep{ep:04d}  group={grp}", flush=True)

        try:
            # --- Observations ---
            clean_obs, pert_obs = _pair_observations(pair, args)
            clean_instr = pair["clean"]["language"]
            pert_instr = (
                pair["pert"]["language"]
                if pair["pert"].get("instruction_mode") == "strict"
                else clean_instr
            )

            # --- 1. Clean forward ---
            clean_action, clean_h = forward_capture(
                cfg=cfg, model=model, stats=stats,
                obs=clean_obs, instruction=clean_instr,
                seed=args.seed, layer=layer, target=target, args=args,
            )

            # --- 2. Perturbed forward ---
            pert_action, pert_h = forward_capture(
                cfg=cfg, model=model, stats=stats,
                obs=pert_obs, instruction=pert_instr,
                seed=args.seed, layer=layer, target=target, args=args,
            )

            # --- 3. True delta ---
            true_delta = clean_h.float() - pert_h.float()

            # --- 4. Oracle injection ---
            oracle_action, oracle_before, oracle_after = forward_oracle(
                cfg=cfg, model=model, stats=stats,
                obs=pert_obs, instruction=pert_instr,
                seed=args.seed, layer=layer, target=target,
                true_delta=true_delta, args=args,
            )

            row: dict[str, Any] = {
                "condition": cond,
                "group": grp,
                "episode": ep,
                "clean_success": bool(pair["clean"]["success"]),
                "pert_success": bool(pair["pert"]["success"]),
            }

            # Baseline (perturbed vs clean — same for all methods)
            base_rel = action_rel_error(pert_action, clean_action)
            base_mse = action_mse(pert_action, clean_action)
            row["baseline_action_rel_error"] = base_rel
            row["baseline_action_mse"] = base_mse

            # Oracle metrics
            row.update(
                compute_metrics(
                    clean_action=clean_action,
                    pert_action=pert_action,
                    candidate_action=oracle_action,
                    clean_h=clean_h,
                    pert_h=pert_h,
                    candidate_h=oracle_after,
                    label="oracle",
                )
            )

            # --- 5. Corrector (optional comparison) ---
            if corrector is not None:
                corr_action, corr_before, corr_after = forward_corrected(
                    cfg=cfg, model=model, stats=stats, corrector=corrector,
                    obs=pert_obs, instruction=pert_instr,
                    seed=args.seed, layer=layer, target=target,
                    target_rms=target_rms, args=args,
                )
                # Compute corrector's direct prediction (no online injection,
                # just the model output on z_pert)
                with torch.no_grad():
                    direct_delta = (
                        corrector(pert_h.to(device).float()).cpu() * target_rms
                    )
                corr_metrics = compute_metrics(
                    clean_action=clean_action,
                    pert_action=pert_action,
                    candidate_action=corr_action,
                    clean_h=clean_h,
                    pert_h=pert_h,
                    candidate_h=corr_after,
                    label="corrector",
                )
                # Also compute direct predictor delta alignment
                corr_metrics.update(
                    _delta_metrics(
                        clean_h.float() - pert_h.float(),
                        direct_delta,
                        prefix="corrector_direct_delta",
                    )
                )
                row.update(corr_metrics)

                # Compute oracle-vs-corrector gap
                for metric in ("action_recovery_vs_pert", "action_mse_recovery",
                               "hidden_recovery_vs_pert"):
                    oracle_val = row.get(f"oracle_{metric}")
                    corr_val = row.get(f"corrector_{metric}")
                    if oracle_val is not None and corr_val is not None:
                        row[f"gap_{metric}"] = float(oracle_val) - float(corr_val)

            rows.append(row)

            # Quick per-case summary
            ora_rec = row.get("oracle_action_recovery_vs_pert", float("nan"))
            ora_hid = row.get("oracle_hidden_recovery_vs_pert", float("nan"))
            print(f"  oracle action_recovery={ora_rec:.6f}  "
                  f"hidden_recovery={ora_hid:.6f}", flush=True)
            if corrector is not None:
                cor_rec = row.get("corrector_action_recovery_vs_pert", float("nan"))
                gap = row.get("gap_action_recovery_vs_pert", float("nan"))
                print(f"  corrector action_recovery={cor_rec:.6f}  "
                      f"oracle−corrector gap={gap:.6f}", flush=True)

        except Exception as exc:
            errors.append({"condition": cond, "episode": ep, "error": repr(exc)})
            print(f"  SKIP: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Aggregate & save ---
    append_jsonl(out_root / "results.jsonl", rows)

    # Print summary tables
    print("\n" + "=" * 70)
    print("ORACLE INTERVENTION RESULTS")
    print("=" * 70)

    for label, prefix in [("oracle", "oracle"), ("corrector", "corrector")]:
        key = f"{prefix}_action_recovery_vs_pert"
        if key not in (rows[0] if rows else {}):
            continue
        print_group_table(group_stats(rows, key), f"{label} action_recovery_vs_pert")
        print_delta_report(rows, prefix, label)

    if corrector is not None and rows:
        print("\n--- Oracle − Corrector gap ---")
        for gap_key in ("gap_action_recovery_vs_pert", "gap_action_mse_recovery",
                         "gap_hidden_recovery_vs_pert"):
            gm = mean_metric(rows, gap_key)
            if gm is not None:
                print(f"  mean {gap_key}: {gm:.6f}", flush=True)

    # Build summary dict
    summary_payload: dict[str, Any] = {
        "summary": str(summary_path),
        "suite": summary.get("suite"),
        "base_task": summary.get("base_task"),
        "layer": layer,
        "target": target,
        "hook_mode": args.hook_mode,
        "num_pairs": len(rows),
        "num_errors": len(errors),
        "elapsed_s": time.time() - start,
        "errors": errors,
    }

    for prefix in ["oracle"] + (["corrector"] if corrector is not None else []):
        summary_payload[f"{prefix}_mean_action_recovery"] = mean_metric(
            rows, f"{prefix}_action_recovery_vs_pert"
        )
        summary_payload[f"{prefix}_mean_action_mse_recovery"] = mean_metric(
            rows, f"{prefix}_action_mse_recovery"
        )
        summary_payload[f"{prefix}_mean_hidden_recovery"] = mean_metric(
            rows, f"{prefix}_hidden_recovery_vs_pert"
        )
        summary_payload[f"{prefix}_mean_delta_cosine"] = mean_metric(
            rows, f"{prefix}_delta_cosine"
        )
        summary_payload[f"{prefix}_mean_delta_angle_deg"] = mean_metric(
            rows, f"{prefix}_delta_angle_deg"
        )
        summary_payload[f"{prefix}_action_recovery_by_group"] = group_stats(
            rows, f"{prefix}_action_recovery_vs_pert"
        )

    write_json(out_root / "summary.json", summary_payload)
    print(f"\nSaved results to {out_root}", flush=True)


def _pair_observations(
    pair: dict[str, Any], args
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    clean_obs = phase2.first_observation(pair["clean"], args)
    if pair["condition"] == "language_instructions":
        return clean_obs, clean_obs
    return clean_obs, phase2.first_observation(pair["pert"], args)


if __name__ == "__main__":
    main()
