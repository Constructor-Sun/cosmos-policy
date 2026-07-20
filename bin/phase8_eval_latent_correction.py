#!/usr/bin/env python3
"""Evaluate a learned Phase 8 latent corrector online on first chunks."""

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


def _instruction_pair(clean: dict[str, Any], pert: dict[str, Any]) -> tuple[str, str]:
    clean_instr = clean["language"]
    pert_instr = pert["language"] if pert.get("instruction_mode") == "strict" else clean_instr
    return clean_instr, pert_instr


def _pair_observations(pair: dict[str, Any], args) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    clean_obs = phase2.first_observation(pair["clean"], args)
    if pair["condition"] == "language_instructions":
        return clean_obs, clean_obs
    return clean_obs, phase2.first_observation(pair["pert"], args)


def forward_capture_target(
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
    intervener = CosmosLayerCaptureIntervener(
        target_indices=target_model_indices(target),
        condition_pass_only=args.condition_pass_only,
    )
    intervener.reset(capture_layers={layer})
    with layer_shift_context(model, intervener, [layer]):
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


def pair_metrics(
    *,
    clean_action: np.ndarray,
    pert_action: np.ndarray,
    corrected_action: np.ndarray,
    clean_h: torch.Tensor,
    pert_h: torch.Tensor,
    corrected_h: torch.Tensor,
) -> dict[str, float]:
    base_rel = action_rel_error(pert_action, clean_action)
    corr_rel = action_rel_error(corrected_action, clean_action)
    base_mse = action_mse(pert_action, clean_action)
    corr_mse = action_mse(corrected_action, clean_action)
    out = {
        "baseline_action_rel_error": base_rel,
        "corrected_action_rel_error": corr_rel,
        "action_recovery_vs_pert": 1.0 - corr_rel / (base_rel + EPS),
        "baseline_action_mse": base_mse,
        "corrected_action_mse": corr_mse,
        "action_mse_recovery_vs_pert": 1.0 - corr_mse / (base_mse + EPS),
    }
    out.update(hidden_recovery(clean_h, pert_h, corrected_h))
    return out


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if key in row]
    return float(sum(vals) / len(vals)) if vals else None


def group_metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for group in ("preserved", "flipped", "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("group") == group]
        vals = [float(row[key]) for row in group_rows if key in row]
        if not vals:
            out[group] = {"n": 0, "mean": None, "num_positive": 0, "frac_positive": None}
            continue
        out[group] = {
            "n": len(vals),
            "mean": float(sum(vals) / len(vals)),
            "num_positive": sum(value > 0.0 for value in vals),
            "frac_positive": sum(value > 0.0 for value in vals) / len(vals),
        }
    return out


def print_action_recovery_by_group(stats: dict[str, dict[str, float | int | None]]) -> None:
    print("action_recovery_vs_pert by group:", flush=True)
    for group in ("preserved", "flipped", "all"):
        row = stats[group]
        mean = row["mean"]
        frac = row["frac_positive"]
        mean_s = "nan" if mean is None else f"{float(mean):.6f}"
        frac_s = "nan" if frac is None else f"{float(frac):.3f}"
        print(
            f"  {group}: n={row['n']} mean={mean_s} "
            f"num_positive={row['num_positive']} frac_positive={frac_s}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    default_summary = phase2.ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    default_output = phase2.ROOT / "experiments/phase8_eval_latent_correction/camera_viewpoints"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--summary", default=str(default_summary))
    p.add_argument("--output-dir", default=str(default_output))
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default=str(phase2.ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--conditions", nargs="*", default=["camera_viewpoints"])
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
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


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    corrector, ckpt = load_corrector(args.checkpoint, device)
    target_cfg = ckpt["target"]
    layer = int(target_cfg["layer"])
    target = str(target_cfg["target"])
    target_rms = float(target_cfg["target_rms"])
    print(f"checkpoint target: layer={layer} target={target} rms={target_rms:.6g}", flush=True)

    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)
    summary_path = pathlib.Path(args.summary)
    summary, pairs = phase2.discover_pairs(summary_path, set(args.conditions or []), set(args.groups))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"Discovered {len(pairs)} eval pairs from {summary_path}", flush=True)
    if not pairs:
        return

    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    start = time.time()
    for idx, pair in enumerate(pairs, 1):
        cond = pair["condition"]
        ep = int(pair["pert"]["episode"])
        print(f"[{idx}/{len(pairs)}] {cond}/ep{ep:04d} {pair['group']}", flush=True)
        try:
            clean_obs, pert_obs = _pair_observations(pair, args)
            clean_instr, pert_instr = _instruction_pair(pair["clean"], pair["pert"])
            clean_action, clean_h = forward_capture_target(
                cfg=cfg, model=model, stats=stats, obs=clean_obs,
                instruction=clean_instr, seed=args.seed, layer=layer, target=target, args=args)
            pert_action, pert_h = forward_capture_target(
                cfg=cfg, model=model, stats=stats, obs=pert_obs,
                instruction=pert_instr, seed=args.seed, layer=layer, target=target, args=args)
            corrected_action, _before, corrected_h = forward_corrected(
                cfg=cfg, model=model, stats=stats, corrector=corrector, obs=pert_obs,
                instruction=pert_instr, seed=args.seed, layer=layer, target=target,
                target_rms=target_rms, args=args)
            metrics = pair_metrics(
                clean_action=clean_action,
                pert_action=pert_action,
                corrected_action=corrected_action,
                clean_h=clean_h,
                pert_h=pert_h,
                corrected_h=corrected_h,
            )
            rows.append({
                "condition": cond,
                "group": pair["group"],
                "episode": ep,
                "clean_success": bool(pair["clean"]["success"]),
                "pert_success": bool(pair["pert"]["success"]),
                **metrics,
            })
        except Exception as exc:
            err = {"condition": cond, "episode": ep, "error": repr(exc)}
            errors.append(err)
            print(f"  skip: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    append_jsonl(out_root / "results.jsonl", rows)
    action_recovery_by_group = group_metric_stats(rows, "action_recovery_vs_pert")
    print_action_recovery_by_group(action_recovery_by_group)
    summary_payload = {
        "checkpoint": str(args.checkpoint),
        "summary": str(summary_path),
        "suite": summary.get("suite"),
        "base_task": summary.get("base_task"),
        "layer": layer,
        "target": target,
        "target_rms": target_rms,
        "alpha": args.alpha,
        "num_pairs": len(rows),
        "num_errors": len(errors),
        "mean_action_recovery_vs_pert": mean_metric(rows, "action_recovery_vs_pert"),
        "action_recovery_by_group": action_recovery_by_group,
        "mean_hidden_recovery_vs_pert": mean_metric(rows, "hidden_recovery_vs_pert"),
        "mean_corrected_action_rel_error": mean_metric(rows, "corrected_action_rel_error"),
        "errors": errors,
        "elapsed_s": time.time() - start,
    }
    write_json(out_root / "summary.json", summary_payload)
    print(f"Saved eval results to {out_root}", flush=True)


if __name__ == "__main__":
    main()
