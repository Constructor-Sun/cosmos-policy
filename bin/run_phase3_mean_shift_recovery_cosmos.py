#!/usr/bin/env python3
"""Run Cosmos Phase 3 action-slot mean-shift recovery."""

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
    alpha_file_key,
    alpha_key,
    build_aggregates,
    direction_consistency,
    group_pairs_by_condition,
    recovery_metrics,
    unique_alphas,
)


def _instruction_pair(clean: dict[str, Any], pert: dict[str, Any]) -> tuple[str, str]:
    clean_instr = clean["language"]
    pert_instr = pert["language"] if pert.get("instruction_mode") == "strict" else clean_instr
    return clean_instr, pert_instr


def _pair_observations(pair: dict[str, Any], args) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    clean = pair["clean"]
    pert = pair["pert"]
    clean_obs = phase2.first_observation(clean, args)
    pert_obs = clean_obs if pair["condition"] == "language_instructions" else phase2.first_observation(pert, args)
    return clean_obs, pert_obs


def _forward_with_layer_hooks(
    *,
    cfg,
    model,
    stats,
    obs: dict[str, np.ndarray],
    instruction: str,
    seed: int,
    layers: list[int],
    args,
    target_indices: list[int],
    intervene_layer: int | None = None,
    direction: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
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
    return np.asarray(out["actions"], dtype=np.float32), intervener.captured, intervener.after


def _capture_pair(
    *,
    cfg,
    model,
    stats,
    pair: dict[str, Any],
    layers: list[int],
    args,
    target_indices: list[int],
) -> dict[str, Any]:
    clean, pert = pair["clean"], pair["pert"]
    clean_obs, pert_obs = _pair_observations(pair, args)
    clean_instr, pert_instr = _instruction_pair(clean, pert)

    clean_action, clean_h, _ = _forward_with_layer_hooks(
        cfg=cfg,
        model=model,
        stats=stats,
        obs=clean_obs,
        instruction=clean_instr,
        seed=args.seed,
        layers=layers,
        args=args,
        target_indices=target_indices,
    )
    pert_action, pert_h, _ = _forward_with_layer_hooks(
        cfg=cfg,
        model=model,
        stats=stats,
        obs=pert_obs,
        instruction=pert_instr,
        seed=args.seed,
        layers=layers,
        args=args,
        target_indices=target_indices,
    )

    missing = [layer for layer in layers if layer not in clean_h or layer not in pert_h]
    if missing:
        raise RuntimeError(f"missing captured target hidden layers {missing}")

    directions = {
        layer: token_mean_shift_direction(clean_h[layer], pert_h[layer])
        for layer in layers
    }
    return {
        "condition": pair["condition"],
        "episode": int(pert["episode"]),
        "clean": clean,
        "pert": pert,
        "clean_obs": clean_obs,
        "pert_obs": pert_obs,
        "clean_instruction": clean_instr,
        "pert_instruction": pert_instr,
        "clean_action": clean_action,
        "pert_action": pert_action,
        "clean_h": clean_h,
        "pert_h": pert_h,
        "directions": directions,
    }


def _save_capture(ep_dir: pathlib.Path, cap: dict[str, Any], layers: list[int]) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.save(ep_dir / "action_clean.npy", cap["clean_action"])
    np.save(ep_dir / "action_pert.npy", cap["pert_action"])
    target_name = cap["target_name"]
    torch.save({str(layer): cap["clean_h"][layer] for layer in layers}, ep_dir / f"hidden_{target_name}_clean.pt")
    torch.save({str(layer): cap["pert_h"][layer] for layer in layers}, ep_dir / f"hidden_{target_name}_pert.pt")
    payload = {
        "condition": cap["condition"],
        "episode": cap["episode"],
        "clean_task_name": cap["clean"]["task_name"],
        "pert_task_name": cap["pert"]["task_name"],
        "clean_instruction": cap["clean_instruction"],
        "pert_instruction": cap["pert_instruction"],
        "instruction_mode": cap["pert"].get("instruction_mode", "task"),
        "layers": layers,
        "target_name": target_name,
        "target_indices": cap["target_indices"],
    }
    (ep_dir / "capture_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _intervene_pair(
    *,
    cfg,
    model,
    stats,
    cap: dict[str, Any],
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    args,
    target_indices: list[int],
) -> tuple[np.ndarray, torch.Tensor]:
    action, _, after = _forward_with_layer_hooks(
        cfg=cfg,
        model=model,
        stats=stats,
        obs=cap["pert_obs"],
        instruction=cap["pert_instruction"],
        seed=args.seed,
        layers=[layer],
        args=args,
        target_indices=target_indices,
        intervene_layer=layer,
        direction=direction,
        alpha=alpha,
    )
    if layer not in after:
        raise RuntimeError(f"intervention at layer {layer} did not produce post-shift hidden")
    return action, after[layer]


def _run_condition(
    *,
    condition: str,
    pairs: list[dict[str, Any]],
    cfg,
    model,
    stats,
    layers: list[int],
    alphas: list[float],
    args,
    target_name: str,
    target_indices: list[int],
    out_root: pathlib.Path,
) -> dict[str, Any]:
    group_dir = out_root / condition
    group_dir.mkdir(parents=True, exist_ok=True)
    captures: dict[int, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    print(f"\n== Phase A capture: {condition} n={len(pairs)} ==", flush=True)
    for idx, pair in enumerate(pairs, 1):
        ep = int(pair["pert"]["episode"])
        print(f"  [{idx}/{len(pairs)}] capture ep{ep:02d}", flush=True)
        try:
            cap = _capture_pair(
                cfg=cfg,
                model=model,
                stats=stats,
                pair=pair,
                layers=layers,
                args=args,
                target_indices=target_indices,
            )
            cap["target_name"] = target_name
            cap["target_indices"] = target_indices
            captures[ep] = cap
            _save_capture(group_dir / f"ep{ep:02d}", cap, layers)
        except Exception as exc:
            skipped[str(ep)] = repr(exc)
            print(f"    skip ep{ep:02d}: {exc}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not captures:
        raise RuntimeError(f"no successful captures for {condition}")

    mean_directions: dict[int, torch.Tensor] = {}
    consistency: dict[str, Any] = {}
    labels = [f"ep{ep:02d}" for ep in sorted(captures)]
    for layer in layers:
        dirs = [captures[ep]["directions"][layer] for ep in sorted(captures)]
        mean_directions[layer] = torch.stack(dirs, dim=0).mean(dim=0)
        consistency[str(layer)] = direction_consistency(dirs, labels)

    torch.save({str(layer): mean_directions[layer] for layer in layers}, group_dir / "mean_shift_directions.pt")
    (group_dir / "direction_consistency.json").write_text(
        json.dumps(consistency, indent=2),
        encoding="utf-8",
    )

    metrics: dict[str, Any] = {
        "condition": condition,
        "direction_source": args.direction_source,
        "direction_formula": f"mean_flipped(clean_{target_name}_h - pert_{target_name}_h)",
        "intervention_formula": f"x[:, {target_indices}] += alpha * direction",
        "stage": f"{target_name}_slot" if len(target_indices) == 1 else f"{target_name}_slots",
        "target_name": target_name,
        "target_indices": target_indices,
        "site": "model.net.blocks[L] forward_pre_hook",
        "condition_pass_only": bool(args.condition_pass_only),
        "layers": layers,
        "alphas": [alpha_key(a) for a in alphas],
        "seed": args.seed,
        "num_denoising_steps": args.num_denoising_steps,
        "skipped": skipped,
        "direction_consistency": consistency,
        "results": {},
    }

    print(f"== Phase B recovery: {condition} captured={len(captures)} ==", flush=True)
    for ep in sorted(captures):
        cap = captures[ep]
        print(f"  intervene ep{ep:02d}", flush=True)
        metrics["results"][str(ep)] = {"episode": ep, "layers": {}}
        for layer in layers:
            if args.direction_source == "same_seed":
                direction = cap["directions"][layer]
                effective_source = "same_seed"
            else:
                direction = mean_directions[layer]
                effective_source = "all_flipped_mean"

            layer_dir = group_dir / f"ep{ep:02d}" / f"L{layer:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            metrics["results"][str(ep)]["layers"][str(layer)] = {
                "effective_direction_source": effective_source,
                "per_alpha": {},
            }
            for alpha in alphas:
                key = alpha_key(alpha)
                if float(alpha) == 0.0:
                    action_np = cap["pert_action"]
                    hidden_after = cap["pert_h"][layer]
                else:
                    action_np, hidden_after = _intervene_pair(
                        cfg=cfg,
                        model=model,
                        stats=stats,
                        cap=cap,
                        layer=layer,
                        direction=direction,
                        alpha=alpha,
                        args=args,
                        target_indices=target_indices,
                    )
                np.save(layer_dir / f"action_alpha_{alpha_file_key(alpha)}.npy", action_np)
                rec = recovery_metrics(
                    action_np=action_np,
                    clean_action=cap["clean_action"],
                    pert_action=cap["pert_action"],
                    hidden_after=hidden_after,
                    clean_h=cap["clean_h"][layer],
                    pert_h=cap["pert_h"][layer],
                )
                rec["alpha"] = float(alpha)
                rec["target_hidden_recovery_vs_pert"] = rec["hidden_recovery_vs_pert"]
                rec[f"{target_name}_hidden_recovery_vs_pert"] = rec["hidden_recovery_vs_pert"]
                metrics["results"][str(ep)]["layers"][str(layer)]["per_alpha"][key] = rec
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    (group_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    default_summary = phase2.ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    default_output = phase2.ROOT / (
        "experiments/phase3_mean_shift_cosmos/"
        "kitchen_scene4_seed7_all_perturb_flipped_action_slot"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default=str(default_summary))
    p.add_argument("--output-dir", default=str(default_output))
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default=str(phase2.ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--layers", nargs="+", default=["0", "mid", "last"])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    p.add_argument("--intervention-target", choices=["action", "video"], default="action")
    p.add_argument("--direction-source", choices=["all_flipped_mean", "same_seed"], default="all_flipped_mean")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--condition-pass-only", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)

    summary_path = pathlib.Path(args.summary)
    summary, pairs = phase2.discover_pairs(summary_path, set(args.conditions or []), {"flipped"})
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    by_condition = group_pairs_by_condition(pairs)
    print(f"Discovered {len(pairs)} flipped pairs from {summary_path}", flush=True)
    for condition, items in by_condition.items():
        print(f"  {condition}: {len(items)}", flush=True)
    if not pairs:
        print("No flipped pairs selected; exiting before model load.", flush=True)
        return

    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    layers = phase2.parse_layers(args.layers, len(model.net.blocks))
    alphas = unique_alphas(args.alphas)
    video_indices, action_index = phase2.latent_indices_for_libero()
    if args.intervention_target == "video":
        target_indices = list(video_indices)
    else:
        target_indices = [int(action_index)]
    print(f"Layers: {layers}  alphas: {[alpha_key(a) for a in alphas]}", flush=True)
    print(f"Action latent slot: {action_index}", flush=True)
    print(
        f"Intervention target: {args.intervention_target} slots={target_indices}",
        flush=True,
    )

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    groups: dict[str, dict[str, Any]] = {}
    for condition, condition_pairs in by_condition.items():
        groups[condition] = _run_condition(
            condition=condition,
            pairs=condition_pairs,
            cfg=cfg,
            model=model,
            stats=stats,
            layers=layers,
            alphas=alphas,
            args=args,
            target_name=args.intervention_target,
            target_indices=target_indices,
            out_root=out_root,
        )

    overall = {
        "suite": summary["suite"],
        "base_task": summary["base_task"],
        "summary": str(summary_path),
        "output_dir": str(out_root),
        "direction_source": args.direction_source,
        "intervention_target": args.intervention_target,
        "target_indices": target_indices,
        "condition_pass_only": bool(args.condition_pass_only),
        "layers": layers,
        "alphas": [alpha_key(a) for a in alphas],
        "num_flipped_pairs": len(pairs),
        "elapsed_s": time.time() - start,
        "groups": groups,
        "by_perturbation": build_aggregates(groups, layers, alphas),
    }
    (out_root / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"Saved: {out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
