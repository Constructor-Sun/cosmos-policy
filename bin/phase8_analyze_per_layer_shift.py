#!/usr/bin/env python3
"""Analyze per-layer shift (perturb latent - clean latent) during first-chunk inference.

For each of the 20 eval cases in phase8_eval_latent_correction, captures hidden states
at EVERY DiT block during specified denoising steps, for both clean and perturb inputs.
Then computes and visualises how the shift evolves across layers and denoising steps.

Usage:
    # First denoising step only (default)
    python bin/phase8_analyze_per_layer_shift.py \
        --checkpoint .../final.pt --capture-steps 0

    # All 5 denoising steps
    python bin/phase8_analyze_per_layer_shift.py \
        --checkpoint .../final.pt --capture-steps 0 1 2 3 4

    # 4-quantile sampling (0%, ~33%, ~67%, 100% of denoising)
    python bin/phase8_analyze_per_layer_shift.py \
        --checkpoint .../final.pt --capture-steps 0 1 2 4
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
    append_jsonl,
    load_corrector,
    target_model_indices,
    write_json,
)


# ── Multi-step intervener ────────────────────────────────────────────────────

class MultiStepCaptureIntervener(CosmosLayerCaptureIntervener):
    """Captures hidden states at specified denoising steps during CFG inference.

    In CFG sampling with N denoising steps, conditioned passes occur at
    pass_idx = 0, 2, 4, ..., 2*(N-1).  This intervener captures at the
    conditioned passes corresponding to the user-specified *denoising step
    indices* (0-indexed), storing results keyed by step index.

    The parent hook MUST always run (for pass tracking at layer 0).  We post-
    filter the captures: desired steps are moved from self.captured into
    self.step_captured; unwanted captures are discarded.
    """

    def __init__(self, capture_steps: set[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self._capture_steps = {int(s) for s in capture_steps}
        # per-step captured dicts: {step: {layer_idx: tensor}}
        self.step_captured: dict[int, dict[int, torch.Tensor]] = {}

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.step_captured = {}

    def _current_step(self) -> int | None:
        """Denoising step index for the current conditioned pass, or None."""
        if not self._active_pass():
            return None
        return self.pass_idx // 2

    def pre_forward_hook(self, layer_idx: int):
        layer_idx = int(layer_idx)
        _parent_hook = super().pre_forward_hook(layer_idx)
        _capture_steps = self._capture_steps

        def hook(_module, inputs):
            # Always let parent run — it handles pass_idx tracking at layer 0
            # and captures into self.captured for every active (conditioned) pass.
            result = _parent_hook(_module, inputs)

            # Post-filter: keep only desired steps, discard the rest.
            step = self._current_step()
            if step is not None and step in _capture_steps:
                if layer_idx in self.captured:
                    self.step_captured.setdefault(step, {})[layer_idx] = self.captured.pop(layer_idx)
            elif layer_idx in self.captured:
                del self.captured[layer_idx]

            return result

        return hook


# ── Forward helpers ──────────────────────────────────────────────────────────

def _instruction_pair(clean: dict[str, Any], pert: dict[str, Any]) -> tuple[str, str]:
    clean_instr = clean["language"]
    pert_instr = pert["language"] if pert.get("instruction_mode") == "strict" else clean_instr
    return clean_instr, pert_instr


def _pair_observations(
    pair: dict[str, Any], args
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    clean_obs = phase2.first_observation(pair["clean"], args)
    if pair["condition"] == "language_instructions":
        return clean_obs, clean_obs
    return clean_obs, phase2.first_observation(pair["pert"], args)


def forward_capture_all_layers(
    *,
    cfg,
    model,
    stats,
    obs: dict[str, np.ndarray],
    instruction: str,
    seed: int,
    target: str,
    num_blocks: int,
    capture_steps: set[int],
    args,
) -> dict[int, dict[int, torch.Tensor]]:
    """Run one forward pass capturing hidden states at EVERY layer, at every step
    in *capture_steps*.  Returns {step_idx: {layer_idx: tensor}}.

    When target="video_action", the captured tensor has shape [B, 3, 14, 14, 2048]
    where slots 0,1 are video (wrist, primary) and slot 2 is action.
    """
    all_layers = set(range(num_blocks))
    intervener = MultiStepCaptureIntervener(
        target_indices=target_model_indices(target),
        condition_pass_only=True,
        capture_steps=capture_steps,
    )
    intervener.reset(capture_layers=all_layers)
    with layer_shift_context(model, intervener, list(all_layers)):
        phase2.set_seed_everywhere(args.reset_seed)
        phase2.get_action(
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
    return {
        step: {layer: t.cpu() for layer, t in layers.items()}
        for step, layers in intervener.step_captured.items()
    }


# ── Per-layer shift metrics ──────────────────────────────────────────────────

def compute_layer_shift_metrics(
    clean_h: torch.Tensor,
    pert_h: torch.Tensor,
) -> dict[str, float]:
    """Compute various shift metrics between clean and perturb hidden states."""
    clean_f = clean_h.detach().float()
    pert_f = pert_h.detach().float()

    diff = pert_f - clean_f  # shift = perturb - clean
    diff_norm = torch.norm(diff).item()
    clean_norm = torch.norm(clean_f).item()
    pert_norm = torch.norm(pert_f).item()

    mse = torch.mean(diff ** 2).item()

    cos_sim = torch.nn.functional.cosine_similarity(
        clean_f.reshape(1, -1), pert_f.reshape(1, -1)
    ).item()

    rel_norm_change = diff_norm / (clean_norm + 1e-8)
    fve = 1.0 - (diff_norm ** 2) / (clean_norm ** 2 + 1e-8)

    return {
        "shift_l2": diff_norm,
        "clean_l2": clean_norm,
        "pert_l2": pert_norm,
        "shift_mse": mse,
        "cosine_similarity": cos_sim,
        "rel_norm_change": rel_norm_change,
        "fraction_var_explained": fve,
    }


# ── Aggregation helper ───────────────────────────────────────────────────────

def _aggregate_layer_rows(
    layer_rows: list[dict[str, Any]],
    layer_idx: int,
    step: int,
    slot_group: str | None,
    layer_summary: dict[str, dict[str, Any]],
) -> None:
    n = len(layer_rows)
    if n == 0:
        return
    if slot_group:
        key = f"step{step}_layer{layer_idx}_{slot_group}"
    else:
        key = f"step{step}_layer{layer_idx}"
    stats_dict = {}
    for metric in ["shift_l2", "clean_l2", "pert_l2", "shift_mse",
                   "cosine_similarity", "rel_norm_change", "fraction_var_explained"]:
        vals = [float(r[metric]) for r in layer_rows]
        stats_dict[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": n,
        }
    for grp in ["preserved", "flipped", "all"]:
        grp_rows = [r for r in layer_rows if grp == "all" or r["group"] == grp]
        stats_dict[f"group_{grp}"] = {}
        for metric in ["shift_l2", "shift_mse", "cosine_similarity"]:
            gvals = [float(r[metric]) for r in grp_rows]
            stats_dict[f"group_{grp}"][metric] = {
                "mean": float(np.mean(gvals)) if gvals else None,
                "std": float(np.std(gvals)) if gvals else None,
                "n": len(gvals),
            }
    layer_summary[key] = stats_dict


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    default_summary = phase2.ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    default_output = (
        phase2.ROOT
        / "experiments/phase8_eval_latent_correction/per_layer_shift_analysis"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--summary", default=str(default_summary))
    p.add_argument("--output-dir", default=str(default_output))
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument(
        "--t5-extra-embeddings",
        default="",
        help="Extra T5 embeddings pickle. Leave empty to skip (default).",
    )
    p.add_argument("--conditions", nargs="*", default=["camera_viewpoints"])
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--condition-pass-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--layer-step", type=int, default=1,
                   help="Capture every N layers (default 1 = all layers)")
    p.add_argument("--capture-steps", type=int, nargs="+", default=[0],
                   help="Denoising step indices to capture (0-indexed). "
                        "Default [0] = first step only.  Use '0 1 2 3 4' for all 5 steps. "
                        "Use '0 1 2 4' for 4-quantile sampling.")
    p.add_argument("--target", default="",
                   help="Target slot override: 'action' (slot 4), 'video' (slots 2,3), "
                        "or 'video_action' (both — reports video & action metrics separately). "
                        "Default: use checkpoint's target (action).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    capture_steps = set(args.capture_steps)
    # Validate against actual number of steps (will be checked at runtime)
    max_step = max(capture_steps) if capture_steps else -1
    if max_step >= args.num_denoising_steps:
        raise ValueError(
            f"capture_steps max ({max_step}) >= num_denoising_steps ({args.num_denoising_steps})"
        )
    print(f"Will capture at denoising steps: {sorted(capture_steps)} "
          f"(conditioned pass indices: {[2*s for s in sorted(capture_steps)]})")

    # Load corrector checkpoint for metadata
    corrector, ckpt = load_corrector(args.checkpoint, device)
    target_cfg = ckpt["target"]
    layer_inject = int(target_cfg["layer"])
    target = args.target or str(target_cfg["target"])
    target_rms = float(target_cfg["target_rms"])
    split_slots = target == "video_action"  # report video & action separately
    print(f"checkpoint target: layer={layer_inject} target={target} rms={target_rms:.6g}")
    if split_slots:
        print("  → will report video (slots 2,3) and action (slot 4) metrics separately")

    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)

    summary_path = pathlib.Path(args.summary)
    summary, pairs = phase2.discover_pairs(summary_path, set(args.conditions or []), set(args.groups))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"Discovered {len(pairs)} eval pairs from {summary_path}")
    if not pairs:
        return

    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)

    # Determine number of blocks
    blocks = model.net.blocks
    num_blocks = len(blocks)
    print(f"Model has {num_blocks} DiT blocks (0-indexed: 0..{num_blocks - 1})")

    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Per-case results ─────────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []  # one row per (case, step, layer)
    errors: list[dict[str, Any]] = []

    start = time.time()
    for idx, pair in enumerate(pairs, 1):
        cond = pair["condition"]
        ep = int(pair["pert"]["episode"])
        group = pair["group"]
        label = f"[{idx}/{len(pairs)}] {cond}/ep{ep:04d} {group}"
        print(f"{label}  ", flush=True)

        try:
            clean_obs, pert_obs = _pair_observations(pair, args)
            clean_instr, pert_instr = _instruction_pair(pair["clean"], pair["pert"])

            # Forward clean  →  {step: {layer: tensor}}
            clean_captured = forward_capture_all_layers(
                cfg=cfg, model=model, stats=stats, obs=clean_obs,
                instruction=clean_instr, seed=args.seed, target=target,
                num_blocks=num_blocks, capture_steps=capture_steps, args=args,
            )

            # Forward perturb  →  {step: {layer: tensor}}
            pert_captured = forward_capture_all_layers(
                cfg=cfg, model=model, stats=stats, obs=pert_obs,
                instruction=pert_instr, seed=args.seed, target=target,
                num_blocks=num_blocks, capture_steps=capture_steps, args=args,
            )

            # Compute per-step per-layer metrics
            common_steps = sorted(set(clean_captured.keys()) & set(pert_captured.keys()))
            for step in common_steps:
                common_layers = sorted(set(clean_captured[step].keys()) & set(pert_captured[step].keys()))
                for layer_idx in common_layers:
                    clean_t = clean_captured[step][layer_idx]
                    pert_t = pert_captured[step][layer_idx]
                    # shape: [B, num_slots, 14, 14, 2048]
                    if split_slots and clean_t.shape[1] >= 3:
                        # Slot 0,1 = video (wrist, primary); Slot 2 = action
                        video_metrics = compute_layer_shift_metrics(
                            clean_t[:, 0:2], pert_t[:, 0:2])
                        action_metrics = compute_layer_shift_metrics(
                            clean_t[:, 2:3], pert_t[:, 2:3])
                        all_rows.append({
                            "condition": cond, "group": group, "episode": ep,
                            "denoising_step": step, "layer": layer_idx,
                            "slot_group": "video", **video_metrics,
                        })
                        all_rows.append({
                            "condition": cond, "group": group, "episode": ep,
                            "denoising_step": step, "layer": layer_idx,
                            "slot_group": "action", **action_metrics,
                        })
                    else:
                        metrics = compute_layer_shift_metrics(clean_t, pert_t)
                        all_rows.append({
                            "condition": cond, "group": group, "episode": ep,
                            "denoising_step": step, "layer": layer_idx,
                            **metrics,
                        })

            n_layers = len(common_layers) if common_steps else 0
            print(f"  captured {len(common_steps)} steps × {n_layers} layers  ", flush=True)

        except Exception as exc:
            err = {"condition": cond, "episode": ep, "group": group, "error": repr(exc)}
            errors.append(err)
            print(f"  skip: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - start

    # ── Aggregate per-layer per-step stats ────────────────────────────────
    layer_summary: dict[str, dict[str, Any]] = {}
    steps = sorted({row["denoising_step"] for row in all_rows})
    unique_layers = sorted({row["layer"] for row in all_rows})
    slot_groups = sorted({row["slot_group"] for row in all_rows if "slot_group" in row})
    has_slots = bool(slot_groups)

    for step in steps:
        step_rows = [r for r in all_rows if r["denoising_step"] == step]
        for layer_idx in unique_layers:
            layer_rows = [r for r in step_rows if r["layer"] == layer_idx]
            if has_slots:
                # Aggregate per slot_group
                for sg in slot_groups:
                    sg_rows = [r for r in layer_rows if r.get("slot_group") == sg]
                    _aggregate_layer_rows(sg_rows, layer_idx, step, sg, layer_summary)
            else:
                _aggregate_layer_rows(layer_rows, layer_idx, step, None, layer_summary)

    # ── Write outputs ────────────────────────────────────────────────────
    append_jsonl(out_root / "per_layer_per_case.jsonl", all_rows)

    write_json(out_root / "per_layer_summary.json", {
        "checkpoint": str(args.checkpoint),
        "summary": str(summary_path),
        "suite": summary.get("suite"),
        "base_task": summary.get("base_task"),
        "num_blocks": num_blocks,
        "injection_layer": layer_inject,
        "target": target,
        "has_split_slots": has_slots,
        "slot_groups": slot_groups,
        "num_pairs": len(pairs),
        "num_cases_processed": len({(r["condition"], r["episode"]) for r in all_rows}),
        "num_layer_records": len(all_rows),
        "num_errors": len(errors),
        "elapsed_s": elapsed,
        "capture_steps": sorted(capture_steps),
        "num_denoising_steps": args.num_denoising_steps,
        "layer_summary": layer_summary,
    })

    # ── Text table (per step) ─────────────────────────────────────────────
    for step in steps:
        table_path = out_root / f"per_layer_shift_table_step{step}.txt"
        lines = []
        header = (
            f"Denoising step {step}  "
            f"{'layer':>6s}  {'shift_L2':>10s} {'clean_L2':>10s} {'shift_MSE':>10s} "
            f"{'cos_sim':>8s} {'rel_delta':>8s} {'FVE':>8s}  {'n':>4s}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for layer_idx in unique_layers:
            key = f"step{step}_layer{layer_idx}"
            if key not in layer_summary:
                continue
            s = layer_summary[key]
            lines.append(
                f"{layer_idx:6d}  "
                f"{s['shift_l2']['mean']:10.4f} {s['clean_l2']['mean']:10.2f} "
                f"{s['shift_mse']['mean']:10.6f} {s['cosine_similarity']['mean']:8.6f} "
                f"{s['rel_norm_change']['mean']:8.6f} {s['fraction_var_explained']['mean']:8.6f}  "
                f"{s['shift_l2']['n']:4d}"
            )
        table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Cross-step summary table ──────────────────────────────────────────
    xstep_path = out_root / "cross_step_summary.txt"
    xlines = []
    xlines.append("CROSS-STEP SHIFT COMPARISON (shift_L2 mean per layer)")
    xlines.append("=" * 80)
    # Header
    xlines.append(f"{'layer':>6s}" + "".join(f"  {'step'+str(s):>12s}" for s in steps))
    xlines.append("-" * (6 + len(steps) * 14))
    for layer_idx in unique_layers:
        vals = []
        for s in steps:
            key = f"step{s}_layer{layer_idx}"
            v = layer_summary[key]["shift_l2"]["mean"] if key in layer_summary else float("nan")
            vals.append(f"{v:12.4f}")
        xlines.append(f"{layer_idx:6d}" + "".join(vals))
    xstep_path.write_text("\n".join(xlines) + "\n", encoding="utf-8")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total records: {len(all_rows)} "
          f"({len(steps)} steps × ~{len(unique_layers)} layers × cases)")
    print(f"Output: {out_root}")
    print(f"  {out_root / 'per_layer_per_case.jsonl'}")
    print(f"  {out_root / 'per_layer_summary.json'}")
    print(f"  {out_root / 'cross_step_summary.txt'}")

    for step in steps:
        print(f"\n--- Step {step} ---")
        for sg in (slot_groups if has_slots else [None]):
            key = f"step{step}_layer{layer_inject}" + (f"_{sg}" if sg else "")
            if key in layer_summary:
                s = layer_summary[key]
                sg_label = f"[{sg}] " if sg else ""
                print(f"  {sg_label}layer {layer_inject}: shift_L2={s['shift_l2']['mean']:.4f} ± {s['shift_l2']['std']:.4f}  "
                      f"cos_sim={s['cosine_similarity']['mean']:.6f}  "
                      f"shift_MSE={s['shift_mse']['mean']:.6f}")

    if errors:
        print(f"\n{len(errors)} errors (see summary.json for details)")


if __name__ == "__main__":
    main()
