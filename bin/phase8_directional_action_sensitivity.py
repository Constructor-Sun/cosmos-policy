#!/usr/bin/env python3
"""Directional action sensitivity for Phase 8 clean/perturb first chunks.

Runs exact layer interventions and estimates JVPs with central differences.
No full Jacobian is constructed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
if LIBERO_PLUS.is_dir():
    sys.path.insert(0, str(LIBERO_PLUS))

import run_phase2_angular_cosmos as phase2
from cosmos_layer_shift import CosmosLayerCaptureIntervener, layer_shift_context
from libero_plus_task_index import clean_base_tasks, index_variants_by_base
from phase8_correction_lib import EPS, append_jsonl, target_model_indices, write_json


class SelectedStepIntervener(CosmosLayerCaptureIntervener):
    """Capture/intervene only on one conditioned denoising pass."""

    def __init__(self, *, denoising_step: int | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.denoising_step = None if denoising_step is None else int(denoising_step)

    def _active_pass(self) -> bool:
        if self.denoising_step is None:
            return self.pass_idx % 2 == 0
        return self.pass_idx == 2 * self.denoising_step


def parser() -> argparse.ArgumentParser:
    default_summary = phase2.ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", nargs="+", default=[str(default_summary)])
    p.add_argument("--all-suites-grid", action="store_true",
                   help="Build first-observation pairs from all LIBERO-Plus suites and perturbations.")
    p.add_argument("--suites", nargs="+",
                   default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    p.add_argument("--grid-conditions", nargs="+", default=None,
                   help="Perturbations selected in grid mode; default is all seven.")
    p.add_argument("--task-limit", type=int, default=0,
                   help="Base tasks per suite in grid mode; 0 means all tasks.")
    p.add_argument("--init-state-indices", nargs="+", type=int, default=[0])
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--merge-inputs", nargs="*", default=[],
                   help="Merge shard results.jsonl files without loading the policy.")
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", nargs="*", default=[])
    p.add_argument("--conditions", nargs="*", default=["camera_viewpoints"])
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--layers", nargs="+", type=int, default=[14, 27])
    p.add_argument("--target", choices=["action", "video", "video_action"], default="action")
    p.add_argument("--jvp-epsilon", type=float, default=0.02)
    p.add_argument("--random-baselines", type=int, default=4)
    p.add_argument("--denoising-step", type=int, default=-1,
                   help="Step to capture/intervene; -1 means the final step.")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--random-seed", type=int, default=20260729)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hook-mode", choices=["pre", "forward"], default="pre")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fail-fast", action="store_true")
    return p


def pair_observations(pair: dict[str, Any], args) -> tuple[dict, dict]:
    clean = phase2.first_observation(pair["clean"], args)
    if pair["condition"] == "language_instructions":
        return clean, clean
    return clean, phase2.first_observation(pair["pert"], args)


def instructions(pair: dict[str, Any]) -> tuple[str, str]:
    clean = pair["clean"]["language"]
    pert = pair["pert"]["language"] if pair["pert"].get("instruction_mode") == "strict" else clean
    return clean, pert


PERTURBATIONS = {
    "camera_viewpoints": "Camera Viewpoints",
    "background_textures": "Background Textures",
    "light_conditions": "Light Conditions",
    "objects_layout": "Objects Layout",
    "robot_initial_states": "Robot Initial States",
    "sensor_noise": "Sensor Noise",
    "language_instructions": "Language Instructions",
}


def grid_pairs(args) -> list[dict[str, Any]]:
    from libero.libero import benchmark as benchmark_module
    from libero.libero.benchmark import grab_language_from_filename

    benchmark_dict = benchmark_module.get_benchmark_dict()
    pairs = []
    for suite in args.suites:
        all_base_tasks = clean_base_tasks(suite)
        base_tasks = all_base_tasks
        if args.task_limit:
            base_tasks = base_tasks[:args.task_limit]
        variants_by_condition = {}
        for condition, category in PERTURBATIONS.items():
            if condition not in set(args.grid_conditions or PERTURBATIONS):
                continue
            benchmark = benchmark_dict[suite](task_order_index=0, category_value=category)
            variants_by_condition[condition] = index_variants_by_base(all_base_tasks, benchmark.tasks)
        for task_index, base_task in enumerate(base_tasks):
            clean_language = grab_language_from_filename(suite, f"{base_task}.bddl")
            selected_conditions = set(args.grid_conditions or PERTURBATIONS)
            for condition, category in PERTURBATIONS.items():
                if condition not in selected_conditions:
                    continue
                variants = variants_by_condition[condition][base_task]
                if not variants:
                    raise RuntimeError(f"no {condition} variant for {suite}/{base_task}")
                variant = variants[task_index % len(variants)]
                pert_language = variant.language if condition == "language_instructions" else clean_language
                instruction_mode = "strict" if condition == "language_instructions" else "base"
                for init_index in args.init_state_indices:
                    common = {
                        "suite": suite, "base_task": base_task, "episode": int(init_index),
                        "init_state_index": int(init_index), "deterministic_reset": True,
                        "deterministic_reset_seed": args.reset_seed, "success": None,
                    }
                    clean = dict(common, condition="clean", task_name=base_task,
                                 language=clean_language, instruction_mode="task")
                    pert = dict(common, condition=condition, task_name=variant.name,
                                language=pert_language, instruction_mode=instruction_mode)
                    pairs.append({
                        "condition": condition, "group": "grid", "clean": clean, "pert": pert,
                        "suite": suite, "base_task": base_task,
                    })
    print(f"Built {len(pairs)} grid pairs across {len(args.suites)} suites", flush=True)
    return pairs


def run_forward(
    *, cfg, model, stats, obs, instruction: str, args,
    capture_layers: list[int] = [], intervene_layer: int | None = None,
    direction: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict[int, torch.Tensor], int]:
    target_indices = target_model_indices(args.target)
    hook_layers = sorted(set(capture_layers) | ({intervene_layer} if intervene_layer is not None else set()))
    iv = SelectedStepIntervener(
        denoising_step=args.selected_step,
        target_indices=target_indices,
        condition_pass_only=True,
    )
    iv.reset_direct(
        capture_layers=set(capture_layers), intervene_layer=intervene_layer,
        direction=direction,
    )
    with layer_shift_context(model, iv, hook_layers, hook_mode=args.hook_mode):
        phase2.set_seed_everywhere(args.reset_seed)
        out = phase2.get_action(
            cfg, model, stats, obs, instruction, seed=args.seed,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    missing = set(capture_layers) - set(iv.captured)
    if missing:
        raise RuntimeError(f"layers not captured at step {args.selected_step}: {sorted(missing)}")
    if intervene_layer is not None and intervene_layer not in iv.after:
        raise RuntimeError(f"layer {intervene_layer} intervention did not run")
    last_conditioned_step = max(0, iv.pass_idx // 2)
    return np.asarray(out["actions"], dtype=np.float32), iv.captured, last_conditioned_step


def orthogonal_delta(clean: torch.Tensor, pert: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clean_f = clean.float().reshape(-1)
    delta_f = (pert.float() - clean.float()).reshape(-1)
    coefficient = torch.dot(delta_f, clean_f) / (torch.dot(clean_f, clean_f) + EPS)
    parallel = coefficient * clean.float()
    return (pert.float() - clean.float()) - parallel, parallel


def random_orthogonal(clean: torch.Tensor, norm: float, generator: torch.Generator) -> torch.Tensor:
    eta = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    clean_f, eta_f = clean.float().reshape(-1), eta.reshape(-1)
    eta = eta - (torch.dot(eta_f, clean_f) / (torch.dot(clean_f, clean_f) + EPS)) * clean.float()
    return eta * (norm / (float(torch.linalg.vector_norm(eta)) + EPS))


def norm(array: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(array, dtype=np.float64).reshape(-1)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (norm(a) * norm(b) + EPS))


def candidate_action(*, cfg, model, stats, obs, instruction, args, layer, direction):
    action, _, _ = run_forward(
        cfg=cfg, model=model, stats=stats, obs=obs, instruction=instruction, args=args,
        intervene_layer=layer, direction=direction,
    )
    return action


def analyze_layer(
    *, cfg, model, stats, clean_obs, pert_obs, clean_instruction, pert_instruction,
    clean_action, pert_action, clean_h, pert_h, layer: int, args, generator,
) -> dict[str, Any]:
    perp, parallel = orthogonal_delta(clean_h, pert_h)
    perp_norm = float(torch.linalg.vector_norm(perp))
    clean_h_norm = float(torch.linalg.vector_norm(clean_h.float()))
    action_delta = pert_action - clean_action
    action_delta_norm = norm(action_delta)
    clean_action_norm = norm(clean_action)
    eps = args.jvp_epsilon

    plus = candidate_action(
        cfg=cfg, model=model, stats=stats, obs=clean_obs, instruction=clean_instruction,
        args=args, layer=layer, direction=eps * perp,
    )
    minus = candidate_action(
        cfg=cfg, model=model, stats=stats, obs=clean_obs, instruction=clean_instruction,
        args=args, layer=layer, direction=-eps * perp,
    )
    jvp = (plus - minus) / (2.0 * eps)
    jvp_norm = norm(jvp)

    row: dict[str, Any] = {
        "layer": layer,
        "clean_hidden_norm": clean_h_norm,
        "orthogonal_delta_norm": perp_norm,
        "orthogonal_relative_shift": perp_norm / (clean_h_norm + EPS),
        "parallel_delta_norm": float(torch.linalg.vector_norm(parallel)),
        "action_delta_norm": action_delta_norm,
        "jvp_norm": jvp_norm,
        "C_cosine_jvp_action_delta": cosine(jvp, action_delta),
        "M_jvp_fraction_of_action_delta": jvp_norm / (action_delta_norm + EPS),
        "S_relative_action_per_relative_latent":
            (jvp_norm / (clean_action_norm + EPS)) / (perp_norm / (clean_h_norm + EPS) + EPS),
        "N_nonlinear_residual": norm(action_delta - jvp) / (action_delta_norm + EPS),
    }

    removed = candidate_action(
        cfg=cfg, model=model, stats=stats, obs=pert_obs, instruction=pert_instruction,
        args=args, layer=layer, direction=-perp,
    )
    removed_error = norm(removed - clean_action) / (action_delta_norm + EPS)
    row["removal_relative_action_error"] = removed_error
    row["removal_action_recovery"] = 1.0 - removed_error

    random_jvp_norms = []
    for index in range(args.random_baselines):
        eta = random_orthogonal(clean_h, perp_norm, generator)
        rplus = candidate_action(
            cfg=cfg, model=model, stats=stats, obs=clean_obs, instruction=clean_instruction,
            args=args, layer=layer, direction=eps * eta,
        )
        rminus = candidate_action(
            cfg=cfg, model=model, stats=stats, obs=clean_obs, instruction=clean_instruction,
            args=args, layer=layer, direction=-eps * eta,
        )
        value = norm((rplus - rminus) / (2.0 * eps))
        row[f"random_{index}_jvp_norm"] = value
        random_jvp_norms.append(value)
    random_mean = float(np.mean(random_jvp_norms)) if random_jvp_norms else float("nan")
    row["random_jvp_norm_mean"] = random_mean
    row["Q_vs_random_direction"] = jvp_norm / (random_mean + EPS)
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(rows)}
    numeric = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
    for key in numeric:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values:
            output[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                           "std": float(np.std(values)), "n": len(values)}
    return output


def grouped_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group = tuple(str(row.get(key)) for key in keys)
        groups.setdefault(group, []).append(row)
    return {"/".join(group): aggregate(items) for group, items in sorted(groups.items())}


def merge_results(args) -> None:
    rows = []
    for input_string in args.merge_inputs:
        path = pathlib.Path(input_string).expanduser()
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise RuntimeError("no rows found in --merge-inputs")
    output = pathlib.Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    append_jsonl(output / "results.jsonl", rows)
    write_json(output / "summary.json", {
        "merged_inputs": args.merge_inputs, "num_rows": len(rows),
        "by_layer": grouped_summary(rows, ("layer",)),
        "by_suite_layer": grouped_summary(rows, ("suite", "layer")),
        "by_perturbation_layer": grouped_summary(rows, ("condition", "layer")),
        "by_suite_perturbation_layer": grouped_summary(rows, ("suite", "condition", "layer")),
    })
    print(f"merged {len(rows)} rows into {output}", flush=True)


def main() -> None:
    args = parser().parse_args()
    if args.merge_inputs:
        merge_results(args)
        return
    args.selected_step = None if args.denoising_step < 0 else args.denoising_step
    if args.selected_step is not None and not 0 <= args.selected_step < args.num_denoising_steps:
        raise ValueError("--denoising-step is outside the denoising schedule")
    if args.jvp_epsilon <= 0 or args.random_baselines < 0:
        raise ValueError("epsilon must be positive and random-baselines nonnegative")

    summaries = []
    if args.all_suites_grid:
        pairs = grid_pairs(args)
        summaries = ["LIBERO-Plus benchmark grid"]
    else:
        pairs = []
        for path_string in args.summary:
            summary, found = phase2.discover_pairs(
                pathlib.Path(path_string), set(args.conditions or []), set(args.groups or [])
            )
            summaries.append(path_string)
            pairs.extend(found)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num-shards)")
    pairs = [pair for index, pair in enumerate(pairs) if index % args.num_shards == args.shard_index]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise RuntimeError("no evaluation pairs discovered")

    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)
    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    for embedding_path in args.t5_extra_embeddings:
        if not pathlib.Path(embedding_path).is_file():
            raise FileNotFoundError(f"T5 embedding file not found: {embedding_path}")
        phase2.load_extra_t5(embedding_path)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    n_blocks = len(model.net.blocks)
    if any(layer < 0 or layer >= n_blocks for layer in args.layers):
        raise ValueError(f"layers must be in [0, {n_blocks - 1}]")

    output = pathlib.Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    if results_path.exists():
        raise FileExistsError(f"refusing to append to existing output: {results_path}")
    rows, errors = [], []
    generator = torch.Generator(device="cpu").manual_seed(args.random_seed)
    start = time.time()
    print(f"pairs={len(pairs)} layers={args.layers} selected_step={args.selected_step}", flush=True)

    for index, pair in enumerate(pairs, 1):
        label = f"[{index}/{len(pairs)}] {pair['condition']}/ep{int(pair['pert']['episode']):04d}"
        print(label, flush=True)
        try:
            clean_obs, pert_obs = pair_observations(pair, args)
            clean_instruction, pert_instruction = instructions(pair)
            clean_action, clean_hidden, detected_step = run_forward(
                cfg=cfg, model=model, stats=stats, obs=clean_obs,
                instruction=clean_instruction, args=args, capture_layers=args.layers,
            )
            if args.selected_step is None:
                args.selected_step = detected_step
                print(f"  auto-detected final conditioned step: {detected_step}", flush=True)
            pert_action, pert_hidden, _ = run_forward(
                cfg=cfg, model=model, stats=stats, obs=pert_obs,
                instruction=pert_instruction, args=args, capture_layers=args.layers,
            )
            base = {
                "condition": pair["condition"], "group": pair.get("group"),
                "episode": int(pair["pert"]["episode"]),
                "suite": pair["pert"].get("suite", pair.get("suite")),
                "base_task": pair.get("base_task"),
                "baseline_action_delta_norm": norm(pert_action - clean_action),
            }
            for layer in args.layers:
                row = dict(base)
                row.update(analyze_layer(
                    cfg=cfg, model=model, stats=stats,
                    clean_obs=clean_obs, pert_obs=pert_obs,
                    clean_instruction=clean_instruction, pert_instruction=pert_instruction,
                    clean_action=clean_action, pert_action=pert_action,
                    clean_h=clean_hidden[layer], pert_h=pert_hidden[layer],
                    layer=layer, args=args, generator=generator,
                ))
                rows.append(row)
                print(f"  L{layer}: C={row['C_cosine_jvp_action_delta']:.3f} "
                      f"M={row['M_jvp_fraction_of_action_delta']:.3f} "
                      f"removal={row['removal_action_recovery']:.3f} "
                      f"Q={row['Q_vs_random_direction']:.3f}", flush=True)
        except Exception as exc:
            errors.append({"label": label, "error": repr(exc)})
            print(f"  SKIP: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    by_layer = {str(layer): aggregate([row for row in rows if row["layer"] == layer])
                for layer in args.layers}
    append_jsonl(results_path, rows)
    write_json(output / "summary.json", {
        "summaries": summaries, "layers": args.layers, "target": args.target,
        "hook_mode": args.hook_mode, "selected_denoising_step": args.selected_step,
        "jvp_epsilon": args.jvp_epsilon,
        "random_baselines": args.random_baselines, "num_pairs_requested": len(pairs),
        "num_rows": len(rows), "num_errors": len(errors), "elapsed_s": time.time() - start,
        "by_layer": by_layer,
        "by_suite_layer": grouped_summary(rows, ("suite", "layer")),
        "by_perturbation_layer": grouped_summary(rows, ("condition", "layer")),
        "by_suite_perturbation_layer": grouped_summary(rows, ("suite", "condition", "layer")),
        "errors": errors,
    })
    print(f"wrote {results_path} and {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
