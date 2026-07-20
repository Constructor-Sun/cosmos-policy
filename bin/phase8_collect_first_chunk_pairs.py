#!/usr/bin/env python3
"""Collect Phase 8 first-chunk clean/perturbed latent-shift pairs."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import zipfile
from typing import Any

import numpy as np
import torch

import run_phase2_angular_cosmos as phase2
from cosmos_layer_shift import CosmosLayerCaptureIntervener, layer_shift_context
from cosmos_policy.experiments.robot.cosmos_utils import duplicate_array, prepare_images_for_model
from phase8_correction_lib import (
    CAPTURE_SLOT_ORDER,
    action_mse,
    action_rel_error,
    append_jsonl,
    write_json,
)


_INIT_STATES_CACHE: dict[tuple[str, str], Any] = {}


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _instruction_pair(clean: dict[str, Any], pert: dict[str, Any]) -> tuple[str, str]:
    clean_instr = clean["language"]
    pert_instr = pert["language"] if pert.get("instruction_mode") == "strict" else clean_instr
    return clean_instr, pert_instr


def _pair_observations(pair: dict[str, Any], args) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    clean_obs = phase2.first_observation(pair["clean"], args)
    pert_obs = phase2.first_observation(pair["pert"], args)
    return clean_obs, pert_obs


def _load_init_states_cached(task_name: str, suite: str):
    key = (suite, task_name)
    if key not in _INIT_STATES_CACHE:
        _INIT_STATES_CACHE[key] = phase2.load_init_states(task_name, suite)
    return _INIT_STATES_CACHE[key]


def first_observation_seeded(
    ep: dict[str, Any],
    args,
    *,
    init_state_index: int,
    env_seed: int,
) -> dict[str, np.ndarray]:
    env = phase2.make_env(ep["task_name"], ep["suite"], args.env_resolution)
    try:
        env.seed(int(env_seed))
        phase2.set_seed_everywhere(int(env_seed))
        env.reset()
        init_task = ep["task_name"] if ep.get("condition") == "objects_layout" else ep.get("base_task", ep["task_name"])
        states = _load_init_states_cached(init_task, ep["suite"])
        obs = env.set_init_state(states[int(init_state_index) % len(states)])
        for _ in range(args.num_warmup):
            obs, _, _, _ = env.step(phase2.DUMMY_ACTION)
        primary = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        if args.flip_images:
            primary = np.flipud(primary)
            wrist = np.flipud(wrist)
        proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
        return {"primary_image": primary, "wrist_image": wrist, "proprio": proprio}
    finally:
        env.close()


def _pair_observations_seeded(pair: dict[str, Any], args) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    init_state_index = int(pair["init_state_index"])
    env_seed = int(pair["env_seed"])
    clean_obs = first_observation_seeded(
        pair["clean"],
        args,
        init_state_index=init_state_index,
        env_seed=env_seed,
    )
    pert_obs = first_observation_seeded(
        pair["pert"],
        args,
        init_state_index=init_state_index,
        env_seed=env_seed,
    )
    return clean_obs, pert_obs


def vae_video_latent(model, obs: dict[str, np.ndarray], cfg) -> torch.Tensor:
    """Encode LIBERO current wrist/primary image slots as [16,2,28,28]."""

    if cfg.suite != "libero":
        raise ValueError("phase8 collector currently assumes cfg.suite='libero'")
    device = next(model.parameters()).device
    images = [obs["wrist_image"], obs["primary_image"]]
    wrist_img, primary_img = prepare_images_for_model(images, cfg)
    blank = np.zeros_like(primary_img)
    bd = duplicate_array(blank, 4)
    wd = duplicate_array(wrist_img, 4)
    pd = duplicate_array(primary_img, 4)
    seq = [
        np.expand_dims(np.zeros_like(blank), axis=0),
        bd,
        wd,
        pd,
        bd.copy(),
        bd.copy(),
        wd.copy(),
        pd.copy(),
        bd.copy(),
    ]
    raw = np.concatenate(seq, axis=0)
    raw = np.expand_dims(raw, 0)
    raw = np.transpose(raw, (0, 4, 1, 2, 3))
    video = torch.from_numpy(raw).to(dtype=torch.uint8, device=device)
    with torch.no_grad():
        latent = model.encode(video).contiguous().float()
    return latent[0, :, [2, 3], :, :].cpu()


def forward_capture(
    *,
    cfg,
    model,
    stats,
    obs: dict[str, np.ndarray],
    instruction: str,
    seed: int,
    layers: list[int],
    args,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    intervener = CosmosLayerCaptureIntervener(
        target_indices=CAPTURE_SLOT_ORDER,
        condition_pass_only=args.condition_pass_only,
    )
    intervener.reset(capture_layers=set(layers))
    with layer_shift_context(model, intervener, layers):
        phase2.set_seed_everywhere(int(args.reset_seed if rng_seed is None else rng_seed))
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
    captured = {str(layer): value.cpu().half() for layer, value in intervener.captured.items()}
    return np.asarray(out["actions"], dtype=np.float32), captured


def sample_relative_path(pair: dict[str, Any]) -> pathlib.Path:
    cond = pair["condition"]
    ep = int(pair["pert"]["episode"])
    clean, pert = pair["clean"], pair["pert"]
    suite = pert.get("suite", pair.get("suite", "unknown_suite"))
    base_task = pair.get("base_task") or clean.get("base_task", clean["task_name"])
    sample_name = pair.get("sample_id", f"ep{ep:04d}")
    return pathlib.Path("samples") / suite / base_task / cond / f"{sample_name}.pt"


def existing_sample_complete(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)
    except OSError:
        return False


def manifest_row_from_pair(
    pair: dict[str, Any],
    *,
    action_rel_error_value: float | None = None,
    action_mse_value: float | None = None,
) -> dict[str, Any]:
    cond = pair["condition"]
    ep = int(pair["pert"]["episode"])
    clean, pert = pair["clean"], pair["pert"]
    suite = pert.get("suite", pair.get("suite", "unknown_suite"))
    base_task = pair.get("base_task") or clean.get("base_task", clean["task_name"])
    sample_name = pair.get("sample_id", f"ep{ep:04d}")
    clean_success = _maybe_bool(clean.get("success"))
    pert_success = _maybe_bool(pert.get("success"))
    perturb_caused_failure = (
        clean_success and not pert_success
        if clean_success is not None and pert_success is not None
        else None
    )
    return {
        "path": str(sample_relative_path(pair)),
        "suite": suite,
        "base_task": base_task,
        "condition": cond,
        "group": pair["group"],
        "source_group": pair.get("source_group"),
        "episode": ep,
        "sample_id": sample_name,
        "clean_success": clean_success,
        "pert_success": pert_success,
        "perturb_caused_failure": perturb_caused_failure,
        "init_state_index": pair.get("init_state_index"),
        "policy_seed": pair.get("policy_seed"),
        "env_seed": pair.get("env_seed"),
        "seed_config_index": pair.get("seed_config_index"),
        "action_rel_error": action_rel_error_value,
        "action_mse": action_mse_value,
    }


def save_pair(
    *,
    pair: dict[str, Any],
    clean_action: np.ndarray,
    pert_action: np.ndarray,
    clean_hidden: dict[str, torch.Tensor],
    pert_hidden: dict[str, torch.Tensor],
    clean_vae: torch.Tensor,
    pert_vae: torch.Tensor,
    layers: list[int],
    out_root: pathlib.Path,
) -> dict[str, Any]:
    cond = pair["condition"]
    ep = int(pair["pert"]["episode"])
    clean, pert = pair["clean"], pair["pert"]
    suite = pert.get("suite", pair.get("suite", "unknown_suite"))
    base_task = pair.get("base_task") or clean.get("base_task", clean["task_name"])
    sample_name = pair.get("sample_id", f"ep{ep:04d}")
    sample_rel = sample_relative_path(pair)
    sample_path = out_root / sample_rel
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    latent_pert = {"vae_video": pert_vae.half(), "layers": {}}
    latent_shift = {"vae_video": (clean_vae.float() - pert_vae.float()).half(), "layers": {}}
    for layer in layers:
        key = str(layer)
        c = clean_hidden[key].float()
        p = pert_hidden[key].float()
        latent_pert["layers"][key] = {
            "video": p[:, :2].half(),
            "action": p[:, 2:3].half(),
        }
        latent_shift["layers"][key] = {
            "video": (c[:, :2] - p[:, :2]).half(),
            "action": (c[:, 2:3] - p[:, 2:3]).half(),
        }
    clean_success = _maybe_bool(clean.get("success"))
    pert_success = _maybe_bool(pert.get("success"))
    perturb_caused_failure = (
        clean_success and not pert_success
        if clean_success is not None and pert_success is not None
        else None
    )
    payload = {
        "meta": {
            "condition": cond,
            "group": pair["group"],
            "source_group": pair.get("source_group"),
            "episode": ep,
            "sample_id": sample_name,
            "clean_success": clean_success,
            "pert_success": pert_success,
            "perturb_caused_failure": perturb_caused_failure,
            "clean_task_name": clean["task_name"],
            "pert_task_name": pert["task_name"],
            "suite": suite,
            "base_task": base_task,
            "summary_path": pair.get("summary_path"),
            "init_state_index_clean": int(clean.get("init_state_index", clean["episode"])),
            "init_state_index_pert": int(pert.get("init_state_index", pert["episode"])),
            "init_state_index": pair.get("init_state_index"),
            "policy_seed": pair.get("policy_seed"),
            "env_seed": pair.get("env_seed"),
            "seed_config_index": pair.get("seed_config_index"),
            "clean_instruction": clean["language"],
            "pert_instruction": pert["language"],
            "instruction_mode": pert.get("instruction_mode", "task"),
            "capture_slot_order": CAPTURE_SLOT_ORDER,
            "layers": layers,
            "shift_formula": "clean - pert",
        },
        "latent_pert": latent_pert,
        "latent_shift": latent_shift,
    }
    torch.save(payload, sample_path)
    return manifest_row_from_pair(
        pair,
        action_rel_error_value=action_rel_error(pert_action, clean_action),
        action_mse_value=action_mse(pert_action, clean_action),
    )


def _resolve_summary_path(summary_arg: str) -> pathlib.Path:
    path = pathlib.Path(summary_arg).expanduser()
    if path.is_absolute():
        return path
    return phase2.ROOT / path


def _summary_condition_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["condition"]: item for item in summary["conditions"]}


def _load_episode_list(summary_path: pathlib.Path, condition_info: dict[str, Any]) -> list[dict[str, Any]]:
    path = phase2.resolve_path(condition_info["episodes_path"], summary_path.parent)
    data = phase2.load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"expected episode list in {path}")
    return data


def _seed_configs(args) -> list[tuple[int, int, int]]:
    if not args.policy_seeds:
        raise ValueError("--sample-mode task_init_policy_grid requires --policy-seeds")
    policy_seeds = [int(seed) for seed in args.policy_seeds]
    env_seeds = [int(seed) for seed in args.env_seeds] if args.env_seeds else list(policy_seeds)
    if len(env_seeds) == 1 and len(policy_seeds) > 1:
        env_seeds = env_seeds * len(policy_seeds)
    if len(env_seeds) != len(policy_seeds):
        raise ValueError("--env-seeds must contain either one seed or the same count as --policy-seeds")
    return [(idx, policy_seed, env_seed) for idx, (policy_seed, env_seed) in enumerate(zip(policy_seeds, env_seeds))]


def _source_groups(clean_eps: list[dict[str, Any]], pert_eps: list[dict[str, Any]]) -> dict[int, str]:
    pert_by_ep = {int(item["episode"]): item for item in pert_eps}
    out = {}
    for clean in clean_eps:
        ep = int(clean["episode"])
        pert = pert_by_ep.get(ep)
        if pert is None:
            continue
        out[ep] = phase2.pair_group(bool(clean["success"]), bool(pert["success"]))
    return out


def build_task_init_policy_grid(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    pairs = []
    seed_configs = _seed_configs(args)
    condition_filter = set(args.conditions or [])
    init_filter = set(args.init_state_indices or [])
    for summary_arg in args.summary:
        summary_path = _resolve_summary_path(summary_arg)
        summary = phase2.load_json(summary_path)
        if not isinstance(summary, dict):
            raise TypeError(f"expected summary object in {summary_path}")
        by_condition = _summary_condition_map(summary)
        clean_info = by_condition["clean"]
        clean_eps = _load_episode_list(summary_path, clean_info)
        if not clean_eps:
            raise RuntimeError(f"no clean episodes in {summary_path}")
        clean_template = dict(clean_eps[0])
        suite = summary["suite"]
        base_task = summary["base_task"]
        init_states = _load_init_states_cached(base_task, suite)
        init_indices = sorted(init_filter) if init_filter else list(range(len(init_states)))
        for condition, info in by_condition.items():
            if condition == "clean" or (condition_filter and condition not in condition_filter):
                continue
            pert_eps = _load_episode_list(summary_path, info)
            if not pert_eps:
                raise RuntimeError(f"no {condition} episodes in {summary_path}")
            pert_template = dict(pert_eps[0])
            source_groups = _source_groups(clean_eps, pert_eps)
            for init_state_index in init_indices:
                for seed_config_index, policy_seed, env_seed in seed_configs:
                    clean = dict(clean_template)
                    clean.update(
                        {
                            "condition": "clean",
                            "episode": init_state_index,
                            "init_state_index": init_state_index,
                            "success": None,
                            "seed": policy_seed,
                            "deterministic_reset": False,
                            "deterministic_reset_seed": env_seed,
                        }
                    )
                    pert = dict(pert_template)
                    pert.update(
                        {
                            "condition": condition,
                            "episode": init_state_index,
                            "init_state_index": init_state_index,
                            "success": None,
                            "seed": policy_seed,
                            "deterministic_reset": False,
                            "deterministic_reset_seed": env_seed,
                        }
                    )
                    sample_id = f"init{init_state_index:04d}_pseed{policy_seed}_eseed{env_seed}"
                    pairs.append(
                        {
                            "condition": condition,
                            "group": "task_init_policy_grid",
                            "source_group": source_groups.get(init_state_index),
                            "source_episode": init_state_index if init_state_index in source_groups else None,
                            "clean": clean,
                            "pert": pert,
                            "suite": suite,
                            "base_task": base_task,
                            "summary_path": str(summary_path),
                            "init_state_index": init_state_index,
                            "policy_seed": policy_seed,
                            "env_seed": env_seed,
                            "seed_config_index": seed_config_index,
                            "sample_id": sample_id,
                        }
                    )
        summaries.append(summary)
        print(f"Built grid pairs from {summary_path}", flush=True)
    return summaries, pairs


def build_summary_pair_samples(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    pairs = []
    for summary_arg in args.summary:
        summary_path = _resolve_summary_path(summary_arg)
        summary, summary_pairs = phase2.discover_pairs(
            summary_path, set(args.conditions or []), set(args.groups)
        )
        for pair in summary_pairs:
            pair["suite"] = summary.get("suite")
            pair["base_task"] = summary.get("base_task")
            pair["summary_path"] = str(summary_path)
        summaries.append(summary)
        pairs.extend(summary_pairs)
        print(f"Discovered {len(summary_pairs)} pairs from {summary_path}", flush=True)
    return summaries, pairs


def apply_shard(pairs: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards == 1:
        return pairs
    return [pair for idx, pair in enumerate(pairs) if idx % args.num_shards == args.shard_index]


def build_parser() -> argparse.ArgumentParser:
    default_summary = phase2.ROOT / (
        "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "__all_conditions__20pair_combined_summary.json"
    )
    default_output = phase2.ROOT / "experiments/phase8_first_chunk_pairs/camera_viewpoints"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", nargs="+", default=[str(default_summary)])
    p.add_argument("--output-dir", default=str(default_output))
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default=str(phase2.ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--conditions", nargs="*", default=["camera_viewpoints"])
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--layers", nargs="+", default=["0", "mid", "last"])
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument(
        "--sample-mode",
        choices=["summary_pairs", "task_init_policy_grid"],
        default="summary_pairs",
    )
    p.add_argument("--policy-seeds", nargs="*", type=int, default=None)
    p.add_argument("--env-seeds", nargs="*", type=int, default=None)
    p.add_argument("--init-state-indices", nargs="*", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--condition-pass-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-existing", action="store_true", help="skip complete .pt samples already present in output-dir")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)
    if args.sample_mode == "task_init_policy_grid":
        summaries, pairs = build_task_init_policy_grid(args)
    else:
        summaries, pairs = build_summary_pair_samples(args)
    total_before_shard = len(pairs)
    pairs = apply_shard(pairs, args)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(
        f"Total selected pairs: {len(pairs)} "
        f"(before shard={total_before_shard}, shard={args.shard_index}/{args.num_shards})",
        flush=True,
    )
    if not pairs:
        return

    cfg = phase2.make_cfg(args)
    phase2.set_seed_everywhere(args.seed)
    phase2.cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    layers = phase2.parse_layers(args.layers, len(model.net.blocks))
    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Layers: {layers}  slots={CAPTURE_SLOT_ORDER}", flush=True)

    rows, errors = [], []
    num_skipped_existing = 0
    start = time.time()
    for idx, pair in enumerate(pairs, 1):
        cond = pair["condition"]
        ep = int(pair["pert"]["episode"])
        sample_id = pair.get("sample_id", f"ep{ep:04d}")
        print(f"[{idx}/{len(pairs)}] {cond}/{sample_id} {pair['group']}", flush=True)
        sample_path = out_root / sample_relative_path(pair)
        if args.skip_existing and existing_sample_complete(sample_path):
            rows.append(manifest_row_from_pair(pair))
            num_skipped_existing += 1
            print(f"  exists: skip {sample_path}", flush=True)
            continue
        if args.skip_existing and sample_path.exists():
            print(f"  existing sample is incomplete or unreadable; regenerating {sample_path}", flush=True)
        try:
            if args.sample_mode == "task_init_policy_grid":
                clean_obs, pert_obs = _pair_observations_seeded(pair, args)
                policy_seed = int(pair["policy_seed"])
            else:
                clean_obs, pert_obs = _pair_observations(pair, args)
                policy_seed = int(args.seed)
            clean_instr, pert_instr = _instruction_pair(pair["clean"], pair["pert"])
            clean_action, clean_h = forward_capture(
                cfg=cfg, model=model, stats=stats, obs=clean_obs,
                instruction=clean_instr, seed=policy_seed, layers=layers, args=args,
                rng_seed=policy_seed)
            pert_action, pert_h = forward_capture(
                cfg=cfg, model=model, stats=stats, obs=pert_obs,
                instruction=pert_instr, seed=policy_seed, layers=layers, args=args,
                rng_seed=policy_seed)
            missing = [layer for layer in map(str, layers) if layer not in clean_h or layer not in pert_h]
            if missing:
                raise RuntimeError(f"missing captured layers: {missing}")
            row = save_pair(
                pair=pair,
                clean_action=clean_action,
                pert_action=pert_action,
                clean_hidden=clean_h,
                pert_hidden=pert_h,
                clean_vae=vae_video_latent(model, clean_obs, cfg),
                pert_vae=vae_video_latent(model, pert_obs, cfg),
                layers=layers,
                out_root=out_root,
            )
            rows.append(row)
        except Exception as exc:
            err = {"condition": cond, "episode": ep, "sample_id": sample_id, "error": repr(exc)}
            errors.append(err)
            print(f"  skip: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    append_jsonl(out_root / "manifest.jsonl", rows)
    write_json(out_root / "summary.json", {
        "summaries": [str(pathlib.Path(item)) for item in args.summary],
        "suites": sorted({str(s.get("suite")) for s in summaries}),
        "base_tasks": sorted({str(s.get("base_task")) for s in summaries}),
        "conditions": args.conditions,
        "groups": args.groups,
        "sample_mode": args.sample_mode,
        "policy_seeds": args.policy_seeds,
        "env_seeds": args.env_seeds,
        "init_state_indices": args.init_state_indices,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_pairs_before_shard": total_before_shard,
        "layers": layers,
        "capture_slot_order": CAPTURE_SLOT_ORDER,
        "skip_existing": args.skip_existing,
        "num_skipped_existing": num_skipped_existing,
        "num_pairs": len(rows),
        "num_errors": len(errors),
        "errors": errors,
        "elapsed_s": time.time() - start,
    })
    print(f"Saved {len(rows)} samples to {out_root}", flush=True)


if __name__ == "__main__":
    main()
