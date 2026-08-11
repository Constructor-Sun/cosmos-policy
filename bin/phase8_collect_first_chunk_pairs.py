#!/usr/bin/env python3
"""Collect Phase 8 first-chunk clean/perturbed latent-shift pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import zipfile
from typing import Any

import h5py
import numpy as np
import torch

import run_phase2_angular_cosmos as phase2
from libero.libero import benchmark, get_libero_path
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
_DEMO_STATES_CACHE: dict[tuple[str, str], list[np.ndarray]] = {}


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


def _load_demo_states_cached(task_name: str, suite: str, demo_root: str) -> list[np.ndarray]:
    key = (suite, task_name)
    if key not in _DEMO_STATES_CACHE:
        path = pathlib.Path(demo_root) / f"{suite}_regen" / f"{task_name}_demo.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"missing demonstration dataset: {path}")
        with h5py.File(path, "r") as handle:
            keys = sorted(handle["data"], key=lambda item: int(item.split("_")[1]))
            _DEMO_STATES_CACHE[key] = [np.asarray(handle[f"data/{item}/states"][0]).copy() for item in keys]
    return _DEMO_STATES_CACHE[key]


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
        states = (_load_demo_states_cached(init_task, ep["suite"], args.demo_root)
                  if ep.get("state_source") == "demo" else _load_init_states_cached(init_task, ep["suite"]))
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
    hook_mode = getattr(args, "hook_mode", "pre")
    with layer_shift_context(model, intervener, layers, hook_mode=hook_mode):
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
        "split": pair.get("split"),
        "state_source": pair.get("state_source"),
        "state_index": pair.get("state_index"),
        "state_hash": pair.get("state_hash"),
        "camera_tuple": pair.get("camera_tuple"),
        "camera_split": pair.get("camera_split"),
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
            "perturb_config_path": pair.get("perturb_config_path"),
            "init_state_index_clean": int(clean.get("init_state_index", clean["episode"])),
            "init_state_index_pert": int(pert.get("init_state_index", pert["episode"])),
            "init_state_index": pair.get("init_state_index"),
            "policy_seed": pair.get("policy_seed"),
            "env_seed": pair.get("env_seed"),
            "seed_config_index": pair.get("seed_config_index"),
            "split": pair.get("split"),
            "state_source": pair.get("state_source"),
            "state_index": pair.get("state_index"),
            "state_hash": pair.get("state_hash"),
            "camera_tuple": pair.get("camera_tuple"),
            "camera_split": pair.get("camera_split"),
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


def _resolve_repo_path(path_arg: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_arg).expanduser()
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


def _one_config_value(info: dict[str, Any], key: str) -> Any:
    values = info.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"expected one {key} value in perturb config, got {values!r}")
    return values[0]


def _base_task_language(suite: str, base_task: str) -> str:
    filename = f"{base_task}.bddl"
    try:
        return benchmark.grab_language_from_filename(suite, filename)
    except TypeError:
        return benchmark.grab_language_from_filename(filename)


def _read_base_task_specs(args) -> list[tuple[str, str, str]]:
    only_tasks = set()
    for value in args.only_task:
        if ":" not in value:
            raise ValueError(f"--only-task must be formatted as suite:task_name, got {value!r}")
        only_tasks.add(tuple(value.split(":", 1)))

    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    specs = []
    for suite in args.suites:
        tasks_info = bddl_root / suite / "tasks_info.txt"
        if not tasks_info.is_file():
            raise FileNotFoundError(f"missing LIBERO task list: {tasks_info}")
        suite_specs = []
        for line in tasks_info.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            base_task = pathlib.Path(line).stem
            if only_tasks and (suite, base_task) not in only_tasks:
                continue
            language = _base_task_language(suite, base_task)
            suite_specs.append((suite, base_task, language))
        specs.extend(suite_specs[: args.task_limit] if args.task_limit else suite_specs)
    if not specs:
        raise RuntimeError("no base tasks selected for the grid")
    return specs


def _camera_splits(args) -> dict[str, list[tuple[int, ...]]]:
    data = phase2.load_json(_resolve_repo_path(args.camera_classification))
    cameras = set()
    for items in data.values():
        for item in items:
            if item.get("category") != "Camera Viewpoints":
                continue
            view, init_state = item["name"].rsplit("_view_", 1)[1].split("_initstate_", 1)
            if init_state == "0":
                cameras.add(tuple(map(int, view.split("_"))))
    if len(cameras) != 450:
        raise RuntimeError(f"expected 450 unique camera tuples, got {len(cameras)}")
    cameras = sorted(cameras)
    order = np.random.default_rng(args.camera_split_seed).permutation(len(cameras))
    ordered = [cameras[int(index)] for index in order]
    return {"train": ordered[:360], "val": ordered[360:405]}


def build_train8000_grid(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = _read_base_task_specs(args)
    if len(specs) != 40 or args.num_train_states != 20 or args.cameras_per_state != 10:
        raise ValueError("train8000 requires 40 tasks, 20 states/task, and 10 cameras/state")
    policy_seeds = list(args.policy_seeds or [])
    if policy_seeds != [1009, 2003, 3001, 4001]:
        raise ValueError("train8000 requires --policy-seeds 1009 2003 3001 4001")
    if args.env_seeds and len(args.env_seeds) != 1:
        raise ValueError("train8000 requires one fixed --env-seeds value")
    env_seed = int(args.env_seeds[0]) if args.env_seeds else 0
    cameras = _camera_splits(args)["train"]
    pairs = []
    for task_index, (suite, base_task, language) in enumerate(specs):
        states = _load_demo_states_cached(base_task, suite, args.demo_root)
        if len(states) < args.num_train_states + 5:
            raise RuntimeError(f"{suite}/{base_task} has only {len(states)} demonstrations")
        demo_indices = np.random.default_rng(args.state_split_seed + task_index).permutation(len(states))[:args.num_train_states]
        for state_position, demo_index_np in enumerate(demo_indices):
            demo_index = int(demo_index_np)
            state_hash = hashlib.sha256(np.ascontiguousarray(states[demo_index][1:]).tobytes()).hexdigest()
            for camera_offset in range(args.cameras_per_state):
                pair_index = len(pairs)
                camera_index = (task_index * 200 + state_position * 10 + camera_offset) % len(cameras)
                camera = cameras[camera_index]
                policy_seed = int(policy_seeds[pair_index % len(policy_seeds)])
                view = "_".join(map(str, camera))
                common = {
                    "suite": suite, "seed": policy_seed, "deterministic_reset": True,
                    "deterministic_reset_seed": env_seed, "base_task": base_task,
                    "episode": demo_index, "init_state_index": demo_index,
                    "state_source": "demo", "success": None,
                }
                clean = {**common, "condition": "clean", "category": "clean", "task_name": base_task,
                         "language": language, "instruction_mode": "task"}
                pert = {**common, "condition": "camera_viewpoints", "category": "Camera Viewpoints",
                        "task_name": f"{base_task}_view_{view}_initstate_0",
                        "language": language, "instruction_mode": "task"}
                pairs.append({
                    "condition": "camera_viewpoints", "group": "train8000", "clean": clean, "pert": pert,
                    "suite": suite, "base_task": base_task, "summary_path": None,
                    "init_state_index": demo_index, "policy_seed": policy_seed, "env_seed": env_seed,
                    "seed_config_index": pair_index % len(policy_seeds), "split": "train",
                    "state_source": "demo", "state_index": demo_index, "state_hash": state_hash,
                    "camera_tuple": camera, "camera_split": "train",
                    "sample_id": f"demo{demo_index:04d}_cam{camera_index:03d}_pseed{policy_seed}_eseed{env_seed}",
                })
    if len(pairs) != 8000:
        raise RuntimeError(f"expected 8000 train pairs, got {len(pairs)}")
    return [], pairs


def build_isolated_val_grid(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = _read_base_task_specs(args)
    if len(specs) != 40:
        raise ValueError("isolated val requires all 40 base tasks")
    policy_seed, cameras, states_per_task, cameras_per_state = 5003, _camera_splits(args)["val"], 5, 2
    env_seed = int(args.env_seeds[0]) if args.env_seeds else 0
    pairs = []
    for task_index, (suite, base_task, language) in enumerate(specs):
        states = _load_demo_states_cached(base_task, suite, args.demo_root)
        permutation = np.random.default_rng(args.state_split_seed + task_index).permutation(len(states))
        state_indices = [int(index) for index in permutation[20:25]]
        if len(state_indices) != states_per_task:
            raise RuntimeError(f"expected {states_per_task} val states for {suite}/{base_task}")

        for state_position, state_index in enumerate(state_indices):
            state = np.asarray(states[state_index])
            state_hash = hashlib.sha256(np.ascontiguousarray(state[1:]).tobytes()).hexdigest()
            for camera_offset in range(cameras_per_state):
                camera_index = (task_index * states_per_task * cameras_per_state
                                + state_position * cameras_per_state + camera_offset) % len(cameras)
                camera = cameras[camera_index]
                view = "_".join(map(str, camera))
                common = {
                    "suite": suite, "seed": policy_seed, "deterministic_reset": True,
                    "deterministic_reset_seed": env_seed, "base_task": base_task,
                    "episode": state_index, "init_state_index": state_index,
                    "state_source": "demo", "success": None,
                }
                clean = {**common, "condition": "clean", "category": "clean", "task_name": base_task,
                         "language": language, "instruction_mode": "task"}
                pert = {**common, "condition": "camera_viewpoints", "category": "Camera Viewpoints",
                        "task_name": f"{base_task}_view_{view}_initstate_0",
                        "language": language, "instruction_mode": "task"}
                pairs.append({
                    "condition": "camera_viewpoints", "group": "val400", "clean": clean, "pert": pert,
                    "suite": suite, "base_task": base_task, "summary_path": None,
                    "init_state_index": state_index, "policy_seed": policy_seed, "env_seed": env_seed,
                    "seed_config_index": 0, "split": "val", "state_source": "demo",
                    "state_index": state_index, "state_hash": state_hash,
                    "camera_tuple": camera, "camera_split": "val",
                    "sample_id": f"state{state_index:04d}_cam{camera_index:03d}_pseed{policy_seed}_eseed{env_seed}",
                })
    if len(pairs) != 400:
        raise RuntimeError(f"expected 400 val pairs, got {len(pairs)}")
    return [], pairs


def build_task_init_policy_grid_from_config(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config_path = _resolve_repo_path(args.perturb_config)
    config = phase2.load_json(config_path)
    if not isinstance(config, dict):
        raise TypeError(f"expected perturb config object in {config_path}")
    by_condition = _summary_condition_map(config)
    condition = args.conditions[0] if len(args.conditions) == 1 else None
    if condition != "camera_viewpoints":
        raise ValueError("--perturb-config currently supports exactly --conditions camera_viewpoints")
    clean_info = by_condition.get("clean")
    pert_info = by_condition.get(condition)
    if clean_info is None or pert_info is None:
        raise ValueError(f"perturb config must contain clean and {condition} entries")
    if _one_config_value(clean_info, "instruction_modes") != "task":
        raise ValueError("expected clean instruction mode 'task' in perturb config")
    pert_instruction_mode = _one_config_value(pert_info, "instruction_modes")
    perturb_parameters = pert_info.get("perturb_parameters")
    if not isinstance(perturb_parameters, dict) or perturb_parameters.get("type") != condition:
        raise ValueError(f"missing {condition} perturb_parameters in {config_path}")
    view_part = perturb_parameters.get("view_part")
    init_state = perturb_parameters.get("parsed_init_state")
    if not isinstance(view_part, str) or not isinstance(init_state, int):
        raise ValueError(f"invalid camera perturb parameters in {config_path}")
    expected_suffix = f"_view_{view_part}_initstate_{init_state}"

    pairs = []
    seed_configs = _seed_configs(args)
    init_filter = set(args.init_state_indices or [])
    for suite, base_task, clean_language in _read_base_task_specs(args):
        pert_task_name = f"{base_task}{expected_suffix}"
        init_states = _load_init_states_cached(base_task, suite)
        init_indices = sorted(init_filter) if init_filter else list(range(len(init_states)))
        for init_state_index in init_indices:
            for seed_config_index, policy_seed, env_seed in seed_configs:
                common = {
                    "suite": suite,
                    "seed": policy_seed,
                    "deterministic_reset": True,
                    "deterministic_reset_seed": env_seed,
                    "base_task": base_task,
                    "episode": init_state_index,
                    "init_state_index": init_state_index,
                    "success": None,
                }
                clean = dict(common)
                clean.update({
                    "condition": "clean",
                    "category": "clean",
                    "task_name": base_task,
                    "language": clean_language,
                    "instruction_mode": "task",
                })
                pert = dict(common)
                pert.update({
                    "condition": condition,
                    "category": "Camera Viewpoints",
                    "task_name": pert_task_name,
                    "language": clean_language,
                    "instruction_mode": pert_instruction_mode,
                })
                sample_id = f"init{init_state_index:04d}_pseed{policy_seed}_eseed{env_seed}"
                pairs.append({
                    "condition": condition,
                    "group": "task_init_policy_grid",
                    "source_group": None,
                    "source_episode": None,
                    "clean": clean,
                    "pert": pert,
                    "suite": suite,
                    "base_task": base_task,
                    "summary_path": None,
                    "perturb_config_path": str(config_path),
                    "init_state_index": init_state_index,
                    "policy_seed": policy_seed,
                    "env_seed": env_seed,
                    "seed_config_index": seed_config_index,
                    "sample_id": sample_id,
                })
        print(f"Built grid pairs from perturb config for {suite}/{base_task}", flush=True)
    return [config], pairs


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
    p.add_argument(
        "--perturb-config",
        default=None,
        help="alignment-audit JSON that defines the camera perturbation without rollout summaries",
    )
    p.add_argument("--train8000", action="store_true", help="use the GetLatent.md 8000-pair training design")
    p.add_argument("--isolated-val", action="store_true")
    p.add_argument("--demo-root", default=str(phase2.ROOT / "LIBERO-Cosmos-Policy/success_only"))
    p.add_argument("--camera-classification", default=str(phase2.ROOT.parent / "LIBERO-plus/libero/libero/benchmark/task_classification.json"))
    p.add_argument("--state-split-seed", type=int, default=0)
    p.add_argument("--camera-split-seed", type=int, default=0)
    p.add_argument("--num-train-states", type=int, default=20)
    p.add_argument("--cameras-per-state", type=int, default=10)
    p.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--task-limit", type=int, default=0, help="limit base tasks per suite; 0 selects all")
    p.add_argument(
        "--only-task",
        action="append",
        default=[],
        help="select one task as suite:task_name; may be repeated",
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
    p.add_argument("--hook-mode", default="pre", choices=["pre", "forward"],
                   help="'pre' = capture at block entry; 'forward' = capture at block exit")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)
    if args.train8000:
        summaries, pairs = build_train8000_grid(args)
    elif args.isolated_val:
        summaries, pairs = build_isolated_val_grid(args)
    elif args.sample_mode == "task_init_policy_grid":
        if args.perturb_config:
            summaries, pairs = build_task_init_policy_grid_from_config(args)
        else:
            summaries, pairs = build_task_init_policy_grid(args)
    else:
        if args.perturb_config:
            raise ValueError("--perturb-config requires --sample-mode task_init_policy_grid")
        summaries, pairs = build_summary_pair_samples(args)
    all_pairs = pairs
    total_before_shard = len(all_pairs)
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
        "summaries": [] if args.train8000 or args.isolated_val or args.perturb_config else [str(pathlib.Path(item)) for item in args.summary],
        "perturb_config": str(_resolve_repo_path(args.perturb_config)) if args.perturb_config and not args.train8000 else None,
        "train8000": args.train8000,
        "isolated_val": args.isolated_val,
        "suites": sorted({str(pair["suite"]) for pair in all_pairs}),
        "base_tasks": sorted({str(pair["base_task"]) for pair in all_pairs}),
        "conditions": args.conditions,
        "groups": args.groups,
        "sample_mode": args.sample_mode,
        "policy_seeds": sorted({int(pair["policy_seed"]) for pair in all_pairs}),
        "env_seeds": sorted({int(pair["env_seed"]) for pair in all_pairs}),
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
