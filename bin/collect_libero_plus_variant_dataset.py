#!/usr/bin/env python3
"""Collect clean-trajectory pairs for five action-invariant LIBERO-Plus conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from collections import deque
from typing import Any

# Keep one collector from claiming most host CPU cores.  These defaults must be
# set before importing NumPy, PyTorch, OpenCV, or FFmpeg-backed helpers.  Users
# can still override any value explicitly in the launch environment.
DEFAULT_CPU_THREADS = os.environ.get("COSMOS_POLICY_CPU_THREADS", "8")
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
):
    os.environ.setdefault(variable, DEFAULT_CPU_THREADS)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DETERMINISTIC", "True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", ROOT.parent / "LIBERO-plus"))
for item in (str(LIBERO_PLUS), str(ROOT), str(ROOT / "bin")):
    if item not in sys.path:
        sys.path.insert(0, item)

import numpy as np
import torch
from PIL import Image
from libero.libero.envs.env_wrapper import fog, gaussian_blur, glass_blur, motion_blur, zoom_blur

import collect_paired_camera_render_dataset as legacy
import run_phase2_angular_cosmos as phase2
from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, TaskSuite
from cosmos_policy.utils.utils import set_seed_everywhere
from phase8_correction_lib import append_jsonl, write_json


torch.set_num_threads(int(DEFAULT_CPU_THREADS))
torch.set_num_interop_threads(max(1, min(4, int(DEFAULT_CPU_THREADS))))


CONDITIONS = ("camera", "background", "light", "noise", "language")
RENDER_CONDITIONS = {"camera", "background", "light"}
FPS = legacy.FPS
WAIT_STEPS = legacy.NUM_STEPS_WAIT


def read_json(path: str | pathlib.Path) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_specs(args) -> tuple[list[legacy.TaskSpec], dict[str, Any], dict[str, Any]]:
    catalog, splits = read_json(args.variant_catalog), read_json(args.variant_splits)
    if splits["catalog_sha256"] != canonical_sha256(catalog):
        raise RuntimeError("variant split file does not match catalog")
    if not splits.get("official_variants_reserved_for_external_test"):
        raise RuntimeError("split file does not reserve official variants")
    only = set(args.only_task)
    specs = []
    for name, task in catalog["tasks"].items():
        if only and name not in only:
            continue
        specs.append(legacy.TaskSpec(len(specs), name, task["instruction"], task["bddl_file"], catalog["suite"]))
    if args.task_limit:
        specs = specs[: args.task_limit]
    return specs, catalog, splits


def variant_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        record["variant_id"]: record
        for task in catalog["tasks"].values()
        for condition in CONDITIONS
        for record in task["variants"][condition]
    }


def choose_variant(pool: list[str], task_id: int, split_local: int) -> str:
    return pool[(task_id * 1_000_003 + split_local) % len(pool)]


def official_noise(image: np.ndarray, noise_id: int) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    severity = (noise_id - 1) % 10 + 1
    if noise_id <= 10:
        out = motion_blur(pil, severity)
    elif noise_id <= 20:
        out = gaussian_blur(pil, severity)
    elif noise_id <= 30:
        out = zoom_blur(pil, severity)
    elif noise_id <= 40:
        out = fog(pil, severity)
    else:
        out = glass_blur(pil, severity)
    return np.ascontiguousarray(np.asarray(out, dtype=np.uint8))


def validate_state_layout(clean_env, pert_env, task: str, variant: str) -> None:
    clean, pert = clean_env.sim.model, pert_env.sim.model
    shape = (clean.nq, clean.nv, clean.na, clean.njnt)
    if shape != (pert.nq, pert.nv, pert.na, pert.njnt):
        raise RuntimeError(f"state layout size mismatch for {task}/{variant}")
    clean_names = [clean.joint_id2name(i) for i in range(clean.njnt)]
    pert_names = [pert.joint_id2name(i) for i in range(pert.njnt)]
    if clean_names != pert_names:
        raise RuntimeError(f"joint order mismatch for {task}/{variant}")


def paired_images(obs, clean_env, pert_env, variant, args):
    front, wrist = legacy.get_obs_images(obs, args.flip_images)
    if variant["condition"] in RENDER_CONDITIONS:
        flat = np.asarray(clean_env.sim.get_state().flatten()).copy()
        pert_front, pert_wrist = legacy.render_from_state(pert_env, flat, args.flip_images)
    elif variant["condition"] == "noise":
        pert_front = official_noise(front, int(variant["parameters"]["noise_id"]))
        pert_wrist = wrist.copy()  # LIBERO-Plus only perturbs agentview_image.
    else:
        pert_front, pert_wrist = front, wrist
    return front, wrist, pert_front, pert_wrist


def rollout(clean_env, pert_env, cfg, model, stats, task, initial_state, policy_seed, variant, args):
    set_seed_everywhere(args.env_seed)
    clean_env.reset()
    obs = clean_env.set_init_state(initial_state)
    if pert_env is not None:
        pert_env.reset()
    queue: deque[np.ndarray] = deque(maxlen=cfg.chunk_size)
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = clean_env.step(phase2.DUMMY_ACTION)
    set_seed_everywhere(policy_seed)
    images = {"clean_front": [], "clean_wrist": [], "pert_front": [], "pert_wrist": []}
    states, sim_states, actions, chunks, timestamps, frame_indices = [], [], [], [], [], []
    query_index, success, success_step, chunk = 0, False, None, None
    for frame_index in range(TASK_MAX_STEPS[TaskSuite.LIBERO_10]):
        cf, cw, pf, pw = paired_images(obs, clean_env, pert_env, variant, args)
        prop = legacy.proprio_from_obs(obs)
        if not queue:
            out = cosmos_utils.get_action(
                cfg, model, stats, {"primary_image": cf, "wrist_image": cw, "proprio": prop},
                task.language, seed=(policy_seed + query_index * 1_000_003) % (2**32),
                randomize_seed=False, num_denoising_steps_action=args.num_denoising_steps,
                generate_future_state_and_value_in_parallel=False,
            )
            chunk = np.asarray(out["actions"], dtype=np.float32)
            queue.extend(chunk)
            query_index += 1
        action = np.asarray(queue.popleft(), dtype=np.float32)
        for key, value in zip(images, (cf, cw, pf, pw)):
            images[key].append(value)
        states.append(prop); sim_states.append(legacy.recoverable_state(clean_env))
        actions.append(action); chunks.append(chunk.copy())
        timestamps.append(frame_index / FPS); frame_indices.append(frame_index)
        obs, _, done, _ = clean_env.step(action.tolist())
        if done:
            success, success_step = True, frame_index
            break
    return {
        "success": success, "success_step": success_step,
        "clean": {"front": np.stack(images["clean_front"]), "wrist": np.stack(images["clean_wrist"])},
        "perturbed": {"front": np.stack(images["pert_front"]), "wrist": np.stack(images["pert_wrist"])},
        "observation.state": np.stack(states).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "action_context": np.stack(chunks).astype(np.float32), "sim_state": sim_states,
        "timestamp": np.asarray(timestamps, dtype=np.float32),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
    }


def encode_latents(model, data, cfg, language: bool):
    clean = torch.stack([legacy.vae_video_latent(model, f, w, cfg) for f, w in zip(data["clean"]["front"], data["clean"]["wrist"])])
    if language:
        return {"vae_video": clean}
    pert = torch.stack([legacy.vae_video_latent(model, f, w, cfg) for f, w in zip(data["perturbed"]["front"], data["perturbed"]["wrist"])])
    return {"clean": {"vae_video": clean}, "perturbed": {"vae_video": pert}}


def save_videos(out_root, episode_index, data, language):
    import imageio.v2 as imageio
    chunk = legacy.episode_chunk(episode_index)
    arrays = {
        "observation.images.front": data["clean"]["front"],
        "observation.images.wrist": data["clean"]["wrist"],
    }
    if not language:
        arrays.update({"perturbed.images.front": data["perturbed"]["front"], "perturbed.images.wrist": data["perturbed"]["wrist"]})
    paths = {}
    for key, array in arrays.items():
        path = out_root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), array.astype(np.uint8), fps=FPS, macro_block_size=None)
        paths[key] = str(path.relative_to(out_root))
    return paths


def save_sample(out_root, split, task, variant, sample_id, policy_seed, init_index, episode_index, frame_start, data, model, cfg, args):
    condition, language = variant["condition"], variant["condition"] == "language"
    path = out_root / "samples" / "libero_10" / task.name / condition / split / f"{sample_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    videos = save_videos(out_root, episode_index, data, language) if args.save_rgb_videos else {}
    parquet = legacy.save_lerobot_parquet(out_root, episode_index, frame_start, task, data)
    meta = {
        "suite": "libero_10", "base_task": task.name, "task_index": task.task_id,
        "condition": condition, "variant_id": variant["variant_id"],
        "variant_task_name": variant["task_name"], "variant_parameters": variant["parameters"],
        "official_variant": variant["official"], "split": split, "sample_id": sample_id,
        "episode_index": episode_index, "init_state_index": init_index,
        "policy_seed": policy_seed, "env_seed": args.env_seed,
        "clean_success": True, "success_step": data["success_step"],
        "clean_instruction": task.language, "perturbed_instruction": variant["instruction"],
        "rgb_video_paths": videos, "lerobot_parquet_path": parquet, "fps": FPS,
        "action_alignment": "shared_clean_action_trajectory",
    }
    payload = {
        "meta": meta, "latent": encode_latents(model, data, cfg, language),
        "observation.state": data["observation.state"], "action": data["action"],
        "action_context": data["action_context"], "sim_state": data["sim_state"],
        "timestamp": data["timestamp"], "frame_index": data["frame_index"],
    }
    torch.save(payload, path)
    return {**meta, "path": str(path.relative_to(out_root)), "num_frames": len(data["frame_index"])}


def parser() -> argparse.ArgumentParser:
    p = legacy.build_parser()
    p.description = __doc__
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--variant-catalog", default=str(ROOT / "configs/libero_plus_variant_catalog.json"))
    p.add_argument("--variant-splits", default=str(ROOT / "configs/libero_plus_variant_splits.json"))
    p.add_argument(
        "--attempt-offset",
        type=int,
        default=0,
        help="Start candidate seed/variant/init selection at this attempt index; useful for segmented continuation.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if not 0 < args.train_per_task < args.examples_per_task:
        raise ValueError("train-per-task must be between zero and examples-per-task")
    num_shards, shard_id, local_rank = legacy.resolve_sharding(args)
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    specs, catalog, splits = load_specs(args)
    tasks = [task for task in specs if task.task_id % num_shards == shard_id]
    if not tasks:
        raise RuntimeError(f"shard {shard_id}/{num_shards} received no tasks")
    records = variant_index(catalog)
    out_root = legacy.shard_output_root(args.output_dir, num_shards, shard_id)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"use a fresh output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "variant_selection.json", {
        "catalog_sha256": splits["catalog_sha256"],
        "condition": args.condition,
        "tasks": {task.name: splits["tasks"][task.name][args.condition] for task in tasks},
    })
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)
    cfg = legacy.make_cfg(args)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    if args.t5_extra_embeddings:
        phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    rows, episodes, episode_stats, errors, counts = [], [], [], [], {"train": 0, "val": 0}
    frame_count, start = 0, time.time()
    state_dim = action_dim = 0
    expected_train = len(tasks) * args.train_per_task
    for task in tasks:
        clean_env = legacy.make_env_from_bddl(task.problem_folder, task.bddl_file, args.env_resolution, args.env_seed)
        init_states, accepted, attempts = legacy.load_init_states_for_task(task), 0, 0
        try:
            while accepted < args.examples_per_task and attempts < args.max_attempts_per_task:
                attempts += 1
                split = "train" if accepted < args.train_per_task else "val"
                local = accepted if split == "train" else accepted - args.train_per_task
                # Advance the rollout candidate after every attempt, including
                # failures.  Using ``local`` here would retry the same variant
                # and policy seed until it succeeds, which can stall forever on
                # a hard task.  ``local`` is still used for the accepted sample
                # index and train/val accounting below.
                attempt_local = args.attempt_offset + attempts - 1
                pool = splits["tasks"][task.name][args.condition][split]
                variant = records[choose_variant(pool, task.task_id, attempt_local)]
                if variant["official"]:
                    raise RuntimeError("official variant entered train/val")
                seeds = args.policy_seeds if split == "train" else args.val_policy_seeds
                policy_seed = legacy.policy_seed_for_sample(
                    seeds, task.name, split, attempt_local
                )
                init_index = attempt_local % len(init_states)
                sample_id = f"{split}_task{task.task_id:02d}_ex{accepted:03d}_init{init_index:04d}_{variant['variant_id']}_pseed{policy_seed}"
                pert_env = None
                try:
                    if args.condition in RENDER_CONDITIONS:
                        pert_env = legacy.make_env_for_task_name(variant["task_name"], "libero_10", args.env_resolution, args.env_seed)
                        validate_state_layout(clean_env, pert_env, task.name, variant["variant_id"])
                    data = rollout(clean_env, pert_env, cfg, model, stats, task, init_states[init_index], policy_seed, variant, args)
                    if not data["success"]:
                        continue
                    episode_index = counts["train"] if split == "train" else expected_train + counts["val"]
                    row = save_sample(out_root, split, task, variant, sample_id, policy_seed, init_index, episode_index, frame_count, data, model, cfg, args)
                    rows.append(row); episodes.append({"episode_index": episode_index, "tasks": [task.language], "length": row["num_frames"]})
                    stats_row = legacy.episode_stats_row(row, data, frame_count)
                    if args.condition == "language":
                        stats_row["stats"].pop("perturbed.images.front", None)
                        stats_row["stats"].pop("perturbed.images.wrist", None)
                    episode_stats.append(stats_row)
                    state_dim, action_dim = data["observation.state"].shape[1], data["action"].shape[1]
                    frame_count += row["num_frames"]; counts[split] += 1; accepted += 1
                except Exception as exc:
                    errors.append({"task": task.name, "attempt": attempts, "error": repr(exc)})
                    if args.fail_fast:
                        raise
                finally:
                    if pert_env is not None:
                        pert_env.close()
        finally:
            clean_env.close()
    rows.sort(key=lambda row: row["episode_index"]); episodes.sort(key=lambda row: row["episode_index"])
    episode_stats.sort(key=lambda row: row["episode_index"])
    legacy.reindex_lerobot_frames(out_root, rows, episode_stats)
    append_jsonl(out_root / "manifest.jsonl", rows)
    append_jsonl(out_root / "meta/episodes.jsonl", episodes)
    append_jsonl(out_root / "meta/episodes_stats.jsonl", episode_stats)
    append_jsonl(out_root / "meta/tasks.jsonl", [{"task_index": task.task_id, "task": task.language} for task in tasks])
    info = legacy.lerobot_info(len(rows), frame_count, len(tasks), args.env_resolution, state_dim, action_dim, counts["train"], counts["val"])
    if args.condition == "language":
        info["total_videos"] = len(rows) * 2
        info["features"].pop("perturbed.images.front")
        info["features"].pop("perturbed.images.wrist")
    write_json(out_root / "meta/info.json", info)
    write_json(out_root / "summary.json", {
        "dataset": "paired_libero_plus_variant", "condition": args.condition,
        "tasks": [task.name for task in tasks], "num_samples": len(rows), "num_frames": frame_count,
        "split_counts": counts, "official_variants_used": False, "errors": errors,
        "num_shards": num_shards, "shard_id": shard_id, "elapsed_s": time.time() - start,
        "attempt_offset": args.attempt_offset,
    })
    if len(rows) != len(tasks) * args.examples_per_task:
        raise RuntimeError("dataset collection was undersampled")


if __name__ == "__main__":
    main()
