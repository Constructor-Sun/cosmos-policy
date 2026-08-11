#!/usr/bin/env python3
"""Collect clean/perturbed camera render pairs from clean LIBERO-10 rollouts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import re
import site
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy-matplotlib")
os.environ.setdefault("DETERMINISTIC", "True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", ROOT.parent / "LIBERO-plus"))
for item in (str(LIBERO_PLUS), str(ROOT), str(ROOT / "bin")):
    if item not in sys.path:
        sys.path.insert(0, item)
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import run_phase2_angular_cosmos as phase2
from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import duplicate_array, prepare_images_for_model
from cosmos_policy.experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, TaskSuite
from cosmos_policy.utils.utils import set_seed_everywhere
from phase8_correction_lib import append_jsonl, write_json


FPS = 20
NUM_STEPS_WAIT = 10
CAMERA_TASK_RE = re.compile(r"^(?P<base>.+)_view_(?P<view>\d+(?:_\d+)*)_initstate_(?P<init_state>\d+)$")


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    language: str
    bddl_file: str
    problem_folder: str


def _resolve_repo_path(path_arg: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_arg).expanduser()
    return path if path.is_absolute() else ROOT / path


def _task_suite():
    with contextlib.redirect_stdout(io.StringIO()):
        return benchmark.get_benchmark_dict()["libero_10"]()


def _category_suite(suite_name: str, category: str):
    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return suite_cls(category_value=category)
        except TypeError:
            return suite_cls()


def _language_from_bddl(problem_folder: str, bddl_file: str) -> str:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / problem_folder / bddl_file
    text = bddl.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+([^)]+)\)", text)
    if match is None:
        raise RuntimeError(f"missing language in BDDL: {bddl}")
    return match.group(1).strip()


def load_libero10_tasks(args) -> list[TaskSpec]:
    data = json.loads(_resolve_repo_path(args.camera_classification).read_text(encoding="utf-8"))
    specs = []
    only = {_base_init_state_name(name) for name in args.only_task}
    seen = set()
    for item in data.get("libero_10", []):
        if item.get("category") != "Camera Viewpoints":
            continue
        parsed = _camera_tuple_from_task_name(item["name"])
        if parsed is None:
            continue
        base_name, _, init_state = parsed
        if init_state != 0 or base_name in seen:
            continue
        seen.add(base_name)
        if only and base_name not in only:
            continue
        bddl_file = f"{base_name}.bddl"
        specs.append(TaskSpec(len(specs), base_name, _language_from_bddl("libero_10", bddl_file), bddl_file, "libero_10"))

    if specs:
        if args.task_limit:
            specs = specs[: args.task_limit]
        return specs

    raise RuntimeError("no LIBERO-10 base tasks with camera variants selected")


def make_env_from_bddl(
    problem_folder: str, bddl_file: str, resolution: int, env_seed: int = 0
) -> OffScreenRenderEnv:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / problem_folder / bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(env_seed)
    return env


def make_env_for_task_name(
    task_name: str, suite: str, resolution: int, env_seed: int = 0
) -> OffScreenRenderEnv:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(env_seed)
    return env


def make_env_for_task(task, resolution: int, env_seed: int = 0) -> OffScreenRenderEnv:
    return make_env_from_bddl(
        task.problem_folder, task.bddl_file, resolution, env_seed=env_seed
    )


def _base_init_state_name(task_name: str) -> str:
    for marker in ("_table_", "_tb_", "_view_", "_language_", "_light_"):
        if marker in task_name:
            return task_name.split(marker, 1)[0]
    return task_name


def load_init_states_for_task(task: TaskSpec):
    init_root = pathlib.Path(get_libero_path("init_states"))
    base_name = _base_init_state_name(task.name)
    candidates = [
        init_root / task.problem_folder / f"{base_name}.pruned_init",
        init_root / "libero_10" / f"{base_name}.pruned_init",
    ]
    for path in candidates:
        if path.is_file():
            return torch.load(str(path), weights_only=False)
    raise FileNotFoundError(
        "missing init state file for "
        f"{task.name}; tried: {', '.join(str(path) for path in candidates)}"
    )


def _camera_tuple_from_task_name(task_name: str) -> tuple[str, tuple[int, ...], int] | None:
    match = CAMERA_TASK_RE.match(task_name)
    if match is None:
        return None
    return (
        match.group("base"),
        tuple(map(int, match.group("view").split("_"))),
        int(match.group("init_state")),
    )


def offset_camera_from_official_test(
    camera: tuple[int, ...], offset_deg: int
) -> tuple[int, ...]:
    """Create a training camera at a fixed spherical-elevation offset.

    LIBERO-Plus encodes camera variants as
    (horizontal, vertical, scale, endpoint_yaw, endpoint_pitch).  The official
    paper constructs training views 5 degrees away from test views on the
    spherical coordinate system.  We offset the vertical spherical angle so
    endpoint-orientation and scale perturbations remain otherwise unchanged.
    """
    if len(camera) != 5:
        raise ValueError(
            f"expected a five-value LIBERO-Plus camera tuple, got {camera}"
        )
    horizontal, vertical, scale, endpoint_yaw, endpoint_pitch = camera
    return (
        horizontal,
        (vertical + offset_deg) % 360,
        scale,
        endpoint_yaw,
        endpoint_pitch,
    )


def parse_camera_splits(args) -> dict[str, dict[str, list[tuple[int, ...]]]]:
    data = json.loads(_resolve_repo_path(args.camera_classification).read_text(encoding="utf-8"))
    by_task: dict[str, set[tuple[int, ...]]] = {}
    for item in data.get("libero_10", []):
        if item.get("category") != "Camera Viewpoints":
            continue
        parsed = _camera_tuple_from_task_name(item["name"])
        if parsed is None:
            continue
        base, camera, init_state = parsed
        if init_state == 0:
            by_task.setdefault(base, set()).add(camera)
    if not by_task:
        raise RuntimeError("no libero_10 camera variants found in task classification")

    if args.camera_train_offset_deg == 0:
        raise ValueError("--camera-train-offset-deg must be non-zero to avoid official-test leakage")
    if not 0.0 < args.camera_val_fraction < 1.0:
        raise ValueError("--camera-val-fraction must be between 0 and 1")

    splits: dict[str, dict[str, list[tuple[int, ...]]]] = {}
    for base_task, official_test_cameras in by_task.items():
        official_test = set(official_test_cameras)
        generated = {
            offset_camera_from_official_test(camera, args.camera_train_offset_deg)
            for camera in official_test
        }
        collisions = generated & official_test
        if collisions:
            raise RuntimeError(
                f"generated cameras overlap official LIBERO-Plus test cameras for {base_task}: "
                f"{sorted(collisions)}"
            )
        if len(generated) != len(official_test):
            raise RuntimeError(f"camera offset produced duplicate views for {base_task}")

        ordered = sorted(generated)
        seed_bytes = hashlib.sha256(f"{args.camera_split_seed}:{base_task}".encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:8], "little") % (2**32)
        order = np.random.default_rng(seed).permutation(len(ordered))
        shuffled = [ordered[int(index)] for index in order]
        val_count = max(
            1,
            min(
                math.floor(len(shuffled) * args.camera_val_fraction + 0.5),
                len(shuffled) - 1,
            ),
        )
        train = shuffled[:-val_count]
        val = shuffled[-val_count:]
        if set(train) & set(val):
            raise RuntimeError(f"train/val camera overlap for {base_task}")
        splits[base_task] = {
            "train": train,
            "val": val,
            "official_test": sorted(official_test),
        }
    return splits


def camera_for_sample(cameras: list[tuple[int, ...]], global_pos: int) -> tuple[int, tuple[int, ...]]:
    camera_index = global_pos % len(cameras)
    return camera_index, cameras[camera_index]


def policy_seed_for_sample(
    seed_bases: list[int], task_name: str, split: str, split_local: int
) -> int:
    """Derive a unique, reproducible policy seed for each accepted sample."""
    if not seed_bases:
        raise ValueError(f"no policy seed bases configured for split={split}")
    base = int(seed_bases[split_local % len(seed_bases)])
    digest = hashlib.sha256(
        f"{base}:{task_name}:{split}:{split_local}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def resolve_sharding(args) -> tuple[int, int, int | None]:
    """Resolve task sharding from CLI flags or torchrun environment variables."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    local_rank_raw = os.environ.get("LOCAL_RANK")

    num_shards = args.num_shards if args.num_shards is not None else env_world_size
    shard_id = args.shard_id if args.shard_id is not None else env_rank
    local_rank = int(local_rank_raw) if local_rank_raw is not None else None

    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"--shard-id must satisfy 0 <= shard_id < num_shards; "
            f"got shard_id={shard_id}, num_shards={num_shards}"
        )
    if env_world_size > 1 and args.num_shards is not None and num_shards != env_world_size:
        raise ValueError(
            f"--num-shards={num_shards} conflicts with torchrun WORLD_SIZE={env_world_size}"
        )
    if env_world_size > 1 and args.shard_id is not None and shard_id != env_rank:
        raise ValueError(
            f"--shard-id={shard_id} conflicts with torchrun RANK={env_rank}"
        )
    return num_shards, shard_id, local_rank


def shard_output_root(
    output_dir: str | pathlib.Path, num_shards: int, shard_id: int
) -> pathlib.Path:
    root = pathlib.Path(output_dir).expanduser()
    if num_shards == 1:
        return root
    return root / f"shard-{shard_id:03d}-of-{num_shards:03d}"


def get_obs_images(obs: dict[str, Any], flip_images: bool) -> tuple[np.ndarray, np.ndarray]:
    front = obs["agentview_image"]
    wrist = obs["robot0_eye_in_hand_image"]
    if flip_images:
        front = np.flipud(front)
        wrist = np.flipud(wrist)
    return np.ascontiguousarray(front), np.ascontiguousarray(wrist)


def proprio_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])).astype(np.float32)


def recoverable_state(env) -> dict[str, np.ndarray]:
    sim = env.sim
    out = {"flattened": np.asarray(sim.get_state().flatten()).copy()}
    data = sim.data
    for key in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        if hasattr(data, key):
            out[key] = np.asarray(getattr(data, key)).copy()
    return out


def set_env_flat_state(env, flat_state: np.ndarray) -> None:
    if hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(np.asarray(flat_state))
    elif hasattr(env.sim.get_state(), "from_flattened"):
        env.sim.set_state(env.sim.get_state().from_flattened(np.asarray(flat_state)))
    else:
        raise RuntimeError("env.sim does not support restoring a flattened MuJoCo state")
    env.sim.forward()


def get_env_observations(env) -> dict[str, Any]:
    if hasattr(env, "_post_process"):
        env._post_process()
    if hasattr(env, "_update_observables"):
        env._update_observables(force=True)
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
        return env.env._get_observations()
    raise RuntimeError(f"{type(env).__name__} does not expose an observation getter")


def render_from_state(env, flat_state: np.ndarray, flip_images: bool) -> tuple[np.ndarray, np.ndarray]:
    set_env_flat_state(env, flat_state)
    obs = get_env_observations(env)
    return get_obs_images(obs, flip_images)


def vae_video_latent(model, front: np.ndarray, wrist: np.ndarray, cfg) -> torch.Tensor:
    device = next(model.parameters()).device
    wrist_img, front_img = prepare_images_for_model([wrist, front], cfg)
    blank = np.zeros_like(front_img)
    raw = np.concatenate(
        [
            np.expand_dims(np.zeros_like(blank), axis=0),
            duplicate_array(blank, 4),
            duplicate_array(wrist_img, 4),
            duplicate_array(front_img, 4),
            duplicate_array(blank.copy(), 4),
            duplicate_array(blank.copy(), 4),
            duplicate_array(wrist_img.copy(), 4),
            duplicate_array(front_img.copy(), 4),
            duplicate_array(blank.copy(), 4),
        ],
        axis=0,
    )
    video = torch.from_numpy(np.transpose(raw[None], (0, 4, 1, 2, 3))).to(dtype=torch.uint8, device=device)
    with torch.no_grad():
        latent = model.encode(video).contiguous().float()
    return latent[0, :, [2, 3], :, :].cpu().half()


def make_cfg(args) -> argparse.Namespace:
    cfg = phase2.make_cfg(args)
    cfg.task_suite_name = "libero_10"
    cfg.num_open_loop_steps = cfg.chunk_size
    cfg.deterministic = True
    cfg.deterministic_reset = True
    cfg.deterministic_reset_seed = args.env_seed
    cfg.env_img_res = args.env_resolution
    cfg.unnorm_key = "libero_10"
    return cfg


def rollout_once(
    *,
    clean_env,
    pert_env,
    cfg,
    model,
    stats,
    task: TaskSpec,
    initial_state: Any,
    policy_seed: int,
    args,
) -> dict[str, Any]:
    set_seed_everywhere(args.env_seed)
    clean_env.reset()
    obs = clean_env.set_init_state(initial_state)
    pert_env.reset()
    action_queue: deque[np.ndarray] = deque(maxlen=cfg.chunk_size)
    max_steps = TASK_MAX_STEPS[TaskSuite.LIBERO_10]

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = clean_env.step(phase2.DUMMY_ACTION)

    # Environment construction/reset remains controlled by env_seed.  Policy
    # sampling uses a distinct per-sample seed after reset so trajectories stay
    # reproducible without collapsing all episodes to the same random stream.
    set_seed_everywhere(policy_seed)

    clean_front, clean_wrist, pert_front, pert_wrist = [], [], [], []
    states, sim_states, actions, action_chunks = [], [], [], []
    timestamps, frame_indices = [], []
    success = False
    success_step = None
    action_query_index = 0

    for frame_index in range(max_steps):
        flat_state = np.asarray(clean_env.sim.get_state().flatten()).copy()
        cf, cw = get_obs_images(obs, args.flip_images)
        pf, pw = render_from_state(pert_env, flat_state, args.flip_images)
        prop = proprio_from_obs(obs)
        observation = {"primary_image": cf, "wrist_image": cw, "proprio": prop}
        if not action_queue:
            query_seed = (policy_seed + action_query_index * 1_000_003) % (2**32)
            out = cosmos_utils.get_action(
                cfg,
                model,
                stats,
                observation,
                task.language,
                seed=query_seed,
                randomize_seed=False,
                num_denoising_steps_action=args.num_denoising_steps,
                generate_future_state_and_value_in_parallel=False,
            )
            chunk = np.asarray(out["actions"], dtype=np.float32)
            action_queue.extend(chunk)
            action_query_index += 1
        action = np.asarray(action_queue.popleft(), dtype=np.float32)

        clean_front.append(cf)
        clean_wrist.append(cw)
        pert_front.append(pf)
        pert_wrist.append(pw)
        states.append(prop)
        sim_states.append(recoverable_state(clean_env))
        actions.append(action)
        action_chunks.append(chunk.copy())
        frame_indices.append(frame_index)
        timestamps.append(frame_index / FPS)

        obs, _, done, _ = clean_env.step(action.tolist())
        if done:
            success = True
            success_step = frame_index
            break

    return {
        "success": success,
        "success_step": success_step,
        "clean": {
            "front": np.stack(clean_front),
            "wrist": np.stack(clean_wrist),
        },
        "perturbed": {
            "front": np.stack(pert_front),
            "wrist": np.stack(pert_wrist),
        },
        "observation.state": np.stack(states).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "action_context": np.stack(action_chunks).astype(np.float32),
        "sim_state": sim_states,
        "timestamp": np.asarray(timestamps, dtype=np.float32),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
    }


def sample_path(out_root: pathlib.Path, split: str, task_name: str, sample_id: str) -> pathlib.Path:
    return out_root / "samples" / "libero_10" / _base_init_state_name(task_name) / "camera_viewpoints" / split / f"{sample_id}.pt"


def video_dir_for_sample(out_root: pathlib.Path, split: str, task_name: str, sample_id: str) -> pathlib.Path:
    return out_root / "videos" / "libero_10" / _base_init_state_name(task_name) / "camera_viewpoints" / split / sample_id


def episode_chunk(episode_index: int, chunks_size: int = 1000) -> int:
    return episode_index // chunks_size


def save_lerobot_videos(out_root: pathlib.Path, episode_index: int, data: dict[str, Any]) -> dict[str, str]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("saving RGB videos requires imageio in the runtime environment") from exc

    chunk = episode_chunk(episode_index)
    videos = {
        "observation.images.front": out_root / "videos" / f"chunk-{chunk:03d}" / "observation.images.front" / f"episode_{episode_index:06d}.mp4",
        "observation.images.wrist": out_root / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist" / f"episode_{episode_index:06d}.mp4",
        "perturbed.images.front": out_root / "videos" / f"chunk-{chunk:03d}" / "perturbed.images.front" / f"episode_{episode_index:06d}.mp4",
        "perturbed.images.wrist": out_root / "videos" / f"chunk-{chunk:03d}" / "perturbed.images.wrist" / f"episode_{episode_index:06d}.mp4",
    }
    arrays = {
        "observation.images.front": data["clean"]["front"],
        "observation.images.wrist": data["clean"]["wrist"],
        "perturbed.images.front": data["perturbed"]["front"],
        "perturbed.images.wrist": data["perturbed"]["wrist"],
    }
    for key, video_path in videos.items():
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(video_path), arrays[key].astype(np.uint8), fps=FPS, macro_block_size=None)
    return {key: str(video_path.relative_to(out_root)) for key, video_path in videos.items()}


def fixed_size_float_array(array: np.ndarray, name: str):
    import pyarrow as pa

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), arr.shape[1])


def save_lerobot_parquet(out_root: pathlib.Path, episode_index: int, global_start_index: int, task: TaskSpec, data: dict[str, Any]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(data["frame_index"])
    chunk = episode_chunk(episode_index)
    path = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "observation.state": fixed_size_float_array(data["observation.state"], "observation.state"),
        "action": fixed_size_float_array(data["action"], "action"),
        "timestamp": pa.array(data["timestamp"].astype(np.float32), type=pa.float32()),
        "frame_index": pa.array(data["frame_index"].astype(np.int64), type=pa.int64()),
        "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
        "index": pa.array(np.arange(global_start_index, global_start_index + n, dtype=np.int64), type=pa.int64()),
        "task_index": pa.array(np.full(n, task.task_id, dtype=np.int64), type=pa.int64()),
    })
    pq.write_table(table, path)
    return str(path.relative_to(out_root))


def feature_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def image_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32) / 255.0
    flat = np.transpose(arr, (0, 3, 1, 2)).reshape(arr.shape[0], 3, -1)
    return {
        "min": flat.min(axis=(0, 2)).reshape(3, 1, 1).tolist(),
        "max": flat.max(axis=(0, 2)).reshape(3, 1, 1).tolist(),
        "mean": flat.mean(axis=(0, 2)).reshape(3, 1, 1).tolist(),
        "std": flat.std(axis=(0, 2)).reshape(3, 1, 1).tolist(),
        "count": [int(min(arr.shape[0], 100))],
    }


def episode_stats_row(row: dict[str, Any], data: dict[str, Any], global_start_index: int) -> dict[str, Any]:
    n = int(len(data["frame_index"]))
    episode_index = int(row["episode_index"])
    task_index = int(row["task_index"])
    index = np.arange(global_start_index, global_start_index + n, dtype=np.int64)
    return {
        "episode_index": episode_index,
        "stats": {
            "observation.images.front": image_stats(data["clean"]["front"]),
            "observation.images.wrist": image_stats(data["clean"]["wrist"]),
            "perturbed.images.front": image_stats(data["perturbed"]["front"]),
            "perturbed.images.wrist": image_stats(data["perturbed"]["wrist"]),
            "observation.state": feature_stats(data["observation.state"]),
            "action": feature_stats(data["action"]),
            "timestamp": feature_stats(data["timestamp"].reshape(-1, 1)),
            "frame_index": feature_stats(data["frame_index"].reshape(-1, 1)),
            "episode_index": feature_stats(np.full((n, 1), episode_index, dtype=np.int64)),
            "index": feature_stats(index.reshape(-1, 1)),
            "task_index": feature_stats(np.full((n, 1), task_index, dtype=np.int64)),
        },
    }


def lerobot_info(
    num_episodes: int,
    num_frames: int,
    num_tasks: int,
    resolution: int,
    state_dim: int,
    action_dim: int,
    num_train_episodes: int,
    num_val_episodes: int,
) -> dict[str, Any]:
    video_info = {
        "video.height": resolution,
        "video.width": resolution,
        "video.codec": "mp4",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": FPS,
        "video.channels": 3,
        "has_audio": False,
    }
    video_feature = {
        "dtype": "video",
        "shape": [resolution, resolution, 3],
        "names": ["height", "width", "channel"],
        "info": video_info,
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": num_episodes,
        "total_frames": num_frames,
        "total_tasks": num_tasks,
        "total_videos": num_episodes * 4,
        "total_chunks": max(1, episode_chunk(max(0, num_episodes - 1)) + 1),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {
            "train": f"0:{num_train_episodes}",
            "val": f"{num_train_episodes}:{num_train_episodes + num_val_episodes}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.front": video_feature,
            "observation.images.wrist": video_feature,
            "perturbed.images.front": video_feature,
            "perturbed.images.wrist": video_feature,
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": [f"state_{i}" for i in range(state_dim)]},
            "action": {"dtype": "float32", "shape": [action_dim], "names": [f"action_{i}" for i in range(action_dim)]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def save_sample(
    *,
    out_root: pathlib.Path,
    split: str,
    task: TaskSpec,
    sample_id: str,
    camera: tuple[int, ...],
    policy_seed: int,
    env_seed: int,
    init_state_index: int,
    episode_index: int,
    global_start_index: int,
    data: dict[str, Any],
    model,
    cfg,
    save_videos: bool,
) -> dict[str, Any]:
    path = sample_path(out_root, split, task.name, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_task_name = _base_init_state_name(task.name)
    rgb_video_paths = save_lerobot_videos(out_root, episode_index, data) if save_videos else {}
    parquet_path = save_lerobot_parquet(out_root, episode_index, global_start_index, task, data)
    clean_latents = [
        vae_video_latent(model, front, wrist, cfg)
        for front, wrist in zip(data["clean"]["front"], data["clean"]["wrist"])
    ]
    pert_latents = [
        vae_video_latent(model, front, wrist, cfg)
        for front, wrist in zip(data["perturbed"]["front"], data["perturbed"]["wrist"])
    ]
    payload = {
        "meta": {
            "suite": "libero_10",
            "base_task": base_task_name,
            "clean_task_name": task.name,
            "task_index": task.task_id,
            "instruction": task.language,
            "condition": "camera_viewpoints",
            "split": split,
            "sample_id": sample_id,
            "episode_index": episode_index,
            "init_state_index": init_state_index,
            "policy_seed": policy_seed,
            "env_seed": env_seed,
            "clean_success": bool(data["success"]),
            "success_step": data["success_step"],
            "camera_tuple": list(camera),
            "rgb_video_paths": rgb_video_paths,
            "lerobot_parquet_path": parquet_path,
            "fps": FPS,
            "time_axis": "one frame per executed clean env step after 10 settle steps",
            "lerobot_alignment": {
                "observation.images.front": "videos/observation.images.front",
                "observation.images.wrist": "videos/observation.images.wrist",
                "perturbed.images.front": "videos/perturbed.images.front",
                "perturbed.images.wrist": "videos/perturbed.images.wrist",
                "observation.state": "observation.state",
                "action": "action",
                "timestamp": "timestamp",
                "frame_index": "frame_index",
            },
        },
        "latent": {
            "clean": {"vae_video": torch.stack(clean_latents)},
            "perturbed": {"vae_video": torch.stack(pert_latents)},
        },
        "observation.state": data["observation.state"],
        "action": data["action"],
        "action_context": data["action_context"],
        "sim_state": data["sim_state"],
        "timestamp": data["timestamp"],
        "frame_index": data["frame_index"],
    }
    torch.save(payload, path)
    return {
        "path": str(path.relative_to(out_root)),
        "suite": "libero_10",
        "base_task": base_task_name,
        "clean_task_name": task.name,
        "task_index": task.task_id,
        "instruction": task.language,
        "condition": "camera_viewpoints",
        "split": split,
        "sample_id": sample_id,
        "episode_index": episode_index,
        "init_state_index": init_state_index,
        "policy_seed": policy_seed,
        "env_seed": env_seed,
        "clean_success": bool(data["success"]),
        "num_frames": int(len(data["frame_index"])),
        "success_step": data["success_step"],
        "camera_tuple": list(camera),
        "rgb_video_paths": rgb_video_paths,
        "lerobot_parquet_path": parquet_path,
    }


def episode_output_paths(out_root: pathlib.Path, episode_index: int) -> list[pathlib.Path]:
    """Return all per-episode files created before the manifest is committed."""
    chunk = episode_chunk(episode_index)
    return [
        out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet",
        *[
            out_root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
            for key in (
                "observation.images.front",
                "observation.images.wrist",
                "perturbed.images.front",
                "perturbed.images.wrist",
            )
        ],
    ]


def cleanup_failed_sample(
    out_root: pathlib.Path, sample_file: pathlib.Path, episode_index: int
) -> None:
    """Remove partial files from a failed save so counts cannot silently diverge."""
    for path in [sample_file, *episode_output_paths(out_root, episode_index)]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def reindex_lerobot_frames(
    out_root: pathlib.Path,
    rows: list[dict[str, Any]],
    episode_stats_rows: list[dict[str, Any]],
) -> None:
    """Make global frame indices follow episode-index order after split sorting."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    stats_by_episode = {
        int(row["episode_index"]): row for row in episode_stats_rows
    }
    global_start = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        parquet_path = out_root / row["lerobot_parquet_path"]
        table = pq.read_table(parquet_path)
        n = table.num_rows
        index = np.arange(global_start, global_start + n, dtype=np.int64)
        column_index = table.schema.get_field_index("index")
        table = table.set_column(
            column_index, "index", pa.array(index, type=pa.int64())
        )
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
        stats_by_episode[episode_index]["stats"]["index"] = feature_stats(
            index.reshape(-1, 1)
        )
        global_start += n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(ROOT / "dataset" / "paired_camera_render_libero10"))
    p.add_argument("--policy-dir", default=str(phase2.POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default="")
    p.add_argument("--camera-classification", default=str(ROOT.parent / "LIBERO-plus/libero/libero/benchmark/task_classification.json"))
    p.add_argument("--camera-split-seed", type=int, default=0)
    p.add_argument("--camera-train-offset-deg", type=int, default=5)
    p.add_argument("--camera-val-fraction", type=float, default=0.1)
    p.add_argument("--policy-seeds", nargs="+", type=int, default=[1009, 2003, 3001, 4001])
    p.add_argument("--val-policy-seeds", nargs="+", type=int, default=[5003])
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--examples-per-task", type=int, default=50)
    p.add_argument("--train-per-task", type=int, default=45)
    p.add_argument("--max-attempts-per-task", type=int, default=500)
    p.add_argument("--task-limit", type=int, default=0)
    p.add_argument("--only-task", action="append", default=[])
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of task shards; defaults to torchrun WORLD_SIZE or 1.",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Zero-based task shard; defaults to torchrun RANK or 0.",
    )
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-rgb-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.examples_per_task <= 0:
        raise ValueError("--examples-per-task must be positive")
    if not 0 < args.train_per_task < args.examples_per_task:
        raise ValueError(
            "--train-per-task must be positive and smaller than --examples-per-task"
        )
    if args.skip_existing:
        raise ValueError(
            "--skip-existing is disabled because the old implementation produced "
            "inconsistent LeRobot metadata. Use a fresh --output-dir."
        )
    num_shards, shard_id, local_rank = resolve_sharding(args)
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(
        f"sharding: shard={shard_id}/{num_shards} "
        f"local_rank={local_rank} cuda_device="
        f"{torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    args.policy_dir = pathlib.Path(args.policy_dir)
    phase2.patch_checkpoint_db(args.policy_dir)
    cfg = make_cfg(args)
    set_seed_everywhere(args.env_seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    if args.t5_extra_embeddings:
        phase2.load_extra_t5(args.t5_extra_embeddings)
    stats = phase2.load_dataset_stats(cfg.dataset_stats_path)
    model, _ = phase2.get_model(cfg)
    all_selected_tasks = load_libero10_tasks(args)
    tasks = [task for task in all_selected_tasks if task.task_id % num_shards == shard_id]
    if not tasks:
        raise RuntimeError(
            f"shard {shard_id}/{num_shards} received no tasks; "
            f"selected task ids={[task.task_id for task in all_selected_tasks]}"
        )
    print(
        f"shard tasks: {[(task.task_id, task.name) for task in tasks]}",
        flush=True,
    )
    camera_splits = parse_camera_splits(args)
    out_root = shard_output_root(args.output_dir, num_shards, shard_id)
    existing_markers = [
        out_root / "camera_splits.json",
        out_root / "manifest.jsonl",
        out_root / "summary.json",
        out_root / "meta" / "info.json",
        out_root / "data",
        out_root / "videos",
        out_root / "samples",
    ]
    if any(path.exists() for path in existing_markers):
        raise FileExistsError(
            f"output directory already contains dataset files: {out_root}. "
            "Use a fresh --output-dir; safe resume is not implemented."
        )
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        out_root / "camera_splits.json",
        {
            task: {split: [list(camera) for camera in cameras] for split, cameras in splits.items()}
            for task, splits in camera_splits.items()
            if task in {spec.name for spec in tasks}
        },
    )

    rows, errors = [], []
    episode_rows, episode_stats_rows = [], []
    expected_train_episodes = len(tasks) * args.train_per_task
    split_episode_counts = {"train": 0, "val": 0}
    accepted_by_task: dict[str, int] = {}
    global_frame_count = 0
    state_dim = 0
    action_dim = 0
    start = time.time()
    for task in tasks:
        clean_env = make_env_from_bddl(
            task.problem_folder,
            task.bddl_file,
            args.env_resolution,
            env_seed=args.env_seed,
        )
        try:
            init_states = load_init_states_for_task(task)
            accepted = 0
            attempts = 0
            while accepted < args.examples_per_task and attempts < args.max_attempts_per_task:
                attempts += 1
                split = "train" if accepted < args.train_per_task else "val"
                seeds = args.policy_seeds if split == "train" else args.val_policy_seeds
                init_state_index = (attempts - 1) % len(init_states)
                split_local = accepted if split == "train" else accepted - args.train_per_task
                policy_seed = policy_seed_for_sample(
                    seeds, task.name, split, split_local
                )
                task_camera_splits = camera_splits.get(task.name)
                if task_camera_splits is None:
                    raise RuntimeError(f"no camera variants found for task {task.name}")
                split_cameras = task_camera_splits[split]
                per_task = args.train_per_task if split == "train" else args.examples_per_task - args.train_per_task
                # Use the stable pre-sharding task id so a sample selects the
                # same camera in single-GPU and multi-GPU runs.
                global_pos = task.task_id * per_task + split_local
                camera_index, camera = camera_for_sample(split_cameras, global_pos)
                view = "_".join(map(str, camera))
                base_task_name = _base_init_state_name(task.name)
                pert_name = f"{base_task_name}_view_{view}_initstate_0"
                sample_id = f"{split}_task{task.task_id:02d}_ex{accepted:03d}_init{init_state_index:04d}_cam{camera_index:03d}_pseed{policy_seed}_eseed{args.env_seed}"
                out_path = sample_path(out_root, split, task.name, sample_id)
                print(
                    f"[{task.name}] attempt={attempts} "
                    f"accepted={accepted}/{args.examples_per_task} "
                    f"split={split} cam={camera}",
                    flush=True,
                )
                # Training/validation cameras are synthesized at a 5-degree
                # spherical offset and therefore are intentionally absent from
                # LIBERO-Plus's registered, test-only task classification.
                pert_env = make_env_for_task_name(
                    pert_name,
                    "libero_10",
                    args.env_resolution,
                    env_seed=args.env_seed,
                )
                try:
                    data = rollout_once(
                        clean_env=clean_env,
                        pert_env=pert_env,
                        cfg=cfg,
                        model=model,
                        stats=stats,
                        task=task,
                        initial_state=init_states[init_state_index],
                        policy_seed=policy_seed,
                        args=args,
                    )
                    if not data["success"]:
                        print(f"  clean failure, retrying ({len(data['frame_index'])} frames)", flush=True)
                        continue
                    episode_index = (
                        split_episode_counts["train"]
                        if split == "train"
                        else expected_train_episodes + split_episode_counts["val"]
                    )
                    try:
                        row = save_sample(
                            out_root=out_root,
                            split=split,
                            task=task,
                            sample_id=sample_id,
                            camera=camera,
                            policy_seed=policy_seed,
                            env_seed=args.env_seed,
                            init_state_index=init_state_index,
                            episode_index=episode_index,
                            global_start_index=global_frame_count,
                            data=data,
                            model=model,
                            cfg=cfg,
                            save_videos=args.save_rgb_videos,
                        )
                    except Exception:
                        cleanup_failed_sample(out_root, out_path, episode_index)
                        raise
                    episode_rows.append({
                        "episode_index": episode_index,
                        "tasks": [task.language],
                        "length": int(len(data["frame_index"])),
                    })
                    episode_stats_rows.append(episode_stats_row(row, data, global_frame_count))
                    state_dim = int(data["observation.state"].shape[1])
                    action_dim = int(data["action"].shape[1])
                    global_frame_count += int(len(data["frame_index"]))
                    rows.append(row)
                    split_episode_counts[split] += 1
                    accepted += 1
                except Exception as exc:
                    err = {"task": task.name, "attempt": attempts, "accepted": accepted, "error": repr(exc)}
                    errors.append(err)
                    print(f"  error: {exc}", flush=True)
                    if args.fail_fast:
                        raise
                finally:
                    pert_env.close()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            clean_env.close()
        accepted_by_task[task.name] = accepted

    rows.sort(key=lambda row: int(row["episode_index"]))
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    episode_stats_rows.sort(key=lambda row: int(row["episode_index"]))
    reindex_lerobot_frames(out_root, rows, episode_stats_rows)
    append_jsonl(out_root / "manifest.jsonl", rows)
    append_jsonl(out_root / "meta" / "episodes.jsonl", episode_rows)
    append_jsonl(out_root / "meta" / "episodes_stats.jsonl", episode_stats_rows)
    task_rows = [{"task_index": task.task_id, "task": task.language} for task in tasks]
    append_jsonl(out_root / "meta" / "tasks.jsonl", task_rows)
    write_json(
        out_root / "meta" / "info.json",
        lerobot_info(
            len(rows),
            global_frame_count,
            len(task_rows),
            args.env_resolution,
            state_dim,
            action_dim,
            split_episode_counts["train"],
            split_episode_counts["val"],
        ),
    )
    write_json(out_root / "summary.json", {
        "dataset": "paired_camera_render_libero10",
        "suite": "libero_10",
        "conditions": ["camera_viewpoints"],
        "tasks": [task.name for task in tasks],
        "examples_per_task": args.examples_per_task,
        "train_per_task": args.train_per_task,
        "val_per_task": args.examples_per_task - args.train_per_task,
        "num_samples": len(rows),
        "num_train_samples": split_episode_counts["train"],
        "num_val_samples": split_episode_counts["val"],
        "accepted_by_task": accepted_by_task,
        "num_frames": global_frame_count,
        "num_errors": len(errors),
        "errors": errors,
        "policy_seeds": args.policy_seeds,
        "val_policy_seeds": args.val_policy_seeds,
        "policy_seed_strategy": "sha256(base_seed, task, split, split_local); unique per sample",
        "action_query_seed_strategy": "policy_seed + query_index * 1000003 (mod 2^32)",
        "env_seed": args.env_seed,
        "camera_split_seed": args.camera_split_seed,
        "camera_train_offset_deg": args.camera_train_offset_deg,
        "camera_val_fraction": args.camera_val_fraction,
        "official_libero_plus_cameras_reserved_for_external_test": True,
        "num_shards": num_shards,
        "shard_id": shard_id,
        "task_sharding_rule": "task_id % num_shards == shard_id",
        "fps": FPS,
        "max_steps": TASK_MAX_STEPS[TaskSuite.LIBERO_10],
        "num_settle_steps": NUM_STEPS_WAIT,
        "elapsed_s": time.time() - start,
    })
    undersampled = {
        task: count
        for task, count in accepted_by_task.items()
        if count != args.examples_per_task
    }
    if undersampled:
        raise RuntimeError(
            "dataset collection was undersampled; "
            f"expected {args.examples_per_task} episodes per task, got {undersampled}"
        )


if __name__ == "__main__":
    main()
