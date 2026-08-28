"""Standalone update of memory ready_frame using distance thresholds.

This script only rewrites memory artifact files:
  - skill_memory_test/libero_10/feasible_recovery_targets.pt
  - skill_memory_test/libero_10/ready3d_targets.pt

It does NOT modify build_recovery.py / build_targets.py / label_boundaries.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
for p in (str(ROOT), str(LIBERO_PLUS)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from memory_system.offline.build_ready3d import (  # noqa: E402
    create_env,
    instance_mask,
    resolve_instance,
    target_xyz_from_obs,
)
from memory_system.offline.build_targets import patch_numpy2_segmentation  # noqa: E402
from memory_system.offline.build_recovery import (  # noqa: E402
    compute_action_scale,
    encode_vae,
    load_lang_map,
    raw_action_chunk,
)
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_model,
    get_t5_embedding_from_cache,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

DEFAULT_MEM_DIR = ROOT / "skill_memory_test/libero_10"
DEFAULT_MANIFEST = DEFAULT_MEM_DIR / "segments_ready_fixed16.json"
DEFAULT_READY3D = DEFAULT_MEM_DIR / "ready3d_targets.pt"
DEFAULT_FEASIBLE = DEFAULT_MEM_DIR / "feasible_recovery_targets.pt"
DEFAULT_DEMO_DIR = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
DEFAULT_CKPT = ROOT.parent.parent / "checkpoints"

# Skill -> desired distance in meters.
SKILL_DISTANCE = {
    "Pick": 0.07,
    "PlaceIn": 0.14,
    "PlaceOn": 0.14,
}
# Acceptable tolerance around the desired distance.
TOLERANCE = 0.03
# Search window before the segment end.
WINDOW = 32
# Fallback frame.
FALLBACK_OFFSET = 16


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def get_segment_map(manifest: dict):
    return {
        (record["task_name"], record["demo_id"]): record
        for record in manifest.get("records", [])
        if record.get("valid")
    }


def load_model_and_stats():
    ckpt = Path(os.environ.get("CHECKPOINT_ROOT", DEFAULT_CKPT))
    pdir = ckpt / "Cosmos-Policy-LIBERO-Predict2-2B"
    cfg = SimpleNamespace(
        suite="libero",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(pdir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True,
        use_wrist_image=True,
        use_proprio=True,
        flip_images=True,
        use_variance_scale=False,
        use_jpeg_compression=True,
        normalize_proprio=True,
        dataset_stats_path=str(pdir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(pdir / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True,
        chunk_size=16,
        randomize_seed=False,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = get_model(cfg)
    model = model.to(dev).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    lang_map = load_lang_map()
    return cfg, model, stats, lang_map, dev


def compute_place_final_target(env, obs, segment, states, frame):
    """Return the 3D position of the placed object at a success frame."""
    # Place skills: role is 'item'; the placed object is the manipulated object.
    arguments = dict(segment.get("arguments", {}))
    # For Place, the manipulated/placed object is the 'item' argument.
    instance = resolve_instance(env, arguments, "Pick")
    if instance is None:
        return None
    # Reuse target_xyz_from_obs, but force instance to the item object.
    mask = instance_mask(env, obs, instance)
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None
    from memory_system.geometry import camera_params, depth_to_metric, flip_depth, pixel_to_world
    cam = camera_params(env.env.sim, "agentview", 256, 256)
    metric = depth_to_metric(obs["agentview_depth"], cam.near, cam.far)
    depth = flip_depth(metric)
    pts = pixel_to_world(np.stack([ys, xs], axis=-1), depth, cam)
    valid = np.isfinite(pts).all(axis=1)
    if int(valid.sum()) < 4:
        return None
    return np.median(pts[valid], axis=0)


def select_ready_frame(segment, ee_states, target_xyz, desired):
    start = min(max(int(segment["start"]), 0), len(ee_states) - 1)
    end = min(max(int(segment["end"]), start + 1), len(ee_states))
    if end <= start:
        return None
    search_start = max(start, end - WINDOW)
    candidates = list(range(search_start, end))
    best_frame = None
    best_score = float("inf")
    for frame in candidates:
        ee = np.asarray(ee_states[frame], dtype=np.float64).reshape(-1)[:3]
        dist = float(np.linalg.norm(ee - np.asarray(target_xyz, dtype=np.float64).reshape(3)))
        score = abs(dist - desired)
        if score < best_score:
            best_score = score
            best_frame = frame
    if best_frame is None:
        return None
    best_dist = float(np.linalg.norm(
        np.asarray(ee_states[best_frame], dtype=np.float64).reshape(-1)[:3]
        - np.asarray(target_xyz, dtype=np.float64).reshape(3)
    ))
    if abs(best_dist - desired) <= TOLERANCE:
        return best_frame
    # Fallback: 16 frames before completion.
    return max(start, end - FALLBACK_OFFSET)


def update_files(
    manifest_path: Path,
    ready3d_path: Path,
    feasible_path: Path,
    demo_dir: Path,
    update_vae: bool,
):
    manifest = load_manifest(manifest_path)
    segment_map = get_segment_map(manifest)
    ready3d = torch.load(ready3d_path, map_location="cpu", weights_only=False)
    feasible = torch.load(feasible_path, map_location="cpu", weights_only=False)

    cfg = model = stats = lang_map = dev = None
    if update_vae:
        cfg, model, stats, lang_map, dev = load_model_and_stats()

    patch_numpy2_segmentation()
    prototypes = ready3d["prototypes"]
    targets = feasible["targets"]
    print(f"loaded {len(prototypes)} ready3d prototypes, {len(targets)} feasible targets", flush=True)

    # Index targets by (task, demo, step) to update together.
    target_index = {}
    for t in targets:
        key = (t["task_name"], t["demo_id"], int(t["planner_step_id"]))
        target_index[key] = t

    # Cache env per task for Place final-target computation.
    env_cache = {}

    # Process unique records.
    for record in manifest.get("records", []):
        if not record.get("valid"):
            continue
        task_name = record["task_name"]
        demo_id = record["demo_id"]
        h5_path = demo_dir / f"{task_name}_demo.hdf5"
        if not h5_path.is_file():
            print(f"missing h5 {h5_path}", flush=True)
            continue
        with h5py.File(h5_path, "r") as handle:
            group = handle["data"][demo_id]
            acts = group["actions"][:]
            rstate = group["robot_states"][:]
            states = group["states"][:]
            ee_states = group["obs"]["ee_states"][:]
            ajpg = group["obs"]["agentview_rgb_jpeg"][:]
            wjpg = group["obs"]["eye_in_hand_rgb_jpeg"][:]

        for segment in record.get("segments", []):
            skill = str(segment["skill"])
            if skill not in SKILL_DISTANCE:
                continue
            key = (task_name, demo_id, int(segment["planner_step_id"]))
            proto = next((p for p in prototypes if p["task_name"] == task_name and p["demo_id"] == demo_id and int(p["planner_step_id"]) == int(segment["planner_step_id"])), None)
            target = target_index.get(key)
            if proto is None or target is None:
                continue

            desired = SKILL_DISTANCE[skill]
            # Determine target_xyz.
            if skill == "Pick":
                target_xyz = np.asarray(proto["target_xyz_world"], dtype=np.float64).reshape(3)
            else:
                # Place: use the placed object's 3D position at success frame.
                success_start = int(segment.get("success_start", min(int(segment["end"]), len(ee_states) - 1)))
                success_frame = min(max(success_start, 0), len(ee_states) - 1)
                env = env_cache.get(task_name)
                if env is None:
                    env = create_env(task_name, 256)
                    env.reset()
                    env_cache[task_name] = env
                obs = env.regenerate_obs_from_state(states[success_frame])
                target_xyz = compute_place_final_target(env, obs, segment, rstate, success_frame)
                if target_xyz is None:
                    print(f"place target unavailable {key}", flush=True)
                    continue

            new_frame = select_ready_frame(segment, ee_states, target_xyz, desired)
            if new_frame is None:
                continue
            old_frame = int(target["ready_frame"])
            if new_frame == old_frame:
                continue

            print(f"update {task_name} {demo_id} step={segment['planner_step_id']} {skill} frame {old_frame}->{new_frame}", flush=True)

            # Update feasible recovery target fields.
            target["ready_frame"] = int(new_frame)
            target["ee_states"] = np.asarray(ee_states[new_frame], dtype=np.float32)
            target["action_scale"] = compute_action_scale(
                acts, rstate, max(int(segment["start"]), new_frame - 16), new_frame
            )
            target["action_chunk_raw"] = raw_action_chunk(
                acts, new_frame - 16, new_frame
            )
            if update_vae:
                t5_emb = get_t5_embedding_from_cache(
                    lang_map.get(task_name, task_name.replace("_", " "))
                )
                target["ready_vae_main"] = encode_vae(
                    new_frame, ajpg, wjpg, rstate, cfg, stats,
                    t5_emb, model, dev,
                )[:, 1:2, :, :].half()
                target["ready_vae_wrist"] = encode_vae(
                    new_frame, ajpg, wjpg, rstate, cfg, stats,
                    t5_emb, model, dev,
                )[:, 0:1, :, :].half()

            # Update ready3d prototype.
            proto["ready_frame"] = int(new_frame)
            proto["target_xyz_world"] = np.asarray(target_xyz, dtype=np.float64)
            proto["eef_pos_ready"] = np.asarray(ee_states[new_frame], dtype=np.float64).reshape(-1)[:3]
            proto["distance_m"] = float(np.linalg.norm(
                np.asarray(ee_states[new_frame], dtype=np.float64).reshape(-1)[:3]
                - np.asarray(target_xyz, dtype=np.float64).reshape(3)
            ))

    for env in env_cache.values():
        env.close()

    ready3d_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ready3d, ready3d_path)
    torch.save(feasible, feasible_path)
    print(f"Wrote {ready3d_path}", flush=True)
    print(f"Wrote {feasible_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ready3d", type=Path, default=DEFAULT_READY3D)
    parser.add_argument("--feasible", type=Path, default=DEFAULT_FEASIBLE)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--no-vae", action="store_true", help="skip VAE re-encoding (not recommended)")
    args = parser.parse_args()
    update_files(args.manifest, args.ready3d, args.feasible, args.demo_dir, update_vae=not args.no_vae)


if __name__ == "__main__":
    main()
