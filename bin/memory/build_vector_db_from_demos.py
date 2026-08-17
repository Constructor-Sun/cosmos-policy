#!/usr/bin/env python3
"""Build vector_db chunks from LIBERO-10 success-only HDF5 demonstrations.
Usage:
  conda activate cosmospolicy
  cd /data1/liu/exp/counterfactual/external/cosmos-policy
  python bin/memory/build_vector_db_from_demos.py
"""
import argparse, io, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import h5py, numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path(os.environ.get("CHECKPOINT_ROOT", ROOT.parent.parent / "checkpoints"))
LP = Path(os.environ.get("LIBERO_PLUS_PATH", ROOT.parent / "LIBERO-plus"))
for p in (str(ROOT), str(LP)):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Patch checkpoint_db: redirect HF URIs to local files
V2W = CKPT / "Cosmos-Predict2-2B-Video2World"
PDIR = CKPT / "Cosmos-Policy-LIBERO-Predict2-2B"
PREFIX = "hf://nvidia/Cosmos-Predict2-2B-Video2World/"
ALOHA = "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
from cosmos_policy._src.imaginaire.utils import checkpoint_db
_orig = checkpoint_db.get_checkpoint_path
def _patch(u):
    u = str(u).rstrip("/")
    if u.startswith(PREFIX): return str(V2W / u[len(PREFIX):])
    if u == ALOHA.rstrip("/"): return str(PDIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    return _orig(u)
checkpoint_db.get_checkpoint_path = _patch

from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_TEMPORAL_COMPRESSION_FACTOR as _TC, get_model,
    get_t5_embedding_from_cache, init_t5_text_embeddings_cache, load_dataset_stats,
    prepare_images_for_model, rescale_proprio,
)
from cosmos_policy.utils.utils import duplicate_array
from libero.libero import benchmark, get_libero_path
def build_batch(primary, wrist, proprio, t5_emb, device):
    blank = np.zeros_like(primary)
    bd = duplicate_array(blank.copy(), total_num_copies=_TC)
    wd, pd = duplicate_array(wrist.copy(), total_num_copies=_TC), duplicate_array(primary.copy(), total_num_copies=_TC)
    seq = [np.expand_dims(np.zeros_like(blank), 0), bd.copy(), wd.copy(), pd.copy(),
           bd.copy(), bd.copy(), wd.copy(), pd.copy(), bd.copy()]
    video = np.transpose(np.tile(np.expand_dims(np.concatenate(seq, 0), 0), (1, 1, 1, 1, 1)), (0, 4, 1, 2, 3))
    v = torch.from_numpy(video).to(device=device, dtype=torch.uint8)
    I = lambda n: torch.tensor([n], dtype=torch.int64, device=device)
    return {"dataset_name": "video_data", "video": v,
            "t5_text_embeddings": t5_emb.to(device=device, dtype=torch.bfloat16),
            "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
            "padding_mask": torch.zeros((1, 1, 224, 224), dtype=torch.bfloat16, device=device),
            "num_conditional_frames": 1,
            "proprio": torch.from_numpy(proprio).reshape(1, -1).to(device=device, dtype=torch.bfloat16),
            "current_proprio_latent_idx": I(1), "current_wrist_image_latent_idx": I(2),
            "current_wrist_image2_latent_idx": I(-1), "current_image_latent_idx": I(3),
            "current_image2_latent_idx": I(-1), "action_latent_idx": I(4),
            "future_proprio_latent_idx": I(5), "future_wrist_image_latent_idx": I(6),
            "future_wrist_image2_latent_idx": I(-1), "future_image_latent_idx": I(7),
            "future_image2_latent_idx": I(-1), "value_latent_idx": I(8)}

def load_lang_map():
    config = ROOT / "configs/libero10_experiment_tasks.json"
    if config.is_file():
        data = json.loads(config.read_text())
        return {task["name"]: task["language"] for task in data["tasks"]}
    bddl = Path(get_libero_path("bddl_files")) / "libero_10"
    tasks = {}
    for line in (bddl / "tasks_info.txt").read_text().splitlines():
        if not (line := line.strip()): continue
        name = Path(line).stem
        tasks[name] = benchmark.grab_language_from_filename("libero_10", f"{name}.bddl")
    return tasks


def normalize_actions(actions, stats):
    """Map raw actions to the normalized action space used during training."""
    actions = np.asarray(actions, dtype=np.float32)
    actions_min = np.asarray(stats["actions_min"], dtype=np.float32)
    actions_max = np.asarray(stats["actions_max"], dtype=np.float32)
    scale = actions_max - actions_min
    if np.any(scale == 0):
        raise ValueError("Action statistics contain a zero-width dimension")
    return 2.0 * (actions - actions_min) / scale - 1.0

def load_segment_manifest(path):
    data = json.loads(path.read_text())
    return {
        (record["task_name"], record["demo_id"]): record
        for record in data["records"]
    }

def encode_vae(frame, ajpg, wjpg, rstate, cfg, stats, t5_emb, model, device):
    primary = np.array(Image.open(io.BytesIO(ajpg[frame])))
    wrist = np.array(Image.open(io.BytesIO(wjpg[frame])))
    imgs = prepare_images_for_model([wrist, primary], cfg, flip_images=cfg.flip_images)
    proprio = rescale_proprio(
        rstate[frame].copy(), stats, non_negative_only=False, scale_multiplier=1.0
    )
    batch = build_batch(imgs[1], imgs[0], proprio, t5_emb, device)
    with torch.no_grad():
        _, lat, _ = model.get_data_and_condition(batch)
    return lat[0, :, [2, 3], :, :].half().cpu()

def make_chunk(t, segment_end, acts, rstate, vae_at, stats, demo_index):
    end = min(t + 16, segment_end)
    valid_length = end - t
    act = acts[t:end].copy()
    if valid_length < 16:
        act = np.concatenate(
            [act, np.tile(act[-1], (16 - valid_length, 1))], axis=0
        )
    raw = torch.from_numpy(act).half()
    return {
        "vae_video": vae_at(t),
        "proprio": torch.from_numpy(rstate[t].copy()).half(),
        "action_chunk": raw,
        "action_chunk_raw": raw,
        "action_chunk_normalized": torch.from_numpy(
            normalize_actions(act, stats)
        ).half(),
        "step_index": t,
        "valid_length": valid_length,
        "init_state_index": demo_index,
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--segments-manifest", type=Path,
                   help="Simulator-labeled JSON from label_libero_skill_segments.py")
    p.add_argument("--max-per-task", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    if args.segments_manifest:
        args.segments_manifest = args.segments_manifest.expanduser().resolve()
    if args.output_dir is None:
        name = "skill_memory_demos" if args.segments_manifest else "vector_db_demos"
        args.output_dir = ROOT / name
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_map = (
        load_segment_manifest(args.segments_manifest) if args.segments_manifest else None
    )

    cfg = SimpleNamespace(suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(PDIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, use_wrist_image=True, use_proprio=True,
        flip_images=args.flip_images, use_variance_scale=False, use_jpeg_compression=True,
        normalize_proprio=True, dataset_stats_path=str(PDIR / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(PDIR / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True, chunk_size=16, randomize_seed=False)
    device = torch.device(args.device)
    print(f"Loading model on {device} …", flush=True)
    model, _ = get_model(cfg)
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    lang_map = load_lang_map()
    print("Model ready.", flush=True)

    total, h5s = 0, sorted(args.input_dir.glob("*_demo.hdf5"))
    if not h5s: raise FileNotFoundError(f"No *_demo.hdf5 in {args.input_dir}")
    for h5_path in h5s:
        task_name = h5_path.stem.replace("_demo", "")
        t5_emb = get_t5_embedding_from_cache(lang_map.get(task_name, task_name.replace("_", " ")))
        with h5py.File(h5_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        selected = demos[:args.max_per_task] if args.max_per_task > 0 else demos
        print(f"\n{task_name}: {len(selected)}/{len(demos)} demos", flush=True)
        for dk in selected:
            with h5py.File(h5_path, "r") as f:
                g = f["data"][dk]
                acts, rstate = g["actions"][:], g["robot_states"][:]
                ajpg, wjpg = g["obs"]["agentview_rgb_jpeg"][:], g["obs"]["eye_in_hand_rgb_jpeg"][:]
            T, demo_index = acts.shape[0], int(dk.split("_")[1])
            cache = {}
            def vae_at(frame):
                frame = min(max(int(frame), 0), T - 1)
                if frame not in cache:
                    cache[frame] = encode_vae(
                        frame, ajpg, wjpg, rstate, cfg, stats, t5_emb, model, device
                    )
                return cache[frame]

            if segment_map is None:
                chunks = [
                    make_chunk(t, T, acts, rstate, vae_at, stats, demo_index)
                    for t in range(0, T, 16)
                ]
                payload, prefix = chunks, "chunks"
                count = len(chunks)
            else:
                record = segment_map.get((task_name, dk))
                if record is None or not record.get("valid"):
                    print(f"  {dk}: skipped (no valid segment labels)", flush=True)
                    continue
                memories = []
                for segment in record["segments"]:
                    start = min(max(int(segment["start"]), 0), T - 1)
                    end = min(max(int(segment["end"]), start), T)
                    terminal_success = int(segment["success_start"]) >= T
                    success_frames = range(
                        min(int(segment["success_start"]), T - 1),
                        min(max(int(segment["success_end"]), 1), T),
                    )
                    success_latents = [vae_at(frame).float() for frame in success_frames]
                    if not success_latents:
                        success_latents = [vae_at(start).float()]
                    chunks = [
                        make_chunk(t, end, acts, rstate, vae_at, stats, demo_index)
                        for t in range(start, end, 16)
                    ]
                    memory = {
                        **segment,
                        "success_state_source": (
                            "replayed_terminal" if terminal_success else "recorded"
                        ),
                        "success_embedding_source": (
                            "last_recorded_observation"
                            if terminal_success else "recorded_success_window"
                        ),
                        "skill_start_vae": vae_at(start),
                        "success_vae": torch.stack(success_latents).mean(0).half(),
                        "chunks": chunks,
                    }
                    terminal_start = segment.get("terminal_start")
                    ready_frame = segment.get("ready_frame")
                    if terminal_start is not None and ready_frame is not None:
                        terminal_start = min(max(int(terminal_start), start), end)
                        ready_frame = min(max(int(ready_frame), start), T - 1)
                        memory["ready_vae"] = vae_at(ready_frame)
                        memory["terminal_chunks"] = [
                            make_chunk(
                                t, end, acts, rstate, vae_at, stats, demo_index
                            )
                            for t in range(terminal_start, end, 16)
                        ]
                    else:
                        memory["ready_vae"] = None
                        memory["terminal_chunks"] = []
                    memories.append(memory)
                has_ready_boundaries = any(
                    "terminal_start" in segment for segment in record["segments"]
                )
                payload = {
                    "format": (
                        "libero_skill_memory_v2"
                        if has_ready_boundaries else "libero_skill_memory_v1"
                    ),
                    "task_name": task_name,
                    "demo_id": dk,
                    "terminal_state_replayed": record.get(
                        "terminal_state_replayed", False
                    ),
                    "segments": memories,
                }
                prefix = "skill_memory"
                count = sum(len(memory["chunks"]) for memory in memories)
            if count or segment_map is not None:
                ts = int(time.time() * 1_000_000)
                out = args.output_dir / f"{prefix}_{task_name}_{dk}_{ts}.pt"
                torch.save(payload, out)
                total += count
                print(f"  {dk}: {count} chunks → {out.name}", flush=True)
    print(f"\nDone. {total} chunks in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
