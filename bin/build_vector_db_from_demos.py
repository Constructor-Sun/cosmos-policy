#!/usr/bin/env python3
"""Build vector_db chunks from LIBERO-10 success-only HDF5 demonstrations.
Usage:
  conda activate cosmospolicy
  cd /data1/liu/exp/counterfactual/external/cosmos-policy
  python bin/build_vector_db_from_demos.py
"""
import argparse, io, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import h5py, numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
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

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen")
    p.add_argument("--output-dir", type=Path, default=ROOT / "vector_db_demos")
    p.add_argument("--max-per-task", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    for k in ("input_dir", "output_dir"): setattr(args, k, getattr(args, k).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        selected = demos[:args.max_per_task]
        print(f"\n{task_name}: {len(selected)}/{len(demos)} demos", flush=True)
        for dk in selected:
            with h5py.File(h5_path, "r") as f:
                g = f["data"][dk]
                acts, rstate = g["actions"][:], g["robot_states"][:]
                ajpg, wjpg = g["obs"]["agentview_rgb_jpeg"][:], g["obs"]["eye_in_hand_rgb_jpeg"][:]
            T, chunks = acts.shape[0], []
            for t in range(0, T, 16):
                primary = np.array(Image.open(io.BytesIO(ajpg[t])))
                wrist = np.array(Image.open(io.BytesIO(wjpg[t])))
                imgs = prepare_images_for_model([wrist, primary], cfg, flip_images=cfg.flip_images)
                proprio = rescale_proprio(rstate[t].copy(), stats, non_negative_only=False, scale_multiplier=1.0)
                batch = build_batch(imgs[1], imgs[0], proprio, t5_emb, device)
                with torch.no_grad():
                    _, lat, _ = model.get_data_and_condition(batch)
                # Pad last partial chunk by repeating the final action
                end = min(t + 16, T)
                act = acts[t:end].copy()
                if end - t < 16:
                    act = np.concatenate([act, np.tile(acts[-1], (16 - (end - t), 1))], axis=0)
                act_normalized = 2.0 * (
                    act.astype(np.float32) - np.asarray(stats["actions_min"], dtype=np.float32)
                ) / (
                    np.asarray(stats["actions_max"], dtype=np.float32)
                    - np.asarray(stats["actions_min"], dtype=np.float32)
                ) - 1.0
                chunks.append({
                    "vae_video": lat[0, :, [2, 3], :, :].half().cpu(),
                    "proprio": torch.from_numpy(rstate[t].copy()).half(),
                    # Keep the legacy field raw for replay compatibility.
                    "action_chunk": torch.from_numpy(act).half(),
                    "action_chunk_raw": torch.from_numpy(act).half(),
                    "action_chunk_normalized": torch.from_numpy(act_normalized).half(),
                    "step_index": t,
                    "init_state_index": int(dk.split("_")[1]),
                })
            if chunks:
                ts = int(time.time() * 1_000_000)
                out = args.output_dir / f"chunks_{task_name}_{dk}_{ts}.pt"
                torch.save(chunks, out)
                total += len(chunks)
                print(f"  {dk}: {len(chunks)} chunks → {out.name}", flush=True)
    print(f"\nDone. {total} chunks in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
