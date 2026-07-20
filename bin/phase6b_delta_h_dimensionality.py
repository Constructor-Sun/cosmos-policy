#!/usr/bin/env python3
"""Quick: effective dimensionality of delta_h_VAE for each perturbation type."""
import sys, os, pathlib, json, pickle, time
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DETERMINISTIC", "True")
# Must be set before importing LIBERO / robosuite / wand / matplotlib.
os.environ["MAGICK_HOME"] = sys.prefix
os.environ["WAND_MAGICK_LIBRARY_SUFFIX"] = "-7.Q16HDRI"
for cache_env, default_path in (
    ("NUMBA_CACHE_DIR", "/tmp/numba_cosmospolicy_cache"),
    ("MPLCONFIGDIR", "/tmp/mpl_cosmospolicy_cache"),
):
    os.environ.setdefault(cache_env, default_path)
    pathlib.Path(os.environ[cache_env]).mkdir(parents=True, exist_ok=True)

from wand.api import library as _wand_library  # noqa: F401

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    prepare_images_for_model, duplicate_array,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CKPT_ROOT = pathlib.Path("/data3/liu/exp/counterfactual/checkpoints")
POLICY_DIR = CKPT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
BASE_MODEL_DIR = CKPT_ROOT / "Cosmos-Predict2-2B-Video2World"
T = 4  # temporal compression
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
EPS = 1e-8

# Patch ckpt
from cosmos_policy._src.imaginaire.utils import checkpoint_db
orig = checkpoint_db.get_checkpoint_path
def patched(uri):
    uri = str(uri).rstrip("/")
    if "Cosmos-Predict2-2B-Video2World" in uri:
        return str(BASE_MODEL_DIR / uri.split("Cosmos-Predict2-2B-Video2World/")[-1])
    if "ALOHA" in uri or "LIBERO" in uri:
        return str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    return orig(uri)
checkpoint_db.get_checkpoint_path = patched

# Load model
cosmos_utils.init_t5_text_embeddings_cache(str(POLICY_DIR / "libero_t5_embeddings.pkl"))
cfg = SimpleNamespace(
    suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path=str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
    config_file="cosmos_policy/config/config.py",
    use_third_person_image=True, num_third_person_images=1,
    use_wrist_image=True, num_wrist_images=1,
    use_proprio=True, flip_images=False,
    use_variance_scale=False, use_jpeg_compression=True,
    num_denoising_steps_action=5,
    unnormalize_actions=True, normalize_proprio=True,
    dataset_stats_path=str(POLICY_DIR / "libero_dataset_statistics.json"),
    t5_text_embeddings_path=str(POLICY_DIR / "libero_t5_embeddings.pkl"),
    trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
)
set_seed_everywhere(7)
print("Loading model...")
model, _ = cosmos_utils.get_model(cfg)
model.eval()
print(f"Loaded: {len(model.net.blocks)} blocks")


def vae_encode(primary, wrist):
    """VAE-encode video frames → latent [16, 2, 28, 28] for indices 2,3."""
    device = next(model.parameters()).device
    images = [primary, wrist]
    processed = prepare_images_for_model(images, SimpleNamespace(
        use_jpeg_compression=True, use_variance_scale=False,
        flip_images=False, trained_with_image_aug=True,
        randomize_seed=False, num_third_person_images=1))
    wrist_img, primary_img = processed[0], processed[1]
    blank = np.zeros_like(primary_img)
    wrist_dup = duplicate_array(wrist_img, T)
    primary_dup = duplicate_array(primary_img, T)
    blank_dup = duplicate_array(blank, T)

    seq = [
        np.expand_dims(np.zeros_like(blank), axis=0),
        blank_dup, wrist_dup, primary_dup,
        blank_dup, blank_dup,
        wrist_dup.copy(), primary_dup.copy(), blank_dup,
    ]
    raw = np.concatenate(seq, axis=0)
    raw = np.expand_dims(raw, 0)
    raw = np.tile(raw, (1, 1, 1, 1, 1))
    raw = np.transpose(raw, (0, 4, 1, 2, 3))
    raw = torch.from_numpy(raw).to(dtype=torch.uint8, device=device)
    with torch.no_grad():
        latent = model.encode(raw).contiguous().float()
    return latent[0, :, [2, 3], :, :].cpu().numpy()


def get_obs(ep, flip=True):
    task = ep["task_name"]
    base_task = ep.get("base_task", task)
    suite = ep.get("suite", "libero_10")
    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        set_seed_everywhere(int(ep.get("deterministic_reset_seed", 0)))
        env.reset()
        init_file = f"{base_task}.pruned_init"
        init_path = pathlib.Path(get_libero_path("init_states")) / suite / init_file
        is_newobj = "_add_" in init_file or "_level" in init_file
        if is_newobj and not init_path.exists():
            init_path = pathlib.Path(get_libero_path("init_states")) / "libero_newobj" / suite / init_file
        states = torch.load(str(init_path), weights_only=False)
        if is_newobj:
            states = states.reshape(1, -1)
        idx = int(ep.get("init_state_index", ep["episode"])) % len(states)
        obs = env.set_init_state(states[idx])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        primary = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        if flip:
            primary = np.flipud(primary); wrist = np.flipud(wrist)
        return primary, wrist
    finally:
        env.close()


def effective_rank(deltas, threshold=0.90):
    """PCA on delta vectors, return minimal k to explain `threshold` variance."""
    N = len(deltas)
    D = deltas[0].size
    if N < 3:
        return None, None, None
    X = np.array([d.reshape(-1) for d in deltas])
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    total = var.sum()
    cum = np.cumsum(var) / total
    r80 = int(np.searchsorted(cum, 0.80) + 1) if total > 0 else 0
    r90 = int(np.searchsorted(cum, 0.90) + 1) if total > 0 else 0
    r95 = int(np.searchsorted(cum, 0.95) + 1) if total > 0 else 0
    return r80, r90, r95, S, cum[:min(20, len(S))]


# Load Phase 1 pairs
summary_path = ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json"
summary = json.loads(summary_path.read_text())

pairs = []
for info in summary["conditions"]:
    condition = info["condition"]
    if condition == "clean":
        continue
    source_root = pathlib.Path(info.get("source_root", str(ROOT)))
    clean_path = source_root / "clean/episodes.json"
    pert_path = ROOT / info["episodes_path"]

    clean_list = json.loads(clean_path.read_text())
    pert_list = json.loads(pert_path.read_text())
    clean_eps = {item["episode"]: item for item in clean_list}
    pert_eps = {item["episode"]: item for item in pert_list}

    for ep_key, clean in clean_eps.items():
        pert = pert_eps.get(ep_key)
        if pert is None:
            continue
        cs, ps = clean["success"], pert["success"]
        if cs and ps:
            group = "preserved"
        elif cs and not ps:
            group = "flipped"
        else:
            continue
        pairs.append({"condition": condition, "group": group, "clean": clean, "pert": pert})

print(f"Found {len(pairs)} pairs")

# Collect delta_h per condition
from collections import defaultdict
deltas_by_cond = defaultdict(lambda: {"flipped": [], "preserved": []})
errors = 0
start = time.time()

for idx, pair in enumerate(pairs):
    cond = pair["condition"]
    grp = pair["group"]
    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{len(pairs)}] {cond} {grp}")
    try:
        cp, cw = get_obs(pair["clean"], flip=True)
        if cond == "language_instructions":
            pp, pw = cp, cw
        else:
            pp, pw = get_obs(pair["pert"], flip=True)
        lc = vae_encode(cp, cw)
        lp = vae_encode(pp, pw)
        deltas_by_cond[cond][grp].append(lp - lc)
    except Exception as e:
        errors += 1
        continue

print(f"\nErrors: {errors}, Elapsed: {time.time()-start:.0f}s")
print(f"\n{'='*65}")
print(f"  Effective dimensionality of delta_h_VAE per perturbation")
print(f"{'='*65}")
print(f"  {'Condition':30s} {'Group':12s} {'N':>4s} {'r80':>5s} {'r90':>5s} {'r95':>5s} {'top5_cum':>10s}")
print(f"  {'-'*65}")

for cond in sorted(deltas_by_cond.keys()):
    for grp in ["flipped", "preserved"]:
        deltas = deltas_by_cond[cond][grp]
        if len(deltas) < 2:
            continue
        result = effective_rank(deltas)
        if result[0] is None:
            continue
        r80, r90, r95, S, cum = result
        top5_cum = cum[min(4, len(cum)-1)] if len(cum) >= 5 else cum[-1]
        print(f"  {cond:30s} {grp:12s} {len(deltas):>4d} {r80:>5d} {r90:>5d} {r95:>5d} {top5_cum:>10.3f}")

print(f"\n  D_total = 25088 (VAE latent × 2 video frames)")
print(f"  Random expected: r80 at 80% of N when N < D")
