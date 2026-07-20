#!/usr/bin/env python3
"""
Phase 6: Project perturbation delta_h (at VAE output) onto the Jacobian-defined
action-sensitive subspace S_action.

Pipeline:
  1. Load pre-computed S_action (right singular vectors of J = ∂a/∂h at VAE output)
  2. For each perturbation pair from Phase 1:
     a. Get clean and perturb observations via LIBERO env
     b. VAE-encode the video frames → latent_clean, latent_pert
     c. Compute delta_h_VAE = latent_pert - latent_clean (video slots only)
     d. Project onto S_action → containment ratio
  3. Report per-perturbation × group containment

Requires: GPU, LIBERO, cosmospolicy conda env, pre-computed Jacobian SVD.
"""

from __future__ import annotations

import argparse, json, math, os, pathlib, pickle, sys, time
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DETERMINISTIC", "True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", str(ROOT.parent / "LIBERO-plus")))
if str(LIBERO_PLUS) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS))

import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    prepare_images_for_model, duplicate_array, load_dataset_stats,
)
from cosmos_policy.utils.utils import set_seed_everywhere
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4

CKPT_ROOT = pathlib.Path("/data3/liu/exp/counterfactual/checkpoints")
POLICY_DIR = CKPT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
BASE_MODEL_DIR = CKPT_ROOT / "Cosmos-Predict2-2B-Video2World"
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
EPS = 1e-8


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def patch_checkpoint(policy_dir: pathlib.Path):
    from cosmos_policy._src.imaginaire.utils import checkpoint_db
    orig = checkpoint_db.get_checkpoint_path
    base_pfx = "hf://nvidia/Cosmos-Predict2-2B-Video2World/"
    aloha_uri = "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
    def patched(uri):
        uri = str(uri).rstrip("/")
        if uri.startswith(base_pfx):
            return str(BASE_MODEL_DIR / uri[len(base_pfx):])
        if uri == aloha_uri:
            return str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        return orig(uri)
    checkpoint_db.get_checkpoint_path = patched


def make_cfg(policy_dir) -> SimpleNamespace:
    return SimpleNamespace(
        suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1,
        use_proprio=True, flip_images=False,
        use_variance_scale=False, use_jpeg_compression=True,
        num_denoising_steps_action=5,
        unnormalize_actions=True, normalize_proprio=True,
        dataset_stats_path=str(policy_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(policy_dir / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
    )


# ---------------------------------------------------------------------------
# LIBERO observation
# ---------------------------------------------------------------------------

def get_observation(ep: dict, flip_images: bool = False) -> dict:
    """Get the first observation for an episode."""
    task = ep["task_name"]  # Use perturbed task name for BDDL (environment setup)
    base_task = ep.get("base_task", task)  # Use base task for init states
    suite = ep.get("suite", "libero_10")
    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        set_seed_everywhere(int(ep.get("deterministic_reset_seed", 0)))
        env.reset()
        init_task = base_task
        init_file = f"{init_task}.pruned_init"
        init_path = pathlib.Path(get_libero_path("init_states")) / suite / init_file
        # Handle newobj variants (like objects_layout) with different state format
        is_newobj = "_add_" in init_file or "_level" in init_file
        if is_newobj and not init_path.exists():
            init_path = pathlib.Path(get_libero_path("init_states")) / "libero_newobj" / suite / init_file
        states = torch.load(str(init_path), weights_only=False)
        if is_newobj:
            states = states.reshape(1, -1)
        init_idx = int(ep.get("init_state_index", ep["episode"])) % len(states)
        obs = env.set_init_state(states[init_idx])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        primary = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        if flip_images:
            primary = np.flipud(primary); wrist = np.flipud(wrist)
        return {"primary_image": primary, "wrist_image": wrist}
    finally:
        env.close()


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------

def vae_encode_video_frames(model, primary: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """VAE-encode the video conditioning frames (indices 2, 3) and return latent.

    Returns: latent_video of shape [16, 2, 28, 28] = [C, num_frames, H, W]
    """
    device = next(model.parameters()).device
    B = 1

    # Prepare image sequence (33 frames → 9 latents)
    images = [primary, wrist]  # primary first for prepare_images_for_model ordering
    processed = prepare_images_for_model(images, SimpleNamespace(
        use_jpeg_compression=True, use_variance_scale=False,
        flip_images=False, trained_with_image_aug=True,
        randomize_seed=False, num_third_person_images=1))
    # processed order: [wrist, primary] (WRIST_IMAGE_IDX=0, IMAGE_IDX=1 in LIBERO)
    wrist_img, primary_img = processed[0], processed[1]

    blank = np.zeros_like(primary_img)
    wrist_dup = duplicate_array(wrist_img, COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_dup = duplicate_array(primary_img, COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    blank_dup = duplicate_array(blank, COSMOS_TEMPORAL_COMPRESSION_FACTOR)

    image_sequence = [
        np.expand_dims(np.zeros_like(blank), axis=0),  # 0: 1 frame
        blank_dup, wrist_dup, primary_dup,               # 1-3: 4 frames each
        blank_dup, blank_dup,                             # 4-5
        wrist_dup.copy(), primary_dup.copy(), blank_dup,  # 6-8
    ]
    raw_seq = np.concatenate(image_sequence, axis=0)
    raw_seq = np.expand_dims(raw_seq, 0)  # (1, T, H, W, C)
    raw_seq = np.tile(raw_seq, (B, 1, 1, 1, 1))
    raw_seq = np.transpose(raw_seq, (0, 4, 1, 2, 3))  # (B, C, T, H, W)
    raw_seq = torch.from_numpy(raw_seq).to(dtype=torch.uint8, device=device)

    # VAE encode
    with torch.no_grad():
        latent = model.encode(raw_seq).contiguous().float()  # [B, 16, 9, 28, 28]

    # Extract video frames only (indices 2, 3)
    latent_video = latent[0, :, [2, 3], :, :].cpu().numpy()  # [16, 2, 28, 28]
    return latent_video


# ---------------------------------------------------------------------------
# Projection onto S_action
# ---------------------------------------------------------------------------

def project_onto_S_action(delta_h: np.ndarray, Vt: np.ndarray, k: int) -> dict:
    """Project delta_h onto top-k action-sensitive subspace.

    Args:
        delta_h: [16, 2, 28, 28] — VAE latent change for video frames
        Vt: [r, 25088] — right singular vectors of Jacobian (r = total rank)
        k: number of top singular vectors to use

    Returns:
        dict with proj_ratio, proj_ratio_full, per_k ratios
    """
    dh_flat = delta_h.reshape(-1)  # [25088]

    total_norm_sq = float(np.dot(dh_flat, dh_flat))
    if total_norm_sq < EPS:
        return {"proj_ratio": 0.0, "proj_ratio_full": 0.0}

    # Project onto top-k subspace
    Vt_k = Vt[:k, :]  # [k, 25088]
    coeffs = Vt_k @ dh_flat  # [k]
    proj_norm_sq = float(np.dot(coeffs, coeffs))

    # Project onto full subspace (all r vectors)
    coeffs_full = Vt @ dh_flat
    proj_full_sq = float(np.dot(coeffs_full, coeffs_full))

    return {
        "proj_ratio": proj_norm_sq / total_norm_sq,
        "proj_ratio_full": proj_full_sq / total_norm_sq,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jacobian-dir",
                   default=str(ROOT / "experiments/phase5_jacobian_autograd"))
    p.add_argument("--output-dir",
                   default=str(ROOT / "experiments/phase6_vae_containment"))
    p.add_argument("--summary", required=True,
                   help="Phase 1 combined summary JSON")
    p.add_argument("--k-list", type=int, nargs="+", default=[1, 3, 5, 10],
                   help="Which k values to report containment for")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Limit number of pairs (0 = all)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load S_action
    jac_dir = pathlib.Path(args.jacobian_dir)
    Vt = np.load(jac_dir / "right_singular_vectors.npy")  # [r, 25088]
    S = np.load(jac_dir / "singular_values.npy")
    with open(jac_dir / "summary.json") as f:
        jac_summary = json.load(f)
    print(f"Loaded S_action: {Vt.shape[0]} singular vectors, "
          f"top-5 S: {S[:5].round(4)}")

    # 2. Load model
    patch_checkpoint(POLICY_DIR)
    cosmos_utils.init_t5_text_embeddings_cache(str(POLICY_DIR / "libero_t5_embeddings.pkl"))
    cfg = make_cfg(POLICY_DIR)
    set_seed_everywhere(args.seed)
    print("Loading model...")
    model, _ = cosmos_utils.get_model(cfg)
    model.eval()
    print(f"Model loaded: {len(model.net.blocks)} blocks")

    # 3. Load Phase 1 pairs
    summary_path = pathlib.Path(args.summary)
    summary = json.loads(summary_path.read_text())
    by_condition = {item["condition"]: item for item in summary["conditions"]}

    pairs = []
    for condition, info in by_condition.items():
        if condition == "clean":
            continue
        source_root = pathlib.Path(info.get("source_root", str(ROOT)))
        clean_path = source_root / "clean/episodes.json"
        pert_path = pathlib.Path(info["episodes_path"])
        if not pert_path.is_absolute():
            pert_path = ROOT / pert_path

        clean_list = json.loads(clean_path.read_text())
        pert_list = json.loads(pert_path.read_text())

        # Convert lists to dicts keyed by episode number
        clean_eps = {item["episode"]: item for item in clean_list}
        pert_eps = {item["episode"]: item for item in pert_list}

        for ep_key, clean in clean_eps.items():
            pert = pert_eps.get(ep_key)
            if pert is None:
                continue
            clean_success = clean["success"]
            pert_success = pert["success"]
            if clean_success and pert_success:
                group = "preserved"
            elif clean_success and not pert_success:
                group = "flipped"
            else:
                continue
            pairs.append({"condition": condition, "group": group,
                          "clean": clean, "pert": pert})

    print(f"Found {len(pairs)} pairs (preserved+flipped)")
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    # 4. Process each pair
    results = []
    errors = []
    start = time.time()
    for idx, pair in enumerate(pairs):
        cond = pair["condition"]
        grp = pair["group"]
        ep = pair["pert"]["episode"]

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(pairs)}] {cond} ep={ep} {grp}")

        # Get observations (skip on env error)
        try:
            clean_obs = get_observation(pair["clean"], flip_images=True)
            pert_obs = (clean_obs if cond == "language_instructions"
                        else get_observation(pair["pert"], flip_images=True))
        except Exception as e:
            errors.append({"idx": idx, "condition": cond, "episode": ep, "error": str(e)})
            continue

        # VAE encode
        latent_clean = vae_encode_video_frames(
            model, clean_obs["primary_image"], clean_obs["wrist_image"])
        latent_pert = vae_encode_video_frames(
            model, pert_obs["primary_image"], pert_obs["wrist_image"])

        # Delta
        delta_h = latent_pert - latent_clean  # [16, 2, 28, 28]

        # Project onto S_action
        proj = project_onto_S_action(delta_h, Vt, max(args.k_list))

        row = {
            "condition": cond, "episode": ep, "group": grp,
            "norm_dh": float(np.linalg.norm(delta_h)),
            "proj_ratio_full": proj["proj_ratio_full"],
        }
        for k in args.k_list:
            proj_k = project_onto_S_action(delta_h, Vt, k)
            row[f"proj_k{k}"] = proj_k["proj_ratio"]
        results.append(row)

    # 5. Aggregate
    print(f"\n{'='*60}")
    print(f"Containment of perturbation delta_h in Jacobian S_action")
    print(f"{'='*60}")

    for k in args.k_list:
        print(f"\n--- k={k} (top {k} action-sensitive directions) ---")
        print(f"  {'Condition':30s} {'Group':12s} {'N':>4s} {'containment':>12s} {'std':>10s}")
        print(f"  {'-'*65}")

        for cond in sorted(set(r["condition"] for r in results)):
            for grp in ["flipped", "preserved"]:
                vals = [r[f"proj_k{k}"] for r in results
                        if r["condition"] == cond and r["group"] == grp]
                if not vals:
                    continue
                print(f"  {cond:30s} {grp:12s} {len(vals):>4d} "
                      f"{np.mean(vals):>12.6f} {np.std(vals):>10.6f}")

    # Random baseline
    D = 16 * 2 * 28 * 28  # 25088
    for k in args.k_list:
        expected_random = k / D
        print(f"\n  Random baseline (k={k}/{D}): {expected_random:.6f}")

    # Save
    import csv
    all_fields = ["condition", "episode", "group", "norm_dh", "proj_ratio_full"] + \
                 [f"proj_k{k}" for k in args.k_list]
    csv_path = out_dir / "containment_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in all_fields})

    # Summary
    summary_out = {
        "jacobian_dir": str(jac_dir),
        "jacobian_sigma": jac_summary.get("sigma", "unknown"),
        "num_pairs": len(results),
        "k_list": args.k_list,
        "video_latent_dim": D,
        "per_perturbation": {},
    }
    for k in args.k_list:
        summary_out["per_perturbation"][f"k{k}"] = {}
        for cond in sorted(set(r["condition"] for r in results)):
            for grp in ["flipped", "preserved"]:
                vals = [r[f"proj_k{k}"] for r in results
                        if r["condition"] == cond and r["group"] == grp]
                if vals:
                    summary_out["per_perturbation"][f"k{k}"][f"{cond}/{grp}"] = {
                        "n": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals))
                    }

    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2))
    print(f"\nDone. {out_dir / 'summary.json'}")
    print(f"Elapsed: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
