#!/usr/bin/env python3
"""
Phase 7 — Multi-layer containment of perturbation δh within action-sensitive subspace.

Core question:
  Do perturbation-induced latent shifts (δh) lie within the model's
  action-sensitive subspace (S_action = top-k singular vectors of ∂a/∂h)?

Measurement layers:
  - VAE output  (DiT input):  autograd Jacobian from Phase 5
  - DiT layer 27 (before FinalLayer):  weight-matrix SVD from Phase 4

Method (per perturbation pair):
  1. Run one denoising step at sigma=80 for clean and perturbed observations.
  2. Capture VAE latent and L27 hidden states.
  3. δh = perturbed − clean.
  4. containment = ‖Proj_{S_action} δh‖² / ‖δh‖².
  5. Compare against random baseline = k / D.

All at sigma=80 — the first denoising step where AdaLN is active and
cross-attention is working.  sigma_min would degrade the FinalLayer
(AdaLN compression), so we avoid it.
"""

import sys, os, pathlib, json, time, csv, ctypes
from types import SimpleNamespace
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
# Native runtime setup (ImageMagick / wand for LIBERO)
# ═══════════════════════════════════════════════════════════════════

def _prepend_env_path(name, value):
    parts = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    if value not in parts:
        os.environ[name] = value + (os.pathsep + os.environ[name] if parts else "")


def _preload_first_existing(lib_dir, names):
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    last_error = None
    for name in names:
        path = lib_dir / name
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=mode)
            return True
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        print(f"[warn] failed to preload ImageMagick library from {lib_dir}: {last_error}")
    return False


def _configure_native_runtime():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("DETERMINISTIC", "True")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy_numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy_mplconfig")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    prefix = pathlib.Path(conda_prefix) if conda_prefix else pathlib.Path(sys.executable).resolve().parents[1]

    lib_dir = prefix / "lib"
    if not lib_dir.exists():
        return

    os.environ.setdefault("MAGICK_HOME", str(prefix))
    os.environ.setdefault("WAND_MAGICK_LIBRARY_SUFFIX", "-7.Q16HDRI")
    _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir))
    _preload_first_existing(lib_dir, ["libMagickCore-7.Q16HDRI.so.10", "libMagickCore-7.Q16HDRI.so"])
    _preload_first_existing(lib_dir, ["libMagickWand-7.Q16HDRI.so.10", "libMagickWand-7.Q16HDRI.so"])


_configure_native_runtime()

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import prepare_images_for_model, duplicate_array
from cosmos_policy.utils.utils import set_seed_everywhere
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

CKPT_ROOT = pathlib.Path("/data3/liu/exp/counterfactual/checkpoints")
POLICY_DIR = CKPT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
BASE_MODEL_DIR = CKPT_ROOT / "Cosmos-Predict2-2B-Video2World"

T = 4               # temporal copies per video slot
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Video slot layout (9 slots, T frames each except slot 0):
#   0: first frame (1)    3: current primary     6: future wrist
#   1: blank              4: action latent        7: future primary
#   2: current wrist      5: proprio latent       8: value latent
# Total temporal frames = 1 + 8*T = 33

SLOT_FIRST = 0
SLOT_WRIST = 2
SLOT_PRIMARY = 3
SLOT_ACTION = 4
SLOT_PROPRIO = 5
SLOT_FUTURE_WRIST = 6
SLOT_FUTURE_PRIMARY = 7
SLOT_VALUE = 8

VAE_PRIMARY_TEMPORAL = [2, 3]   # temporal indices in VAE latent for primary image region
L27_ACTION_TEMPORAL = 4          # temporal index in L27 for action slot
L27_GRID_H = [0, 1]             # grid rows that contribute to action output
L27_GRID_W = 14                 # all columns contribute
L27_DIM = 2048                  # hidden dimension per grid position

DEFAULT_TEXT = "put the black bowl in the bottom drawer of the cabinet and close it"


# ═══════════════════════════════════════════════════════════════════
# Checkpoint path monkey-patch
# ═══════════════════════════════════════════════════════════════════

from cosmos_policy._src.imaginaire.utils import checkpoint_db

_orig_get_checkpoint_path = checkpoint_db.get_checkpoint_path

def _get_checkpoint_path(uri):
    s = str(uri)
    if "Cosmos-Predict2-2B-Video2World" in s:
        return str(BASE_MODEL_DIR / s.split("Cosmos-Predict2-2B-Video2World/")[-1])
    if "ALOHA" in s or "LIBERO" in s:
        return str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    return _orig_get_checkpoint_path(uri)

checkpoint_db.get_checkpoint_path = _get_checkpoint_path


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

cosmos_utils.init_t5_text_embeddings_cache(str(POLICY_DIR / "libero_t5_embeddings.pkl"))

cfg = SimpleNamespace(
    suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path=str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
    config_file="cosmos_policy/config/config.py",
    use_third_person_image=True, num_third_person_images=1,
    use_wrist_image=True, num_wrist_images=1, use_proprio=True, flip_images=False,
    use_variance_scale=False, use_jpeg_compression=True,
    num_denoising_steps_action=5, unnormalize_actions=True, normalize_proprio=True,
    dataset_stats_path=str(POLICY_DIR / "libero_dataset_statistics.json"),
    t5_text_embeddings_path=str(POLICY_DIR / "libero_t5_embeddings.pkl"),
    trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
)

set_seed_everywhere(7)
print("Loading model ...")
model, _ = cosmos_utils.get_model(cfg)
model.eval()
device = next(model.parameters()).device
print(f"Loaded: {len(model.net.blocks)} DiT blocks, device={device}")


# ═══════════════════════════════════════════════════════════════════
# Video construction
# ═══════════════════════════════════════════════════════════════════

def build_video_tensor(primary, wrist, *, T=T):
    """Build the multi-slot video tensor expected by the Cosmos model.

    Constructs a 9-slot video where image slots are replicated T times
    along the temporal axis.  Slot 0 is a single zero frame.

    Returns (torch.Tensor): shape (1, C, 1+8*T, H, W), dtype uint8.
    """
    images = [primary, wrist]
    processed = prepare_images_for_model(images, SimpleNamespace(
        use_jpeg_compression=True, use_variance_scale=False,
        flip_images=False, trained_with_image_aug=True,
        randomize_seed=False, num_third_person_images=1,
    ))
    wi, pi = processed[0], processed[1]
    blank = np.zeros_like(pi)

    wd = duplicate_array(wi, T)   # (T, H, W, C)
    pd = duplicate_array(pi, T)
    bd = duplicate_array(blank, T)

    seq = [
        np.expand_dims(np.zeros_like(blank), axis=0),  # slot 0: first frame (1, H, W, C)
        bd,                                              # slot 1
        wd,                                              # slot 2: current wrist
        pd,                                              # slot 3: current primary
        bd,                                              # slot 4: action
        bd,                                              # slot 5: proprio
        wd.copy(),                                       # slot 6: future wrist
        pd.copy(),                                       # slot 7: future primary
        bd,                                              # slot 8: value
    ]
    raw = np.concatenate(seq, axis=0)          # (1+8*T, H, W, C)
    raw = np.expand_dims(raw, 0)                # (1, 1+8*T, H, W, C)
    raw = np.transpose(raw, (0, 4, 1, 2, 3))   # (1, C, 1+8*T, H, W)
    return torch.from_numpy(raw).to(dtype=torch.uint8, device=device)


# ═══════════════════════════════════════════════════════════════════
# Observation helpers
# ═══════════════════════════════════════════════════════════════════

def get_obs(ep, flip=True):
    """Instantiate a LIBERO env, reset to the episode init state, return images."""
    task = ep["task_name"]
    base = ep.get("base_task", task)
    suite = ep.get("suite", "libero_10")
    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        set_seed_everywhere(int(ep.get("deterministic_reset_seed", 0)))
        env.reset()
        init_f = f"{base}.pruned_init"
        ip = pathlib.Path(get_libero_path("init_states")) / suite / init_f
        is_new = "_add_" in init_f or "_level" in init_f
        if is_new and not ip.exists():
            ip = pathlib.Path(get_libero_path("init_states")) / "libero_newobj" / suite / init_f
        states = torch.load(str(ip), weights_only=False)
        if is_new:
            states = states.reshape(1, -1)
        idx = int(ep.get("init_state_index", ep["episode"])) % len(states)
        obs = env.set_init_state(states[idx])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        primary, wrist = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
        if flip:
            primary, wrist = np.flipud(primary), np.flipud(wrist)
        return primary, wrist
    finally:
        env.close()


# ═══════════════════════════════════════════════════════════════════
# DiT forward with intermediate capture
# ═══════════════════════════════════════════════════════════════════

class DiTForward:
    """Run one denoising step at a given sigma, capture VAE latent and L27 state."""

    def __init__(self, model, primary, wrist, *, sigma_val=80.0, text=None):
        self.model = model
        self.sigma_val = sigma_val

        raw_video = build_video_tensor(primary, wrist)
        if text is None:
            text = DEFAULT_TEXT

        B = 1
        text_emb = cosmos_utils.get_t5_embedding_from_cache(text)
        data_batch = {
            "dataset_name": "video_data",
            "video": raw_video,
            "t5_text_embeddings": text_emb,
            "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
            "padding_mask": torch.zeros(B, 1, 256, 256, dtype=torch.bfloat16, device=device),
            "num_conditional_frames": model.config.min_num_conditional_frames,
            "current_wrist_image_latent_idx":  torch.tensor([SLOT_WRIST],          dtype=torch.int64, device=device),
            "current_image_latent_idx":         torch.tensor([SLOT_PRIMARY],        dtype=torch.int64, device=device),
            "action_latent_idx":                torch.tensor([SLOT_ACTION],         dtype=torch.int64, device=device),
            "future_image_latent_idx":          torch.tensor([SLOT_FUTURE_PRIMARY], dtype=torch.int64, device=device),
            "future_wrist_image_latent_idx":    torch.tensor([SLOT_FUTURE_WRIST],   dtype=torch.int64, device=device),
            "future_proprio_latent_idx":        torch.tensor([SLOT_PROPRIO],        dtype=torch.int64, device=device),
            "value_latent_idx":                 torch.tensor([SLOT_VALUE],          dtype=torch.int64, device=device),
        }
        for k in ("current_wrist_image2_latent_idx", "current_image2_latent_idx",
                  "future_wrist_image2_latent_idx", "future_image2_latent_idx",
                  "current_proprio_latent_idx"):
            data_batch[k] = torch.tensor([-1], dtype=torch.int64, device=device)

        raw_state, latent_state, condition = model.get_data_and_condition(data_batch)
        self.latent_state = latent_state
        self.xt = latent_state + torch.randn_like(latent_state) * sigma_val
        self.condition = condition

        # Outputs populated by run()
        self.h_L27 = None
        self.vae_latent = None
        self.action = None
        self._handles = []

    def _hook_L27(self, _m, _in, out):
        x = out[0] if isinstance(out, tuple) else out
        self.h_L27 = x.detach().clone()

    def run(self):
        """Execute one denoising step and capture intermediates."""
        blocks = self.model.net.blocks
        self._handles.append(blocks[27].register_forward_hook(self._hook_L27))
        try:
            with torch.no_grad():
                sigma_t = torch.full((1, 1), self.sigma_val, device=device, dtype=torch.float32)
                out = self.model.denoise(self.xt, sigma_t.squeeze(-1), self.condition)
                self.vae_latent = self.latent_state.detach().cpu().numpy()   # [1, 16, 9, 28, 28]
                al = out.x0[:, :, [SLOT_ACTION], :, :]
                self.action = al.reshape(1, -1)[:, :112].cpu().numpy()
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
        return self


# ═══════════════════════════════════════════════════════════════════
# Jacobian / projection utilities
# ═══════════════════════════════════════════════════════════════════

def l27_weight_jacobian(ckpt):
    """Extract action-sensitive directions from FinalLayer weight matrix.

    The FinalLayer maps h [B,1,14,14,2048] → action [112].
    Only 28 grid positions (h∈{0,1}, w∈{0..13}) × 4 output channels
    (0,16,32,48) contribute to the action output.

    Returns:
      Vt:  [r, 2048]  top right singular vectors (action-sensitive directions)
      S:   [r]         singular values
      r:   int         effective rank
    """
    W = ckpt["net.final_layer.linear.weight"].float().numpy()   # [64, 2048]
    action_ch = [0, 16, 32, 48]
    W_action = W[action_ch]                                      # [4, 2048]
    U, S, Vt = np.linalg.svd(W_action, full_matrices=False)
    r = int(sum(S > S[0] * 1e-6))
    return Vt[:r], S[:r], r


def project_VAE(delta_h_vae, Vt, k):
    """Project VAE-level δh onto top-k action-sensitive singular vectors.

    Args:
      delta_h_vae: [16, 2, 28, 28] — δh in the primary-image region of VAE latent.
      Vt: [D, D] or [k, D] — right singular vectors of ∂a/∂h at VAE output.
      k:  number of top singular vectors to use.

    Returns:
      containment = ‖Proj δh‖² / ‖δh‖² ∈ [0, 1].
    """
    dh_flat = delta_h_vae.reshape(-1)
    Vt_k = Vt[:k]
    coeffs = Vt_k @ dh_flat
    return float(np.sum(coeffs ** 2)) / (float(np.sum(dh_flat ** 2)) + 1e-12)


def project_L27(delta_h_L27, Vt_per_pos):
    """Project L27-level δh onto action-sensitive directions.

    Only grid positions that feed into the action output are considered:
    temporal index = 4 (action slot), h ∈ {0,1}, all w ∈ {0..13}.

    The *total* δh norm is computed over all positions (full tensor);
    the *projected* norm only over the contributing positions.

    Args:
      delta_h_L27: [1, 9, 14, 14, 2048] or broadcastable.
      Vt_per_pos:  [r, 2048] — action-sensitive direction vectors per position.

    Returns:
      containment ∈ [0, 1].
    """
    dh = delta_h_L27.reshape(9, 14, 14, L27_DIM)
    total_sq = float(np.sum(dh ** 2))
    proj_sq = 0.0
    for h in L27_GRID_H:
        for w in range(L27_GRID_W):
            dh_hw = dh[L27_ACTION_TEMPORAL, h, w]   # [2048]
            coeffs = dh_hw @ Vt_per_pos.T             # [r]
            proj_sq += float(np.sum(coeffs ** 2))
    return proj_sq / (total_sq + 1e-12)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 7 — Multi-layer containment analysis")
    p.add_argument("--output-dir", default=str(ROOT / "experiments/phase7_multilayer"))
    p.add_argument("--sigma", type=float, default=80.0)
    p.add_argument("--max-pairs", type=int, default=0)
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # ── Load pre-computed Jacobians ─────────────────────────────

    # L27: from FinalLayer weight SVD (Phase 4)
    ckpt = torch.load(str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
                      map_location="cpu", weights_only=False)
    Vt_L27, S_L27, r_L27 = l27_weight_jacobian(ckpt)
    print(f"L27 (weights): rank={r_L27}, S={S_L27.round(4)}")

    # VAE: from autograd Jacobian (Phase 5)
    vae_dir = ROOT / "experiments/phase5_jacobian_autograd"
    Vt_vae = np.load(vae_dir / "right_singular_vectors.npy")     # [112, 25088]
    S_vae = np.load(vae_dir / "singular_values.npy")
    print(f"VAE (autograd): rank={len(S_vae)}, top-5 S={S_vae[:5].round(4)}")

    # ── Load perturbation pairs ─────────────────────────────────

    summary_path = ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json"
    summary = json.loads(summary_path.read_text())

    pairs = []
    for info in summary["conditions"]:
        cond = info["condition"]
        if cond == "clean":
            continue
        src = pathlib.Path(info.get("source_root", str(ROOT)))
        cp = src / "clean/episodes.json"
        pp = ROOT / info["episodes_path"]
        cl = json.loads(cp.read_text())
        pl = json.loads(pp.read_text())
        ce = {it["episode"]: it for it in cl}
        pe = {it["episode"]: it for it in pl}
        for ek, clean in ce.items():
            pert = pe.get(ek)
            if pert is None:
                continue
            cs, ps = clean["success"], pert["success"]
            if cs and ps:
                grp = "preserved"
            elif cs and not ps:
                grp = "flipped"
            else:
                continue
            pairs.append({"condition": cond, "group": grp, "clean": clean, "pert": pert})

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    print(f"\nProcessing {len(pairs)} pairs (sigma={args.sigma}) ...")

    # ── Per-pair containment ────────────────────────────────────

    results = []
    errors = []
    t_start = time.time()

    for idx, pair in enumerate(pairs):
        cond = pair["condition"]
        grp = pair["group"]

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(pairs)}]  {cond}  {grp}")

        try:
            # Get observations
            cp_img, cw_img = get_obs(pair["clean"], flip=True)
            if cond == "language_instructions":
                # Language perturbation only affects text, not pixels.
                # δh_VAE ≡ 0 (same images), so containment is trivially 0.
                # These pairs are included for completeness but add no VAE signal.
                pp_img, pw_img = cp_img, cw_img
            else:
                pp_img, pw_img = get_obs(pair["pert"], flip=True)

            # Run denoising for both clean and perturbed
            cap_c = DiTForward(model, cp_img, cw_img, sigma_val=args.sigma).run()
            cap_p = DiTForward(model, pp_img, pw_img, sigma_val=args.sigma).run()

            row = {"condition": cond, "episode": pair["pert"]["episode"],
                   "group": grp}

            # --- VAE-level containment ---
            lc = cap_c.vae_latent   # [1, 16, 9, 28, 28]
            lp = cap_p.vae_latent
            # δh over the primary-image temporal region only (indices 2,3).
            # This matches the Jacobian domain from Phase 5 (D = 16×2×28×28 = 25088).
            dh_vae = (lp - lc)[0, :, VAE_PRIMARY_TEMPORAL, :, :]   # [16, 2, 28, 28]
            for k in (1, 3, 5, 10):
                row[f"vae_k{k}"] = project_VAE(dh_vae, Vt_vae, k)

            # --- L27-level containment ---
            if cap_c.h_L27 is not None and cap_p.h_L27 is not None:
                dh_L27 = (cap_p.h_L27 - cap_c.h_L27).float().cpu().numpy()
                for k_vec in (1, 2, 3, 4):
                    row[f"L27_k{k_vec}"] = project_L27(dh_L27, Vt_L27[:k_vec])

            results.append(row)

        except Exception as e:
            errors.append({"cond": cond, "ep": pair["pert"]["episode"], "err": str(e)})
            if len(errors) <= 3:
                import traceback
                print(f"  ERROR [{len(errors)}]  {cond}  ep={pair['pert']['episode']}:")
                traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s  |  results={len(results)}  errors={len(errors)}")
    for e in errors[:5]:
        print(f"  [{e['cond']}  ep={e['ep']}]:  {e['err'][:120]}")

    # ── Report ──────────────────────────────────────────────────

    K_VAE  = [("k1", 1), ("k3", 3), ("k5", 5), ("k10", 10)]
    K_L27  = [("k1", 1), ("k2", 2), ("k3", 3), ("k4", 4)]
    DIMS   = {"vae": 25088, "L27": 401408}

    for layer_name, key_prefix, k_list in [("VAE", "vae", K_VAE),
                                            ("L27", "L27", K_L27)]:
        print(f"\n{'='*70}")
        print(f"  {layer_name} — containment in action-sensitive subspace  (sigma={args.sigma})")
        print(f"{'='*70}")

        for k_label, k_val in k_list:
            k_col = f"{key_prefix}_{k_label}"
            if not results or k_col not in results[0]:
                continue

            D = DIMS[key_prefix]
            rnd = k_val / D
            print(f"\n  --- {k_label}  (top {k_val}, random baseline: {rnd:.6f}) ---")
            print(f"  {'Condition':30s} {'Group':12s} {'N':>4s}  {'containment':>12s}  {'×random':>8s}")
            print(f"  {'-'*68}")

            by_cond = defaultdict(lambda: {"flipped": [], "preserved": []})
            for r in results:
                if k_col in r:
                    by_cond[r["condition"]][r["group"]].append(r[k_col])

            all_flipped, all_preserved = [], []
            for cond in sorted(by_cond):
                for grp in ("flipped", "preserved"):
                    vals = by_cond[cond][grp]
                    if not vals:
                        continue
                    m = np.mean(vals)
                    print(f"  {cond:30s} {grp:12s} {len(vals):>4d}  {m:>12.6f}  {m / (rnd + 1e-12):>8.2f}x")
                    if grp == "flipped":
                        all_flipped.extend(vals)
                    else:
                        all_preserved.extend(vals)

            if all_flipped and all_preserved:
                mf, mp = np.mean(all_flipped), np.mean(all_preserved)
                print(f"  {'Flipped avg':30s} {'':12s} {'':>4s}  {mf:>12.6f}")
                print(f"  {'Preserved avg':30s} {'':12s} {'':>4s}  {mp:>12.6f}")
                print(f"  {'Ratio F/P':30s} {'':12s} {'':>4s}  {mf / (mp + 1e-12):>12.2f}x")

    # ── Save ────────────────────────────────────────────────────

    all_cols = ["condition", "episode", "group"]
    for r in results:
        for k in r:
            if k not in all_cols:
                all_cols.append(k)

    with (out_dir / "containment.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set(all_cols)))
        w.writeheader()
        for r in results:
            w.writerow(r)

    (out_dir / "summary.json").write_text(json.dumps({
        "sigma": args.sigma,
        "n_pairs": len(results),
        "n_errors": len(errors),
        "l27_rank": r_L27,
        "vae_rank": len(S_vae),
    }, indent=2))

    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
