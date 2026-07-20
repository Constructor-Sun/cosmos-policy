#!/usr/bin/env python3
"""M1+M6b fragility analysis for Cosmos-Policy DiT (Fawzi et al. 2018 style).

M1 — Jacobian Sensitivity Profile: is the perturbation direction Δh
     disproportionately more sensitive than random directions?
M6b — Boundary Curvature (gradient rotation): is the action prediction
      boundary flatter along Δh for flipped cases?

Uses Phase 2 saved hidden states + model FinalLayer weights.
No new full-model forward passes needed.
"""

from __future__ import annotations

import argparse, csv, json, math, os, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]

# EDM constants
SIGMA_DATA = 0.5
SIGMA_MIN = 0.002  # final clean-step sigma


# ---------------------------------------------------------------------------
# Data loading (same pattern as phase3_fragility_m2_m4.py)
# ---------------------------------------------------------------------------

def load_phase2_pairs(root: Path, groups: set[str]) -> list[dict]:
    pairs = []
    for mpath in sorted(root.glob("*/*/metrics.json")):
        rec = json.loads(mpath.read_text())
        if rec["group"] not in groups:
            continue
        rec["_dir"] = mpath.parent
        pairs.append(rec)
    print(f"Loaded {len(pairs)} pairs from {root}")
    return pairs


def load_hidden_action(pair: dict, clean_or_pert: str, device="cpu") -> torch.Tensor:
    path = pair["_dir"] / f"hidden_action_{clean_or_pert}.pt"
    d = torch.load(str(path), map_location=device, weights_only=True)
    t = list(d.values())[0]  # shape: [1, 1, 14, 14, 2048]
    return t.float().squeeze(0)  # → [1, 14, 14, 2048]


# ---------------------------------------------------------------------------
# Sub-network: block-27 hidden state → FinalLayer output
# ---------------------------------------------------------------------------

def build_sub_network(model) -> dict:
    """Extract FinalLayer + timestep components from the DiT model."""
    net = model.net
    final_layer = net.final_layer  # FinalLayer module
    t_embedder = net.t_embedder    # Timesteps → TimestepEmbedding
    t_norm = net.t_embedding_norm  # RMSNorm on t_embedding
    use_adaln_lora = net.use_adaln_lora

    def get_affline_scale_log_info(t_emb):
        # For use_adaln_lora=False path: adaln is computed per-block, not globally
        # For use_adaln_lora=True, there's a global adaln_lora projection
        # We recompute from sigma as needed
        if hasattr(net, 'get_affline_scale_log_info'):
            return net.get_affline_scale_log_info(t_emb)
        return None

    return {
        "final_layer": final_layer,
        "t_embedder": t_embedder,
        "t_norm": t_norm,
        "use_adaln_lora": use_adaln_lora,
        "get_adaln": get_affline_scale_log_info,
    }


def compute_timestep_embedding(sub_net: dict, sigma: float, device="cpu", dtype=torch.float32):
    """Compute t_embedding and adaln_lora at a given sigma."""
    # c_noise = 0.25 * log(sigma / sigma_data)
    c_noise = 0.25 * math.log(sigma / SIGMA_DATA)
    t_B = torch.tensor([[c_noise]], device=device, dtype=torch.bfloat16)  # (1, 1)

    # t_embedder = Sequential(Timesteps, TimestepEmbedding)
    # TimestepEmbedding.forward returns (emb, adaln_lora_B_T_3D) when use_adaln_lora=True
    t_out = sub_net["t_embedder"](t_B)
    if isinstance(t_out, tuple):
        t_emb, adaln_lora = t_out
    else:
        t_emb, adaln_lora = t_out, None

    t_emb = sub_net["t_norm"](t_emb)              # (1, 1, 2048)
    return t_emb, adaln_lora


def final_layer_forward(h: torch.Tensor, t_emb: torch.Tensor,
                        adaln: torch.Tensor | None,
                        final_layer: nn.Module) -> torch.Tensor:
    """Run FinalLayer on hidden state at the action slot.

    h:      [1, 14, 14, 2048]  (T=1, action slot only)
    t_emb:  [1, 1, 2048]       (T=1)
    adaln:  [1, 1, 6144] or None
    Returns: [1, 14, 14, 64]   (64 = patch_spatial² * patch_temporal * out_channels)
    """
    # Add batch dim and match dtype of FinalLayer weights
    x = h.unsqueeze(0)  # (1, 1, 14, 14, 2048)
    # FinalLayer expects bfloat16 on cuda
    target_dtype = next(final_layer.parameters()).dtype
    x = x.to(dtype=target_dtype)
    output = final_layer(x, t_emb, adaln)  # (1, 1, 14, 14, 64)
    return output.float().squeeze(0)  # (1, 14, 14, 64)


# ---------------------------------------------------------------------------
# M1: Jacobian Sensitivity Profile (via exact JVP, no finite-diff)
# ---------------------------------------------------------------------------

def m1_sensitivity_ratio(h_clean: torch.Tensor, h_pert: torch.Tensor,
                          sub_net: dict, t_emb: torch.Tensor,
                          adaln: torch.Tensor | None,
                          n_random: int = 10) -> dict:
    """Sensitivity of FinalLayer output along Δh vs random directions.

    Uses torch.autograd.functional.jvp for exact Jacobian-vector products,
    avoiding bfloat16 finite-difference precision issues.
    """
    fl = sub_net["final_layer"]

    dh = (h_pert - h_clean).reshape(-1)
    dh_norm = dh / (dh.norm() + EPS)

    # Cast to float32 for precise JVP computation
    h_clean_f32 = h_clean.float()
    t_emb_f32 = t_emb.float()
    adaln_f32 = adaln.float() if adaln is not None else None

    # Temporarily cast FinalLayer to float32
    orig_dtype = next(fl.parameters()).dtype
    fl_float = fl.float()

    def f_output_f32(h_in: torch.Tensor) -> torch.Tensor:
        """FinalLayer output flattened, float32."""
        out = final_layer_forward_f32(h_in, t_emb_f32, adaln_f32, fl_float)
        return out.reshape(-1)

    # JVP along a direction v: J @ v = directional derivative of f at h along v
    def sensitivity_jvp(h: torch.Tensor, v: torch.Tensor) -> float:
        _, jvp_val = torch.autograd.functional.jvp(f_output_f32, h, (v,))
        return float(jvp_val.norm())

    f0 = f_output_f32(h_clean_f32)
    norm_f0 = float(f0.norm())

    # Sensitivity along Δh
    v_dh = dh_norm.reshape_as(h_clean_f32).float()
    s_dh = sensitivity_jvp(h_clean_f32, v_dh)

    # Sensitivity along random directions
    s_rand = []
    for _ in range(n_random):
        r = torch.randn_like(h_clean_f32)
        r = r / (r.norm() + EPS)
        s_rand.append(sensitivity_jvp(h_clean_f32, r))

    s_random_mean = float(np.mean(s_rand))
    s_random_std = float(np.std(s_rand))
    ratio = s_dh / (s_random_mean + EPS)

    # Restore original dtype
    fl_float.to(orig_dtype)

    return {
        "s_dh": s_dh, "s_random_mean": s_random_mean, "s_random_std": s_random_std,
        "fragility_ratio": ratio, "norm_f0": norm_f0,
        "per_random": s_rand,
    }


def final_layer_forward_f32(h: torch.Tensor, t_emb: torch.Tensor,
                             adaln: torch.Tensor | None,
                             final_layer: nn.Module) -> torch.Tensor:
    """Float32 version of final_layer_forward."""
    x = h.unsqueeze(0)  # (1, 1, 14, 14, 2048)
    output = final_layer(x, t_emb, adaln)
    return output.squeeze(0)  # (1, 14, 14, 64)


# ---------------------------------------------------------------------------
# M6b: Gradient Rotation Curvature
# ---------------------------------------------------------------------------

def m6b_curvature(h_clean: torch.Tensor, h_pert: torch.Tensor,
                   sub_net: dict, t_emb: torch.Tensor,
                   adaln: torch.Tensor | None,
                   n_points: int = 5) -> dict:
    """Gradient rotation along h(α) = h_clean + α * Δh_norm (float32 for precision)."""
    fl = sub_net["final_layer"]
    orig_dtype = next(fl.parameters()).dtype
    fl_float = fl.float()

    h_clean_f32 = h_clean.float()
    t_emb_f32 = t_emb.float()
    adaln_f32 = adaln.float() if adaln is not None else None

    dh = (h_pert.float() - h_clean_f32).reshape(-1)
    dh_norm_vec = dh / (dh.norm() + EPS)

    def f_output_f32(h_in: torch.Tensor) -> torch.Tensor:
        out = final_layer_forward_f32(h_in, t_emb_f32, adaln_f32, fl_float)
        return out.reshape(-1)

    f0 = f_output_f32(h_clean_f32).detach()

    def loss_fn(h_in: torch.Tensor) -> torch.Tensor:
        return ((f_output_f32(h_in) - f0) ** 2).sum()

    alphas = np.linspace(0, 1, n_points)
    grads = []
    for alpha in alphas:
        h_a = h_clean_f32 + alpha * dh_norm_vec.reshape_as(h_clean_f32)
        h_a = h_a.detach().clone().requires_grad_(True)
        loss = loss_fn(h_a)
        g = torch.autograd.grad(loss, h_a, create_graph=False)[0]
        grads.append(g.reshape(-1))

    fl_float.to(orig_dtype)

    cosines = []
    for k in range(len(grads) - 1):
        gk, gk1 = grads[k], grads[k + 1]
        denom = (gk.norm() * gk1.norm() + EPS)
        cos_val = float((gk @ gk1) / denom)
        cos_val = max(-1.0, min(1.0, cos_val))
        cosines.append(cos_val)

    angles = [math.acos(c) for c in cosines]
    curvatures = [a / 0.25 for a in angles]  # per-unit-step
    mean_curv = float(np.mean(curvatures)) if curvatures else float("nan")
    total_angle = float(np.sum(angles))

    return {
        "cosines": cosines, "angles_deg": [math.degrees(a) for a in angles],
        "curvature_mean_rad_per_unit": mean_curv,
        "total_rotation_deg": math.degrees(total_angle),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_results = str(ROOT / "experiments/phase2_angular_cosmos/"
                          "kitchen_scene4_seed7_all_perturb_preserved_flipped_last")
    p.add_argument("--results-dir", default=default_results)
    p.add_argument("--output-dir",
                   default=str(ROOT / "experiments/phase3_fragility_m1_m6b/kitchen_scene4_seed7"))
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--policy-dir",
                   default=str(ROOT.parent.parent / "checkpoints" / "Cosmos-Policy-LIBERO-Predict2-2B"))
    p.add_argument("--n-random", type=int, default=10, help="Random directions for M1")
    p.add_argument("--eps", type=float, default=1e-3, help="Finite-difference step")
    p.add_argument("--m1-only", action="store_true")
    p.add_argument("--m6b-only", action="store_true")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    do_m1 = not args.m6b_only
    do_m6b = not args.m1_only

    # Load pairs
    pairs = load_phase2_pairs(Path(args.results_dir), set(args.groups))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        print("No pairs found."); return

    # Counts
    from collections import Counter
    for (cond, grp), n in sorted(Counter((p["condition"], p["group"]) for p in pairs).items()):
        print(f"  {cond:30s} {grp:12s} {n}")

    # Load model (one-time) and extract sub-network
    print("\nLoading model...")
    device = torch.device(args.device)

    # We need cosmos_utils for model loading
    sys.path.insert(0, str(ROOT))
    from cosmos_policy.experiments.robot import cosmos_utils

    policy_dir = Path(args.policy_dir)
    checkpoint_db_path = ROOT / "cosmos_policy/_src/imaginaire/utils/checkpoint_db.py"
    # patch checkpoint_db for offline loading
    from cosmos_policy._src.imaginaire.utils import checkpoint_db as ckpt_db
    BASE_MODEL_DIR = ROOT.parent.parent / "checkpoints" / "Cosmos-Predict2-2B-Video2World"
    orig_get = ckpt_db.get_checkpoint_path

    def offline_get(uri):
        uri = str(uri).rstrip("/")
        base_prefix = "hf://nvidia/Cosmos-Predict2-2B-Video2World/"
        aloha_uri = "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
        if uri.startswith(base_prefix):
            return str(BASE_MODEL_DIR / uri[len(base_prefix):])
        if uri == aloha_uri:
            return str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        return orig_get(uri)
    ckpt_db.get_checkpoint_path = offline_get

    cfg = SimpleNamespace(
        suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1, use_proprio=True,
        flip_images=True, use_variance_scale=False, use_jpeg_compression=True,
        num_denoising_steps_action=5, unnormalize_actions=True, normalize_proprio=True,
        dataset_stats_path=str(policy_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(policy_dir / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
    )
    model, _ = cosmos_utils.get_model(cfg)
    model = model.to(device)
    model.eval()
    sub_net = build_sub_network(model)
    print(f"  Model loaded. DiT blocks: {len(model.net.blocks)}")

    # Compute timestep embedding at sigma_min (final clean step)
    # Note: cosmospolicy sampler uses sigma_min=0.002 for the final sample_clean step
    t_emb, adaln = compute_timestep_embedding(sub_net, SIGMA_MIN, device=device, dtype=torch.float32)
    print(f"  Timestep embedding computed at sigma_min={SIGMA_MIN}")

    # Process each pair
    m1_rows = []
    m6b_rows = []
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    import time
    start = time.time()
    for idx, pair in enumerate(pairs, 1):
        cond, ep, grp = pair["condition"], int(pair["episode"]), pair["group"]
        label = f"[{idx}/{len(pairs)}] {cond}/ep{ep:02d} {grp}"
        print(label)

        h_clean = load_hidden_action(pair, "clean", device=device)
        h_pert = load_hidden_action(pair, "pert", device=device)

        pair_dir = out_root / cond / f"ep{ep:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # M1
        if do_m1:
            try:
                m1 = m1_sensitivity_ratio(h_clean, h_pert, sub_net, t_emb, adaln,
                                          n_random=args.n_random)
                (pair_dir / "m1_fragility.json").write_text(json.dumps(m1, indent=2, default=str))
                m1_rows.append({
                    "condition": cond, "episode": ep, "group": grp,
                    "s_dh": m1["s_dh"], "s_random_mean": m1["s_random_mean"],
                    "s_random_std": m1["s_random_std"], "fragility_ratio": m1["fragility_ratio"],
                })
                print(f"    M1 ratio={m1['fragility_ratio']:.3f} (s_dh={m1['s_dh']:.4f}, s_rand={m1['s_random_mean']:.4f})")
            except Exception as e:
                print(f"    M1 ERROR: {e}")

        # M6b
        if do_m6b:
            try:
                m6b = m6b_curvature(h_clean, h_pert, sub_net, t_emb, adaln, n_points=5)
                (pair_dir / "m6b_curvature.json").write_text(json.dumps(m6b, indent=2, default=str))
                m6b_rows.append({
                    "condition": cond, "episode": ep, "group": grp,
                    "curvature_mean": m6b["curvature_mean_rad_per_unit"],
                    "total_rotation_deg": m6b["total_rotation_deg"],
                    "cosine_01": m6b["cosines"][0] if len(m6b["cosines"]) > 0 else float("nan"),
                    "cosine_12": m6b["cosines"][1] if len(m6b["cosines"]) > 1 else float("nan"),
                    "cosine_23": m6b["cosines"][2] if len(m6b["cosines"]) > 2 else float("nan"),
                    "cosine_34": m6b["cosines"][3] if len(m6b["cosines"]) > 3 else float("nan"),
                })
                print(f"    M6b curv={m6b['curvature_mean_rad_per_unit']:.4f} rad/unit (total_rot={m6b['total_rotation_deg']:.2f}°)")
            except Exception as e:
                print(f"    M6b ERROR: {e}")

    elapsed = time.time() - start
    print(f"\nCompleted {len(pairs)} pairs in {elapsed:.1f}s")

    # ---- Summaries ----
    def _summary(rows, fields):
        grouped = defaultdict(list)
        for r in rows:
            grouped[(r["condition"], r["group"])].append(r)
        out = []
        for (cond, grp), items in sorted(grouped.items()):
            entry = {"condition": cond, "group": grp, "n": len(items)}
            for f in fields:
                vals = [item[f] for item in items if not math.isnan(item[f])]
                entry[f"{f}_mean"] = float(np.mean(vals)) if vals else float("nan")
                entry[f"{f}_std"] = float(np.std(vals)) if vals else float("nan")
            out.append(entry)
        return out

    def _save_csv(path, fieldnames, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    if do_m1 and m1_rows:
        fields = ["s_dh", "s_random_mean", "s_random_std", "fragility_ratio"]
        s = _summary(m1_rows, fields)
        _save_csv(out_root / "m1_sensitivity_summary.csv",
                  ["condition", "group", "n"] + [f"{x}_mean" for x in fields] + [f"{x}_std" for x in fields], s)
        _save_csv(out_root / "m1_sensitivity_detail.csv",
                  ["condition", "episode", "group"] + fields, m1_rows)

        print("\n[M1] Fragility Ratio (higher = Δh is more sensitive than random → Fawzi predicts flipped > preserved)")
        print(f"{'condition':30s} {'group':12s} {'n':>3s} {'ratio':>8s} {'s_dh':>8s} {'s_rand':>8s}")
        for r in s:
            print(f"{r['condition']:30s} {r['group']:12s} {r['n']:>3d} "
                  f"{r['fragility_ratio_mean']:>8.3f} {r['s_dh_mean']:>8.4f} {r['s_random_mean_mean']:>8.4f}")

    if do_m6b and m6b_rows:
        fields = ["curvature_mean", "total_rotation_deg"]
        s = _summary(m6b_rows, fields)
        _save_csv(out_root / "m6b_curvature_summary.csv",
                  ["condition", "group", "n"] + [f"{x}_mean" for x in fields] + [f"{x}_std" for x in fields], s)
        _save_csv(out_root / "m6b_curvature_detail.csv",
                  ["condition", "episode", "group", "curvature_mean", "total_rotation_deg",
                   "cosine_01", "cosine_12", "cosine_23", "cosine_34"], m6b_rows)

        print("\n[M6b] Curvature (lower = flatter boundary → Fawzi predicts flipped < preserved)")
        print(f"{'condition':30s} {'group':12s} {'n':>3s} {'curvature':>10s} {'total_rot':>10s}")
        for r in s:
            print(f"{r['condition']:30s} {r['group']:12s} {r['n']:>3d} "
                  f"{r['curvature_mean_mean']:>10.6f} {r['total_rotation_deg_mean']:>10.4f}")

    # Combined CSV
    if do_m1 and do_m6b and m1_rows and m6b_rows:
        m6b_lookup = {(r["condition"], r["episode"]): r for r in m6b_rows}
        combined = []
        for r in m1_rows:
            key = (r["condition"], r["episode"])
            m6b_r = m6b_lookup.get(key, {})
            combined.append({**r, **{f"m6b_{k}": m6b_r.get(k, float("nan"))
                                     for k in ["curvature_mean", "total_rotation_deg"]}})
        _save_csv(out_root / "m1_m6b_combined.csv",
                  ["condition", "episode", "group", "fragility_ratio", "s_dh", "s_random_mean",
                   "m6b_curvature_mean", "m6b_total_rotation_deg"], combined)

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
