#!/usr/bin/env python3
"""
Phase 4 — Jacobian: Define the action-sensitive subspace S_action via the
Jacobian ∂a/∂h of the FinalLayer linear readout.

This is a MODEL PROPERTY, independent of any perturbation.

Method:
  1. Extract W = final_layer.linear.weight from the Cosmos-Policy checkpoint
  2. Determine which output channels and grid positions map to the action
  3. S_action = span of the relevant rows of W, replicated across grid positions
  4. Project each perturbation's delta_h onto S_action
  5. Report containment per perturbation type

The FinalLayer chain:
  h [B,1,14,14,2048] → LayerNorm → AdaLN(scale,shift) → Linear(W:64×2048)
  → output [B,1,14,14,64] → unpatchify → latent [B,16,1,28,28]
  → flatten → take first 112 → action

Grid-to-action mapping:
  - Only h∈{0,1}, w∈{0..13} contribute (28 of 196 grid positions)
  - For each such position, only W rows 0, 16, 32, 48 (c=0 for each p1,p2) contribute
  - Each grid position independently maps through these 4 weight vectors
"""

from __future__ import annotations

import argparse, json, math, os, sys, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]

CKPT_PATH = "/data3/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
RESULTS_DIR_DEFAULT = str(
    ROOT / "experiments/phase2_angular_cosmos/"
    "kitchen_scene4_seed7_all_perturb_preserved_flipped_last"
)
OUT_DIR_DEFAULT = str(ROOT / "experiments/phase4_jacobian")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(results_dir: Path):
    dh_list, da_list, meta_list = [], [], []
    for mpath in sorted(results_dir.glob("*/*/metrics.json")):
        rec = json.loads(mpath.read_text())
        if rec["group"] not in ("preserved", "flipped"):
            continue
        ep_dir = mpath.parent
        for slot in ["action"]:
            h_clean = _load_hidden(ep_dir, slot, "clean")
            h_pert = _load_hidden(ep_dir, slot, "pert")
        a_clean = np.load(ep_dir / "action_clean.npy").astype(np.float64).reshape(-1)
        a_pert = np.load(ep_dir / "action_pert.npy").astype(np.float64).reshape(-1)
        dh_list.append(h_pert - h_clean)
        da_list.append(a_pert - a_clean)
        meta_list.append({
            "condition": rec["condition"], "episode": rec["episode"],
            "group": rec["group"],
        })
    dh = np.array(dh_list, dtype=np.float64)
    da = np.array(da_list, dtype=np.float64)
    print(f"Loaded {len(dh)} pairs: dh {dh.shape}, da {da.shape}")
    return dh, da, meta_list


def _load_hidden(ep_dir, slot_name, clean_or_pert):
    d = torch.load(str(ep_dir / f"hidden_{slot_name}_{clean_or_pert}.pt"),
                   map_location="cpu", weights_only=True)
    t = list(d.values())[0]
    return t.float().reshape(-1).numpy()


# ---------------------------------------------------------------------------
# Jacobian analysis
# ---------------------------------------------------------------------------

def compute_jacobian_action_subspace(ckpt_path: str):
    """Extract the action-sensitive subspace from the FinalLayer weights.

    Returns:
        basis_vectors: (4, 2048) — the 4 weight vectors defining S_action
        contributing_positions: list of (h, w) tuples for the 28 grid positions
    """
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    W = ckpt["net.final_layer.linear.weight"].float().numpy()  # [64, 2048]
    print(f"  FinalLayer linear weight: {W.shape}")

    # The 4 output channels that map to action (c=0 for each p1,p2 combo):
    # p1=0,p2=0 → output channel (0*2+0)*16+0 = 0
    # p1=0,p2=1 → output channel (0*2+1)*16+0 = 16
    # p1=1,p2=0 → output channel (1*2+0)*16+0 = 32
    # p1=1,p2=1 → output channel (1*2+1)*16+0 = 48
    action_channels = [0, 16, 32, 48]
    basis_vectors = W[action_channels, :]  # [4, 2048]
    print(f"  Action-relevant W rows: {action_channels} → basis {basis_vectors.shape}")

    # Check effective rank
    U, S, Vt = np.linalg.svd(basis_vectors, full_matrices=False)
    print(f"  Singular values of basis: {S}")
    print(f"  Effective rank: {sum(S > S[0] * 1e-6)}")

    # Contributing grid positions: h ∈ {0, 1}, w ∈ {0..13}
    contributing = [(h, w) for h in [0, 1] for w in range(14)]
    print(f"  Contributing grid positions: {len(contributing)} out of 196 (14×14)")

    return basis_vectors, contributing


def project_delta_h_onto_jacobian_subspace(
    dh: np.ndarray,           # (N, 401408) flattened delta_h
    basis_vectors: np.ndarray,  # (4, 2048)
    contributing: list,       # list of (h, w) tuples
) -> np.ndarray:
    """Project each delta_h onto the Jacobian-defined action-sensitive subspace.

    Returns proj_ratio per sample: fraction of ||delta_h||² in S_action.

    S_action = for each contributing grid position, span of 4 basis vectors.
    For non-contributing positions, sensitivity is zero.
    """
    N = dh.shape[0]
    # Reshape to [N, 196, 2048]
    dh_grid = dh.reshape(N, 14, 14, 2048)

    # Orthonormalize basis for clean projection
    # basis_vectors: [4, 2048], SVD → U:[4,4], S:[4], Vt:[4,2048]
    # Right singular vectors Vt rows are directions in 2048-dim hidden space
    U, S, Vt = np.linalg.svd(basis_vectors, full_matrices=False)
    r = int(sum(S > S[0] * 1e-6))
    Q = Vt[:r, :].T  # [2048, r] — orthonormal basis for S_action per grid position
    print(f"  Action subspace rank per grid position: {r}")

    total_sq = np.sum(dh_grid ** 2, axis=(1, 2, 3))  # [N]

    proj_sq = np.zeros(N)
    for h, w in contributing:
        dh_hw = dh_grid[:, h, w, :]  # [N, 2048]
        # Project onto span(Q)
        coeffs = dh_hw @ Q  # [N, r]
        proj = coeffs @ Q.T  # [N, 2048]
        proj_sq += np.sum(proj ** 2, axis=1)

    proj_ratio = proj_sq / (total_sq + EPS)
    return proj_ratio


def project_delta_h_weighted(
    dh: np.ndarray,
    basis_vectors: np.ndarray,
    contributing: list,
) -> np.ndarray:
    """Variant: weight each grid position by its Jacobian sensitivity.

    Uses the singular values of the basis as weights — grid positions whose
    hidden state more strongly affects the action get higher weight.
    """
    N = dh.shape[0]
    dh_grid = dh.reshape(N, 14, 14, 2048)

    U, S, Vt = np.linalg.svd(basis_vectors, full_matrices=False)
    r = int(sum(S > S[0] * 1e-6))
    Q = Vt[:r, :].T  # [2048, r]
    weights = S[:r]  # [r]

    total_sq = np.sum(dh_grid ** 2, axis=(1, 2, 3))

    proj_sq = np.zeros(N)
    for h, w in contributing:
        dh_hw = dh_grid[:, h, w, :]
        coeffs = dh_hw @ Q  # [N, r]
        weighted_coeffs = coeffs * weights[None, :]  # scale by singular values
        proj = weighted_coeffs @ Q.T
        proj_sq += np.sum(proj ** 2, axis=1)

    proj_ratio = proj_sq / (total_sq + EPS)
    return proj_ratio


# ---------------------------------------------------------------------------
# Full Jacobian: include LayerNorm gradient at each operating point
# ---------------------------------------------------------------------------

def layernorm_jacobian(h: np.ndarray) -> np.ndarray:
    """Compute ∂LN(h)/∂h for a single 2048-dim vector.

    LayerNorm (elementwise_affine=False):
      y = (h - μ) / σ
    where μ = mean(h), σ = std(h).

    Jacobian: ∂y/∂h = (I - 1/D * 1@1^T - y@y^T) / σ
    """
    D = h.shape[-1]
    mu = h.mean(axis=-1, keepdims=True)
    sigma = h.std(axis=-1, keepdims=True)
    y = (h - mu) / (sigma + EPS)

    # For batched input
    I = np.eye(D)
    ones = np.ones((D, 1)) / D
    yyT = y[..., None] @ y[..., None].swapaxes(-2, -1) / D  # handle batch dims

    # [batch, D, D] Jacobian
    J_ln = (I[None, :, :] - np.outer(ones.squeeze(), ones.squeeze()) - yyT) / (sigma[..., None] + EPS)
    return J_ln


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--output-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--ckpt", default=CKPT_PATH)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    dh, da, meta = load_all_data(Path(args.results_dir))

    # 2. Compute Jacobian-based action subspace
    basis, contributing = compute_jacobian_action_subspace(args.ckpt)

    # 3. Project all delta_h
    print(f"\n{'='*60}")
    print(f"Projecting delta_h onto Jacobian-defined S_action")
    print(f"{'='*60}")

    proj_ratio = project_delta_h_onto_jacobian_subspace(dh, basis, contributing)
    proj_ratio_w = project_delta_h_weighted(dh, basis, contributing)

    # 4. Random baseline
    # S_action uses 28 grid positions × rank(basis) ≤ 4 dimensions per position
    U, S, _ = np.linalg.svd(basis, full_matrices=False)
    r = int(sum(S > S[0] * 1e-6))
    total_dim = 196 * 2048
    subspace_dim = len(contributing) * r
    expected_random = subspace_dim / total_dim
    print(f"\n  S_action dimension: {subspace_dim} / {total_dim} = {expected_random:.6f}")
    print(f"  This is the expected random containment (any subspace of this dimension)")

    # Also empirical random baseline
    rand_ratios = []
    for _ in range(100):
        rand_basis = np.random.randn(4, 2048)
        rand_basis = rand_basis / np.linalg.norm(rand_basis, axis=1, keepdims=True)
        rand_r = project_delta_h_onto_jacobian_subspace(dh, rand_basis, contributing)
        rand_ratios.append(np.mean(rand_r))
    rand_mean, rand_std = np.mean(rand_ratios), np.std(rand_ratios)

    print(f"  Empirical random baseline: {rand_mean:.6f} ± {rand_std:.6f}")
    print(f"  Jacobian S_action containment (unweighted): {np.mean(proj_ratio):.6f}")
    print(f"  Jacobian S_action containment (weighted):   {np.mean(proj_ratio_w):.6f}")
    print(f"  Ratio (unweighted / random): {np.mean(proj_ratio)/(rand_mean+EPS):.2f}x")

    # 5. Per-perturbation × group breakdown
    print(f"\n{'='*60}")
    print(f"Per-perturbation containment in Jacobian S_action (unweighted)")
    print(f"{'='*60}")

    conditions = sorted(set(m["condition"] for m in meta))
    print(f"  {'Condition':30s} {'Group':12s} {'N':>4s} {'containment':>12s}")
    print(f"  {'-'*60}")

    rows = []
    for cond in conditions:
        for grp in ["flipped", "preserved"]:
            idx = [i for i, m in enumerate(meta) if m["condition"] == cond and m["group"] == grp]
            if not idx:
                continue
            ratios = proj_ratio[idx]
            ratios_w = proj_ratio_w[idx]
            print(f"  {cond:30s} {grp:12s} {len(idx):>4d} {np.mean(ratios):>12.6f}  (w: {np.mean(ratios_w):.6f})")
            rows.append({
                "condition": cond, "group": grp, "n": len(idx),
                "containment": float(np.mean(ratios)),
                "containment_std": float(np.std(ratios)),
                "containment_weighted": float(np.mean(ratios_w)),
            })

    _save_csv(out_dir / "jacobian_containment.csv",
              ["condition", "group", "n", "containment", "containment_std", "containment_weighted"],
              rows)

    # 6. Compare: which grid positions contribute most to perturbation delta_h?
    print(f"\n{'='*60}")
    print(f"Per-grid-position analysis: where do perturbations hit?")
    print(f"{'='*60}")

    dh_grid = dh.reshape(140, 14, 14, 2048)
    pos_norm = np.linalg.norm(dh_grid, axis=3)  # [140, 14, 14]
    mean_pos_norm = pos_norm.mean(axis=0)  # [14, 14]

    # Normalize
    norm_14x14 = mean_pos_norm / mean_pos_norm.sum()

    print(f"  Mean ||delta_h|| per grid position (14×14), normalized:")
    print(f"  {'':>6s}", end="")
    for w in range(14):
        flag = "c" if w < 14 else " "
        print(f"{f'w={w}' + flag:>10s}", end="")
    print()
    for h in range(14):
        flag = "c" if h < 2 else " "
        print(f"  {f'h={h}' + flag:>6s}", end="")
        for w in range(14):
            marker = " *" if h < 2 else "  "
            print(f"{norm_14x14[h, w]:>8.4f}{marker}", end="")
        print()
    print(f"  (* = contributing to action, c = contributing row/col)")

    contrib_norm = sum(norm_14x14[h, w] for h, w in contributing)
    print(f"\n  Total norm in contributing grid positions: {contrib_norm:.4f} ({contrib_norm*100:.1f}%)")
    print(f"  Total norm in non-contributing positions: {1-contrib_norm:.4f} ({(1-contrib_norm)*100:.1f}%)")

    # 7. Summary
    summary = {
        "method": "Jacobian ∂a/∂h via FinalLayer linear weights",
        "action_channels": [0, 16, 32, 48],
        "contributing_grid_positions": len(contributing),
        "subspace_rank_per_position": r,
        "total_subspace_dim": subspace_dim,
        "total_hidden_dim": total_dim,
        "expected_random_containment": float(expected_random),
        "empirical_random_containment": float(rand_mean),
        "jacobian_containment_unweighted": float(np.mean(proj_ratio)),
        "jacobian_containment_weighted": float(np.mean(proj_ratio_w)),
        "ratio_vs_random": float(np.mean(proj_ratio) / (rand_mean + EPS)),
        "contributing_grid_norm_fraction": float(contrib_norm),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. {out_dir / 'summary.json'}")


def _save_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
