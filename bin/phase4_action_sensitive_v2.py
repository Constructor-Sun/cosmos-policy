#!/usr/bin/env python3
"""
Phase 4 v2: Does the action-sensitive latent subspace contain all 7 perturbations?

Tests H3 from Subspace.md:
  "The harmfulness of perturb-specific subspaces comes from their overlap
   with action-critical directions."

Two methods to define the action-sensitive subspace S_action:
  A. PLS: find latent directions in delta_h that maximally covary with delta_a
  B. PCA+sensitivity: PCA on delta_h, rank PCs by correlation with action_error

Then for each perturbation type, measure containment:
  R[p] = ||Proj_{S_action} delta_h_p||^2 / ||delta_h_p||^2

Reads Phase 2 artifacts; no new forward passes.
"""

from __future__ import annotations

import argparse, json, math, os, sys, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR_DEFAULT = str(
    ROOT / "experiments/phase2_angular_cosmos/"
    "kitchen_scene4_seed7_all_perturb_preserved_flipped_last"
)
OUT_DIR_DEFAULT = str(ROOT / "experiments/phase4_action_sensitive_v2")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(results_dir: Path):
    """Load delta_h, delta_a, and metadata for all pairs."""
    delta_h_list, delta_a_list, meta_list = [], [], []
    for mpath in sorted(results_dir.glob("*/*/metrics.json")):
        rec = json.loads(mpath.read_text())
        if rec["group"] not in ("preserved", "flipped"):
            continue
        ep_dir = mpath.parent

        # Load hidden states
        for slot in ["action"]:
            h_clean = _load_hidden(ep_dir, slot, "clean")
            h_pert = _load_hidden(ep_dir, slot, "pert")
        a_clean = np.load(ep_dir / "action_clean.npy").astype(np.float64).reshape(-1)
        a_pert = np.load(ep_dir / "action_pert.npy").astype(np.float64).reshape(-1)

        delta_h_list.append(h_pert - h_clean)
        delta_a_list.append(a_pert - a_clean)
        meta_list.append({
            "condition": rec["condition"], "episode": rec["episode"],
            "group": rec["group"],
        })

    dh = np.array(delta_h_list, dtype=np.float64)
    da = np.array(delta_a_list, dtype=np.float64)
    print(f"Loaded {len(dh)} pairs: dh {dh.shape}, da {da.shape}")
    return dh, da, meta_list


def _load_hidden(ep_dir: Path, slot_name: str, clean_or_pert: str):
    d = torch.load(str(ep_dir / f"hidden_{slot_name}_{clean_or_pert}.pt"),
                   map_location="cpu", weights_only=True)
    t = list(d.values())[0]
    return t.float().reshape(-1).numpy()


# ---------------------------------------------------------------------------
# PCA via Gram matrix
# ---------------------------------------------------------------------------

def gram_pca(X: np.ndarray, max_k: int = 100):
    """PCA when N < D via Gram matrix. Returns scores, loadings, evr."""
    N, D = X.shape
    K = min(N, max_k)
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T  # (N, N)
    evals, evecs = np.linalg.eigh(G)
    evals = evals[::-1][:K]
    evecs = evecs[:, ::-1][:, :K]
    mask = evals > evals.max() * 1e-10
    K = int(mask.sum())
    evals, evecs = evals[:K], evecs[:, :K]
    inv_sqrt = 1.0 / np.sqrt(evals * (N - 1) + EPS)
    loadings = Xc.T @ evecs @ np.diag(inv_sqrt)
    scores = evecs @ np.diag(np.sqrt(evals * (N - 1)))
    total = float(evals.sum())
    evr = np.array([float(evals[i] / total) for i in range(K)])
    return scores.astype(np.float64), loadings.astype(np.float64), evr


# ---------------------------------------------------------------------------
# Method A: PLS-based action-sensitive subspace
# ---------------------------------------------------------------------------

def method_a_pls(dh, da, n_pca=50, n_pls=10):
    """Define S_action via PLS(delta_h -> delta_a)."""
    print("\n" + "=" * 60)
    print(f"Method A: PLS action-sensitive subspace (PCA={n_pca}, PLS={n_pls})")
    print("=" * 60)

    # Reduce dh via PCA
    scores, loadings_pca, evr = gram_pca(dh, max_k=n_pca)
    print(f"  PCA: {scores.shape[1]} PCs, top-5 evr: {evr[:5].round(4)}")

    # Standardize
    X = (scores - scores.mean(0)) / (scores.std(0) + EPS)
    Y = (da - da.mean(0)) / (da.std(0) + EPS)

    # PLS
    pls = PLSRegression(n_components=n_pls, scale=False)
    pls.fit(X, Y)
    pls_loadings = pls.x_loadings_  # (K_pca, n_pls)

    # R^2
    Y_pred = pls.predict(X)
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum(Y ** 2)
    r2 = 1 - ss_res / (ss_tot + EPS)
    print(f"  PLS R² (delta_a prediction): {r2:.4f}")

    # Per-dimension R²
    r2_per_dim = []
    for j in range(Y.shape[1]):
        ss_res_j = np.sum((Y[:, j] - Y_pred[:, j]) ** 2)
        ss_tot_j = np.sum(Y[:, j] ** 2)
        r2_per_dim.append(1 - ss_res_j / (ss_tot_j + EPS))
    print(f"  Per-dim R²: mean={np.mean(r2_per_dim):.4f}, "
          f"min={np.min(r2_per_dim):.4f}, max={np.max(r2_per_dim):.4f}")

    # Projection: for each sample, what fraction of delta_h (in PCA space) is in S_action?
    # S_action is spanned by pls_loadings columns
    proj_sq = np.sum((scores @ pls_loadings) ** 2, axis=1)
    total_sq = np.sum(scores ** 2, axis=1)
    proj_ratio = proj_sq / (total_sq + EPS)

    # Random baseline
    rand_ratios = []
    for _ in range(200):
        Q, _ = np.linalg.qr(np.random.randn(scores.shape[1], n_pls))
        rand_proj = np.sum((scores @ Q) ** 2, axis=1)
        rand_ratios.append(np.mean(rand_proj / (total_sq + EPS)))
    rand_mean, rand_std = np.mean(rand_ratios), np.std(rand_ratios)

    print(f"  Random baseline proj_ratio: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  PLS proj_ratio mean: {np.mean(proj_ratio):.4f}")
    print(f"  Ratio: {np.mean(proj_ratio) / (rand_mean + EPS):.2f}x")

    return {
        "scores": scores, "pls_loadings": pls_loadings, "proj_ratio": proj_ratio,
        "r2": r2, "r2_per_dim_mean": float(np.mean(r2_per_dim)),
        "rand_mean": rand_mean, "rand_std": rand_std,
        "pls_mean_ratio": float(np.mean(proj_ratio)),
        "ratio_vs_random": float(np.mean(proj_ratio) / (rand_mean + EPS)),
    }


# ---------------------------------------------------------------------------
# Method B: PCA + action-error-weighted sensitivity
# ---------------------------------------------------------------------------

def method_b_pca_sensitivity(dh, da, n_pca=50, top_k=10):
    """Define S_action via PCA sensitivity ranking.

    PC directions where projection magnitude correlates with action_error
    are considered action-sensitive.
    """
    print("\n" + "=" * 60)
    print(f"Method B: PCA + sensitivity ranking (PCA={n_pca}, top_k={top_k})")
    print("=" * 60)

    scores, loadings, evr = gram_pca(dh, max_k=n_pca)
    K = scores.shape[1]
    print(f"  PCA: {K} PCs")

    # Action error per sample
    action_error = np.linalg.norm(da, axis=1)

    # For each PC, compute correlation between |score| and action_error
    sensitivity = []
    for k in range(K):
        corr = np.corrcoef(np.abs(scores[:, k]), action_error)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        sensitivity.append((k, corr, evr[k]))

    # Sort by |correlation| descending
    sensitivity.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  Top-{top_k} PCs by action_error correlation:")
    print(f"  {'Rank':<6} {'PC':<6} {'corr':<10} {'evr':<10}")
    for rank, (pc_idx, corr, ev) in enumerate(sensitivity[:top_k]):
        print(f"  {rank+1:<6} {pc_idx:<6} {corr:<10.4f} {ev:<10.4f}")

    # Build S_action from top_k most sensitive PCs
    top_indices = [s[0] for s in sensitivity[:top_k]]
    sensitive_loadings = loadings[:, top_indices]  # (D, top_k) in original space
    # In PCA score space, S_action basis = identity on top_indices
    basis_in_score_space = np.zeros((K, top_k))
    for i, idx in enumerate(top_indices):
        basis_in_score_space[idx, i] = 1.0

    # Projection
    proj_sq = np.sum((scores @ basis_in_score_space) ** 2, axis=1)
    total_sq = np.sum(scores ** 2, axis=1)
    proj_ratio = proj_sq / (total_sq + EPS)

    # Random baseline
    rand_ratios = []
    for _ in range(200):
        rand_indices = np.random.choice(K, top_k, replace=False)
        rand_basis = np.zeros((K, top_k))
        for i, idx in enumerate(rand_indices):
            rand_basis[idx, i] = 1.0
        rand_proj = np.sum((scores @ rand_basis) ** 2, axis=1)
        rand_ratios.append(np.mean(rand_proj / (total_sq + EPS)))
    rand_mean, rand_std = np.mean(rand_ratios), np.std(rand_ratios)

    print(f"  Random ({top_k} PCs) proj_ratio: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  Sensitivity-ranked proj_ratio mean: {np.mean(proj_ratio):.4f}")
    print(f"  Ratio: {np.mean(proj_ratio) / (rand_mean + EPS):.2f}x")

    return {
        "scores": scores, "basis": basis_in_score_space, "proj_ratio": proj_ratio,
        "sensitivity": sensitivity,
        "rand_mean": rand_mean, "rand_std": rand_std,
        "mean_ratio": float(np.mean(proj_ratio)),
        "ratio_vs_random": float(np.mean(proj_ratio) / (rand_mean + EPS)),
    }


# ---------------------------------------------------------------------------
# Per-perturbation containment analysis
# ---------------------------------------------------------------------------

def per_perturbation_containment(
    dh, da, meta, result_a, result_b, out_dir: Path
):
    """For each perturbation type, report containment in S_action."""
    print("\n" + "=" * 60)
    print("Per-perturbation containment in S_action")
    print("=" * 60)

    conditions = sorted(set(m["condition"] for m in meta))

    for method_name, res in [("PLS (A)", result_a), ("PCA+sens (B)", result_b)]:
        print(f"\n--- {method_name} ---")
        proj_ratio = res["proj_ratio"]

        rows = []
        print(f"  {'Condition':30s} {'Group':12s} {'N':>4s} {'mean_proj':>10s} {'std_proj':>10s}")
        print(f"  {'-'*65}")

        for cond in conditions:
            for grp in ["flipped", "preserved"]:
                idx = [i for i, m in enumerate(meta)
                       if m["condition"] == cond and m["group"] == grp]
                if not idx:
                    continue
                ratios = proj_ratio[idx]
                print(f"  {cond:30s} {grp:12s} {len(idx):>4d} "
                      f"{np.mean(ratios):>10.4f} {np.std(ratios):>10.4f}")
                rows.append({
                    "method": method_name,
                    "condition": cond, "group": grp, "n": len(idx),
                    "mean_proj_ratio": float(np.mean(ratios)),
                    "std_proj_ratio": float(np.std(ratios)),
                })

        _save_csv(out_dir / f"containment_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                  ["method", "condition", "group", "n", "mean_proj_ratio", "std_proj_ratio"], rows)

    # Also: cross-perturbation containment matrix (Phase B from Subspace.md)
    print("\n" + "=" * 60)
    print("Cross-perturbation containment matrix (Phase B style)")
    print("  R[p -> q] = ||Proj_{S_p} delta_q||^2 / ||delta_q||^2")
    print("=" * 60)

    # For each perturbation p, compute S_p = PCA of its delta_h (all samples)
    scores_all, _, evr_all = gram_pca(dh, max_k=50)
    K_scores = scores_all.shape[1]

    # Per-perturbation PCA subspace (top-k PCs of that perturbation's delta_h)
    k_subspace = 4
    pert_subspaces = {}
    for cond in conditions:
        idx = [i for i, m in enumerate(meta) if m["condition"] == cond]
        if len(idx) < 5:
            continue
        pert_scores = scores_all[idx]
        pert_scores_c = pert_scores - pert_scores.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(pert_scores_c, full_matrices=False)
        pert_subspaces[cond] = Vt[:k_subspace, :].T  # (K_scores, k)

    # Cross-reconstruction matrix
    pert_list = sorted(pert_subspaces.keys())
    n_pert = len(pert_list)
    cross_matrix = np.zeros((n_pert, n_pert))
    cross_details = []

    for i, p_train in enumerate(pert_list):
        basis = pert_subspaces[p_train]  # (K, k)
        for j, p_test in enumerate(pert_list):
            idx = [ii for ii, m in enumerate(meta) if m["condition"] == p_test]
            test_scores = scores_all[idx]
            proj_sq = np.sum((test_scores @ basis) ** 2, axis=1)
            total_sq = np.sum(test_scores ** 2, axis=1)
            ratios = proj_sq / (total_sq + EPS)
            cross_matrix[i, j] = np.mean(ratios)
            cross_details.append({
                "train_pert": p_train, "test_pert": p_test, "k": k_subspace,
                "mean_proj_ratio": float(np.mean(ratios)),
                "n_test": len(idx),
            })

    _save_csv(out_dir / "cross_reconstruction.csv",
              ["train_pert", "test_pert", "k", "mean_proj_ratio", "n_test"],
              cross_details)

    header = "train\\test"
    print(f"\n  R[train → test] (k={k_subspace}):")
    print(f"  {header:25s}", end="")
    for p in pert_list:
        print(f"{p[:15]:>15s}", end="")
    print()
    for i, p_train in enumerate(pert_list):
        print(f"  {p_train:25s}", end="")
        for j in range(n_pert):
            diag = " *" if i == j else "  "
            print(f"{cross_matrix[i, j]:>15.4f}{diag}", end="")
        print()

    mean_diag = np.mean(np.diag(cross_matrix))
    mean_off = (cross_matrix.sum() - np.trace(cross_matrix)) / (n_pert * (n_pert - 1) + EPS)
    print(f"\n  Mean diagonal (self): {mean_diag:.4f}")
    print(f"  Mean off-diagonal (cross): {mean_off:.4f}")
    print(f"  Diagonal / off-diagonal ratio: {mean_diag / (mean_off + EPS):.2f}x")

    if mean_diag / (mean_off + EPS) > 2:
        print(f"  → Perturb-SPECIFIC subspaces (diagonal dominates)")
    else:
        print(f"  → Shared/global subspace (off-diagonal comparable to diagonal)")

    return cross_matrix, pert_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--output-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--n-pca", type=int, default=50)
    p.add_argument("--n-pls", type=int, default=10)
    p.add_argument("--top-k-sens", type=int, default=10)
    p.add_argument("--k-subspace", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    dh, da, meta = load_all_data(Path(args.results_dir))
    if len(dh) == 0:
        print("No data."); return

    # Count
    from collections import Counter
    for (c, g), n in Counter((m["condition"], m["group"]) for m in meta).items():
        print(f"  {c:30s} {g:12s} {n}")

    # Method A: PLS
    result_a = method_a_pls(dh, da, n_pca=args.n_pca, n_pls=args.n_pls)

    # Method B: PCA + sensitivity
    result_b = method_b_pca_sensitivity(dh, da, n_pca=args.n_pca, top_k=args.top_k_sens)

    # Per-perturbation containment
    cross_matrix, pert_list = per_perturbation_containment(
        dh, da, meta, result_a, result_b, out_dir
    )

    # Summary
    summary = {
        "method_a_pls": {
            "r2": result_a["r2"],
            "r2_per_dim_mean": result_a["r2_per_dim_mean"],
            "pls_mean_proj_ratio": result_a["pls_mean_ratio"],
            "random_mean_proj_ratio": result_a["rand_mean"],
            "ratio_vs_random": result_a["ratio_vs_random"],
        },
        "method_b_pca_sensitivity": {
            "mean_proj_ratio": result_b["mean_ratio"],
            "random_mean_proj_ratio": result_b["rand_mean"],
            "ratio_vs_random": result_b["ratio_vs_random"],
            "top_sensitive_pcs": [
                {"pc": int(s[0]), "corr": float(s[1]), "evr": float(s[2])}
                for s in result_b["sensitivity"][:args.top_k_sens]
            ],
        },
        "cross_reconstruction": {
            "k": args.k_subspace,
            "mean_diagonal": float(np.mean(np.diag(cross_matrix))),
            "mean_off_diagonal": float(
                (cross_matrix.sum() - np.trace(cross_matrix)) /
                (len(pert_list) * (len(pert_list) - 1) + EPS)
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
