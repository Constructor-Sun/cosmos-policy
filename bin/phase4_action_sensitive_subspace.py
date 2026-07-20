#!/usr/bin/env python3
"""
Phase 4: Action-Sensitive Subspace Analysis.

Core hypothesis:
  "All perturbations (Camera, Lighting, Robot Pose, etc.) ultimately push the
   latent state along shared 'action-critical directions' — directions in the
   DiT hidden space that are maximally sensitive for action prediction."

Two-part test:
  1. FIND the action-sensitive subspace: a low-dimensional subspace S of the
     hidden-state space where perturbations cause maximal action-output change.
  2. CHECK whether flipped Δh vectors from ALL perturbation types project
     strongly onto S (vs. preserved Δh).

Method:
  - PLS (Partial Least Squares) between Δh and Δa to find latent directions
    that covary maximally with action output changes.
  - Cross-perturbation validation: hold out each perturbation type in turn,
    fit PLS on the rest, test on the held-out.
  - Compare projection ratios for flipped vs. preserved.

Reads Phase 2 artifacts; no new forward passes needed.
"""

from __future__ import annotations

import argparse, json, math, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR_DEFAULT = str(
    ROOT / "experiments/phase2_angular_cosmos/"
    "kitchen_scene4_seed7_all_perturb_preserved_flipped_last"
)
OUT_DIR_DEFAULT = str(ROOT / "experiments/phase4_action_sensitive")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_pairs(results_dir: Path) -> list[dict]:
    """Load all Phase 2 pairs (preserved + flipped)."""
    pairs = []
    for mpath in sorted(results_dir.glob("*/*/metrics.json")):
        rec = json.loads(mpath.read_text())
        if rec["group"] not in ("preserved", "flipped"):
            continue
        rec["_dir"] = mpath.parent
        pairs.append(rec)
    print(f"Loaded {len(pairs)} pairs from {results_dir}")
    return pairs


def load_hidden(pair: dict, slot_name: str, clean_or_pert: str, device="cpu") -> torch.Tensor:
    """Load hidden_{slot_name}_{clean_or_pert}.pt, return flattened float32 tensor."""
    path = pair["_dir"] / f"hidden_{slot_name}_{clean_or_pert}.pt"
    d = torch.load(str(path), map_location=device, weights_only=True)
    t = list(d.values())[0]
    return t.float().reshape(-1)


def load_action(pair: dict, slot: str) -> np.ndarray:
    """Load action_{slot}.npy, return flattened float64 array."""
    return np.load(pair["_dir"] / f"action_{slot}.npy").astype(np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Core: action-sensitive subspace via PLS
# ---------------------------------------------------------------------------

def compute_delta_h_pca(
    delta_h: np.ndarray,  # (N, D) with D large (401408)
    max_components: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on delta_h via Gram matrix (N << D case).

    Returns:
        scores:   (N, K) — PC scores
        loadings: (D, K) — PC loadings (directions in original space)
        evr:      (K,)   — explained variance ratio
    """
    N, D = delta_h.shape
    K = min(N, max_components)

    # Center
    mu = delta_h.mean(axis=0, keepdims=True)
    Xc = delta_h - mu  # (N, D)

    # Gram matrix G = Xc @ Xc.T  → (N, N)
    G = Xc @ Xc.T  # (N, N)
    eigenvals, eigenvecs = np.linalg.eigh(G)  # ascending

    # Descending order
    eigenvals = eigenvals[::-1][:K]
    eigenvecs = eigenvecs[:, ::-1][:, :K]  # (N, K)

    # Drop near-zero
    mask = eigenvals > eigenvals.max() * 1e-10
    K = int(mask.sum())
    eigenvals = eigenvals[:K]
    eigenvecs = eigenvecs[:, :K]

    # Loadings = Xc.T @ V @ diag(1/sqrt(lambda * (N-1)))
    # PCA scores = Xc @ loadings = V * sqrt(lambda*(N-1))
    inv_sqrt_lambda = 1.0 / np.sqrt(eigenvals * (N - 1) + EPS)
    loadings = Xc.T @ eigenvecs @ np.diag(inv_sqrt_lambda)  # (D, K)
    scores = eigenvecs @ np.diag(np.sqrt(eigenvals * (N - 1)))  # (N, K)

    total_var = float(eigenvals.sum())
    evr = np.array([float(eigenvals[i] / total_var) for i in range(K)])

    # Keep all eigenvalues for total_variance reference
    all_eigenvals = np.linalg.eigvalsh(G)[::-1]
    all_total = float(all_eigenvals.sum())
    evr_full = np.array([float(all_eigenvals[i] / all_total) for i in range(K)])

    return scores.astype(np.float64), loadings.astype(np.float64), evr_full


def fit_action_sensitive_subspace(
    delta_h_scores: np.ndarray,  # (N_train, K_pca) — PC scores of delta_h
    delta_a: np.ndarray,          # (N_train, D_action) — action deltas
    n_pls_components: int = 10,
    scale: bool = True,
) -> tuple[PLSRegression, np.ndarray, np.ndarray]:
    """Fit PLS to find subspace of delta_h that maximally covaries with delta_a.

    Returns:
        pls: fitted PLSRegression model
        X_scores: (N_train, n_pls_components) — training PLS scores
        pls_loadings_in_pca_space: (K_pca, n_pls_components) — PLS directions in PCA space
    """
    X = delta_h_scores.copy()
    Y = delta_a.copy()

    if scale:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + EPS)

    pls = PLSRegression(n_components=n_pls_components, scale=False)
    pls.fit(X, Y)

    # PLS x_loadings_ gives the directions in X (PCA) space
    pls_loadings = pls.x_loadings_  # (K_pca, n_components)

    return pls, pls.x_scores_, pls_loadings


def project_onto_subspace(
    delta_h_scores: np.ndarray,       # (N, K_pca)
    pls_loadings_pca: np.ndarray,     # (K_pca, n_pls)
) -> np.ndarray:
    """Project delta_h (in PCA space) onto the PLS action-sensitive subspace.

    Returns:
        proj_norm_sq: (N,) — squared norm of projection onto subspace
    """
    # Projection: X @ W where W is the PLS loading
    proj = delta_h_scores @ pls_loadings_pca  # (N, n_pls)
    proj_norm_sq = np.sum(proj ** 2, axis=1)  # (N,)
    return proj_norm_sq


# ---------------------------------------------------------------------------
# Cross-perturbation analysis
# ---------------------------------------------------------------------------

def cross_perturbation_analysis(
    pairs: list[dict],
    out_dir: Path,
    n_pca: int = 50,
    n_pls: int = 10,
    random_seed: int = 42,
) -> dict:
    """Main cross-perturbation analysis."""
    np.random.seed(random_seed)

    # ── 1. Load all data ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Loading all delta_h and delta_a")
    print("=" * 70)

    all_delta_h = []
    all_delta_a = []
    all_meta = []

    for i, pair in enumerate(pairs):
        if (i + 1) % 20 == 0:
            print(f"  Loading {i+1}/{len(pairs)}...")

        h_clean = load_hidden(pair, "action", "clean").numpy()
        h_pert = load_hidden(pair, "action", "pert").numpy()
        a_clean = load_action(pair, "clean")
        a_pert = load_action(pair, "pert")

        dh = h_pert - h_clean
        da = a_pert - a_clean

        all_delta_h.append(dh)
        all_delta_a.append(da)
        all_meta.append({
            "condition": pair["condition"],
            "episode": pair["episode"],
            "group": pair["group"],
        })

    delta_h = np.array(all_delta_h, dtype=np.float64)   # (140, 401408)
    delta_a = np.array(all_delta_a, dtype=np.float64)   # (140, 112)
    meta = all_meta
    N, D = delta_h.shape
    D_action = delta_a.shape[1]
    print(f"  delta_h: {delta_h.shape}, delta_a: {delta_a.shape}")
    print(f"  Memory: {delta_h.nbytes / 1024**2:.1f} MB + {delta_a.nbytes / 1024:.1f} MB")

    # ── 2. Global PCA on delta_h ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: PCA on delta_h (Gram matrix, 140 x 140)")
    print("=" * 70)

    scores, loadings, evr = compute_delta_h_pca(delta_h, max_components=n_pca)
    K_pca = scores.shape[1]
    print(f"  Retained {K_pca} PCs, top-5 evr: {evr[:5]}")
    print(f"  PC1: {evr[0]:.3f}, PC2: {evr[1]:.3f}, PC3: {evr[2]:.3f}, "
          f"PC1-5 cum: {evr[:5].sum():.3f}, PC1-10 cum: {evr[:10].sum():.3f}")

    # ── 3. Global PLS ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 3: Global PLS (all data, {K_pca} PCs → {n_pls} components)")
    print("=" * 70)

    pls, pls_scores, pls_loadings_pca = fit_action_sensitive_subspace(
        scores, delta_a, n_pls_components=n_pls
    )
    print(f"  PLS X scores shape: {pls_scores.shape}")
    print(f"  PLS loadings in PCA space: {pls_loadings_pca.shape}")

    # ── 4. Project all samples onto action-sensitive subspace ─────────
    print("\n" + "=" * 70)
    print("STEP 4: Projection onto action-sensitive subspace")
    print("=" * 70)

    proj_sq = project_onto_subspace(scores, pls_loadings_pca)  # (N,)
    total_norm_sq = np.sum(scores ** 2, axis=1)  # (N,)
    proj_ratio = proj_sq / (total_norm_sq + EPS)  # fraction of delta_h in subspace

    # Per-condition × group summary
    results_rows = []
    for i in range(N):
        results_rows.append({
            "condition": meta[i]["condition"],
            "episode": meta[i]["episode"],
            "group": meta[i]["group"],
            "proj_norm_sq": float(proj_sq[i]),
            "total_norm_sq": float(total_norm_sq[i]),
            "proj_ratio": float(proj_ratio[i]),
        })

    # Print summary
    summary_rows = _group_aggregate(results_rows, ["proj_norm_sq", "proj_ratio"])
    print(f"\n  {'Condition':30s} {'Group':12s} {'N':>4s} {'proj_ratio':>10s} {'proj_norm_sq':>12s}")
    print(f"  {'-'*70}")
    for r in summary_rows:
        print(f"  {r['condition']:30s} {r['group']:12s} {r['n']:>4d} "
              f"{r.get('proj_ratio_mean', float('nan')):>10.4f} "
              f"{r.get('proj_norm_sq_mean', float('nan')):>12.4f}")

    # ── 5. Cross-perturbation holdout analysis ────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Cross-perturbation holdout analysis")
    print("  (For each perturbation held out, learn subspace from others)")
    print("=" * 70)

    perturbations = sorted(set(m["condition"] for m in meta))
    cv_rows = []

    for heldout_pert in perturbations:
        train_idx = [i for i, m in enumerate(meta) if m["condition"] != heldout_pert]
        test_idx = [i for i, m in enumerate(meta) if m["condition"] == heldout_pert]

        # Fit PLS on training perturbations only
        train_scores = scores[train_idx]
        train_da = delta_a[train_idx]
        pls_cv, _, pls_load_cv = fit_action_sensitive_subspace(
            train_scores, train_da, n_pls_components=n_pls
        )

        # Project test samples
        test_scores = scores[test_idx]
        proj_sq_cv = project_onto_subspace(test_scores, pls_load_cv)
        total_sq_cv = np.sum(test_scores ** 2, axis=1)
        proj_ratio_cv = proj_sq_cv / (total_sq_cv + EPS)

        # Aggregate by group
        for grp in ["preserved", "flipped"]:
            grp_idx = [j for j in test_idx if meta[j]["group"] == grp]
            if not grp_idx:
                continue
            grp_proj = proj_ratio_cv[[test_idx.index(j) for j in grp_idx]]
            cv_rows.append({
                "heldout": heldout_pert,
                "group": grp,
                "n": len(grp_idx),
                "proj_ratio_mean": float(np.mean(grp_proj)),
                "proj_ratio_std": float(np.std(grp_proj)),
            })

    _save_csv(out_dir / "cv_projection_by_perturbation.csv",
              ["heldout", "group", "n", "proj_ratio_mean", "proj_ratio_std"],
              cv_rows)

    print(f"\n  {'Heldout':30s} {'Group':12s} {'N':>4s} {'proj_ratio_mean':>16s}")
    print(f"  {'-'*70}")
    for r in cv_rows:
        print(f"  {r['heldout']:30s} {r['group']:12s} {r['n']:>4d} {r['proj_ratio_mean']:>16.6f}")

    # Flipped vs preserved test
    flipped_ratios = [r["proj_ratio_mean"] for r in cv_rows if r["group"] == "flipped" and r["n"] >= 3]
    preserved_ratios = [r["proj_ratio_mean"] for r in cv_rows if r["group"] == "preserved" and r["n"] >= 3]
    if flipped_ratios and preserved_ratios:
        diff = np.mean(flipped_ratios) - np.mean(preserved_ratios)
        print(f"\n  Mean flipped proj_ratio: {np.mean(flipped_ratios):.6f}")
        print(f"  Mean preserved proj_ratio: {np.mean(preserved_ratios):.6f}")
        print(f"  Difference (flipped - preserved): {diff:+.6f}")
        if diff > 0:
            print(f"  → Flipped samples project MORE onto action-sensitive subspace (supports hypothesis)")
        else:
            print(f"  → Flipped samples do NOT project more (contradicts hypothesis)")

    # ── 6. Cross-perturbation delta-h centroid alignment ──────────────
    print("\n" + "=" * 70)
    print("STEP 6: Cross-perturbation flipped delta-h subspace overlap")
    print("=" * 70)

    # For each perturbation, compute mean flipped delta_h centroid (in PCA space)
    centroids = {}
    for pert in perturbations:
        pert_idx = [i for i, m in enumerate(meta) if m["condition"] == pert and m["group"] == "flipped"]
        if len(pert_idx) < 2:
            continue
        centroids[pert] = np.mean(scores[pert_idx], axis=0)

    # Pairwise cosine between flipped centroids
    pert_list = sorted(centroids.keys())
    cos_matrix = np.zeros((len(pert_list), len(pert_list)))
    for i, p1 in enumerate(pert_list):
        for j, p2 in enumerate(pert_list):
            c1, c2 = centroids[p1], centroids[p2]
            cos_matrix[i, j] = float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + EPS))

    _save_csv(out_dir / "cross_perturbation_centroid_cosine.csv",
              ["perturbation"] + pert_list,
              [{"perturbation": p, **{pp: cos_matrix[i, j] for j, pp in enumerate(pert_list)}}
               for i, p in enumerate(pert_list)])

    print(f"\n  Pairwise cosine between flipped delta-h centroids (in PCA space):")
    print(f"  {'':25s}", end="")
    for p in pert_list:
        print(f"{p[:12]:>12s}", end="")
    print()
    for i, p in enumerate(pert_list):
        print(f"  {p:25s}", end="")
        for j in range(len(pert_list)):
            marker = " *" if i != j and cos_matrix[i, j] > 0.3 else "  "
            print(f"{cos_matrix[i, j]:>10.4f}{marker}", end="")
        print()
    print(f"  (* marks cos > 0.3 — evidence of shared subspace)")

    mean_off_diag = (cos_matrix.sum() - np.trace(cos_matrix)) / (len(pert_list) * (len(pert_list) - 1))
    print(f"\n  Mean off-diagonal cosine: {mean_off_diag:.4f}")
    if mean_off_diag > 0.3:
        print(f"  → Strong evidence for shared flipped subspace across perturbations")
    elif mean_off_diag > 0.1:
        print(f"  → Moderate evidence for shared flipped subspace")
    else:
        print(f"  → Weak or no evidence for shared flipped subspace")

    # ── 7. Action-space cross-perturbation alignment ─────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Cross-perturbation flipped delta-a alignment (action space)")
    print("=" * 70)

    da_centroids = {}
    for pert in perturbations:
        pert_idx = [i for i, m in enumerate(meta) if m["condition"] == pert and m["group"] == "flipped"]
        if len(pert_idx) < 2:
            continue
        das = delta_a[pert_idx]
        da_norms = np.linalg.norm(das, axis=1, keepdims=True)
        da_dirs = das / (da_norms + EPS)  # unit directions
        da_centroids[pert] = np.mean(da_dirs, axis=0)
        da_centroids[pert] /= (np.linalg.norm(da_centroids[pert]) + EPS)

    da_pert_list = sorted(da_centroids.keys())
    da_cos = np.zeros((len(da_pert_list), len(da_pert_list)))
    for i, p1 in enumerate(da_pert_list):
        for j, p2 in enumerate(da_pert_list):
            da_cos[i, j] = float(np.dot(da_centroids[p1], da_centroids[p2]))

    _save_csv(out_dir / "cross_perturbation_action_centroid_cosine.csv",
              ["perturbation"] + da_pert_list,
              [{"perturbation": p, **{pp: da_cos[i, j] for j, pp in enumerate(da_pert_list)}}
               for i, p in enumerate(da_pert_list)])

    print(f"\n  Pairwise cosine between flipped delta-a centroids (action space):")
    print(f"  {'':25s}", end="")
    for p in da_pert_list:
        print(f"{p[:12]:>12s}", end="")
    print()
    for i, p in enumerate(da_pert_list):
        print(f"  {p:25s}", end="")
        for j in range(len(da_pert_list)):
            marker = " *" if i != j and da_cos[i, j] > 0.3 else "  "
            print(f"{da_cos[i, j]:>10.4f}{marker}", end="")
        print()

    da_mean_off_diag = (da_cos.sum() - np.trace(da_cos)) / (len(da_pert_list) * (len(da_pert_list) - 1))
    print(f"\n  Mean off-diagonal cosine: {da_mean_off_diag:.4f}")

    # ── 8. Baseline: random subspace test ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8: Random subspace baseline")
    print("=" * 70)

    n_rand = 100
    rand_proj_ratios = []
    for _ in range(n_rand):
        # Random orthonormal subspace of same dimension
        rand_basis = np.random.randn(K_pca, n_pls)
        Q, _ = np.linalg.qr(rand_basis)
        rand_proj = project_onto_subspace(scores, Q)
        rand_ratio = np.mean(rand_proj / (total_norm_sq + EPS))
        rand_proj_ratios.append(rand_ratio)

    rand_mean = np.mean(rand_proj_ratios)
    rand_std = np.std(rand_proj_ratios)
    pls_mean_ratio = np.mean(proj_ratio)

    print(f"  Random subspace (dim={n_pls}): mean ratio = {rand_mean:.6f} ± {rand_std:.6f}")
    print(f"  PLS action-sensitive subspace: mean ratio = {pls_mean_ratio:.6f}")
    print(f"  Ratio (PLS / random): {pls_mean_ratio / (rand_mean + EPS):.2f}x")

    # ── 9. Save all outputs ──────────────────────────────────────────
    _save_csv(out_dir / "per_sample_projection.csv",
              ["condition", "episode", "group", "proj_norm_sq", "total_norm_sq", "proj_ratio"],
              results_rows)

    summary = {
        "N": N,
        "D_hidden": D,
        "D_action": D_action,
        "K_pca": K_pca,
        "n_pls": n_pls,
        "pca_top5_evr": evr[:5].tolist(),
        "pls_mean_proj_ratio": float(pls_mean_ratio),
        "random_mean_proj_ratio": float(rand_mean),
        "random_std_proj_ratio": float(rand_std),
        "pls_random_ratio": float(pls_mean_ratio / (rand_mean + EPS)),
        "mean_off_diag_hidden_centroid": float(mean_off_diag),
        "mean_off_diag_action_centroid": float(da_mean_off_diag),
        "cv_summary": cv_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Summary written to {out_dir / 'summary.json'}")

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_aggregate(rows: list[dict], fields: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["group"])].append(r)
    out = []
    for (cond, grp), items in sorted(grouped.items()):
        entry = {"condition": cond, "group": grp, "n": len(items)}
        for f in fields:
            vals = [it[f] for it in items if not math.isnan(it.get(f, float("nan")))]
            entry[f"{f}_mean"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{f}_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(entry)
    return out


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import csv
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
    p.add_argument("--n-pca", type=int, default=50,
                   help="Number of PCA components for delta_h reduction")
    p.add_argument("--n-pls", type=int, default=10,
                   help="Number of PLS components for action-sensitive subspace")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_all_pairs(results_dir)
    if not pairs:
        print("No pairs found."); return

    # Count by condition × group
    from collections import Counter
    for (cond, grp), n in sorted(Counter((p["condition"], p["group"]) for p in pairs).items()):
        print(f"  {cond:30s} {grp:12s} {n}")

    summary = cross_perturbation_analysis(
        pairs, out_dir,
        n_pca=args.n_pca, n_pls=args.n_pls,
        random_seed=args.seed,
    )

    return summary


if __name__ == "__main__":
    main()
