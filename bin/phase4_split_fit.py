#!/usr/bin/env python3
"""
Phase 4 v3: Clean separation — define S_action on one group, test on the other.

Key question:
  If we define the action-sensitive subspace S_action using ONLY preserved samples,
  does it contain flipped samples' delta_h?

  And vice versa: does S_flipped contain preserved?

Approach:
  1. Split data into preserved (87) and flipped (53)
  2. On TRAIN group: PCA → PLS(delta_h, delta_a) → S_action
  3. Project TEST group delta_h onto S_action
  4. Compare: containment ratios, R² cross-prediction, principal angles

This removes the circularity of defining and testing on the same data.
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
OUT_DIR_DEFAULT = str(ROOT / "experiments/phase4_split")


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
# PCA via Gram matrix
# ---------------------------------------------------------------------------

def fit_gram_pca(X_train: np.ndarray, max_k: int = 100):
    """Fit PCA on X_train (N < D). Returns (mu, loadings, evr, K)."""
    N, D = X_train.shape
    K = min(N, max_k)
    mu = X_train.mean(axis=0, keepdims=True)
    Xc = X_train - mu
    G = Xc @ Xc.T
    evals, evecs = np.linalg.eigh(G)
    evals = evals[::-1][:K]
    evecs = evecs[:, ::-1][:, :K]
    mask = evals > evals.max() * 1e-10
    K = int(mask.sum())
    evals, evecs = evals[:K], evecs[:, :K]
    inv_sqrt = 1.0 / np.sqrt(evals * (N - 1) + EPS)
    loadings = Xc.T @ evecs @ np.diag(inv_sqrt)  # (D, K)
    total = float(evals.sum())
    evr = np.array([float(evals[i] / total) for i in range(K)])
    return mu, loadings.astype(np.float64), evr, K


def transform_pca(X: np.ndarray, mu: np.ndarray, loadings: np.ndarray):
    """Project X onto pre-fitted PCA basis."""
    Xc = X - mu
    return (Xc @ loadings).astype(np.float64)


# ---------------------------------------------------------------------------
# Fit PLS on train, evaluate on test
# ---------------------------------------------------------------------------

def fit_pls_and_evaluate(
    train_dh: np.ndarray,   # (N_train, D)
    train_da: np.ndarray,   # (N_train, D_action)
    test_dh: np.ndarray,    # (N_test, D)
    test_da: np.ndarray,    # (N_test, D_action)
    n_pca: int = 50,
    n_pls: int = 10,
    label: str = "",
) -> dict:
    """Fit PCA+PLS on train, evaluate containment and prediction on test."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train: {train_dh.shape[0]} samples, Test: {test_dh.shape[0]} samples")
    print(f"{'='*60}")

    # 1. Fit PCA on train
    mu, pca_loadings, evr, K = fit_gram_pca(train_dh, max_k=n_pca)
    print(f"  PCA: {K} PCs retained, top-5 evr: {evr[:5].round(4)}")

    # 2. Transform both train and test
    train_scores = transform_pca(train_dh, mu, pca_loadings)
    test_scores = transform_pca(test_dh, mu, pca_loadings)

    # 3. Standardize
    train_mean = train_scores.mean(axis=0, keepdims=True)
    train_std = train_scores.std(axis=0, keepdims=True) + EPS
    da_mean = train_da.mean(axis=0, keepdims=True)
    da_std = train_da.std(axis=0, keepdims=True) + EPS

    X_train = (train_scores - train_mean) / train_std
    Y_train = (train_da - da_mean) / da_std
    X_test = (test_scores - train_mean) / train_std
    Y_test = (test_da - da_mean) / da_std

    # 4. Fit PLS on train
    pls = PLSRegression(n_components=n_pls, scale=False)
    pls.fit(X_train, Y_train)
    pls_loadings = pls.x_loadings_  # (K, n_pls)

    # 5. Train R²
    Y_train_pred = pls.predict(X_train)
    ss_res_train = np.sum((Y_train - Y_train_pred) ** 2)
    ss_tot_train = np.sum(Y_train ** 2)
    r2_train = 1 - ss_res_train / (ss_tot_train + EPS)

    # 6. Test R²
    Y_test_pred = pls.predict(X_test)
    ss_res_test = np.sum((Y_test - Y_test_pred) ** 2)
    ss_tot_test = np.sum(Y_test ** 2)
    r2_test = 1 - ss_res_test / (ss_tot_test + EPS)

    print(f"  PLS R²: train={r2_train:.4f}, test={r2_test:.4f}")

    # 7. Containment: what fraction of test delta_h (in PCA space) is in S_action?
    # S_action = span(pls_loadings)
    # Projection of X_test onto S_action
    test_proj_sq = np.sum((X_test @ pls_loadings) ** 2, axis=1)
    test_total_sq = np.sum(X_test ** 2, axis=1)
    test_proj_ratio = test_proj_sq / (test_total_sq + EPS)

    train_proj_sq = np.sum((X_train @ pls_loadings) ** 2, axis=1)
    train_total_sq = np.sum(X_train ** 2, axis=1)
    train_proj_ratio = train_proj_sq / (train_total_sq + EPS)

    # 8. Random baseline on test data
    rand_ratios = []
    for _ in range(200):
        Q, _ = np.linalg.qr(np.random.randn(K, n_pls))
        rand_proj = np.sum((X_test @ Q) ** 2, axis=1)
        rand_ratios.append(np.mean(rand_proj / (test_total_sq + EPS)))
    rand_mean, rand_std = np.mean(rand_ratios), np.std(rand_ratios)

    print(f"  Containment (mean proj_ratio):")
    print(f"    Train: {np.mean(train_proj_ratio):.4f}")
    print(f"    Test:  {np.mean(test_proj_ratio):.4f}")
    print(f"    Random baseline: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"    Test / Random: {np.mean(test_proj_ratio) / (rand_mean + EPS):.2f}x")

    return {
        "label": label,
        "n_train": train_dh.shape[0],
        "n_test": test_dh.shape[0],
        "K_pca": K,
        "n_pls": n_pls,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "train_proj_ratio_mean": float(np.mean(train_proj_ratio)),
        "test_proj_ratio_mean": float(np.mean(test_proj_ratio)),
        "rand_mean": float(rand_mean),
        "rand_std": float(rand_std),
        "ratio_vs_random": float(np.mean(test_proj_ratio) / (rand_mean + EPS)),
        "test_proj_ratio": test_proj_ratio,
        "train_proj_ratio": train_proj_ratio,
        "pls": pls,
        "pls_loadings": pls_loadings,
        "pca_mu": mu,
        "pca_loadings": pca_loadings,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_test_pred": Y_test_pred,
    }


# ---------------------------------------------------------------------------
# Principal angles between two PLS subspaces
# ---------------------------------------------------------------------------

def principal_angles(loadings_a: np.ndarray, loadings_b: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces defined by their basis vectors.

    loadings_a: (K, n_pls) — basis for subspace A
    loadings_b: (K, n_pls) — basis for subspace B

    Returns cos(theta) for each principal angle.
    """
    # Orthonormalize
    Qa, _ = np.linalg.qr(loadings_a)
    Qb, _ = np.linalg.qr(loadings_b)
    # SVD of Qa.T @ Qb gives cosines of principal angles
    cross = Qa.T @ Qb
    _, S, _ = np.linalg.svd(cross)
    # Clamp to [0, 1]
    S = np.clip(S, 0, 1)
    return S  # cos(theta_1), cos(theta_2), ...


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--output-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--n-pca", type=int, default=50)
    p.add_argument("--n-pls", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dh, da, meta = load_all_data(Path(args.results_dir))

    # Split
    preserved_idx = [i for i, m in enumerate(meta) if m["group"] == "preserved"]
    flipped_idx = [i for i, m in enumerate(meta) if m["group"] == "flipped"]

    dh_pres, da_pres = dh[preserved_idx], da[preserved_idx]
    dh_flip, da_flip = dh[flipped_idx], da[flipped_idx]
    meta_pres = [meta[i] for i in preserved_idx]
    meta_flip = [meta[i] for i in flipped_idx]

    print(f"\nPreserved: {len(dh_pres)} samples, Flipped: {len(dh_flip)} samples")

    # ── Experiment 1: Train on preserved, test on flipped ──────────
    result_p2f = fit_pls_and_evaluate(
        dh_pres, da_pres, dh_flip, da_flip,
        n_pca=args.n_pca, n_pls=args.n_pls,
        label="EXP 1: Train=preserved (→ S_preserved), Test=flipped",
    )

    # ── Experiment 2: Train on flipped, test on preserved ──────────
    result_f2p = fit_pls_and_evaluate(
        dh_flip, da_flip, dh_pres, da_pres,
        n_pca=args.n_pca, n_pls=args.n_pls,
        label="EXP 2: Train=flipped (→ S_flipped), Test=preserved",
    )

    # ── Experiment 3: Train/test on preserved (baseline) ───────────
    # Split preserved in half
    n_pres = len(dh_pres)
    perm = np.random.permutation(n_pres)
    half = n_pres // 2
    result_p2p = fit_pls_and_evaluate(
        dh_pres[perm[:half]], da_pres[perm[:half]],
        dh_pres[perm[half:]], da_pres[perm[half:]],
        n_pca=args.n_pca, n_pls=args.n_pls,
        label="EXP 3: Train=preserved(A), Test=preserved(B) [baseline]",
    )

    # ── Experiment 4: Train/test on flipped (baseline) ─────────────
    n_flip = len(dh_flip)
    perm_f = np.random.permutation(n_flip)
    half_f = n_flip // 2
    result_f2f = fit_pls_and_evaluate(
        dh_flip[perm_f[:half_f]], da_flip[perm_f[:half_f]],
        dh_flip[perm_f[half_f:]], da_flip[perm_f[half_f:]],
        n_pca=args.n_pca, n_pls=args.n_pls,
        label="EXP 4: Train=flipped(A), Test=flipped(B) [baseline]",
    )

    # ── Principal angles between S_preserved and S_flipped ─────────
    print(f"\n{'='*60}")
    print(f"  Principal angles between S_preserved and S_flipped")
    print(f"{'='*60}")

    # Get PLS loadings (need to be in a common space for comparison)
    # Both are in their own PCA spaces, so we compare the canonical
    # correlation instead: how well does S_preserved explain flipped delta_a?

    # We already have r2_test from p2f and f2p
    print(f"  R²(preserved → flipped actions): {result_p2f['r2_test']:.4f}")
    print(f"  R²(flipped → preserved actions): {result_f2p['r2_test']:.4f}")
    print(f"  R²(preserved → preserved actions): {result_p2p['r2_test']:.4f}")
    print(f"  R²(flipped → flipped actions): {result_f2f['r2_test']:.4f}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Cross-group containment & prediction")
    print(f"{'='*60}")

    header = f"  {'Experiment':<35s} {'Test Contain':>12s} {'vs Rand':>8s} {'R² test':>8s}"
    print(header)
    print(f"  {'-'*65}")
    for res in [result_p2f, result_f2p, result_p2p, result_f2f]:
        print(f"  {res['label']:<35s} "
              f"{res['test_proj_ratio_mean']:>12.4f} "
              f"{res['ratio_vs_random']:>8.2f}x "
              f"{res['r2_test']:>8.4f}")

    # ── Per-perturbation breakdown for flipped ────────────────────
    print(f"\n{'='*60}")
    print(f"  Per-perturbation flipped containment in S_preserved")
    print(f"{'='*60}")

    conditions = sorted(set(m["condition"] for m in meta_flip))
    test_ratios = result_p2f["test_proj_ratio"]
    test_meta = meta_flip

    print(f"  {'Condition':30s} {'N':>4s} {'containment':>12s}")
    print(f"  {'-'*50}")
    for cond in conditions:
        idx = [i for i, m in enumerate(test_meta) if m["condition"] == cond]
        if not idx:
            continue
        ratios = test_ratios[idx]
        print(f"  {cond:30s} {len(idx):>4d} {np.mean(ratios):>12.4f}")

    # Save
    summary = {
        "preserved_to_flipped": {
            "test_containment": result_p2f["test_proj_ratio_mean"],
            "ratio_vs_random": result_p2f["ratio_vs_random"],
            "r2_test": result_p2f["r2_test"],
            "r2_train": result_p2f["r2_train"],
        },
        "flipped_to_preserved": {
            "test_containment": result_f2p["test_proj_ratio_mean"],
            "ratio_vs_random": result_f2p["ratio_vs_random"],
            "r2_test": result_f2p["r2_test"],
            "r2_train": result_f2p["r2_train"],
        },
        "preserved_to_preserved": {
            "test_containment": result_p2p["test_proj_ratio_mean"],
            "ratio_vs_random": result_p2p["ratio_vs_random"],
            "r2_test": result_p2p["r2_test"],
            "r2_train": result_p2p["r2_train"],
        },
        "flipped_to_flipped": {
            "test_containment": result_f2f["test_proj_ratio_mean"],
            "ratio_vs_random": result_f2f["ratio_vs_random"],
            "r2_test": result_f2f["r2_test"],
            "r2_train": result_f2f["r2_train"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
