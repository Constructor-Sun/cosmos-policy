#!/usr/bin/env python3
"""
Phase 4 v4: PCA on ALL data (unsupervised), PLS on train group only.

Fixes the issue in v3 where PCA was also fit on train-only, potentially
missing variance directions that exist only in the test group.

Strategy:
  1. PCA on all 140 delta_h → shared PC space
  2. Split scores into train/test groups
  3. PLS on train group → S_action
  4. Containment & R² on test group

Also adds: per-perturbation flipped containment in S_preserved.
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
OUT_DIR_DEFAULT = str(ROOT / "experiments/phase4_split_v2")


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
    meta = meta_list
    print(f"Loaded {len(dh)} pairs: dh {dh.shape}, da {da.shape}")
    return dh, da, meta


def _load_hidden(ep_dir, slot_name, clean_or_pert):
    d = torch.load(str(ep_dir / f"hidden_{slot_name}_{clean_or_pert}.pt"),
                   map_location="cpu", weights_only=True)
    t = list(d.values())[0]
    return t.float().reshape(-1).numpy()


# ---------------------------------------------------------------------------
# PCA on all data via Gram matrix
# ---------------------------------------------------------------------------

def pca_all_data(dh, max_k=50):
    """PCA on all delta_h. Returns scores (N, K), loadings (D, K), evr."""
    N, D = dh.shape
    K = min(N, max_k)
    mu = dh.mean(axis=0, keepdims=True)
    Xc = dh - mu
    G = Xc @ Xc.T  # (N, N)
    evals, evecs = np.linalg.eigh(G)
    evals = evals[::-1][:K]
    evecs = evecs[:, ::-1][:, :K]
    mask = evals > evals.max() * 1e-10
    K = int(mask.sum())
    evals, evecs = evals[:K], evecs[:, :K]
    inv_sqrt = 1.0 / np.sqrt(evals * (N - 1) + EPS)
    loadings = Xc.T @ evecs @ np.diag(inv_sqrt)  # (D, K)
    scores = evecs @ np.diag(np.sqrt(evals * (N - 1)))  # (N, K)
    total = float(evals.sum())
    evr = np.array([float(evals[i] / total) for i in range(K)])
    return scores.astype(np.float64), loadings.astype(np.float64), evr


# ---------------------------------------------------------------------------
# Fit PLS on train, evaluate on test
# ---------------------------------------------------------------------------

def fit_pls_on_split(
    all_scores, all_da, all_meta,
    train_group: str,
    test_group: str,
    n_pls: int = 10,
    label: str = "",
) -> dict:
    """Train PLS on train_group samples, test on test_group samples."""
    train_idx = [i for i, m in enumerate(all_meta) if m["group"] == train_group]
    test_idx = [i for i, m in enumerate(all_meta) if m["group"] == test_group]

    X_train = all_scores[train_idx]
    Y_train_raw = all_da[train_idx]
    X_test = all_scores[test_idx]
    Y_test_raw = all_da[test_idx]
    test_meta = [all_meta[i] for i in test_idx]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train ({train_group}): {len(train_idx)}, Test ({test_group}): {len(test_idx)}")
    print(f"{'='*60}")

    # Standardize using train stats
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + EPS
    Y_mean = Y_train_raw.mean(axis=0, keepdims=True)
    Y_std = Y_train_raw.std(axis=0, keepdims=True) + EPS

    Xt = (X_train - X_mean) / X_std
    Yt = (Y_train_raw - Y_mean) / Y_std
    Xe = (X_test - X_mean) / X_std
    Ye = (Y_test_raw - Y_mean) / Y_std

    # PLS
    pls = PLSRegression(n_components=n_pls, scale=False)
    pls.fit(Xt, Yt)
    pls_loadings = pls.x_loadings_  # (K_pca, n_pls)

    # R²
    Yt_pred = pls.predict(Xt)
    Ye_pred = pls.predict(Xe)
    r2_train = 1 - np.sum((Yt - Yt_pred) ** 2) / (np.sum(Yt ** 2) + EPS)
    r2_test = 1 - np.sum((Ye - Ye_pred) ** 2) / (np.sum(Ye ** 2) + EPS)

    # Containment: ||Proj_{S} X||^2 / ||X||^2
    # Note: X is already standardized, so each dimension has ~equal weight
    proj_train = np.sum((Xt @ pls_loadings) ** 2, axis=1)
    total_train = np.sum(Xt ** 2, axis=1)
    ratio_train = proj_train / (total_train + EPS)

    proj_test = np.sum((Xe @ pls_loadings) ** 2, axis=1)
    total_test = np.sum(Xe ** 2, axis=1)
    ratio_test = proj_test / (total_test + EPS)

    # Random baseline
    K = Xt.shape[1]
    rand_ratios = []
    for _ in range(200):
        Q, _ = np.linalg.qr(np.random.randn(K, n_pls))
        rp = np.sum((Xe @ Q) ** 2, axis=1)
        rand_ratios.append(np.mean(rp / (total_test + EPS)))
    rand_mean, rand_std = np.mean(rand_ratios), np.std(rand_ratios)

    print(f"  PLS R²: train={r2_train:.4f}, test={r2_test:.4f}")
    print(f"  Containment: train={np.mean(ratio_train):.4f}, test={np.mean(ratio_test):.4f}")
    print(f"  Random baseline: {rand_mean:.4f}±{rand_std:.4f}")
    print(f"  Test/Random: {np.mean(ratio_test)/(rand_mean+EPS):.2f}x")

    # Per-condition breakdown on test group
    conditions = sorted(set(m["condition"] for m in test_meta))
    print(f"\n  Per-condition containment:")
    print(f"  {'Condition':30s} {'N':>4s} {'containment':>12s} {'R² (action pred)':>15s}")
    print(f"  {'-'*65}")
    cond_rows = []
    for cond in conditions:
        ci = [j for j, m in enumerate(test_meta) if m["condition"] == cond]
        if not ci:
            continue
        cr = ratio_test[ci]
        # Per-condition R²
        ye_cond = Ye[ci]
        yep_cond = Ye_pred[ci]
        r2_cond = 1 - np.sum((ye_cond - yep_cond)**2) / (np.sum(ye_cond**2) + EPS)
        print(f"  {cond:30s} {len(ci):>4d} {np.mean(cr):>12.4f} {r2_cond:>15.4f}")
        cond_rows.append({
            "condition": cond, "n": len(ci),
            "mean_containment": float(np.mean(cr)),
            "r2_action": float(r2_cond),
        })

    return {
        "label": label,
        "train_group": train_group, "test_group": test_group,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "r2_train": r2_train, "r2_test": r2_test,
        "containment_train": float(np.mean(ratio_train)),
        "containment_test": float(np.mean(ratio_test)),
        "rand_mean": float(rand_mean), "rand_std": float(rand_std),
        "ratio_vs_random": float(np.mean(ratio_test) / (rand_mean + EPS)),
        "ratio_test": ratio_test,
        "test_meta": test_meta,
        "cond_rows": cond_rows,
        "pls_loadings": pls_loadings,
    }


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

    n_pres = sum(1 for m in meta if m["group"] == "preserved")
    n_flip = sum(1 for m in meta if m["group"] == "flipped")
    print(f"Preserved: {n_pres}, Flipped: {n_flip}")

    # ── PCA on ALL data ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PCA on all {len(dh)} samples → shared PC space")
    print(f"{'='*60}")
    scores, loadings, evr = pca_all_data(dh, max_k=args.n_pca)
    print(f"  {scores.shape[1]} PCs, top-5 evr: {evr[:5].round(4)}")
    print(f"  PC1-5 cum: {evr[:5].sum():.3f}, PC1-10 cum: {evr[:10].sum():.3f}")

    # ── Experiment 1: preserved → flipped ──────────────────────────
    r_p2f = fit_pls_on_split(
        scores, da, meta,
        train_group="preserved", test_group="flipped",
        n_pls=args.n_pls,
        label="EXP 1: S_preserved tested on flipped",
    )

    # ── Experiment 2: flipped → preserved ──────────────────────────
    r_f2p = fit_pls_on_split(
        scores, da, meta,
        train_group="flipped", test_group="preserved",
        n_pls=args.n_pls,
        label="EXP 2: S_flipped tested on preserved",
    )

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<38s} {'Contain':>8s} {'vsRand':>6s} {'R²':>8s}")
    print(f"  {'-'*60}")
    for res in [r_p2f, r_f2p]:
        print(f"  {res['label']:<38s} "
              f"{res['containment_test']:>8.4f} "
              f"{res['ratio_vs_random']:>6.2f}x "
              f"{res['r2_test']:>8.4f}")

    # ── Key metric: flipped containment in S_preserved ─────────────
    print(f"\n  Key: Flipped delta_h is {r_p2f['containment_test']:.1%} contained in S_preserved")
    print(f"  (random = {r_p2f['rand_mean']:.1%}, ratio = {r_p2f['ratio_vs_random']:.2f}x)")
    print(f"  S_preserved predicts flipped delta_a with R² = {r_p2f['r2_test']:.4f}")

    # Save
    summary = {
        "pca": {"K": scores.shape[1], "top5_evr": evr[:5].tolist()},
        "preserved_to_flipped": {
            "containment": r_p2f["containment_test"],
            "ratio_vs_random": r_p2f["ratio_vs_random"],
            "r2_test": r_p2f["r2_test"],
            "r2_train": r_p2f["r2_train"],
            "per_condition": r_p2f["cond_rows"],
        },
        "flipped_to_preserved": {
            "containment": r_f2p["containment_test"],
            "ratio_vs_random": r_f2p["ratio_vs_random"],
            "r2_test": r_f2p["r2_test"],
            "r2_train": r_f2p["r2_train"],
            "per_condition": r_f2p["cond_rows"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
