#!/usr/bin/env python3
"""
Phase 4 — Video Slot: PLS-based action-sensitive subspace on video-slot hidden states.

Compares video slot (where perturbations enter) vs action slot (where actions
are read out) in terms of action-sensitive subspace containment.

Key question: does the video-slot hidden state at layer 27 show different
action-sensitive containment patterns than the action slot?

Reads existing Phase 2 data (video + action hidden states at layer 27).
"""

from __future__ import annotations

import argparse, json, math, os, sys, csv
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
OUT_DIR_DEFAULT = str(ROOT / "experiments/phase4_video_slot")


def load_all_data(results_dir: Path):
    """Load both video-slot and action-slot delta_h, plus delta_a."""
    dh_v_list, dh_a_list, da_list, meta_list = [], [], [], []
    for mpath in sorted(results_dir.glob("*/*/metrics.json")):
        rec = json.loads(mpath.read_text())
        if rec["group"] not in ("preserved", "flipped"):
            continue
        ep_dir = mpath.parent

        # Video slot hidden
        hv_clean = _load_hidden(ep_dir, "video", "clean")
        hv_pert = _load_hidden(ep_dir, "video", "pert")
        # Action slot hidden
        ha_clean = _load_hidden(ep_dir, "action", "clean")
        ha_pert = _load_hidden(ep_dir, "action", "pert")
        # Action output
        a_clean = np.load(ep_dir / "action_clean.npy").astype(np.float64).reshape(-1)
        a_pert = np.load(ep_dir / "action_pert.npy").astype(np.float64).reshape(-1)

        dh_v_list.append(hv_pert - hv_clean)
        dh_a_list.append(ha_pert - ha_clean)
        da_list.append(a_pert - a_clean)
        meta_list.append({
            "condition": rec["condition"], "episode": rec["episode"],
            "group": rec["group"],
        })

    dh_v = np.array(dh_v_list, dtype=np.float64)  # (N, 802816)
    dh_a = np.array(dh_a_list, dtype=np.float64)  # (N, 401408)
    da = np.array(da_list, dtype=np.float64)      # (N, 112)
    print(f"Loaded {len(dh_v)} pairs:")
    print(f"  dh_video: {dh_v.shape}, dh_action: {dh_a.shape}, da: {da.shape}")
    return dh_v, dh_a, da, meta_list


def _load_hidden(ep_dir, slot_name, clean_or_pert):
    d = torch.load(str(ep_dir / f"hidden_{slot_name}_{clean_or_pert}.pt"),
                   map_location="cpu", weights_only=True)
    t = list(d.values())[0]
    return t.float().reshape(-1).numpy()


def gram_pca(X, max_k=50):
    N, D = X.shape
    K = min(N, max_k)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    G = Xc @ Xc.T
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


def fit_pls_split(all_scores, all_da, all_meta, train_group, test_group, n_pls=10, label=""):
    """Train PLS on train_group, test on test_group."""
    train_idx = [i for i, m in enumerate(all_meta) if m["group"] == train_group]
    test_idx = [i for i, m in enumerate(all_meta) if m["group"] == test_group]

    Xt_raw = all_scores[train_idx]
    Yt_raw = all_da[train_idx]
    Xe_raw = all_scores[test_idx]
    Ye_raw = all_da[test_idx]
    test_meta = [all_meta[i] for i in test_idx]

    X_mean = Xt_raw.mean(axis=0, keepdims=True)
    X_std = Xt_raw.std(axis=0, keepdims=True) + EPS
    Y_mean = Yt_raw.mean(axis=0, keepdims=True)
    Y_std = Yt_raw.std(axis=0, keepdims=True) + EPS

    Xt = (Xt_raw - X_mean) / X_std
    Yt = (Yt_raw - Y_mean) / Y_std
    Xe = (Xe_raw - X_mean) / X_std
    Ye = (Ye_raw - Y_mean) / Y_std

    pls = PLSRegression(n_components=n_pls, scale=False)
    pls.fit(Xt, Yt)
    pls_loadings = pls.x_loadings_

    Yt_pred = pls.predict(Xt)
    Ye_pred = pls.predict(Xe)
    r2_train = 1 - np.sum((Yt - Yt_pred) ** 2) / (np.sum(Yt ** 2) + EPS)
    r2_test = 1 - np.sum((Ye - Ye_pred) ** 2) / (np.sum(Ye ** 2) + EPS)

    # Containment
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

    print(f"\n  [{label}] Train=({train_group}, {len(train_idx)}), Test=({test_group}, {len(test_idx)})")
    print(f"    R²: train={r2_train:.4f}, test={r2_test:.4f}")
    print(f"    Containment: train={np.mean(ratio_train):.4f}, test={np.mean(ratio_test):.4f}")
    print(f"    Random baseline: {rand_mean:.4f}, Ratio: {np.mean(ratio_test)/(rand_mean+EPS):.2f}x")

    # Per-condition
    conditions = sorted(set(m["condition"] for m in test_meta))
    cond_rows = []
    for cond in conditions:
        ci = [j for j, m in enumerate(test_meta) if m["condition"] == cond]
        if not ci:
            continue
        cr = ratio_test[ci]
        ye_cond = Ye[ci]; yep_cond = Ye_pred[ci]
        r2_cond = 1 - np.sum((ye_cond - yep_cond)**2) / (np.sum(ye_cond**2) + EPS)
        cond_rows.append({"condition": cond, "n": len(ci),
                         "containment": float(np.mean(cr)), "r2": float(r2_cond)})

    return {
        "label": label, "r2_train": r2_train, "r2_test": r2_test,
        "containment_train": float(np.mean(ratio_train)),
        "containment_test": float(np.mean(ratio_test)),
        "rand_mean": float(rand_mean), "ratio_vs_random": float(np.mean(ratio_test)/(rand_mean+EPS)),
        "cond_rows": cond_rows,
    }


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

    dh_v, dh_a, da, meta = load_all_data(Path(args.results_dir))

    # PCA on all data for each slot
    print(f"\n=== PCA: video slot ({dh_v.shape[1]} dim) ===")
    scores_v, loadings_v, evr_v = gram_pca(dh_v, max_k=args.n_pca)
    print(f"  {scores_v.shape[1]} PCs, top-5 evr: {evr_v[:5].round(4)}")

    print(f"\n=== PCA: action slot ({dh_a.shape[1]} dim) ===")
    scores_a, loadings_a, evr_a = gram_pca(dh_a, max_k=args.n_pca)
    print(f"  {scores_a.shape[1]} PCs, top-5 evr: {evr_a[:5].round(4)}")

    # PLS split analyses
    results = []
    for slot_name, scores in [("video", scores_v), ("action", scores_a)]:
        print(f"\n{'='*60}")
        print(f"  SLOT: {slot_name}")
        print(f"{'='*60}")

        r_p2f = fit_pls_split(scores, da, meta, "preserved", "flipped",
                              n_pls=args.n_pls, label=f"{slot_name}: pres→flip")
        r_f2p = fit_pls_split(scores, da, meta, "flipped", "preserved",
                              n_pls=args.n_pls, label=f"{slot_name}: flip→pres")
        results.append({"slot": slot_name, "p2f": r_p2f, "f2p": r_f2p})

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  VIDEO SLOT vs ACTION SLOT")
    print(f"{'='*60}")
    header = f"  {'Slot':10s} {'Direction':15s} {'R² test':>8s} {'Contain':>8s} {'vsRand':>6s}"
    print(header)
    print(f"  {'-'*55}")
    for r in results:
        for d in ["p2f", "f2p"]:
            rd = r[d]
            print(f"  {r['slot']:10s} {rd['label']:15s} "
                  f"{rd['r2_test']:>8.4f} {rd['containment_test']:>8.4f} "
                  f"{rd['ratio_vs_random']:>6.2f}x")

    # Per-condition for flipped in video slot
    print(f"\n  Video slot: per-perturbation flipped containment in S_preserved:")
    for r in results:
        if r["slot"] == "video":
            for d in ["p2f"]:
                print(f"\n  {r[d]['label']}:")
                for cr in r[d]["cond_rows"]:
                    print(f"    {cr['condition']:30s} n={cr['n']:>3d}  "
                          f"containment={cr['containment']:.4f}  R²={cr['r2']:.4f}")

    # Summary
    summary_rows = []
    for r in results:
        for d in ["p2f", "f2p"]:
            rd = r[d]
            summary_rows.append({
                "slot": r["slot"], "direction": rd["label"],
                "r2_test": rd["r2_test"], "r2_train": rd["r2_train"],
                "containment_test": rd["containment_test"],
                "ratio_vs_random": rd["ratio_vs_random"],
            })
    _save_csv(out_dir / "video_vs_action_comparison.csv",
              ["slot", "direction", "r2_test", "r2_train", "containment_test", "ratio_vs_random"],
              summary_rows)
    print(f"\nDone. {out_dir / 'video_vs_action_comparison.csv'}")


def _save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
