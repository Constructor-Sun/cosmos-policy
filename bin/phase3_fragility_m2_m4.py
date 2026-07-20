#!/usr/bin/env python3
"""M2+M4 fragility analysis for Cosmos-Policy DiT (Fawzi et al. 2018 style).

M2 — Hidden State PCA: do perturbation-induced hidden shifts align with
     low-variance PCA directions of the clean data distribution?
M4 — Action Space Clustering: do flipped Δa directions cluster on the
     action sphere (indicating a consistent "fragile action dimension")?

Reads Phase 2 artifacts; no new forward passes needed.
"""

from __future__ import annotations

import argparse, json, math, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase2(root: Path, groups: set[str]) -> list[dict]:
    """Load all Phase 2 pairs matching *groups*."""
    pairs = []
    for mpath in sorted(root.glob("*/*/metrics.json")):
        rec = json.loads(mpath.read_text())
        if rec["group"] not in groups:
            continue
        ep_dir = mpath.parent
        rec["_dir"] = ep_dir
        pairs.append(rec)
    print(f"Loaded {len(pairs)} pairs from {root}")
    return pairs


def load_hidden(pair: dict, slot_name: str, clean_or_pert: str, device="cpu") -> torch.Tensor:
    """Load hidden_{slot_name}_{clean_or_pert}.pt, return flattened float32 tensor.

    slot_name: 'video' or 'action'
    """
    path = pair["_dir"] / f"hidden_{slot_name}_{clean_or_pert}.pt"
    d = torch.load(str(path), map_location=device, weights_only=True)
    t = list(d.values())[0]  # take the only layer
    return t.float().reshape(-1)


def load_action(pair: dict, slot: str) -> np.ndarray:
    """Load action_{slot}.npy, return flattened float64 array."""
    return np.load(pair["_dir"] / f"action_{slot}.npy").astype(np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# M2: Hidden State PCA — Low-Variance Direction Alignment
# ---------------------------------------------------------------------------

def m2_pca_analysis(pairs: list[dict], out_dir: Path, slot_name: str) -> dict:
    """Gram-matrix PCA on clean hidden states of *slot_name* ('video' or 'action')."""

    # 1. Collect clean hidden states → matrix X (N x D)
    X = torch.stack([load_hidden(p, slot_name, "clean") for p in pairs])  # (N, D)
    N, D = X.shape
    print(f"\n[M2-{slot_name}] N={N}, D={D:,}")

    # Center
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu  # (N, D)

    # Gram matrix G = Xc @ Xc.T / (N-1)  → (N, N)
    G = (Xc @ Xc.T) / (N - 1)
    eigenvals, eigenvecs = torch.linalg.eigh(G)  # ascending: λ₀ ≤ ... ≤ λ_{N-1}
    eigenvals = eigenvals.flip(0)       # descending: λ₁ ≥ λ₂ ≥ ...
    eigenvecs = eigenvecs.flip(1)       # columns = α_i for descending λ

    # Drop near-zero eigenvalues
    mask = eigenvals > eigenvals.max() * 1e-10
    K = int(mask.sum().item())
    eigenvals = eigenvals[:K]
    eigenvecs = eigenvecs[:, :K]  # (N, K)

    total_var = float(eigenvals.sum())
    _save_csv(out_dir / "pca_eigenvalues.csv",
              ["rank", "eigenvalue", "variance_ratio", "cumulative_ratio"],
              [[i + 1, float(eigenvals[i]), float(eigenvals[i] / total_var),
                float(eigenvals[:i + 1].sum() / total_var)] for i in range(K)])

    # 2. Collect Δh = h_pert - h_clean, project onto PCs
    rows = []
    cum_curves: dict[str, list[list[float]]] = defaultdict(list)

    for pair in pairs:
        dh = load_hidden(pair, slot_name, "pert") - load_hidden(pair, slot_name, "clean")
        dh_norm_sq = float(dh.dot(dh))
        if dh_norm_sq < EPS:
            continue

        Xc_dh = Xc @ dh  # (N,)
        proj_sq = []
        for i in range(K):
            dot = float((Xc_dh @ eigenvecs[:, i]) / math.sqrt(float(eigenvals[i]) * (N - 1) + EPS))
            proj_sq.append(dot * dot)

        total = sum(proj_sq) + EPS
        proj_norm = [p / total for p in proj_sq]
        cum = np.cumsum(proj_norm).tolist()
        cum_curves[pair["group"]].append(cum)

        tail_start = max(1, int(K * 0.8))
        tail_energy = sum(proj_norm[tail_start:])
        top5_energy = sum(proj_norm[:5])
        eff_rank = next((i for i, c in enumerate(cum) if c >= 0.8), K)

        rows.append({
            "condition": pair["condition"], "episode": pair["episode"],
            "group": pair["group"], "K": K,
            "tail_energy_0.8": tail_energy, "top5_energy": top5_energy,
            "eff_rank_0.8": eff_rank,
        })

    # 3. Per-condition × group summary
    tail_summary = _group_summary(rows, ["tail_energy_0.8", "top5_energy", "eff_rank_0.8"])
    _save_csv(out_dir / "tail_energy_summary.csv",
              ["condition", "group", "n", "tail_energy_mean", "tail_energy_std",
               "top5_energy_mean", "top5_energy_std", "eff_rank_mean", "eff_rank_std"],
              tail_summary)
    _save_csv(out_dir / "pc_projection_detail.csv",
              ["condition", "episode", "group", "K", "tail_energy_0.8", "top5_energy", "eff_rank_0.8"],
              rows)
    _save_cum_curves(out_dir / "cum_energy_by_group.csv", cum_curves, K)

    # Print summary
    print(f"\n[M2-{slot_name}] Tail Energy (higher = more in low-variance PCs → flipped > preserved)")
    print(f"{'condition':30s} {'group':12s} {'n':>4s} {'tail_energy':>10s} {'top5_energy':>10s} {'eff_rank':>8s}")
    for r in tail_summary:
        te = r.get("tail_energy_0.8_mean", float("nan"))
        t5 = r.get("top5_energy_mean", float("nan"))
        er = r.get("eff_rank_0.8_mean", float("nan"))
        print(f"{r['condition']:30s} {r['group']:12s} {r['n']:>4d} "
              f"{te:>10.4f} {t5:>10.4f} {er:>8.1f}")

    return {"N": N, "D": D, "K": K, "total_variance": total_var,
            "tail_summary": tail_summary}


# ---------------------------------------------------------------------------
# M4: Action Space Clustering — Fragile Action Dimension Discovery
# ---------------------------------------------------------------------------

def m4_action_clustering(pairs: list[dict], out_dir: Path) -> dict:
    """Analyse clustering of Δa = a_pert - a_clean in action space."""
    print("\n[M4] Action space clustering")

    # Collect all Δa
    all_da = {}
    for pair in pairs:
        da = load_action(pair, "pert") - load_action(pair, "clean")
        key = (pair["condition"], pair["group"])
        all_da.setdefault(key, []).append(da)

    D_action = len(da)
    print(f"  Action dimension: {D_action}")

    # 1. Per-condition×group pairwise cosine + concentration
    cs_rows = []
    for (cond, grp), das in sorted(all_da.items()):
        norms = np.array([np.linalg.norm(a) for a in das])
        valid = [a / (n + EPS) for a, n in zip(das, norms) if n > EPS]
        if len(valid) < 2:
            cs_rows.append({"condition": cond, "group": grp, "n": len(valid),
                           "mean_pairwise_cos": float("nan"),
                           "R": 0.0, "kappa": float("nan")})
            continue
        V = np.stack(valid)  # (n, D)
        # Mean resultant length
        R = float(np.linalg.norm(V.sum(axis=0)) / len(V))
        # von Mises-Fisher κ (Banerjee 2005 approximation)
        kappa = float(R * (D_action - R * R) / (1 - R * R + EPS)) if R < 0.99 else 1e6
        # Mean pairwise cosine
        cosmat = V @ V.T
        np.fill_diagonal(cosmat, 0)
        mean_cos = float(cosmat.sum() / (len(V) * (len(V) - 1)))
        cs_rows.append({"condition": cond, "group": grp, "n": len(valid),
                       "mean_pairwise_cos": mean_cos, "R": R, "kappa": kappa})

    _save_csv(out_dir / "concentration_summary.csv",
              ["condition", "group", "n", "mean_pairwise_cos", "R", "kappa"], cs_rows)

    # 2. PCA on Δa (unnormalized) — does flipped have a dominant PC?
    pca_rows = []
    pc1_loadings = None
    for (cond, grp), das in sorted(all_da.items()):
        stacked = np.stack([a for a in das if np.linalg.norm(a) > EPS])
        if len(stacked) < 2:
            continue
        X = stacked - stacked.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var_ratio = float(S[0] ** 2 / (S ** 2).sum())
        pca_rows.append({"condition": cond, "group": grp, "n": len(stacked),
                        "pc1_var_ratio": var_ratio,
                        "pc1_pc2_ratio": float(S[0] / (S[1] + EPS)),
                        "top3_cumulative": float((S[:3] ** 2).sum() / (S ** 2).sum())})
        if grp == "flipped" and pc1_loadings is None:
            pc1_loadings = Vt[0]  # keep first flipped PC1 for later inspection

    _save_csv(out_dir / "delta_action_pca.csv",
              ["condition", "group", "n", "pc1_var_ratio", "pc1_pc2_ratio", "top3_cumulative"], pca_rows)

    # 3. Save PC1 loadings reshaped as (chunks, dof) for inspection
    if pc1_loadings is not None:
        chunk_size, dof = 16, 7
        loading_2d = pc1_loadings.reshape(chunk_size, dof)
        _save_csv(out_dir / "pc1_loadings.csv",
                  ["chunk"] + [f"dof_{j}" for j in range(dof)],
                  [{"chunk": i, **{f"dof_{j}": float(loading_2d[i, j]) for j in range(dof)}}
                   for i in range(chunk_size)])

    # 4. Save per-pair normalized Δa directions
    dir_rows = []
    for pair in pairs:
        da = load_action(pair, "pert") - load_action(pair, "clean")
        n = np.linalg.norm(da)
        if n < EPS:
            continue
        da_norm = da / n
        dir_rows.append({"condition": pair["condition"], "episode": pair["episode"],
                        "group": pair["group"], "norm": float(n),
                        **{f"da_{j}": float(da_norm[j]) for j in range(min(D_action, 112))}})
    _save_csv(out_dir / "delta_action_directions.csv",
              ["condition", "episode", "group", "norm"] + [f"da_{j}" for j in range(min(D_action, 112))],
              dir_rows)

    # Print summary
    print(f"\n[M4] Action Direction Clustering")
    print(f"{'condition':30s} {'group':12s} {'n':>4s} {'mean_pair_cos':>12s} {'R':>8s} {'pc1_var':>8s}")
    for c in cs_rows:
        pcr = next((r["pc1_var_ratio"] for r in pca_rows
                    if r["condition"] == c["condition"] and r["group"] == c["group"]), float("nan"))
        print(f"{c['condition']:30s} {c['group']:12s} {c['n']:>4d} "
              f"{c['mean_pairwise_cos']:>12.6f} {c['R']:>8.4f} {pcr:>8.4f}")

    # Random baseline: what does uniform on S^{D-1} look like?
    n_baseline = 20
    rand_samples = np.random.randn(n_baseline, D_action)
    rand_norm = rand_samples / np.linalg.norm(rand_samples, axis=1, keepdims=True)
    rand_cosmat = rand_norm @ rand_norm.T
    np.fill_diagonal(rand_cosmat, 0)
    baseline_cos = float(rand_cosmat.sum() / (n_baseline * (n_baseline - 1)))
    baseline_std = 1.0 / math.sqrt(D_action)
    print(f"\n  Random baseline (n={n_baseline}, D={D_action}):")
    print(f"    mean pairwise cos = {baseline_cos:.6f} (expected ~0, theoretical σ ≈ {baseline_std:.4f})")
    print(f"    → if flipped mean_pairwise_cos >> {baseline_cos + 2*baseline_std:.4f}, evidence for clustering")

    return {"action_dim": D_action, "concentration": cs_rows, "pca": pca_rows,
            "random_baseline_cos": baseline_cos, "random_baseline_std": baseline_std}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_summary(rows: list[dict], fields: list[str]) -> list[dict]:
    """Aggregate *fields* by condition × group, reporting mean/std."""
    from collections import defaultdict
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


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if isinstance(r, dict):
                w.writerow({k: r.get(k, "") for k in fieldnames})
            else:
                w.writerow(dict(zip(fieldnames, r)))
    print(f"  Wrote {path}")


def _save_cum_curves(path: Path, curves: dict[str, list[list[float]]], K: int) -> None:
    """Save mean±std cumulative energy curves per group."""
    rows = []
    for grp, curvelist in sorted(curves.items()):
        arr = np.array([c + [1.0] * (K - len(c)) for c in curvelist])  # pad to K
        for i in range(K):
            rows.append({"group": grp, "pc_rank": i + 1,
                        "mean_cum_energy": float(np.mean(arr[:, i])),
                        "std_cum_energy": float(np.std(arr[:, i])),
                        "pc_variance_ratio": 0.0})  # placeholder
    _save_csv(path, ["group", "pc_rank", "mean_cum_energy", "std_cum_energy", "pc_variance_ratio"], rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_results = str(ROOT / "experiments/phase2_angular_cosmos/"
                          "kitchen_scene4_seed7_all_perturb_preserved_flipped_last")
    p.add_argument("--results-dir", default=default_results)
    p.add_argument("--output-dir",
                   default=str(ROOT / "experiments/phase3_fragility_m2_m4/kitchen_scene4_seed7"))
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)

    pairs = load_phase2(results_dir, set(args.groups))
    if not pairs:
        print("No pairs found."); return

    # Counts
    from collections import Counter
    for (cond, grp), n in sorted(Counter((p["condition"], p["group"]) for p in pairs).items()):
        print(f"  {cond:30s} {grp:12s} {n}")

    # M2 — video hidden PCA
    m2_video_out = out_dir / "m2_hidden_pca_video"
    m2_video_out.mkdir(parents=True, exist_ok=True)
    m2_video = m2_pca_analysis(pairs, m2_video_out, "video")

    # M2 — action hidden PCA
    m2_action_out = out_dir / "m2_hidden_pca_action"
    m2_action_out.mkdir(parents=True, exist_ok=True)
    m2_action = m2_pca_analysis(pairs, m2_action_out, "action")

    # M4
    m4_out = out_dir / "m4_action_clustering"
    m4_out.mkdir(parents=True, exist_ok=True)
    m4_result = m4_action_clustering(pairs, m4_out)

    # Summary
    summary = {
        "results_dir": str(results_dir),
        "num_pairs": len(pairs),
        "m2_video": m2_video,
        "m2_action": m2_action,
        "m4": {k: v for k, v in m4_result.items() if k in ("action_dim", "random_baseline_cos")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
