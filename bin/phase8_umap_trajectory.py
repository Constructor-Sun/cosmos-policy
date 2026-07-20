#!/usr/bin/env python3
"""UMAP visualization for Phase-8 trajectory shift embeddings.

Reads ``collect_shift_embeddings.py`` HDF5 outputs, pairs clean/perturbed
rollouts, computes per-step shifts, filters out any trajectory that contains a
zero shift, samples up to N trajectories per perturbation with suite balancing,
and plots time-connected UMAP trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy_mplconfig")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

EPS = 1e-8
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments/phase8_shift_embeddings/all_suites_full"
DEFAULT_OUTPUT = ROOT / "experiments/phase8_umap_trajectory/all_suites_full"
CONDITIONS = [
    "background_textures",
    "camera_viewpoints",
    "language_instructions",
    "light_conditions",
    "objects_layout",
    "robot_initial_states",
    "sensor_noise",
]
COLORS = {
    "background_textures": "#1f77b4",
    "camera_viewpoints": "#ff7f0e",
    "language_instructions": "#2ca02c",
    "light_conditions": "#d62728",
    "objects_layout": "#9467bd",
    "robot_initial_states": "#8c564b",
    "sensor_noise": "#17becf",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--layer", default="L27", choices=["L0", "L14", "L27"])
    p.add_argument("--spaces", nargs="+", default=["hidden_video", "hidden_action"],
                   choices=["hidden_video", "hidden_action", "hidden_video_action", "final_action"])
    p.add_argument("--per-condition", type=int, default=8)
    p.add_argument("--min-nonzero-steps", type=int, default=4)
    p.add_argument("--zero-eps", type=float, default=EPS)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="Optional cap per trajectory after filtering.")
    return p.parse_args()


def read_manifest(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "manifest.parquet")


def pair_records(df: pd.DataFrame, root: Path) -> list[dict]:
    clean = {
        (r.suite, r.task_name): r
        for r in df[df.condition == "clean"].itertuples(index=False)
    }
    rows = []
    for r in df[df.condition != "clean"].itertuples(index=False):
        c = clean.get((r.suite, r.task_name))
        if c is None or not bool(c.success):
            continue
        group = "preserved" if bool(r.success) else "flipped"
        rows.append({
            "suite": str(r.suite),
            "task_name": str(r.task_name),
            "condition": str(r.condition),
            "group": group,
            "clean_h5": root / str(c.h5_path),
            "pert_h5": root / str(r.h5_path),
            "variant_name": str(r.variant_name),
            "total_steps": int(r.total_steps),
            "num_snapshots": int(r.num_snapshots),
        })
    return rows


def shared_steps(clean_h5: Path, pert_h5: Path) -> list[str]:
    with h5py.File(clean_h5, "r") as c, h5py.File(pert_h5, "r") as p:
        steps = sorted(set(c["data"].keys()) & set(p["data"].keys()), key=lambda x: int(x.split("_")[1]))
    return steps


def select_space(arr: np.ndarray, space: str) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if space == "final_action":
        return arr.reshape(-1)
    if arr.shape != (3, 14, 14, 2048):
        raise ValueError(f"expected (3,14,14,2048), got {arr.shape}")
    if space == "hidden_video":
        return arr[:2].reshape(-1)
    if space == "hidden_action":
        return arr[2].reshape(-1)
    if space == "hidden_video_action":
        return arr.reshape(-1)
    raise ValueError(space)


def read_delta(handle_c, handle_p, step: str, layer: str, space: str) -> np.ndarray:
    key = "action" if space == "final_action" else layer
    c = select_space(handle_c[f"data/{step}/{key}"][()], space)
    p = select_space(handle_p[f"data/{step}/{key}"][()], space)
    return p - c


def trajectory_meta(records: list[dict], layer: str, spaces: list[str], zero_eps: float,
                    min_nonzero_steps: int) -> pd.DataFrame:
    """Filter on all requested spaces: any zero in any space drops trajectory."""
    rows = []
    for i, r in enumerate(records):
        steps = shared_steps(r["clean_h5"], r["pert_h5"])
        ok_steps = len(steps)
        has_zero = False
        with h5py.File(r["clean_h5"], "r") as c, h5py.File(r["pert_h5"], "r") as p:
            for step in steps:
                for space in spaces:
                    d = read_delta(c, p, step, layer, space)
                    if float(np.linalg.norm(d)) <= zero_eps:
                        has_zero = True
                        break
                if has_zero:
                    break
        rows.append({**{k: r[k] for k in ("suite", "task_name", "condition", "group", "variant_name")},
                     "record_index": i, "n_steps": ok_steps, "has_zero": has_zero,
                     "eligible": (not has_zero) and ok_steps >= min_nonzero_steps})
    return pd.DataFrame(rows)


def suite_balanced_sample(meta: pd.DataFrame, per_condition: int) -> pd.DataFrame:
    picked = []
    for cond in CONDITIONS:
        g = meta[(meta.condition == cond) & (meta.eligible)].copy()
        if g.empty:
            continue
        g = g.sort_values(["n_steps", "suite", "task_name"], ascending=[False, True, True])
        suites = sorted(g.suite.unique())
        base = per_condition // max(len(suites), 1)
        rem = per_condition % max(len(suites), 1)
        chosen = []
        for j, suite in enumerate(suites):
            take = base + (1 if j < rem else 0)
            chosen.append(g[g.suite == suite].head(take))
        chosen = pd.concat(chosen) if chosen else g.head(0)
        if len(chosen) < per_condition:
            rest = g.drop(index=chosen.index, errors="ignore").head(per_condition - len(chosen))
            chosen = pd.concat([chosen, rest])
        picked.append(chosen.head(per_condition))
    return pd.concat(picked, ignore_index=True) if picked else meta.head(0)


def load_points(records: list[dict], sample: pd.DataFrame, layer: str, space: str,
                normalize: bool, max_steps: int) -> tuple[np.ndarray, list[dict]]:
    vecs, rows = [], []
    by_index = {i: r for i, r in enumerate(records)}
    for traj_i, m in enumerate(sample.itertuples(index=False)):
        r = by_index[int(m.record_index)]
        steps = shared_steps(r["clean_h5"], r["pert_h5"])
        if max_steps:
            steps = steps[:max_steps]
        with h5py.File(r["clean_h5"], "r") as c, h5py.File(r["pert_h5"], "r") as p:
            for snap_i, step in enumerate(steps):
                delta = read_delta(c, p, step, layer, space)
                norm = float(np.linalg.norm(delta))
                if norm <= EPS:
                    raise RuntimeError(f"zero point survived filtering: {m.condition}/{m.task_name}/{step}")
                vec = delta / norm if normalize else delta
                vecs.append(vec.astype(np.float32, copy=False))
                rows.append({
                    "trajectory_id": f"{m.condition}::{m.suite}::{m.task_name}",
                    "sample_trajectory_index": traj_i,
                    "record_index": int(m.record_index),
                    "suite": m.suite,
                    "task_name": m.task_name,
                    "condition": m.condition,
                    "group": m.group,
                    "variant_name": m.variant_name,
                    "step": step,
                    "step_idx": int(step.split("_")[1]),
                    "snapshot_idx": snap_i,
                    "delta_norm": norm,
                })
    return np.stack(vecs, axis=0), rows


def reduce_umap(X: np.ndarray, pca_dim: int, n_neighbors: int, min_dist: float,
                metric: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import umap
    dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
    if dim < 2:
        raise RuntimeError(f"not enough points for dimensionality reduction: {X.shape}")
    reducer = TruncatedSVD(n_components=dim, random_state=seed) if X.shape[1] > 1000 else PCA(n_components=dim, random_state=seed)
    Xr = reducer.fit_transform(X)
    emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                    metric=metric, random_state=seed).fit_transform(Xr)
    return Xr, emb


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_overview(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    for tid, g in df.groupby("trajectory_id"):
        cond = g.condition.iloc[0]
        g = g.sort_values("snapshot_idx")
        ax.plot(g.umap_x, g.umap_y, color=COLORS.get(cond, "gray"), alpha=0.25, linewidth=0.8)
    for cond, g in df.groupby("condition"):
        ax.scatter(g.umap_x, g.umap_y, s=8, color=COLORS.get(cond, "gray"), alpha=0.62, label=cond)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_by_condition(df: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond, sub in df.groupby("condition"):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        max_snap = max(int(sub.snapshot_idx.max()), 1)
        for tid, g in sub.groupby("trajectory_id"):
            g = g.sort_values("snapshot_idx")
            ax.plot(g.umap_x, g.umap_y, color="0.55", alpha=0.35, linewidth=0.8)
            sc = ax.scatter(g.umap_x, g.umap_y, c=g.snapshot_idx / max_snap,
                            cmap="viridis", s=12, alpha=0.85)
            ax.scatter(g.umap_x.iloc[0], g.umap_y.iloc[0], facecolors="none", edgecolors="black", s=38, linewidths=0.8)
            ax.scatter(g.umap_x.iloc[-1], g.umap_y.iloc[-1], marker="s", facecolors="none", edgecolors="black", s=38, linewidths=0.8)
        ax.set_title(f"{title_prefix} - {cond}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="normalized time")
        fig.tight_layout()
        fig.savefig(out_dir / f"{cond}.png")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_manifest(in_dir)
    records = pair_records(df, in_dir)
    meta = trajectory_meta(records, args.layer, args.spaces, args.zero_eps, args.min_nonzero_steps)
    sample = suite_balanced_sample(meta, args.per_condition)
    meta.to_csv(out_dir / "trajectory_filter_manifest.csv", index=False)
    sample.to_csv(out_dir / "sampled_trajectories.csv", index=False)

    manifest = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "layer": args.layer,
        "spaces": args.spaces,
        "per_condition": args.per_condition,
        "sampled_trajectories": int(len(sample)),
        "eligible_by_condition": meta.groupby("condition")["eligible"].sum().astype(int).to_dict(),
        "sampled_by_condition": sample.groupby("condition")["record_index"].count().astype(int).to_dict(),
        "zero_rule": "drop entire trajectory if any requested space has a zero-shift step",
        "normalized_deltas": not args.no_normalize,
    }

    for space in args.spaces:
        print(f"[load] {args.layer} {space}: {len(sample)} trajectories", flush=True)
        X, rows = load_points(records, sample, args.layer, space, not args.no_normalize, args.max_steps)
        _, emb = reduce_umap(X, args.pca_dim, args.n_neighbors, args.min_dist, args.metric, args.seed)
        for r, xy in zip(rows, emb):
            r["layer"] = "action" if space == "final_action" else args.layer
            r["space"] = space
            r["umap_x"] = float(xy[0])
            r["umap_y"] = float(xy[1])
        space_dir = out_dir / f"{args.layer}_{space}"
        space_dir.mkdir(parents=True, exist_ok=True)
        write_csv(space_dir / "points.csv", rows)
        pts = pd.DataFrame(rows)
        plot_overview(pts, space_dir / "overview.png", f"{args.layer} {space} trajectory UMAP")
        plot_by_condition(pts, space_dir / "by_condition", f"{args.layer} {space}")
        manifest[f"{space}_points"] = int(len(rows))
        manifest[f"{space}_outputs"] = {
            "points": str(space_dir / "points.csv"),
            "overview": str(space_dir / "overview.png"),
            "by_condition": str(space_dir / "by_condition"),
        }

    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
