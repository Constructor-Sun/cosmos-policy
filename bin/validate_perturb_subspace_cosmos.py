#!/usr/bin/env python3
"""Offline validation for perturbation-specific subspaces.

This script reads Phase-2 paired artifacts and validates whether perturbation
shift subspaces generalize beyond the exact samples used to fit PCA.

It runs three checks:
  1. episode-level train/test projection within each perturbation,
  2. bootstrap stability of the fitted top-k PCA subspace,
  3. N-scaling via repeated subsampling.

No policy rollout or model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from perturb_subspace_probe_cosmos import (
    EPS,
    Subspace,
    discover_records,
    effective_rank_at_energy,
    explained_energy,
    fit_subspace,
    grouped_indices,
    load_space_matrix,
    projection_energy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    ROOT
    / "experiments/phase2_angular_cosmos/"
    "vla_jepa_kitchen_scene4_seed7_50case_corepert_preserved_flipped_last"
)
DEFAULT_OUT_DIR = ROOT / "experiments/phase3_perturb_subspace_validation/seed7_50case_last"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def aggregate(
    rows: list[dict[str, Any]],
    *,
    key_fields: list[str],
    value_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)

    out = []
    for key, items in sorted(grouped.items()):
        result = {field: value for field, value in zip(key_fields, key)}
        result["repeats"] = len(items)
        for field in value_fields:
            mu, sd = mean_std([float(item[field]) for item in items])
            result[f"{field}_mean"] = mu
            result[f"{field}_std"] = sd
        out.append(result)
    return out


def basis_from_subspace(subspace: Subspace, k: int) -> np.ndarray | None:
    kk = min(int(k), subspace.rank)
    if kk <= 0:
        return None

    U = subspace.eigvecs[:, :kk]
    S = subspace.singular_values[:kk]
    valid = S > EPS
    if not np.any(valid):
        return None

    U = U[:, valid]
    S = S[valid]
    basis = (U.T @ subspace.row_matrix) / S.reshape(-1, 1)
    basis = basis.astype(np.float64, copy=False)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + EPS
    return basis


def principal_angle_degrees(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    if basis_a.size == 0 or basis_b.size == 0:
        return np.asarray([], dtype=np.float64)
    cross = basis_a @ basis_b.T
    singular_values = np.linalg.svd(cross, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.degrees(np.arccos(singular_values))


def split_indices(
    rng: np.random.Generator,
    indices: list[int],
    *,
    test_frac: float,
    min_train: int,
    min_test: int,
) -> tuple[list[int], list[int]] | None:
    n = len(indices)
    test_n = max(min_test, int(round(n * test_frac)))
    train_n = n - test_n
    if train_n < min_train or test_n < min_test:
        return None
    perm = rng.permutation(indices)
    test = [int(i) for i in perm[:test_n]]
    train = [int(i) for i in perm[test_n:]]
    return train, test


def run_heldout(
    X_all: np.ndarray,
    records,
    group_keys: dict[tuple[str, str], list[int]],
    *,
    space: str,
    groups: list[str],
    ks: list[int],
    repeats: int,
    test_frac: float,
    min_train: int,
    min_test: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for (condition, group), indices in sorted(group_keys.items()):
        if group not in groups:
            continue
        for repeat in range(repeats):
            split = split_indices(
                rng,
                indices,
                test_frac=test_frac,
                min_train=min_train,
                min_test=min_test,
            )
            if split is None:
                continue
            train_idx, test_idx = split
            subspace = fit_subspace(
                X_all,
                records,
                space=space,
                condition=condition,
                group=group,
                indices=train_idx,
                min_n=min_train,
            )
            if subspace is None:
                continue

            cross_idx = [
                idx
                for (other_condition, other_group), other_indices in group_keys.items()
                if other_group == group and other_condition != condition
                for idx in other_indices
            ]
            for k in ks:
                same = projection_energy(subspace, X_all[test_idx], k)
                if cross_idx:
                    cross = projection_energy(subspace, X_all[cross_idx], k)
                else:
                    cross = np.asarray([float("nan")], dtype=np.float64)
                same_mu, same_sd = mean_std([float(v) for v in same])
                cross_mu, cross_sd = mean_std([float(v) for v in cross])
                rows.append(
                    {
                        "space": space,
                        "condition": condition,
                        "group": group,
                        "repeat": repeat,
                        "k": k,
                        "train_n": len(train_idx),
                        "test_n": len(test_idx),
                        "same_test_projection": same_mu,
                        "same_test_projection_std": same_sd,
                        "cross_projection": cross_mu,
                        "cross_projection_std": cross_sd,
                        "diag_over_cross": same_mu / cross_mu if cross_mu > EPS else float("nan"),
                    }
                )
    return rows


def run_bootstrap(
    X_all: np.ndarray,
    records,
    group_keys: dict[tuple[str, str], list[int]],
    *,
    space: str,
    groups: list[str],
    ks: list[int],
    repeats: int,
    min_n: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for (condition, group), indices in sorted(group_keys.items()):
        if group not in groups or len(indices) < min_n:
            continue
        full = fit_subspace(
            X_all,
            records,
            space=space,
            condition=condition,
            group=group,
            indices=indices,
            min_n=min_n,
        )
        if full is None:
            continue
        for repeat in range(repeats):
            sample = [int(i) for i in rng.choice(indices, size=len(indices), replace=True)]
            boot = fit_subspace(
                X_all,
                records,
                space=space,
                condition=condition,
                group=group,
                indices=sample,
                min_n=min_n,
            )
            if boot is None:
                continue
            for k in ks:
                full_basis = basis_from_subspace(full, k)
                boot_basis = basis_from_subspace(boot, k)
                if full_basis is None or boot_basis is None:
                    continue
                angles = principal_angle_degrees(full_basis, boot_basis)
                rows.append(
                    {
                        "space": space,
                        "condition": condition,
                        "group": group,
                        "repeat": repeat,
                        "k": min(k, len(angles)),
                        "n": len(indices),
                        "mean_angle_deg": float(angles.mean()) if len(angles) else float("nan"),
                        "max_angle_deg": float(angles.max()) if len(angles) else float("nan"),
                        "top_angle_deg": float(angles[0]) if len(angles) else float("nan"),
                    }
                )
    return rows


def run_n_scaling(
    X_all: np.ndarray,
    records,
    group_keys: dict[tuple[str, str], list[int]],
    *,
    space: str,
    groups: list[str],
    ks: list[int],
    sample_ns: list[int],
    repeats: int,
    min_test: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for (condition, group), indices in sorted(group_keys.items()):
        if group not in groups:
            continue
        max_n = len(indices)
        for train_n in [n for n in sample_ns if 2 <= n <= max_n]:
            for repeat in range(repeats):
                train_idx = [int(i) for i in rng.choice(indices, size=train_n, replace=False)]
                train_set = set(train_idx)
                test_idx = [idx for idx in indices if idx not in train_set]
                subspace = fit_subspace(
                    X_all,
                    records,
                    space=space,
                    condition=condition,
                    group=group,
                    indices=train_idx,
                    min_n=2,
                )
                if subspace is None:
                    continue
                r80 = effective_rank_at_energy(subspace.singular_values, 0.80)
                r90 = effective_rank_at_energy(subspace.singular_values, 0.90)
                has_test = len(test_idx) >= min_test
                for k in ks:
                    ev_values = {
                        f"ev_top{kk}": explained_energy(subspace.singular_values, kk)
                        for kk in ks
                    }
                    test_projection = float("nan")
                    if has_test:
                        test_projection = float(
                            np.nanmean(projection_energy(subspace, X_all[test_idx], k))
                        )
                    rows.append(
                        {
                            "space": space,
                            "condition": condition,
                            "group": group,
                            "repeat": repeat,
                            "train_n": train_n,
                            "test_n": len(test_idx),
                            "k": k,
                            "r80": r80,
                            "r90": r90,
                            **ev_values,
                            "test_projection": test_projection,
                        }
                    )
    return rows


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.replace(",", " ").split() if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--spaces", nargs="+", default=["hidden_video"])
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--layer", default="last")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-repeats", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=50)
    parser.add_argument("--subsample-repeats", type=int, default=50)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--min-train", type=int, default=8)
    parser.add_argument("--min-test", type=int, default=4)
    parser.add_argument("--subsample-ns", default="8 12 16 24 32 40 46")
    parser.add_argument("--normalize-deltas", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    results_dir = results_dir.resolve()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir.resolve()

    record_groups = {group for group in args.groups if group != "all"}
    if "all" in args.groups:
        record_groups.update({"flipped", "preserved"})
    records = discover_records(results_dir, record_groups)
    if not records:
        raise RuntimeError(f"No Phase-2 records found under {results_dir}")

    group_keys = grouped_indices(records, args.groups)
    ks = sorted({int(k) for k in args.ks if int(k) > 0})
    sample_ns = parse_ints(args.subsample_ns)
    rng = np.random.default_rng(args.seed)

    manifest = {
        "results_dir": str(results_dir),
        "output_dir": str(out_dir),
        "spaces": args.spaces,
        "groups": args.groups,
        "ks": ks,
        "layer": args.layer,
        "seed": args.seed,
        "split_repeats": args.split_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "subsample_repeats": args.subsample_repeats,
        "subsample_ns": sample_ns,
        "num_records": len(records),
        "record_counts": {},
    }
    for record in records:
        key = f"{record.condition}/{record.group}"
        manifest["record_counts"][key] = manifest["record_counts"].get(key, 0) + 1

    for space in args.spaces:
        print(f"[load] space={space} records={len(records)}", flush=True)
        X_all = load_space_matrix(
            records,
            space=space,
            layer=args.layer,
            normalize_deltas=args.normalize_deltas,
        )

        heldout = run_heldout(
            X_all,
            records,
            group_keys,
            space=space,
            groups=args.groups,
            ks=ks,
            repeats=args.split_repeats,
            test_frac=args.test_frac,
            min_train=args.min_train,
            min_test=args.min_test,
            rng=rng,
        )
        bootstrap = run_bootstrap(
            X_all,
            records,
            group_keys,
            space=space,
            groups=args.groups,
            ks=ks,
            repeats=args.bootstrap_repeats,
            min_n=args.min_train,
            rng=rng,
        )
        n_scaling = run_n_scaling(
            X_all,
            records,
            group_keys,
            space=space,
            groups=args.groups,
            ks=ks,
            sample_ns=sample_ns,
            repeats=args.subsample_repeats,
            min_test=args.min_test,
            rng=rng,
        )

        write_csv(
            out_dir / f"heldout_projection_detail_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "repeat",
                "k",
                "train_n",
                "test_n",
                "same_test_projection",
                "same_test_projection_std",
                "cross_projection",
                "cross_projection_std",
                "diag_over_cross",
            ],
            heldout,
        )
        write_csv(
            out_dir / f"heldout_projection_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "k",
                "repeats",
                "same_test_projection_mean",
                "same_test_projection_std",
                "cross_projection_mean",
                "cross_projection_std",
                "diag_over_cross_mean",
                "diag_over_cross_std",
            ],
            aggregate(
                heldout,
                key_fields=["space", "condition", "group", "k"],
                value_fields=["same_test_projection", "cross_projection", "diag_over_cross"],
            ),
        )

        write_csv(
            out_dir / f"bootstrap_stability_detail_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "repeat",
                "k",
                "n",
                "mean_angle_deg",
                "max_angle_deg",
                "top_angle_deg",
            ],
            bootstrap,
        )
        write_csv(
            out_dir / f"bootstrap_stability_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "k",
                "repeats",
                "mean_angle_deg_mean",
                "mean_angle_deg_std",
                "max_angle_deg_mean",
                "max_angle_deg_std",
                "top_angle_deg_mean",
                "top_angle_deg_std",
            ],
            aggregate(
                bootstrap,
                key_fields=["space", "condition", "group", "k"],
                value_fields=["mean_angle_deg", "max_angle_deg", "top_angle_deg"],
            ),
        )

        ev_fieldnames = [f"ev_top{k}" for k in ks]
        write_csv(
            out_dir / f"n_scaling_detail_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "repeat",
                "train_n",
                "test_n",
                "k",
                "r80",
                "r90",
                *ev_fieldnames,
                "test_projection",
            ],
            n_scaling,
        )
        write_csv(
            out_dir / f"n_scaling_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "train_n",
                "k",
                "repeats",
                "r80_mean",
                "r80_std",
                "r90_mean",
                "r90_std",
                *[f"{name}_mean" for name in ev_fieldnames],
                *[f"{name}_std" for name in ev_fieldnames],
                "test_projection_mean",
                "test_projection_std",
            ],
            aggregate(
                n_scaling,
                key_fields=["space", "condition", "group", "train_n", "k"],
                value_fields=["r80", "r90", *ev_fieldnames, "test_projection"],
            ),
        )

    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
