#!/usr/bin/env python3
"""Offline perturbation subspace probe for Cosmos Policy Phase-2 artifacts.

This script does not run the policy model. It reads the Phase-2 saved paired
artifacts described in COSMOS_POLICY_COUNTERFACTUAL_PHASES.md:

  action_clean.npy / action_pert.npy
  hidden_action_clean.pt / hidden_action_pert.pt
  hidden_video_clean.pt / hidden_video_pert.pt
  metrics.json

For every perturbation condition and group, it estimates a shift subspace from
delta = pert - clean, then reports:

  1. Low-rank structure of each condition/group subspace.
  2. Cross-reconstruction energy: how well a train subspace explains held-out
     deltas from another condition/group.

Groups include preserved, flipped, and all. The primary scientific comparison
should use preserved and flipped separately; all is a diagnostic view of the
overall perturbation shift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

EPS = 1e-12
DEFAULT_PHASES_MD = Path(__file__).resolve().parents[1] / "COSMOS_POLICY_COUNTERFACTUAL_PHASES.md"
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments/phase3_perturb_subspace_probe/from_phase2_preserved_flipped_last"
)


@dataclass
class Record:
    condition: str
    group: str
    episode: int
    pair_dir: Path
    action_error: float


@dataclass
class Subspace:
    key: tuple[str, str, str]
    indices: list[int]
    n: int
    dim: int
    rank: int
    singular_values: np.ndarray
    eigvecs: np.ndarray
    row_matrix: np.ndarray
    mean_pairwise_cos: float
    mean_resultant_length: float
    action_error_mean: float
    action_error_std: float


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_phase2_results_dir(phases_md: Path, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = phases_md.parent / path
        return path.resolve()

    text = phases_md.read_text(encoding="utf-8")
    candidates = []
    pattern = r"experiments/phase2_angular_cosmos/[^\s`]+preserved_flipped[^\s`]*"
    for match in re.finditer(pattern, text):
        rel = match.group(0).rstrip("/.,)")
        path = (phases_md.parent / rel).resolve()
        if path.exists() and (path / "summary.json").exists():
            candidates.append(path)

    if not candidates:
        raise RuntimeError(
            f"Could not find an existing Phase-2 preserved/flipped results dir in {phases_md}"
        )
    return candidates[0]


def discover_records(results_dir: Path, groups: set[str]) -> list[Record]:
    records = []
    for metrics_path in sorted(results_dir.glob("*/*/metrics.json")):
        metrics = load_json(metrics_path)
        group = str(metrics.get("group", ""))
        if group not in groups:
            continue
        action_error = float(metrics.get("action_error", float("nan")))
        records.append(
            Record(
                condition=str(metrics["condition"]),
                group=group,
                episode=int(metrics["episode"]),
                pair_dir=metrics_path.parent,
                action_error=action_error,
            )
        )
    return records


def load_action_delta(pair_dir: Path) -> np.ndarray:
    clean = np.load(pair_dir / "action_clean.npy").astype(np.float32).reshape(-1)
    pert = np.load(pair_dir / "action_pert.npy").astype(np.float32).reshape(-1)
    return pert - clean


def load_hidden_delta(pair_dir: Path, slot: str, layer: str) -> np.ndarray:
    clean = torch.load(pair_dir / f"hidden_{slot}_clean.pt", map_location="cpu", weights_only=True)
    pert = torch.load(pair_dir / f"hidden_{slot}_pert.pt", map_location="cpu", weights_only=True)
    clean_t = select_layer_tensor(clean, layer)
    pert_t = select_layer_tensor(pert, layer)
    return (pert_t.float().reshape(-1) - clean_t.float().reshape(-1)).numpy().astype(np.float32)


def select_layer_tensor(payload: dict[str, torch.Tensor], layer: str) -> torch.Tensor:
    if layer == "last":
        key = sorted(payload.keys(), key=lambda item: int(item))[-1]
        return payload[key]
    if layer not in payload:
        raise KeyError(f"layer {layer!r} not found; available={sorted(payload.keys())}")
    return payload[layer]


def load_space_matrix(
    records: list[Record],
    *,
    space: str,
    layer: str,
    normalize_deltas: bool,
) -> np.ndarray:
    rows = []
    for record in records:
        if space == "action":
            delta = load_action_delta(record.pair_dir)
        elif space == "hidden_action":
            delta = load_hidden_delta(record.pair_dir, "action", layer)
        elif space == "hidden_video":
            delta = load_hidden_delta(record.pair_dir, "video", layer)
        else:
            raise ValueError(f"unknown space: {space}")

        if normalize_deltas:
            delta = delta / (float(np.linalg.norm(delta)) + EPS)
        rows.append(delta.astype(np.float32, copy=False))

    return np.stack(rows, axis=0)


def grouped_indices(records: list[Record], groups: list[str]) -> dict[tuple[str, str], list[int]]:
    by_key: dict[tuple[str, str], list[int]] = {}
    conditions = sorted({record.condition for record in records})
    for condition in conditions:
        for group in groups:
            if group == "all":
                idx = [i for i, record in enumerate(records) if record.condition == condition]
            else:
                idx = [
                    i
                    for i, record in enumerate(records)
                    if record.condition == condition and record.group == group
                ]
            if idx:
                by_key[(condition, group)] = idx
    return by_key


def fit_subspace(
    X_all: np.ndarray,
    records: list[Record],
    *,
    space: str,
    condition: str,
    group: str,
    indices: list[int],
    min_n: int,
) -> Subspace | None:
    if len(indices) < min_n:
        return None

    X = X_all[indices].astype(np.float64, copy=False)
    n, dim = X.shape
    gram = X @ X.T
    evals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    eigvecs = eigvecs[:, order]
    tol = max(float(evals[0]) * 1e-10 if len(evals) else 0.0, EPS)
    rank = int(np.sum(evals > tol))
    singular_values = np.sqrt(evals)

    unit_rows = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    if n >= 2:
        cos = unit_rows @ unit_rows.T
        mean_pairwise_cos = float((cos.sum() - np.trace(cos)) / (n * (n - 1)))
    else:
        mean_pairwise_cos = float("nan")
    mean_resultant_length = float(np.linalg.norm(unit_rows.mean(axis=0)))

    action_errors = np.asarray([records[i].action_error for i in indices], dtype=np.float64)
    action_errors = action_errors[np.isfinite(action_errors)]
    action_error_mean = float(action_errors.mean()) if len(action_errors) else float("nan")
    action_error_std = float(action_errors.std()) if len(action_errors) else float("nan")

    return Subspace(
        key=(space, condition, group),
        indices=indices,
        n=n,
        dim=dim,
        rank=rank,
        singular_values=singular_values,
        eigvecs=eigvecs,
        row_matrix=X,
        mean_pairwise_cos=mean_pairwise_cos,
        mean_resultant_length=mean_resultant_length,
        action_error_mean=action_error_mean,
        action_error_std=action_error_std,
    )


def explained_energy(singular_values: np.ndarray, k: int) -> float:
    total = float(np.sum(singular_values**2))
    if total <= EPS:
        return float("nan")
    kk = min(k, len(singular_values))
    return float(np.sum(singular_values[:kk] ** 2) / total)


def effective_rank_at_energy(singular_values: np.ndarray, threshold: float) -> int | None:
    total = float(np.sum(singular_values**2))
    if total <= EPS:
        return None
    cumulative = np.cumsum(singular_values**2) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def projection_energy(train: Subspace, Y: np.ndarray, k: int) -> np.ndarray:
    kk = min(int(k), train.rank)
    if kk <= 0:
        return np.full((Y.shape[0],), np.nan, dtype=np.float64)

    U = train.eigvecs[:, :kk]
    S = train.singular_values[:kk]
    valid = S > EPS
    if not np.any(valid):
        return np.full((Y.shape[0],), np.nan, dtype=np.float64)

    U = U[:, valid]
    S = S[valid]
    coeff = (Y.astype(np.float64, copy=False) @ train.row_matrix.T) @ U
    coeff = coeff / S.reshape(1, -1)
    numerator = np.sum(coeff**2, axis=1)
    denominator = np.sum(Y.astype(np.float64, copy=False) ** 2, axis=1) + EPS
    return numerator / denominator


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(subspaces: list[Subspace], ks: list[int]) -> list[dict[str, Any]]:
    rows = []
    for subspace in subspaces:
        space, condition, group = subspace.key
        row: dict[str, Any] = {
            "space": space,
            "condition": condition,
            "group": group,
            "n": subspace.n,
            "dim": subspace.dim,
            "rank": subspace.rank,
            "mean_pairwise_cos": subspace.mean_pairwise_cos,
            "mean_resultant_length": subspace.mean_resultant_length,
            "effective_rank_80": effective_rank_at_energy(subspace.singular_values, 0.80),
            "effective_rank_90": effective_rank_at_energy(subspace.singular_values, 0.90),
            "action_error_mean": subspace.action_error_mean,
            "action_error_std": subspace.action_error_std,
        }
        for k in ks:
            row[f"ev_top{k}"] = explained_energy(subspace.singular_values, k)
        rows.append(row)
    return rows


def build_cross_rows(
    X_all: np.ndarray,
    subspaces: list[Subspace],
    test_groups: dict[tuple[str, str], list[int]],
    *,
    space: str,
    ks: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for train in subspaces:
        train_space, train_condition, train_group = train.key
        if train_space != space:
            continue
        for (test_condition, test_group), indices in sorted(test_groups.items()):
            Y = X_all[indices]
            for k in ks:
                energies = projection_energy(train, Y, k)
                finite = energies[np.isfinite(energies)]
                rows.append(
                    {
                        "space": space,
                        "k": k,
                        "train_condition": train_condition,
                        "train_group": train_group,
                        "train_n": train.n,
                        "test_condition": test_condition,
                        "test_group": test_group,
                        "test_n": len(indices),
                        "mean_projection_energy": float(finite.mean()) if len(finite) else float("nan"),
                        "std_projection_energy": float(finite.std()) if len(finite) else float("nan"),
                    }
                )
    return rows


def save_basis(
    out_dir: Path,
    subspace: Subspace,
    *,
    k: int,
) -> dict[str, Any] | None:
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
    basis = basis.astype(np.float32, copy=False)

    space, condition, group = subspace.key
    path = out_dir / "bases" / f"{space}__{condition}__{group}__top{basis.shape[0]}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        basis=basis,
        singular_values=S.astype(np.float32, copy=False),
        space=space,
        condition=condition,
        group=group,
        n=np.asarray([subspace.n], dtype=np.int64),
        dim=np.asarray([subspace.dim], dtype=np.int64),
    )
    return {
        "space": space,
        "condition": condition,
        "group": group,
        "k": int(basis.shape[0]),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases-md", default=str(DEFAULT_PHASES_MD))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--spaces", nargs="+", default=["action", "hidden_action", "hidden_video"])
    parser.add_argument("--groups", nargs="+", default=["flipped", "preserved", "all"])
    parser.add_argument("--fit-groups", nargs="+", default=["flipped", "preserved", "all"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layer", default="last")
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--normalize-deltas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-bases", action="store_true")
    parser.add_argument(
        "--basis-k",
        type=int,
        default=None,
        help="Top-k basis vectors to save when --save-bases is set. Defaults to max(--ks).",
    )
    args = parser.parse_args()

    phases_md = Path(args.phases_md).expanduser().resolve()
    results_dir = parse_phase2_results_dir(phases_md, args.results_dir)
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = phases_md.parent / out_dir
    out_dir = out_dir.resolve()

    record_groups = {group for group in args.groups if group != "all"}
    record_groups.update({group for group in args.fit_groups if group != "all"})
    records = discover_records(results_dir, record_groups)
    if not records:
        raise RuntimeError(f"No Phase-2 records found under {results_dir}")

    ks = sorted({int(k) for k in args.ks if int(k) > 0})
    fit_group_keys = grouped_indices(records, args.fit_groups)
    test_group_keys = grouped_indices(records, args.groups)

    all_summary_rows: list[dict[str, Any]] = []
    all_cross_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "phases_md": str(phases_md),
        "results_dir": str(results_dir),
        "output_dir": str(out_dir),
        "spaces": args.spaces,
        "groups": args.groups,
        "fit_groups": args.fit_groups,
        "ks": ks,
        "layer": args.layer,
        "min_n": args.min_n,
        "normalize_deltas": args.normalize_deltas,
        "save_bases": args.save_bases,
        "basis_k": args.basis_k if args.basis_k is not None else max(ks),
        "num_records": len(records),
        "record_counts": {},
        "basis_files": [],
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

        subspaces = []
        for (condition, group), indices in sorted(fit_group_keys.items()):
            subspace = fit_subspace(
                X_all,
                records,
                space=space,
                condition=condition,
                group=group,
                indices=indices,
                min_n=args.min_n,
            )
            if subspace is not None:
                subspaces.append(subspace)

        summary_rows = build_summary_rows(subspaces, ks)
        cross_rows = build_cross_rows(X_all, subspaces, test_group_keys, space=space, ks=ks)
        all_summary_rows.extend(summary_rows)
        all_cross_rows.extend(cross_rows)

        if args.save_bases:
            basis_k = args.basis_k if args.basis_k is not None else max(ks)
            for subspace in subspaces:
                basis_info = save_basis(out_dir, subspace, k=basis_k)
                if basis_info is not None:
                    manifest["basis_files"].append(basis_info)

        write_csv(
            out_dir / f"subspace_summary_{space}.csv",
            [
                "space",
                "condition",
                "group",
                "n",
                "dim",
                "rank",
                "mean_pairwise_cos",
                "mean_resultant_length",
                "effective_rank_80",
                "effective_rank_90",
                "action_error_mean",
                "action_error_std",
                *[f"ev_top{k}" for k in ks],
            ],
            summary_rows,
        )
        write_csv(
            out_dir / f"cross_reconstruction_{space}.csv",
            [
                "space",
                "k",
                "train_condition",
                "train_group",
                "train_n",
                "test_condition",
                "test_group",
                "test_n",
                "mean_projection_energy",
                "std_projection_energy",
            ],
            cross_rows,
        )

    write_csv(
        out_dir / "subspace_summary_all_spaces.csv",
        [
            "space",
            "condition",
            "group",
            "n",
            "dim",
            "rank",
            "mean_pairwise_cos",
            "mean_resultant_length",
            "effective_rank_80",
            "effective_rank_90",
            "action_error_mean",
            "action_error_std",
            *[f"ev_top{k}" for k in ks],
        ],
        all_summary_rows,
    )
    write_csv(
        out_dir / "cross_reconstruction_all_spaces.csv",
        [
            "space",
            "k",
            "train_condition",
            "train_group",
            "train_n",
            "test_condition",
            "test_group",
            "test_n",
            "mean_projection_energy",
            "std_projection_energy",
        ],
        all_cross_rows,
    )

    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
