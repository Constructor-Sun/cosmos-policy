#!/usr/bin/env python3
"""Phase 8 first-chunk shift subspace probe.

This is an offline reader for ``collect_shift_embeddings.py`` outputs.  It
uses the first saved snapshot (``data/step_0000`` by default), pairs each
perturbed HDF5 with the matching clean HDF5, computes

    delta = perturbed - clean

and then fits the same uncentered SVD/PCA-style subspaces used by
``perturb_subspace_probe_cosmos.py``:

    R[p -> q] = ||Proj_{S_p(k)} delta_q||^2 / ||delta_q||^2

Only the subspace diagnostics are implemented here; no rollout, intervention,
or model loading is performed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "experiments/phase8_shift_embeddings/all_suites_full"
DEFAULT_OUTPUT_DIR = ROOT / "experiments/phase8_first_chunk_subspace/all_suites_full"
DEFAULT_LAYERS = ["L0", "L14", "L27"]
DEFAULT_SPACES = ["hidden_video", "hidden_action", "final_action"]
DEFAULT_KS = [1, 2, 4, 8, 16, 32]


@dataclass(frozen=True)
class Record:
    index: int
    suite: str
    task_name: str
    condition: str
    group: str
    clean_h5: Path
    pert_h5: Path
    clean_success: bool
    pert_success: bool
    clean_total_steps: int
    pert_total_steps: int
    variant_name: str


@dataclass
class Subspace:
    axis: str
    key: str
    indices: list[int]
    n: int
    n_nonzero: int
    n_zero: int
    dim: int
    rank: int
    singular_values: np.ndarray
    eigvecs: np.ndarray
    row_matrix: np.ndarray
    mean_pairwise_cos: float
    mean_resultant_length: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--step", default="step_0000")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=DEFAULT_SPACES,
        choices=["hidden_video", "hidden_action", "hidden_video_action", "final_action"],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument(
        "--success-filter",
        choices=["clean-success", "all"],
        default="clean-success",
        help="Use only rows whose clean counterpart succeeded, or all pairable rows.",
    )
    parser.add_argument(
        "--normalize-deltas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each delta vector before fitting subspaces. This matches the Phase-3 probe default.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["global", "condition", "suite", "condition_suite", "condition_group"],
        choices=["global", "condition", "suite", "condition_suite", "condition_group"],
        help="Which grouping axes to summarize.",
    )
    parser.add_argument(
        "--cross-axes",
        nargs="+",
        default=["condition", "suite"],
        choices=["condition", "suite", "condition_suite", "condition_group"],
        help="Which grouping axes to use for cross-reconstruction matrices.",
    )
    return parser.parse_args()


def load_manifest(input_dir: Path) -> pd.DataFrame:
    manifest_path = input_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return pd.read_parquet(manifest_path)


def make_records(df: pd.DataFrame, input_dir: Path, *, success_filter: str) -> list[Record]:
    clean_rows = {
        (str(row.suite), str(row.task_name)): row
        for row in df[df["condition"] == "clean"].itertuples(index=False)
    }

    records: list[Record] = []
    for row in df[df["condition"] != "clean"].itertuples(index=False):
        key = (str(row.suite), str(row.task_name))
        clean = clean_rows.get(key)
        if clean is None:
            continue
        clean_success = bool(clean.success)
        pert_success = bool(row.success)
        if success_filter == "clean-success" and not clean_success:
            continue
        if clean_success and pert_success:
            group = "preserved"
        elif clean_success and not pert_success:
            group = "flipped"
        else:
            group = "other"

        records.append(
            Record(
                index=len(records),
                suite=str(row.suite),
                task_name=str(row.task_name),
                condition=str(row.condition),
                group=group,
                clean_h5=input_dir / str(clean.h5_path),
                pert_h5=input_dir / str(row.h5_path),
                clean_success=clean_success,
                pert_success=pert_success,
                clean_total_steps=int(clean.total_steps),
                pert_total_steps=int(row.total_steps),
                variant_name=str(row.variant_name),
            )
        )
    return records


def read_step_dataset(path: Path, step: str, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset_path = f"data/{step}/{key}"
        if dataset_path not in handle:
            raise KeyError(f"{dataset_path} not found in {path}")
        return handle[dataset_path][()]


def select_space(payload: np.ndarray, *, layer: str, space: str) -> np.ndarray:
    if space == "final_action":
        return payload.astype(np.float32, copy=False).reshape(-1)

    arr = payload.astype(np.float32, copy=False)
    if arr.shape != (3, 14, 14, 2048):
        raise ValueError(f"{layer} expected shape (3,14,14,2048), got {arr.shape}")

    # collect_shift_embeddings.py saves CAPTURE_TEMPORAL_SLOTS = [2, 3, 4]:
    #   0,1 = video slots (wrist, primary), 2 = action slot.
    if space == "hidden_video":
        return arr[:2].reshape(-1)
    if space == "hidden_action":
        return arr[2].reshape(-1)
    if space == "hidden_video_action":
        return arr.reshape(-1)
    raise ValueError(f"unknown space: {space}")


def load_delta_matrix(
    records: list[Record],
    *,
    layer: str,
    space: str,
    step: str,
    normalize_deltas: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    key = "action" if space == "final_action" else layer
    rows = []
    row_meta = []
    for record in records:
        try:
            clean = read_step_dataset(record.clean_h5, step, key)
            pert = read_step_dataset(record.pert_h5, step, key)
            delta = select_space(pert, layer=layer, space=space) - select_space(clean, layer=layer, space=space)
            norm = float(np.linalg.norm(delta))
            if normalize_deltas:
                delta = delta / (norm + EPS)
            rows.append(delta.astype(np.float32, copy=False))
            row_meta.append(
                {
                    "index": record.index,
                    "suite": record.suite,
                    "task_name": record.task_name,
                    "condition": record.condition,
                    "group": record.group,
                    "delta_norm": norm,
                    "pert_success": record.pert_success,
                    "variant_name": record.variant_name,
                }
            )
        except Exception as exc:  # keep the script useful on partially-written runs
            row_meta.append(
                {
                    "index": record.index,
                    "suite": record.suite,
                    "task_name": record.task_name,
                    "condition": record.condition,
                    "group": record.group,
                    "error": str(exc),
                }
            )
    if not rows:
        raise RuntimeError(f"no readable deltas for layer={layer} space={space} step={step}")
    return np.stack(rows, axis=0), row_meta


def group_indices(records: list[Record], axis: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        if axis == "global":
            key = "all"
        elif axis == "condition":
            key = record.condition
        elif axis == "suite":
            key = record.suite
        elif axis == "condition_suite":
            key = f"{record.condition}/{record.suite}"
        elif axis == "condition_group":
            key = f"{record.condition}/{record.group}"
        else:
            raise ValueError(f"unknown axis: {axis}")
        groups.setdefault(key, []).append(i)
    return groups


def fit_subspace(X_all: np.ndarray, *, axis: str, key: str, indices: list[int], min_n: int) -> Subspace | None:
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

    row_norms = np.linalg.norm(X, axis=1)
    nonzero = row_norms > EPS
    n_nonzero = int(np.sum(nonzero))
    n_zero = int(n - n_nonzero)
    unit_rows = X[nonzero] / (row_norms[nonzero, None] + EPS)
    if n_nonzero >= 2:
        cos = unit_rows @ unit_rows.T
        mean_pairwise_cos = float((cos.sum() - np.trace(cos)) / (n_nonzero * (n_nonzero - 1)))
    else:
        mean_pairwise_cos = float("nan")
    mean_resultant_length = float(np.linalg.norm(unit_rows.mean(axis=0))) if n_nonzero else float("nan")

    return Subspace(
        axis=axis,
        key=key,
        indices=indices,
        n=n,
        n_nonzero=n_nonzero,
        n_zero=n_zero,
        dim=dim,
        rank=rank,
        singular_values=singular_values,
        eigvecs=eigvecs,
        row_matrix=X,
        mean_pairwise_cos=mean_pairwise_cos,
        mean_resultant_length=mean_resultant_length,
    )


def effective_rank_at_energy(singular_values: np.ndarray, threshold: float) -> int | None:
    total = float(np.sum(singular_values**2))
    if total <= EPS:
        return None
    cumulative = np.cumsum(singular_values**2) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def explained_energy(singular_values: np.ndarray, k: int) -> float:
    total = float(np.sum(singular_values**2))
    if total <= EPS:
        return float("nan")
    kk = min(int(k), len(singular_values))
    return float(np.sum(singular_values[:kk] ** 2) / total)


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
    denominator = np.sum(Y.astype(np.float64, copy=False) ** 2, axis=1)
    out = np.full((Y.shape[0],), np.nan, dtype=np.float64)
    valid_rows = denominator > EPS
    out[valid_rows] = numerator[valid_rows] / denominator[valid_rows]
    return out


def summarize_subspaces(
    subspaces: list[Subspace],
    *,
    layer: str,
    space: str,
    ks: list[int],
    records: list[Record],
) -> list[dict[str, Any]]:
    rows = []
    for subspace in subspaces:
        group_records = [records[i] for i in subspace.indices]
        row: dict[str, Any] = {
            "layer": layer,
            "space": space,
            "axis": subspace.axis,
            "key": subspace.key,
            "n": subspace.n,
            "n_nonzero": subspace.n_nonzero,
            "n_zero": subspace.n_zero,
            "dim": subspace.dim,
            "rank": subspace.rank,
            "r80": effective_rank_at_energy(subspace.singular_values, 0.80),
            "r90": effective_rank_at_energy(subspace.singular_values, 0.90),
            "r95": effective_rank_at_energy(subspace.singular_values, 0.95),
            "mean_pairwise_cos": subspace.mean_pairwise_cos,
            "mean_resultant_length": subspace.mean_resultant_length,
            "n_flipped": sum(1 for record in group_records if record.group == "flipped"),
            "n_preserved": sum(1 for record in group_records if record.group == "preserved"),
            "n_other": sum(1 for record in group_records if record.group == "other"),
            "n_suites": len({record.suite for record in group_records}),
            "n_conditions": len({record.condition for record in group_records}),
        }
        for k in ks:
            row[f"ev_top{k}"] = explained_energy(subspace.singular_values, k)
        rows.append(row)
    return rows


def cross_reconstruction_rows(
    X_all: np.ndarray,
    subspaces: list[Subspace],
    test_groups: dict[str, list[int]],
    *,
    layer: str,
    space: str,
    ks: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for train in subspaces:
        for test_key, indices in sorted(test_groups.items()):
            Y = X_all[indices]
            for k in ks:
                energies = projection_energy(train, Y, k)
                finite = energies[np.isfinite(energies)]
                rows.append(
                    {
                        "layer": layer,
                        "space": space,
                        "axis": train.axis,
                        "k": k,
                        "train_key": train.key,
                        "train_n": train.n,
                        "test_key": test_key,
                        "test_n": len(indices),
                        "is_diagonal": train.key == test_key,
                        "mean_projection_energy": float(finite.mean()) if len(finite) else float("nan"),
                        "std_projection_energy": float(finite.std()) if len(finite) else float("nan"),
                        "n_finite": int(len(finite)),
                    }
                )
    return rows


def cross_axis_summary(cross_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}
    for row in cross_rows:
        if not np.isfinite(row["mean_projection_energy"]):
            continue
        key = (row["layer"], row["space"], row["axis"], int(row["k"]))
        side = "diag" if row["is_diagonal"] else "offdiag"
        buckets.setdefault(key, {"diag": [], "offdiag": []})[side].append(float(row["mean_projection_energy"]))

    out = []
    for (layer, space, axis, k), values in sorted(buckets.items()):
        diag = values["diag"]
        offdiag = values["offdiag"]
        diag_mean = float(np.mean(diag)) if diag else float("nan")
        offdiag_mean = float(np.mean(offdiag)) if offdiag else float("nan")
        out.append(
            {
                "layer": layer,
                "space": space,
                "axis": axis,
                "k": k,
                "diagonal_mean": diag_mean,
                "off_diagonal_mean": offdiag_mean,
                "diag_offdiag_ratio": diag_mean / (offdiag_mean + EPS)
                if np.isfinite(diag_mean) and np.isfinite(offdiag_mean)
                else float("nan"),
                "n_diagonal_cells": len(diag),
                "n_off_diagonal_cells": len(offdiag),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = input_dir.resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ks = sorted({int(k) for k in args.ks if int(k) > 0})
    df = load_manifest(input_dir)
    records = make_records(df, input_dir, success_filter=args.success_filter)
    if not records:
        raise RuntimeError("no pairable perturbation records found")

    all_summary_rows: list[dict[str, Any]] = []
    all_cross_rows: list[dict[str, Any]] = []
    all_cross_axis_rows: list[dict[str, Any]] = []
    readable_counts: list[dict[str, Any]] = []

    print(f"[input] {input_dir}")
    print(f"[records] pairable perturb rows={len(records)}")
    print(f"[output] {output_dir}")

    for layer in args.layers:
        for space in args.spaces:
            if space == "final_action" and layer != args.layers[0]:
                # final_action is layer-independent; compute it once.
                continue
            layer_label = "action" if space == "final_action" else layer
            print(f"[load] layer={layer_label} space={space} step={args.step}", flush=True)
            X_all, row_meta = load_delta_matrix(
                records,
                layer=layer,
                space=space,
                step=args.step,
                normalize_deltas=args.normalize_deltas,
            )
            if X_all.shape[0] != len(records):
                raise RuntimeError("partial HDF5 read support is not implemented for grouped indices")
            readable_counts.append({"layer": layer_label, "space": space, "n": int(X_all.shape[0]), "dim": int(X_all.shape[1])})
            write_csv(output_dir / f"delta_rows__{layer_label}__{space}.csv", row_meta)

            ordered_axes = []
            for axis in list(args.axes) + list(args.cross_axes):
                if axis not in ordered_axes:
                    ordered_axes.append(axis)

            for axis in ordered_axes:
                subspaces = []
                for key, indices in sorted(group_indices(records, axis).items()):
                    subspace = fit_subspace(X_all, axis=axis, key=key, indices=indices, min_n=args.min_n)
                    if subspace is not None:
                        subspaces.append(subspace)

                if axis in args.axes:
                    all_summary_rows.extend(
                        summarize_subspaces(subspaces, layer=layer_label, space=space, ks=ks, records=records)
                    )

                if axis in args.cross_axes:
                    current_cross_rows = cross_reconstruction_rows(
                        X_all,
                        subspaces,
                        group_indices(records, axis),
                        layer=layer_label,
                        space=space,
                        ks=ks,
                    )
                    all_cross_rows.extend(current_cross_rows)
                    all_cross_axis_rows.extend(cross_axis_summary(current_cross_rows))

                del subspaces
                gc.collect()

            del X_all
            gc.collect()

    write_csv(output_dir / "subspace_summary.csv", all_summary_rows)
    write_csv(output_dir / "cross_reconstruction.csv", all_cross_rows)
    write_csv(output_dir / "cross_axis_summary.csv", all_cross_axis_rows)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "step": args.step,
        "layers": args.layers,
        "spaces": args.spaces,
        "ks": ks,
        "min_n": args.min_n,
        "success_filter": args.success_filter,
        "normalize_deltas": args.normalize_deltas,
        "axes": args.axes,
        "cross_axes": args.cross_axes,
        "manifest_rows": int(len(df)),
        "pairable_perturb_rows": len(records),
        "record_counts": {
            "by_condition": {k: len(v) for k, v in sorted(group_indices(records, "condition").items())},
            "by_suite": {k: len(v) for k, v in sorted(group_indices(records, "suite").items())},
            "by_group": {k: len(v) for k, v in sorted(group_indices(records, "condition_group").items())},
        },
        "matrices": readable_counts,
        "outputs": {
            "subspace_summary": str(output_dir / "subspace_summary.csv"),
            "cross_reconstruction": str(output_dir / "cross_reconstruction.csv"),
            "cross_axis_summary": str(output_dir / "cross_axis_summary.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[done] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
