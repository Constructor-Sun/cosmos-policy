#!/usr/bin/env python3
"""Measure paired radial/orthogonal latent shifts from Phase 8 saved samples."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
import torch


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--layers", type=int, nargs=2, default=[14, 27])
    parser.add_argument("--target", choices=["action", "video"], default="action")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_rows(root: pathlib.Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    with manifest.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_target(sample: dict[str, Any], layer: int, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(layer)
    pert = sample["latent_pert"]["layers"][key][target].float().reshape(-1)
    # Files store clean - perturb; reconstruct clean without another model pass.
    clean = pert + sample["latent_shift"]["layers"][key][target].float().reshape(-1)
    return clean, pert


def geometry(clean: torch.Tensor, pert: torch.Tensor) -> dict[str, float]:
    delta = pert - clean
    clean_sq = float(torch.dot(clean, clean))
    clean_norm = clean_sq**0.5
    pert_norm = float(torch.linalg.vector_norm(pert))
    delta_sq = float(torch.dot(delta, delta))
    dot = float(torch.dot(delta, clean))
    parallel_sq = dot * dot / max(clean_sq, EPS)
    perp_sq = max(0.0, delta_sq - parallel_sq)
    signed_parallel_rel = dot / max(clean_sq, EPS)
    r_perp = perp_sq**0.5 / max(clean_norm, EPS)
    cosine = float(torch.dot(clean, pert)) / max(clean_norm * pert_norm, EPS)
    cosine_pred = (1.0 + signed_parallel_rel) / max(
        ((1.0 + signed_parallel_rel) ** 2 + r_perp**2) ** 0.5, EPS
    )
    return {
        "clean_norm": clean_norm,
        "pert_norm": pert_norm,
        "delta_norm": delta_sq**0.5,
        "rel_delta": delta_sq**0.5 / max(clean_norm, EPS),
        "signed_parallel_rel": signed_parallel_rel,
        "abs_parallel_rel": abs(signed_parallel_rel),
        "r_perp": r_perp,
        "orthogonal_energy_fraction": perp_sq / max(delta_sq, EPS),
        "cosine": cosine,
        "cosine_from_decomposition": cosine_pred,
        "cosine_identity_abs_error": abs(cosine - cosine_pred),
    }


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    means = np.empty(draws, dtype=np.float64)
    # Chunk draws so the index matrix remains small.
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(records: list[dict[str, Any]], metrics: list[str], draws: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(records)}
    rng = np.random.default_rng(seed)
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in records], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "mean_bootstrap_95ci": bootstrap_ci(values, draws, rng),
            "n": int(values.size),
        }
    return result


def main() -> None:
    args = parse_args()
    root = pathlib.Path(args.input_dir).expanduser().resolve()
    output = pathlib.Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    if args.max_samples:
        rows = rows[: args.max_samples]

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    low, high = args.layers
    for index, row in enumerate(rows, 1):
        try:
            sample_path = (root / row["path"]).resolve()
            sample = torch.load(sample_path, map_location="cpu", weights_only=False)
            record = {key: row.get(key) for key in (
                "suite", "base_task", "condition", "sample_id", "init_state_index",
                "policy_seed", "env_seed", "action_mse", "action_rel_error"
            )}
            for layer in (low, high):
                clean, pert = flatten_target(sample, layer, args.target)
                record.update({f"l{layer}_{k}": v for k, v in geometry(clean, pert).items()})
            denominator = record[f"l{low}_r_perp"]
            record[f"r_perp_ratio_l{high}_over_l{low}"] = (
                record[f"l{high}_r_perp"] / denominator if denominator > EPS else float("nan")
            )
            records.append(record)
        except Exception as exc:
            errors.append({"path": str(row.get("path")), "error": repr(exc)})
        if index == 1 or index % 250 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)} (valid={len(records)}, errors={len(errors)})", flush=True)

    csv_path = output / "per_sample_geometry.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    layer_metrics = [
        "clean_norm", "pert_norm", "delta_norm", "rel_delta", "signed_parallel_rel",
        "abs_parallel_rel", "r_perp", "orthogonal_energy_fraction", "cosine",
        "cosine_identity_abs_error",
    ]
    metrics = [f"l{layer}_{metric}" for layer in (low, high) for metric in layer_metrics]
    ratio_metric = f"r_perp_ratio_l{high}_over_l{low}"
    metrics.append(ratio_metric)

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group_key in ("suite", "base_task", "condition"):
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(group_key))].append(record)
        groups[group_key] = {
            name: summarize(items, metrics, args.bootstrap, args.seed + i + 1)
            for i, (name, items) in enumerate(sorted(grouped.items()))
        }

    summary = {
        "input_dir": str(root),
        "target": args.target,
        "layers": [low, high],
        "num_manifest_rows": len(rows),
        "num_valid": len(records),
        "num_errors": len(errors),
        "aggregation_note": "Geometry is computed per sample before aggregation; bootstrap CIs are for arithmetic means.",
        "overall": summarize(records, metrics, args.bootstrap, args.seed),
        "by": groups,
        "errors": errors,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
        handle.write("\n")
    print(f"wrote {csv_path}")
    print(f"wrote {output / 'summary.json'}")


if __name__ == "__main__":
    main()
