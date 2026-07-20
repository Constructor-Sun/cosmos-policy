#!/usr/bin/env python3
"""Analyze Cosmos Phase 2 angular hidden/action metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any

import numpy as np

EPS = 1e-12
STAGES = ("video", "action_hidden")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        item = load_json(metrics_path)
        item["path"] = str(metrics_path)
        rows.append(item)
    return rows


def pick_layer(records: list[dict[str, Any]], layer: str) -> str:
    keys = set()
    for rec in records:
        keys.update(rec.get("angle_video_deg", {}).keys())
        keys.update(rec.get("angle_action_hidden_deg", {}).keys())
    if not keys:
        raise RuntimeError("No layer metrics found")
    if layer == "last":
        return max(keys, key=int)
    if layer == "mid":
        vals = sorted(int(k) for k in keys)
        return str(vals[len(vals) // 2])
    return str(int(layer))


def value_for(rec: dict[str, Any], stage: str, layer: str) -> float | None:
    key = "angle_video_deg" if stage == "video" else "angle_action_hidden_deg"
    val = rec.get(key, {}).get(layer)
    return None if val is None else float(val)


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def std(xs: list[float]) -> float:
    return float(np.std(xs)) if xs else float("nan")


def fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.6g}"


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xa, ya = np.asarray(x), np.asarray(y)
    if np.std(xa) < EPS or np.std(ya) < EPS:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def ranks(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals)
    order = np.argsort(arr)
    out = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(ranks(x).tolist(), ranks(y).tolist())


def write_points_csv(records: list[dict[str, Any]], layer: str, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "condition", "episode", "group", "layer", "video_angle_deg",
            "action_hidden_angle_deg", "action_error", "action_mse",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "condition": rec["condition"],
                "episode": rec["episode"],
                "group": rec["group"],
                "layer": layer,
                "video_angle_deg": value_for(rec, "video", layer),
                "action_hidden_angle_deg": value_for(rec, "action_hidden", layer),
                "action_error": rec.get("action_error"),
                "action_mse": rec.get("action_mse"),
            })


def preserved_flipped_table(records: list[dict[str, Any]], layer: str) -> list[dict[str, Any]]:
    rows = []
    conditions = sorted({rec["condition"] for rec in records})
    for condition in conditions:
        subset = [rec for rec in records if rec["condition"] == condition]
        for stage in STAGES:
            pv = [value_for(r, stage, layer) for r in subset if r["group"] == "preserved"]
            fv = [value_for(r, stage, layer) for r in subset if r["group"] == "flipped"]
            pv = [v for v in pv if v is not None]
            fv = [v for v in fv if v is not None]
            pm, fm = mean(pv), mean(fv)
            rows.append({
                "condition": condition,
                "stage": stage,
                "n_preserved": len(pv),
                "n_flipped": len(fv),
                "preserved_mean": pm,
                "preserved_std": std(pv),
                "flipped_mean": fm,
                "flipped_std": std(fv),
                "diff": fm - pm if not (math.isnan(pm) or math.isnan(fm)) else float("nan"),
                "ratio": fm / pm if not math.isnan(pm) and abs(pm) > EPS else float("nan"),
            })
    return rows


def correlation_rows(records: list[dict[str, Any]], layer: str) -> list[dict[str, Any]]:
    rows = []
    buckets = {"POOLED": records}
    for condition in sorted({r["condition"] for r in records}):
        buckets[condition] = [r for r in records if r["condition"] == condition]
    for name, subset in buckets.items():
        for stage in STAGES:
            pts = [(value_for(r, stage, layer), r.get("action_error")) for r in subset]
            pts = [(x, y) for x, y in pts if x is not None and y is not None]
            xs, ys = [x for x, _ in pts], [float(y) for _, y in pts]
            rows.append({
                "condition": name,
                "stage": stage,
                "n": len(xs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
                "mean_angle": mean(xs),
                "mean_action_error": mean(ys),
            })
    return rows


def print_table(title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    print(f"\n{title}")
    print(",".join(fields))
    for row in rows:
        vals = []
        for field in fields:
            val = row[field]
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        print(",".join(vals))


def write_summary(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    root = pathlib.Path(args.results_dir)
    records = collect(root)
    if args.groups:
        allowed = set(args.groups)
        records = [rec for rec in records if rec.get("group") in allowed]
    if not records:
        raise RuntimeError(f"No metrics found under {root}")
    layer = pick_layer(records, args.layer)
    pf_rows = preserved_flipped_table(records, layer)
    corr_rows = correlation_rows(records, layer)
    print(f"Loaded {len(records)} metrics from {root}; layer={layer}")
    print_table(
        "Preserved vs Flipped Angular",
        pf_rows,
        ["condition", "stage", "n_preserved", "n_flipped", "preserved_mean",
         "preserved_std", "flipped_mean", "flipped_std", "diff", "ratio"],
    )
    print_table(
        "Angular vs Action Error Correlation",
        corr_rows,
        ["condition", "stage", "n", "pearson", "spearman", "mean_angle", "mean_action_error"],
    )
    if args.csv:
        write_points_csv(records, layer, pathlib.Path(args.csv))
        print(f"\npoints_csv={args.csv}")
    if args.out_json:
        write_summary(pathlib.Path(args.out_json), {
            "results_dir": str(root),
            "layer": layer,
            "num_records": len(records),
            "preserved_flipped": pf_rows,
            "correlations": corr_rows,
        })
        print(f"summary_json={args.out_json}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--layer", default="last", help="last, mid, or integer layer")
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped", "recovery"])
    p.add_argument("--csv", default="")
    p.add_argument("--out-json", default="")
    return p


if __name__ == "__main__":
    main()
