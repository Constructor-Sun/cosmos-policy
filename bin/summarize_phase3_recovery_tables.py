#!/usr/bin/env python3
"""Print Phase 3 latent/action recovery tables from summary.json."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from statistics import mean
from typing import Any


def _fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.3f}"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _alpha_float(key: str) -> float:
    return float(key)


def collect_rows(
    summary: dict[str, Any],
    *,
    layer: str,
    latent_field: str,
    include_alpha_zero: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, group in summary.get("groups", {}).items():
        for ep, ep_data in group.get("results", {}).items():
            layer_data = ep_data.get("layers", {}).get(layer, {})
            for alpha_key, alpha_data in layer_data.get("per_alpha", {}).items():
                alpha = _alpha_float(alpha_key)
                if alpha == 0.0 and not include_alpha_zero:
                    continue
                latent_rec = alpha_data.get(latent_field)
                action_rec = alpha_data.get("recovery_vs_pert")
                if latent_rec is None or action_rec is None:
                    continue
                rows.append(
                    {
                        "condition": condition,
                        "episode": int(ep),
                        "alpha": alpha,
                        "latent_recovery": float(latent_rec),
                        "action_recovery": float(action_rec),
                    }
                )
    return rows


def print_correlation_table(rows: list[dict[str, Any]]) -> None:
    print("**Intervention / Correlation**")
    print()
    print("| Intervention | Correlation r |")
    print("|---|---:|")

    overall_r = _pearson(
        [row["latent_recovery"] for row in rows],
        [row["action_recovery"] for row in rows],
    )
    print(f"| Overall | {_fmt(overall_r)} |")

    for condition in sorted({row["condition"] for row in rows}):
        cond_rows = [row for row in rows if row["condition"] == condition]
        r = _pearson(
            [row["latent_recovery"] for row in cond_rows],
            [row["action_recovery"] for row in cond_rows],
        )
        print(f"| {condition} | {_fmt(r)} |")
    print()


def print_alpha_table(rows: list[dict[str, Any]]) -> None:
    print("**Alpha Sweep**")
    print()
    print("| Alpha | Mean video latent recovery | Final action recovery |")
    print("|---:|---:|---:|")
    for alpha in sorted({row["alpha"] for row in rows}):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        latent_mean = mean(row["latent_recovery"] for row in alpha_rows)
        action_mean = mean(row["action_recovery"] for row in alpha_rows)
        print(f"| {alpha:g} | {latent_mean:.3f} | {action_mean:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--latent-field", default="video_hidden_recovery_vs_pert")
    parser.add_argument("--include-alpha-zero", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    layer = args.layer
    if layer is None:
        layers = [str(layer) for layer in summary.get("layers", [])]
        if not layers:
            raise SystemExit("summary has no layers; pass --layer explicitly")
        layer = layers[-1]

    rows = collect_rows(
        summary,
        layer=str(layer),
        latent_field=args.latent_field,
        include_alpha_zero=True,
    )
    if not rows:
        raise SystemExit(
            f"no rows found for layer={layer} latent_field={args.latent_field}; "
            "make sure this is a video-target Phase 3 run"
        )

    corr_rows = rows if args.include_alpha_zero else [row for row in rows if row["alpha"] != 0.0]
    print_correlation_table(corr_rows)
    print_alpha_table(rows)


if __name__ == "__main__":
    main()
