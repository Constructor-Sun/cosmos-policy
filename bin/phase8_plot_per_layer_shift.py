#!/usr/bin/env python3
"""Visualise per-layer latent shift from phase8_analyze_per_layer_shift.py output.

Supports multi-step data: when the JSONL contains a "denoising_step" field, plots
show separate curves for each step, enabling comparison of how the perturbation
shift evolves across denoising steps.

Usage:
    python bin/phase8_plot_per_layer_shift.py \
        --input-dir experiments/phase8_eval_latent_correction/per_layer_shift_analysis \
        --output-dir experiments/phase8_eval_latent_correction/per_layer_shift_analysis/plots
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def load_per_case(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_ascii_plot(
    layers: list[int],
    values: dict[int, float],
    stds: dict[int, float] | None = None,
    title: str = "",
    ylabel: str = "",
    width: int = 70,
    height: int = 20,
) -> str:
    """Generate an ASCII plot that renders directly in the terminal."""
    y_vals = [values[l] for l in layers]
    if stds:
        y_min = min(values[l] - stds[l] for l in layers)
        y_max = max(values[l] + stds[l] for l in layers)
    else:
        y_min, y_max = min(y_vals), max(y_vals)

    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01

    y_range = y_max - y_min
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    y_range = y_max - y_min

    x_min, x_max = min(layers), max(layers)
    plot_w = width - 10

    canvas = [[" "] * width for _ in range(height)]
    for row in range(height):
        canvas[row][7] = "│"
    for col in range(8, width):
        canvas[height - 1][col] = "─"

    for l in layers:
        col = 8 + int((l - x_min) / max(x_max - x_min, 1) * (plot_w - 1))
        row = height - 1 - int((values[l] - y_min) / y_range * (height - 1))
        row = max(0, min(height - 1, row))
        canvas[row][col] = "●"
        if stds:
            val_hi = values[l] + stds[l]
            val_lo = values[l] - stds[l]
            row_hi = height - 1 - int((val_hi - y_min) / y_range * (height - 1))
            row_lo = height - 1 - int((val_lo - y_min) / y_range * (height - 1))
            row_hi = max(0, min(height - 1, row_hi))
            row_lo = max(0, min(height - 1, row_lo))
            for r in range(row_lo, row_hi + 1):
                if canvas[r][col] == " ":
                    canvas[r][col] = "│"

    for i in range(5):
        row = i * (height - 1) // 4
        val = y_max - i * y_range / 4
        tick = f"{val:7.4f}"
        for j, ch in enumerate(tick):
            if row < height:
                canvas[row][j] = ch

    for i, ch in enumerate(title):
        if i < width:
            canvas[0][i] = ch

    sorted_layers = sorted(layers)
    for a, b in zip(sorted_layers[:-1], sorted_layers[1:]):
        col_a = 8 + int((a - x_min) / max(x_max - x_min, 1) * (plot_w - 1))
        col_b = 8 + int((b - x_min) / max(x_max - x_min, 1) * (plot_w - 1))
        row_a = height - 1 - int((values[a] - y_min) / y_range * (height - 1))
        row_b = height - 1 - int((values[b] - y_min) / y_range * (height - 1))
        row_a = max(0, min(height - 1, row_a))
        row_b = max(0, min(height - 1, row_b))
        steps = max(abs(col_b - col_a), abs(row_b - row_a), 1)
        for s in range(steps + 1):
            c = col_a + (col_b - col_a) * s // steps
            r = row_a + (row_b - row_a) * s // steps
            c = max(0, min(width - 1, c))
            r = max(0, min(height - 1, r))
            if canvas[r][c] == " ":
                canvas[r][c] = "·"

    return "\n".join("".join(row) for row in canvas)


# ── Multi-step ASCII overlay ─────────────────────────────────────────────────

def make_ascii_plot_multi_step(
    layers: list[int],
    step_values: dict[int, dict[int, float]],  # {step: {layer: value}}
    title: str = "",
    width: int = 70,
    height: int = 20,
) -> str:
    """ASCII plot overlaying multiple step curves with different markers."""
    markers = ["●", "▲", "■", "◆", "★"]
    colors_hint = {0: "step0", 1: "step1", 2: "step2", 3: "step3", 4: "step4"}

    all_vals = [v for sv in step_values.values() for v in sv.values()]
    y_min, y_max = min(all_vals), max(all_vals)
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    y_range = y_max - y_min
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    y_range = y_max - y_min

    x_min, x_max = min(layers), max(layers)
    plot_w = width - 10

    canvas = [[" "] * width for _ in range(height)]
    for row in range(height):
        canvas[row][7] = "│"
    for col in range(8, width):
        canvas[height - 1][col] = "─"

    for si, (step, values) in enumerate(sorted(step_values.items())):
        marker = markers[si % len(markers)]
        for l in layers:
            if l not in values:
                continue
            col = 8 + int((l - x_min) / max(x_max - x_min, 1) * (plot_w - 1))
            row = height - 1 - int((values[l] - y_min) / y_range * (height - 1))
            row = max(0, min(height - 1, row))
            canvas[row][col] = marker

        # Connect with lines
        valid = sorted([l for l in layers if l in values])
        for a, b in zip(valid[:-1], valid[1:]):
            col_a = 8 + int((a - x_min) / max(x_max - x_min, 1) * (plot_w - 1))
            col_b = 8 + int((b - x_min) / max(x_max - x_min, 1) * (plot_w - 1))
            row_a = height - 1 - int((values[a] - y_min) / y_range * (height - 1))
            row_b = height - 1 - int((values[b] - y_min) / y_range * (height - 1))
            row_a = max(0, min(height - 1, row_a))
            row_b = max(0, min(height - 1, row_b))
            st = max(abs(col_b - col_a), abs(row_b - row_a), 1)
            for s in range(st + 1):
                c = col_a + (col_b - col_a) * s // st
                r = row_a + (row_b - row_a) * s // st
                c = max(0, min(width - 1, c))
                r = max(0, min(height - 1, r))
                if canvas[r][c] == " ":
                    canvas[r][c] = "·"

    for i in range(5):
        row = i * (height - 1) // 4
        val = y_max - i * y_range / 4
        tick = f"{val:7.4f}"
        for j, ch in enumerate(tick):
            if row < height:
                canvas[row][j] = ch

    for i, ch in enumerate(title):
        if i < width:
            canvas[0][i] = ch

    # Legend
    legend_parts = []
    for si, step in enumerate(sorted(step_values.keys())):
        legend_parts.append(f"{markers[si]} step{step}")
    legend = "  ".join(legend_parts)
    for i, ch in enumerate(legend):
        if i < width and (height - 1) < height:
            canvas[height - 1][i] = ch

    return "\n".join("".join(row) for row in canvas)


# ── Matplotlib helpers ───────────────────────────────────────────────────────

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


_STEP_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]


def make_matplotlib_plots(
    layers: list[int],
    summary_data: dict[str, Any],
    per_case: list[dict[str, Any]],
    output_dir: pathlib.Path,
) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_summary = summary_data.get("layer_summary", {})
    if isinstance(list(layer_summary.keys())[0], str):
        layer_summary = {k: v for k, v in layer_summary.items()}

    injection_layer = summary_data.get("injection_layer", -1)
    steps = sorted({r["denoising_step"] for r in per_case}) if per_case else [None]

    # 1) Per-step overlay: shift_L2 vs layer (one line per step)
    fig, ax = plt.subplots(figsize=(12, 6))
    for si, step in enumerate(steps):
        vals = {}
        errs = {}
        for layer_idx in layers:
            key = f"step{step}_layer{layer_idx}"
            if key in layer_summary:
                vals[layer_idx] = layer_summary[key]["shift_l2"]["mean"]
                errs[layer_idx] = layer_summary[key]["shift_l2"]["std"]
        if not vals:
            continue
        ls, ls_sorted = sorted(vals.keys()), [vals[l] for l in sorted(vals.keys())]
        color = _STEP_COLORS[si % len(_STEP_COLORS)]
        ax.plot(ls_sorted, ls, "o-", color=color, linewidth=1.5, markersize=3,
                label=f"step {step}")
        ax.fill_between(ls_sorted,
                        [vals[l] - errs[l] for l in ls_sorted],
                        [vals[l] + errs[l] for l in ls_sorted],
                        alpha=0.1, color=color)
    if injection_layer in layers:
        ax.axvline(x=injection_layer, color="red", linestyle=":", alpha=0.5,
                   label=f"injection L{injection_layer}")
    ax.set_title("Shift L2 vs Layer (by denoising step)")
    ax.set_xlabel("DiT Block (Layer)")
    ax.set_ylabel("Shift L2 Norm")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_layer_shift_by_step.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_layer_shift_by_step.png'}")

    # 2) Per-step overlay: cosine_similarity vs layer
    fig, ax = plt.subplots(figsize=(12, 6))
    for si, step in enumerate(steps):
        vals, errs = {}, {}
        for layer_idx in layers:
            key = f"step{step}_layer{layer_idx}"
            if key in layer_summary:
                vals[layer_idx] = layer_summary[key]["cosine_similarity"]["mean"]
                errs[layer_idx] = layer_summary[key]["cosine_similarity"]["std"]
        if not vals:
            continue
        ls = sorted(vals.keys())
        color = _STEP_COLORS[si % len(_STEP_COLORS)]
        ax.plot(ls, [vals[l] for l in ls], "s-", color=color, linewidth=1.5, markersize=3,
                label=f"step {step}")
        ax.fill_between(ls,
                        [vals[l] - errs[l] for l in ls],
                        [vals[l] + errs[l] for l in ls],
                        alpha=0.1, color=color)
    if injection_layer in layers:
        ax.axvline(x=injection_layer, color="red", linestyle=":", alpha=0.5)
    ax.set_title("Cosine Similarity vs Layer (by denoising step)")
    ax.set_xlabel("DiT Block (Layer)")
    ax.set_ylabel("Cosine Similarity (clean vs perturb)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_layer_cosim_by_step.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_layer_cosim_by_step.png'}")

    # 3) Shift evolution across steps (at selected layers)
    # Pick layers: 0, injection_layer//2, injection_layer, num_blocks-1
    fig, ax = plt.subplots(figsize=(10, 6))
    highlight_layers = sorted(set([0, injection_layer // 2, injection_layer, layers[-1]]))
    for li, layer_idx in enumerate(highlight_layers):
        if layer_idx not in layers:
            continue
        vals = []
        for step in steps:
            key = f"step{step}_layer{layer_idx}"
            if key in layer_summary:
                vals.append(layer_summary[key]["shift_l2"]["mean"])
        if len(vals) == len(steps):
            color = plt.cm.plasma(li / max(len(highlight_layers) - 1, 1))
            ax.plot(steps, vals, "o-", color=color, linewidth=1.5, markersize=5,
                    label=f"layer {layer_idx}")
    ax.set_title("Shift L2 Across Denoising Steps (at selected layers)")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Shift L2 Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "shift_across_steps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'shift_across_steps.png'}")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-ascii", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_dir = pathlib.Path(args.input_dir).expanduser()
    per_case_path = input_dir / "per_layer_per_case.jsonl"
    summary_path = input_dir / "per_layer_summary.json"

    if not per_case_path.exists():
        print(f"ERROR: {per_case_path} not found.")
        sys.exit(1)
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found.")
        sys.exit(1)

    per_case = load_per_case(per_case_path)
    summary_data = load_summary(summary_path)
    layer_summary = summary_data.get("layer_summary", {})
    injection_layer = summary_data.get("injection_layer", -1)

    # Extract layers and steps
    layers = sorted({r["layer"] for r in per_case})
    has_steps = any("denoising_step" in r for r in per_case)
    steps = sorted({r["denoising_step"] for r in per_case}) if has_steps else []

    print(f"Loaded {len(per_case)} records across {len(layers)} layers ({layers[0]}..{layers[-1]})")
    if has_steps:
        print(f"Denoising steps: {steps}")
    print(f"Injection layer: {injection_layer}")

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else (input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ASCII plots ──────────────────────────────────────────────────────
    if not args.no_ascii:
        if has_steps:
            # Multi-step overlay for shift_L2
            step_vals = {}
            for step in steps:
                sv = {}
                for l in layers:
                    key = f"step{step}_layer{l}"
                    if key in layer_summary:
                        sv[l] = layer_summary[key]["shift_l2"]["mean"]
                if sv:
                    step_vals[step] = sv
            print("\n=== Shift L2 vs Layer (multi-step overlay) ===")
            print(make_ascii_plot_multi_step(layers, step_vals,
                                             title="Shift L2 vs Layer by denoising step"))

            print("\n=== Cosine Similarity vs Layer (multi-step overlay) ===")
            step_cos = {}
            for step in steps:
                sv = {}
                for l in layers:
                    key = f"step{step}_layer{l}"
                    if key in layer_summary:
                        sv[l] = layer_summary[key]["cosine_similarity"]["mean"]
                if sv:
                    step_cos[step] = sv
            print(make_ascii_plot_multi_step(layers, step_cos,
                                             title="Cosine Similarity vs Layer by denoising step"))
        else:
            # Single-step plots
            for key, title in [("shift_l2", "Shift L2 Norm vs Layer"),
                               ("cosine_similarity", "Cosine Similarity vs Layer"),
                               ("shift_mse", "Shift MSE vs Layer")]:
                # Reconstruct from layer_summary (which may have step prefix)
                vals = {}
                stds = {}
                for l in layers:
                    # Try various key formats
                    for lk in layer_summary:
                        if str(l) in lk and isinstance(layer_summary[lk], dict) and key in layer_summary[lk]:
                            vals[l] = layer_summary[lk][key]["mean"]
                            stds[l] = layer_summary[lk][key]["std"]
                            break
                if vals:
                    print(f"\n{make_ascii_plot(layers, vals, stds, title=title)}")

    # ── Matplotlib plots ─────────────────────────────────────────────────
    ok = make_matplotlib_plots(layers, summary_data, per_case, output_dir)
    if not ok:
        print("\n[INFO] matplotlib not available; PNG plots not generated.")
        print(f"       See {output_dir} for text-based output.")


if __name__ == "__main__":
    main()
