#!/usr/bin/env python3
"""Analyze whether denoising Hessian curvature tracks perturbation fragility.

This is a lightweight post-processing script for outputs from
``run_denoise_hessian_cosmos.py``.  It joins pair-level perturbation outcomes
from ``summary.json`` with per-step/site Hessian estimates from
``denoise_hessian_delta.csv`` and reports:

  - correlations between curvature metrics and action deviation;
  - condition-centered correlations to reduce perturbation-type confounding;
  - AUC for preserved/flipped prediction, both pooled and within condition;
  - condition/group summary tables for reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

EPS = 1e-12
DEFAULT_RESULTS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "experiments/phase3_denoise_hessian/kitchen_scene4_seed7_all_perturb_clean_final_action_fd"
)


def finite_mask(*arrays: Iterable[float]) -> np.ndarray:
    mask = None
    for arr in arrays:
        vals = np.asarray(arr, dtype=float)
        valid = np.isfinite(vals)
        mask = valid if mask is None else (mask & valid)
    if mask is None:
        return np.asarray([], dtype=bool)
    return mask


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = finite_mask(xa, ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 3 or np.std(xa) < EPS or np.std(ya) < EPS:
        return float("nan"), float("nan")
    res = stats.pearsonr(xa, ya)
    return float(res.statistic), float(res.pvalue)


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = finite_mask(xa, ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 3 or np.std(xa) < EPS or np.std(ya) < EPS:
        return float("nan"), float("nan")
    res = stats.spearmanr(xa, ya)
    return float(res.statistic), float(res.pvalue)


def signed_log1p(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def log1p_clipped(values: pd.Series) -> pd.Series:
    return np.log1p(values.astype(float).clip(lower=0.0))


def demean_by_condition(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column] - frame.groupby("condition")[column].transform("mean")


def residualize_condition(frame: pd.DataFrame, column: str) -> pd.Series:
    """Residualize a column on condition dummies.

    Demeaning by condition is equivalent to residualizing on a categorical fixed
    effect for a single variable.  This helper keeps the operation explicit.
    """

    return demean_by_condition(frame, column)


def fixed_effect_delta_r2(frame: pd.DataFrame, predictor: str, outcome: str) -> tuple[float, float, float]:
    """Return baseline R^2, full R^2, and incremental R^2 over condition FE."""

    sub = frame[["condition", predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5 or sub[predictor].std() < EPS or sub[outcome].std() < EPS:
        return float("nan"), float("nan"), float("nan")

    try:
        enc = OneHotEncoder(drop=None, sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(drop=None, sparse=False)
    x_cond = enc.fit_transform(sub[["condition"]])
    y = sub[outcome].to_numpy(dtype=float)
    y_centered = y - y.mean()
    total = float(np.dot(y_centered, y_centered))
    if total < EPS:
        return float("nan"), float("nan"), float("nan")

    beta_base = np.linalg.lstsq(x_cond, y, rcond=None)[0]
    pred_base = x_cond @ beta_base
    r2_base = 1.0 - float(np.sum((y - pred_base) ** 2)) / total

    x_pred = sub[[predictor]].to_numpy(dtype=float)
    x_full = np.concatenate([x_cond, x_pred], axis=1)
    beta_full = np.linalg.lstsq(x_full, y, rcond=None)[0]
    pred_full = x_full @ beta_full
    r2_full = 1.0 - float(np.sum((y - pred_full) ** 2)) / total
    return r2_base, r2_full, r2_full - r2_base


def load_pair_frame(results_dir: pathlib.Path) -> pd.DataFrame:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for pair in summary["pairs"]:
        rows.append({
            "condition": pair["condition"],
            "episode": int(pair["episode"]),
            "group": pair["group"],
            "flipped_label": 1 if pair["group"] == "flipped" else 0,
            "pert_success": bool(pair["pert_success"]),
            "failure_label": 0 if bool(pair["pert_success"]) else 1,
            "action_error": float(pair["action_error_from_final_latent"]),
            "action_mse": float(pair.get("pert_action_mse_to_clean_final", float("nan"))),
        })
    return pd.DataFrame(rows)


def load_merged(results_dir: pathlib.Path, sites: set[str], calls: set[int] | None) -> pd.DataFrame:
    pairs = load_pair_frame(results_dir)
    delta = pd.read_csv(results_dir / "denoise_hessian_delta.csv")
    delta["episode"] = delta["episode"].astype(int)
    delta["denoise_call_index"] = delta["denoise_call_index"].astype(int)
    if sites:
        delta = delta[delta["site"].isin(sites)].copy()
    if calls is not None:
        delta = delta[delta["denoise_call_index"].isin(calls)].copy()

    merged = delta.merge(pairs, on=["condition", "episode", "group"], how="left", validate="many_to_one")
    if merged["action_error"].isna().any():
        missing = int(merged["action_error"].isna().sum())
        raise RuntimeError(f"{missing} Hessian rows did not match pair-level summary rows")

    for col in ["lambda_clean", "lambda_perturbed", "delta_lambda", "ratio_lambda", "loss_clean", "loss_perturbed", "delta_loss"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["log_lambda_clean"] = log1p_clipped(merged["lambda_clean"])
    merged["log_lambda_perturbed"] = log1p_clipped(merged["lambda_perturbed"])
    merged["signed_log_delta_lambda"] = signed_log1p(merged["delta_lambda"])
    merged["log_ratio_lambda"] = np.log(merged["ratio_lambda"].where(merged["ratio_lambda"] > 0.0))
    merged["log_action_mse"] = np.log1p(merged["action_mse"])
    merged["log_action_error"] = np.log1p(merged["action_error"])
    return merged


def correlation_rows(merged: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("lambda_clean", "raw clean curvature"),
        ("log_lambda_clean", "log clean curvature"),
        ("lambda_perturbed", "raw perturbed curvature"),
        ("log_lambda_perturbed", "log perturbed curvature"),
        ("delta_lambda", "raw curvature increase"),
        ("signed_log_delta_lambda", "signed log curvature increase"),
    ]
    outcomes = [
        ("action_mse", "clean-action MSE"),
        ("action_error", "relative action error"),
        ("log_action_mse", "log clean-action MSE"),
        ("log_action_error", "log relative action error"),
    ]
    rows = []
    for (call, site), frame in merged.groupby(["denoise_call_index", "site"], sort=True):
        frame = frame.copy()
        for metric, metric_desc in metrics:
            for outcome, outcome_desc in outcomes:
                sub = frame[["condition", metric, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                pearson, pearson_p = safe_pearson(sub[metric], sub[outcome])
                spearman, spearman_p = safe_spearman(sub[metric], sub[outcome])

                centered = sub.copy()
                centered[f"{metric}_cond_resid"] = residualize_condition(centered, metric)
                centered[f"{outcome}_cond_resid"] = residualize_condition(centered, outcome)
                c_metric = f"{metric}_cond_resid"
                c_outcome = f"{outcome}_cond_resid"
                c_pearson, c_pearson_p = safe_pearson(centered[c_metric], centered[c_outcome])
                c_spearman, c_spearman_p = safe_spearman(centered[c_metric], centered[c_outcome])
                r2_base, r2_full, delta_r2 = fixed_effect_delta_r2(sub, metric, outcome)

                rows.append({
                    "denoise_call_index": call,
                    "site": site,
                    "metric": metric,
                    "metric_desc": metric_desc,
                    "outcome": outcome,
                    "outcome_desc": outcome_desc,
                    "n": len(sub),
                    "pearson": pearson,
                    "pearson_p": pearson_p,
                    "spearman": spearman,
                    "spearman_p": spearman_p,
                    "condition_centered_pearson": c_pearson,
                    "condition_centered_pearson_p": c_pearson_p,
                    "condition_centered_spearman": c_spearman,
                    "condition_centered_spearman_p": c_spearman_p,
                    "condition_fe_r2": r2_base,
                    "condition_plus_metric_r2": r2_full,
                    "metric_delta_r2_over_condition": delta_r2,
                })
    return pd.DataFrame(rows)


def auc_rows(merged: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "lambda_clean",
        "log_lambda_clean",
        "lambda_perturbed",
        "log_lambda_perturbed",
        "delta_lambda",
        "signed_log_delta_lambda",
    ]
    rows = []
    for (call, site), frame in merged.groupby(["denoise_call_index", "site"], sort=True):
        for metric in metrics:
            sub = frame[[metric, "flipped_label", "condition"]].replace([np.inf, -np.inf], np.nan).dropna()
            y = sub["flipped_label"].to_numpy(dtype=int)
            x = sub[metric].to_numpy(dtype=float)
            if len(np.unique(y)) == 2 and np.std(x) >= EPS:
                overall_auc = float(roc_auc_score(y, x))
            else:
                overall_auc = float("nan")

            cond_aucs = []
            cond_weights = []
            cond_names = []
            for condition, cond in sub.groupby("condition", sort=True):
                cy = cond["flipped_label"].to_numpy(dtype=int)
                cx = cond[metric].to_numpy(dtype=float)
                if len(np.unique(cy)) != 2 or np.std(cx) < EPS:
                    continue
                cond_aucs.append(float(roc_auc_score(cy, cx)))
                cond_weights.append(len(cond))
                cond_names.append(condition)
            if cond_aucs:
                weighted_auc = float(np.average(cond_aucs, weights=cond_weights))
                mean_auc = float(np.mean(cond_aucs))
            else:
                weighted_auc = float("nan")
                mean_auc = float("nan")
            rows.append({
                "denoise_call_index": call,
                "site": site,
                "metric": metric,
                "n": len(sub),
                "n_flipped": int(y.sum()) if len(y) else 0,
                "n_preserved": int(len(y) - y.sum()) if len(y) else 0,
                "pooled_auc_flipped": overall_auc,
                "within_condition_auc_weighted": weighted_auc,
                "within_condition_auc_mean": mean_auc,
                "within_condition_auc_num_conditions": len(cond_aucs),
                "within_condition_auc_conditions": ";".join(cond_names),
            })
    return pd.DataFrame(rows)


def condition_group_summary(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "action_mse",
        "action_error",
        "lambda_clean",
        "lambda_perturbed",
        "delta_lambda",
        "loss_clean",
        "loss_perturbed",
        "delta_loss",
    ]
    grouped = merged.groupby(["condition", "group", "denoise_call_index", "site"], sort=True)
    rows = []
    for key, frame in grouped:
        row = {
            "condition": key[0],
            "group": key[1],
            "denoise_call_index": key[2],
            "site": key[3],
            "n": len(frame),
        }
        for col in cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_median"] = float(frame[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def top_rows_for_markdown(corr: pd.DataFrame, auc: pd.DataFrame) -> list[str]:
    lines = []
    focus = corr[
        (corr["outcome"] == "action_mse")
        & (corr["metric"].isin(["log_lambda_clean", "log_lambda_perturbed", "signed_log_delta_lambda"]))
    ].copy()
    focus["abs_condition_centered_spearman"] = focus["condition_centered_spearman"].abs()
    focus = focus.sort_values("abs_condition_centered_spearman", ascending=False).head(12)
    lines.append("## Top condition-centered correlations with action_mse")
    lines.append("")
    lines.append("| call | site | metric | n | pooled rho | condition-centered rho | delta R2 over condition |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in focus.iterrows():
        lines.append(
            f"| {int(row['denoise_call_index'])} | {row['site']} | {row['metric']} | {int(row['n'])} | "
            f"{row['spearman']:.4f} | {row['condition_centered_spearman']:.4f} | "
            f"{row['metric_delta_r2_over_condition']:.4f} |"
        )

    lines.append("")
    auc_focus = auc[auc["metric"].isin(["log_lambda_clean", "log_lambda_perturbed", "signed_log_delta_lambda"])].copy()
    auc_focus["auc_score"] = auc_focus["within_condition_auc_weighted"].fillna(auc_focus["pooled_auc_flipped"])
    auc_focus = auc_focus.sort_values("auc_score", ascending=False).head(12)
    lines.append("## Top flipped/preserved AUC")
    lines.append("")
    lines.append("| call | site | metric | pooled AUC | within-condition weighted AUC | within-condition conditions |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    for _, row in auc_focus.iterrows():
        lines.append(
            f"| {int(row['denoise_call_index'])} | {row['site']} | {row['metric']} | "
            f"{row['pooled_auc_flipped']:.4f} | {row['within_condition_auc_weighted']:.4f} | "
            f"{row['within_condition_auc_conditions']} |"
        )
    return lines


def write_markdown(
    path: pathlib.Path,
    results_dir: pathlib.Path,
    merged: pd.DataFrame,
    corr: pd.DataFrame,
    auc: pd.DataFrame,
    sites: list[str],
) -> None:
    lines = [
        "# Denoise Hessian Curvature Analysis",
        "",
        f"- results_dir: `{results_dir}`",
        f"- rows: `{len(merged)}`",
        f"- sites: `{', '.join(sites)}`",
        "- pair-level outcome: `action_mse = pert_action_mse_to_clean_final`",
        "- causal-leaning test: `lambda_clean` vs perturbation outcome",
        "- association test: `lambda_perturbed` / `delta_lambda` vs perturbation outcome",
        "- condition-centered columns subtract per-condition means before correlation",
        "",
    ]
    lines.extend(top_rows_for_markdown(corr, auc))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `curvature_pair_table.csv`: merged pair/outcome/curvature table")
    lines.append("- `curvature_correlations.csv`: Pearson/Spearman and condition-centered correlations")
    lines.append("- `curvature_auc.csv`: flipped/preserved AUC")
    lines.append("- `condition_group_summary.csv`: condition/group medians and means")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else results_dir / "curvature_analysis"
    sites = list(args.sites)
    calls = None if args.calls is None else set(args.calls)

    merged = load_merged(results_dir, set(sites), calls)
    corr = correlation_rows(merged)
    auc = auc_rows(merged)
    cond_summary = condition_group_summary(merged)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "curvature_pair_table.csv", index=False)
    corr.to_csv(out_dir / "curvature_correlations.csv", index=False)
    auc.to_csv(out_dir / "curvature_auc.csv", index=False)
    cond_summary.to_csv(out_dir / "condition_group_summary.csv", index=False)
    write_markdown(out_dir / "summary.md", results_dir, merged, corr, auc, sites)

    print(f"merged_rows={len(merged)}")
    print(f"out_dir={out_dir}")
    print("focus: clean curvature predicts susceptibility; perturbed/delta curvature tracks post-perturbation sharpness")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--sites", nargs="+", default=["vae", "layer0", "mid"])
    parser.add_argument("--calls", nargs="*", type=int, default=None)
    return parser


if __name__ == "__main__":
    main()
