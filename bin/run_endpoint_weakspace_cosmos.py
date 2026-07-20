#!/usr/bin/env python3
"""Endpoint weak-space comparison for Cosmos Policy preserved vs flipped pairs.

No recovery intervention. No obs(alpha) intermediate probes.

Default analysis:
  - Rebuild true clean/pert first observations.
  - Load true Phase-2 clean/pert action and hidden endpoint artifacts.
  - For each condition, compare preserved vs flipped endpoint-shift geometry
    directly with group centroids, pairwise cosine, and per-sample centroid
    margins.
  - Keep the older flipped weak-direction score for backward compatibility.

Optional Hessian-free endpoint analysis:
  - With --hessian-final-layer, compute a JVP/Rayleigh proxy at h_clean in the
    actual endpoint shift direction, using both final action-slot and video-slot
    hidden maps.
  - The action proxy measures action-slot hidden -> final action-slot output.
  - The video proxy measures video-slot hidden -> final video-slot output. It is
    not video-to-action sensitivity, because at the final layer there is no
    further cross-slot attention.
  - This does not materialize a Hessian matrix and does not create synthetic
    intermediate observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = Path(__file__).resolve().parent
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_phase2_angular_cosmos import (  # noqa: E402
    discover_pairs,
    first_observation,
    load_extra_t5,
    make_cfg,
    patch_checkpoint_db,
)
from cosmos_policy.experiments.robot import cosmos_utils  # noqa: E402
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_model,
    load_dataset_stats,
    rescale_proprio,
)
from cosmos_policy.utils.utils import set_seed_everywhere  # noqa: E402

EPS = 1e-12


DETAIL_FIELDS = [
    "condition", "episode", "group",
    "obs_norm", "obs_cos_weak", "obs_proj_weak", "obs_energy_weak",
    "obs_cos_flipped_centroid", "obs_cos_preserved_centroid", "obs_centroid_margin_fp",
    "action_error", "action_norm", "action_cos_weak", "action_proj_weak", "action_energy_weak",
    "action_cos_flipped_centroid", "action_cos_preserved_centroid", "action_centroid_margin_fp",
    "hidden_action_norm", "hidden_action_cos_weak", "hidden_action_proj_weak", "hidden_action_energy_weak",
    "hidden_action_cos_flipped_centroid", "hidden_action_cos_preserved_centroid",
    "hidden_action_centroid_margin_fp",
    "hidden_video_norm", "hidden_video_cos_weak", "hidden_video_proj_weak", "hidden_video_energy_weak",
    "hidden_video_cos_flipped_centroid", "hidden_video_cos_preserved_centroid",
    "hidden_video_centroid_margin_fp",
    "hessian_rayleigh_shift", "hessian_rayleigh_random_mean", "hessian_rayleigh_ratio",
    "hessian_action_rayleigh_shift", "hessian_action_rayleigh_random_mean", "hessian_action_rayleigh_ratio",
    "hessian_video_rayleigh_shift", "hessian_video_rayleigh_random_mean", "hessian_video_rayleigh_ratio",
    "clean_success", "pert_success", "instruction_mode", "clean_task_name", "pert_task_name",
]


DIRECTION_SPACES = [
    ("obs", "obs_delta"),
    ("action", "action_delta"),
    ("hidden_action", "hidden_action_delta"),
    ("hidden_video", "hidden_video_delta"),
]


GEOMETRY_FIELDS = [
    "condition", "space", "n_flipped", "n_preserved",
    "flipped_centroid_coherence", "preserved_centroid_coherence",
    "flipped_preserved_centroid_cos", "flipped_preserved_centroid_angle_deg",
    "within_flipped_cos_mean", "within_flipped_cos_std", "within_flipped_n",
    "within_preserved_cos_mean", "within_preserved_cos_std", "within_preserved_n",
    "between_flipped_preserved_cos_mean", "between_flipped_preserved_cos_std",
    "between_flipped_preserved_n",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def phase2_pair_dir(results_dir: Path, condition: str, episode: int) -> Path:
    return results_dir / condition / f"ep{episode:02d}"


def load_action_delta(pair_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = np.load(pair_dir / "action_clean.npy").astype(np.float64).reshape(-1)
    pert = np.load(pair_dir / "action_pert.npy").astype(np.float64).reshape(-1)
    return clean, pert, pert - clean


def load_hidden_delta(pair_dir: Path, slot: str) -> np.ndarray:
    clean = torch.load(str(pair_dir / f"hidden_{slot}_clean.pt"), map_location="cpu", weights_only=True)
    pert = torch.load(str(pair_dir / f"hidden_{slot}_pert.pt"), map_location="cpu", weights_only=True)
    clean_t = list(clean.values())[0].float().reshape(-1)
    pert_t = list(pert.values())[0].float().reshape(-1)
    return (pert_t - clean_t).numpy().astype(np.float32)


def load_hidden_slot_tensors(pair_dir: Path, slot: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.load(str(pair_dir / f"hidden_{slot}_clean.pt"), map_location=device, weights_only=True)
    pert = torch.load(str(pair_dir / f"hidden_{slot}_pert.pt"), map_location=device, weights_only=True)
    clean_t = list(clean.values())[0].float().squeeze(0).to(device)
    pert_t = list(pert.values())[0].float().squeeze(0).to(device)
    return clean_t, pert_t


def rel_action_error(pert: np.ndarray, clean: np.ndarray) -> float:
    return float(np.linalg.norm(pert - clean) / (np.linalg.norm(clean) + EPS))


def obs_to_vec(obs: dict[str, np.ndarray], dataset_stats: dict[str, Any]) -> np.ndarray:
    primary = obs["primary_image"].astype(np.float32).reshape(-1) / 255.0
    wrist = obs["wrist_image"].astype(np.float32).reshape(-1) / 255.0
    proprio = obs["proprio"].astype(np.float32)
    try:
        proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0).astype(np.float32)
    except Exception:
        proprio = proprio.astype(np.float32)
    return np.concatenate([primary, wrist, proprio.reshape(-1)]).astype(np.float32)


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray) -> np.ndarray | None:
    n = norm(vec)
    if n <= EPS:
        return None
    return (vec / n).astype(np.float32)


def finite(value: float) -> bool:
    return not math.isnan(float(value))


def clipped_cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return max(-1.0, min(1.0, float(np.dot(a, b))))


def direction_features(delta: np.ndarray, weak: np.ndarray | None) -> dict[str, float]:
    n = norm(delta)
    if weak is None or n <= EPS:
        return {"norm": n, "cos": float("nan"), "proj": float("nan"), "energy": float("nan")}
    proj = float(np.dot(delta.astype(np.float64), weak.astype(np.float64)))
    cos = max(-1.0, min(1.0, proj / (n + EPS)))
    return {"norm": n, "cos": cos, "proj": proj, "energy": cos * cos}


def weak_direction(items: list[dict[str, Any]], field: str, leave_out_key: tuple[str, int, str] | None = None) -> np.ndarray | None:
    acc = None
    count = 0
    for item in items:
        if item["group"] != "flipped":
            continue
        if leave_out_key is not None and item["key"] == leave_out_key:
            continue
        u = unit(item[field])
        if u is not None:
            if acc is None:
                acc = np.zeros_like(u, dtype=np.float64)
            acc += u.astype(np.float64, copy=False)
            count += 1
    if count == 0 or acc is None:
        return None
    return unit(acc / float(count))


def group_unit_directions(
    items: list[dict[str, Any]],
    field: str,
    group: str,
    exclude_key: tuple[str, int, str] | None = None,
) -> list[np.ndarray]:
    dirs = []
    for item in items:
        if item["group"] != group:
            continue
        if exclude_key is not None and item["key"] == exclude_key:
            continue
        u = unit(item[field])
        if u is not None:
            dirs.append(u)
    return dirs


def centroid_direction(dirs: list[np.ndarray]) -> tuple[np.ndarray | None, float]:
    if not dirs:
        return None, float("nan")
    acc = np.zeros_like(dirs[0], dtype=np.float64)
    for vec in dirs:
        acc += vec.astype(np.float64, copy=False)
    mean = acc / float(len(dirs))
    coherence = float(np.linalg.norm(mean))
    if coherence <= EPS:
        return None, coherence
    return (mean / coherence).astype(np.float32), coherence


def pairwise_cos_stats(a_dirs: list[np.ndarray], b_dirs: list[np.ndarray] | None = None) -> tuple[float, float, int]:
    values = []
    if b_dirs is None:
        for i, a in enumerate(a_dirs):
            for b in a_dirs[i + 1:]:
                values.append(clipped_cosine(a, b))
    else:
        for a in a_dirs:
            for b in b_dirs:
                values.append(clipped_cosine(a, b))

    values = [value for value in values if finite(value)]
    if not values:
        return float("nan"), float("nan"), 0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def centroid_comparison_features(delta: np.ndarray, flipped_centroid: np.ndarray | None, preserved_centroid: np.ndarray | None) -> dict[str, float]:
    cos_flipped = direction_features(delta, flipped_centroid)["cos"]
    cos_preserved = direction_features(delta, preserved_centroid)["cos"]
    margin = cos_flipped - cos_preserved if finite(cos_flipped) and finite(cos_preserved) else float("nan")
    return {
        "cos_flipped_centroid": cos_flipped,
        "cos_preserved_centroid": cos_preserved,
        "centroid_margin_fp": margin,
    }


def add_direction_geometry(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    centroid_leave_one_out: bool = False,
) -> list[dict[str, Any]]:
    row_by_key = {row["key"]: row for row in rows}
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_condition[item["condition"]].append(item)

    geometry_rows = []
    for condition, cond_items in sorted(by_condition.items()):
        for space, field in DIRECTION_SPACES:
            flipped_dirs = group_unit_directions(cond_items, field, "flipped")
            preserved_dirs = group_unit_directions(cond_items, field, "preserved")
            flipped_centroid, flipped_coherence = centroid_direction(flipped_dirs)
            preserved_centroid, preserved_coherence = centroid_direction(preserved_dirs)

            centroid_cos = clipped_cosine(flipped_centroid, preserved_centroid)
            centroid_angle = math.degrees(math.acos(centroid_cos)) if finite(centroid_cos) else float("nan")
            within_flipped_mu, within_flipped_sd, within_flipped_n = pairwise_cos_stats(flipped_dirs)
            within_preserved_mu, within_preserved_sd, within_preserved_n = pairwise_cos_stats(preserved_dirs)
            between_mu, between_sd, between_n = pairwise_cos_stats(flipped_dirs, preserved_dirs)

            geometry_rows.append({
                "condition": condition,
                "space": space,
                "n_flipped": len(flipped_dirs),
                "n_preserved": len(preserved_dirs),
                "flipped_centroid_coherence": flipped_coherence,
                "preserved_centroid_coherence": preserved_coherence,
                "flipped_preserved_centroid_cos": centroid_cos,
                "flipped_preserved_centroid_angle_deg": centroid_angle,
                "within_flipped_cos_mean": within_flipped_mu,
                "within_flipped_cos_std": within_flipped_sd,
                "within_flipped_n": within_flipped_n,
                "within_preserved_cos_mean": within_preserved_mu,
                "within_preserved_cos_std": within_preserved_sd,
                "within_preserved_n": within_preserved_n,
                "between_flipped_preserved_cos_mean": between_mu,
                "between_flipped_preserved_cos_std": between_sd,
                "between_flipped_preserved_n": between_n,
            })

            for item in cond_items:
                row = row_by_key[item["key"]]
                row_flipped_centroid = flipped_centroid
                row_preserved_centroid = preserved_centroid
                if centroid_leave_one_out:
                    if item["group"] == "flipped":
                        row_flipped_centroid, _ = centroid_direction(
                            group_unit_directions(cond_items, field, "flipped", exclude_key=item["key"])
                        )
                    elif item["group"] == "preserved":
                        row_preserved_centroid, _ = centroid_direction(
                            group_unit_directions(cond_items, field, "preserved", exclude_key=item["key"])
                        )

                feats = centroid_comparison_features(item[field], row_flipped_centroid, row_preserved_centroid)
                row[f"{space}_cos_flipped_centroid"] = feats["cos_flipped_centroid"]
                row[f"{space}_cos_preserved_centroid"] = feats["cos_preserved_centroid"]
                row[f"{space}_centroid_margin_fp"] = feats["centroid_margin_fp"]

    return geometry_rows


def build_endpoint_items(args, dataset_stats: dict[str, Any]) -> list[dict[str, Any]]:
    _, pairs = discover_pairs(Path(args.summary), set(args.conditions or []), set(args.groups))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError(f"No pairs selected from {args.summary}")

    print(f"Discovered {len(pairs)} pairs", flush=True)
    for (condition, group), count in sorted(Counter((p["condition"], p["group"]) for p in pairs).items()):
        print(f"  {condition:24s} {group:10s} {count:3d}", flush=True)

    clean_obs_cache: dict[tuple, dict[str, np.ndarray]] = {}
    out = []
    results_dir = Path(args.results_dir)
    for idx, pair in enumerate(pairs, 1):
        condition = pair["condition"]
        clean = pair["clean"]
        pert = pair["pert"]
        episode = int(pert["episode"])
        pair_dir = phase2_pair_dir(results_dir, condition, episode)
        if not (pair_dir / "metrics.json").exists():
            raise FileNotFoundError(f"Missing Phase-2 artifacts for {condition}/ep{episode:02d}: {pair_dir}")

        print(f"[{idx}/{len(pairs)}] endpoint load {condition}/ep{episode:02d} {pair['group']}", flush=True)

        clean_key = (
            clean["suite"],
            clean["task_name"],
            int(clean.get("init_state_index", clean["episode"])),
            int(clean["episode"]),
            bool(clean.get("deterministic_reset", True)),
        )
        if clean_key not in clean_obs_cache:
            clean_obs_cache[clean_key] = first_observation(clean, args)
        clean_obs = clean_obs_cache[clean_key]
        pert_obs = clean_obs if condition == "language_instructions" else first_observation(pert, args)

        obs_delta = obs_to_vec(pert_obs, dataset_stats) - obs_to_vec(clean_obs, dataset_stats)
        action_clean, action_pert, action_delta = load_action_delta(pair_dir)
        metrics = load_json(pair_dir / "metrics.json")

        out.append({
            "key": (condition, episode, pair["group"]),
            "condition": condition,
            "episode": episode,
            "group": pair["group"],
            "pair_dir": pair_dir,
            "obs_delta": obs_delta,
            "action_delta": action_delta.astype(np.float32),
            "hidden_action_delta": load_hidden_delta(pair_dir, "action"),
            "hidden_video_delta": load_hidden_delta(pair_dir, "video"),
            "action_error": float(metrics.get("action_error", rel_action_error(action_pert, action_clean))),
            "clean": clean,
            "pert": pert,
        })
    return out


def summarize_by_condition(items: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_condition[item["condition"]].append(item)

    rows = []
    for condition, cond_items in sorted(by_condition.items()):
        for item in cond_items:
            key = item["key"]
            # Leave-one-out for flipped samples avoids using the sample to define its own weak direction.
            loo_key = key if item["group"] == "flipped" and args.leave_one_out else None
            obs_weak = weak_direction(cond_items, "obs_delta", leave_out_key=loo_key)
            action_weak = weak_direction(cond_items, "action_delta", leave_out_key=loo_key)
            hidden_action_weak = weak_direction(cond_items, "hidden_action_delta", leave_out_key=loo_key)
            hidden_video_weak = weak_direction(cond_items, "hidden_video_delta", leave_out_key=loo_key)

            obs = direction_features(item["obs_delta"], obs_weak)
            action = direction_features(item["action_delta"], action_weak)
            hidden_action = direction_features(item["hidden_action_delta"], hidden_action_weak)
            hidden_video = direction_features(item["hidden_video_delta"], hidden_video_weak)

            clean = item["clean"]
            pert = item["pert"]
            rows.append({
                "key": key,
                "condition": condition,
                "episode": item["episode"],
                "group": item["group"],
                "obs_norm": obs["norm"],
                "obs_cos_weak": obs["cos"],
                "obs_proj_weak": obs["proj"],
                "obs_energy_weak": obs["energy"],
                "action_error": item["action_error"],
                "action_norm": action["norm"],
                "action_cos_weak": action["cos"],
                "action_proj_weak": action["proj"],
                "action_energy_weak": action["energy"],
                "hidden_action_norm": hidden_action["norm"],
                "hidden_action_cos_weak": hidden_action["cos"],
                "hidden_action_proj_weak": hidden_action["proj"],
                "hidden_action_energy_weak": hidden_action["energy"],
                "hidden_video_norm": hidden_video["norm"],
                "hidden_video_cos_weak": hidden_video["cos"],
                "hidden_video_proj_weak": hidden_video["proj"],
                "hidden_video_energy_weak": hidden_video["energy"],
                "hessian_rayleigh_shift": float("nan"),
                "hessian_rayleigh_random_mean": float("nan"),
                "hessian_rayleigh_ratio": float("nan"),
                "hessian_action_rayleigh_shift": float("nan"),
                "hessian_action_rayleigh_random_mean": float("nan"),
                "hessian_action_rayleigh_ratio": float("nan"),
                "hessian_video_rayleigh_shift": float("nan"),
                "hessian_video_rayleigh_random_mean": float("nan"),
                "hessian_video_rayleigh_ratio": float("nan"),
                "clean_success": bool(clean["success"]),
                "pert_success": bool(pert["success"]),
                "instruction_mode": pert.get("instruction_mode", "task"),
                "clean_task_name": clean["task_name"],
                "pert_task_name": pert["task_name"],
            })
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def finite_mean_std(values: list[Any]) -> tuple[float, float]:
    vals = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if not math.isnan(f):
            vals.append(f)
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def group_summary(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["group"])].append(row)

    out = []
    for (condition, group), items in sorted(grouped.items()):
        rec: dict[str, Any] = {"condition": condition, "group": group, "n": len(items)}
        for field in fields:
            mu, sd = finite_mean_std([item.get(field, float("nan")) for item in items])
            rec[f"{field}_mean"] = mu
            rec[f"{field}_std"] = sd
        out.append(rec)
    return out


def best_balanced_accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    items = []
    for row in rows:
        if row["group"] not in {"preserved", "flipped"}:
            continue
        try:
            value = float(row[field])
        except Exception:
            continue
        if not math.isnan(value):
            items.append((value, 1 if row["group"] == "flipped" else 0))
    if len(items) < 3 or len({label for _, label in items}) < 2:
        return None

    values = sorted({value for value, _ in items})
    candidates = [values[0] - 1e-12, values[-1] + 1e-12]
    candidates.extend((a + b) / 2.0 for a, b in zip(values, values[1:]))
    best = None
    for sign in (1.0, -1.0):
        for threshold in candidates:
            tp = fp = tn = fn = 0
            for value, label in items:
                pred = int(sign * value >= sign * threshold)
                if pred and label:
                    tp += 1
                elif pred and not label:
                    fp += 1
                elif not pred and not label:
                    tn += 1
                else:
                    fn += 1
            tpr = tp / (tp + fn) if tp + fn else float("nan")
            tnr = tn / (tn + fp) if tn + fp else float("nan")
            rec = {
                "field": field,
                "balanced_acc": 0.5 * (tpr + tnr),
                "accuracy": (tp + tn) / len(items),
                "rule": ">=" if sign > 0 else "<=",
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
            if best is None or (rec["balanced_acc"], rec["accuracy"]) > (best["balanced_acc"], best["accuracy"]):
                best = rec
    return best


def add_final_layer_hessian_proxy(rows: list[dict[str, Any]], args) -> None:
    from phase3_fragility_m1_m6b import (  # noqa: WPS433
        SIGMA_MIN,
        build_sub_network,
        compute_timestep_embedding,
        final_layer_forward_f32,
    )

    print("Loading model for endpoint Hessian-free final-layer proxy...", flush=True)
    args.policy_dir = Path(args.policy_dir)
    patch_checkpoint_db(args.policy_dir)
    cfg = make_cfg(args)
    set_seed_everywhere(args.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    model, _ = get_model(cfg)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    sub_net = build_sub_network(model)
    t_emb, adaln = compute_timestep_embedding(sub_net, SIGMA_MIN, device=device, dtype=torch.float32)
    fl = sub_net["final_layer"].float()
    t_emb_f32 = t_emb.float()
    adaln_f32 = adaln.float() if adaln is not None else None

    def expand_time_condition(tensor: torch.Tensor | None, t_len: int) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.shape[1] == t_len:
            return tensor
        if tensor.shape[1] == 1:
            return tensor.repeat(1, t_len, 1)
        raise ValueError(f"Cannot expand timestep condition from T={tensor.shape[1]} to T={t_len}")

    def rayleigh_for_pair(pair_dir: Path, slot: str) -> tuple[float, float, float]:
        h_clean, h_pert = load_hidden_slot_tensors(pair_dir, slot, device)
        h_clean_f32 = h_clean.float()
        dh = (h_pert.float() - h_clean_f32)
        dh_norm = torch.linalg.vector_norm(dh)
        if float(dh_norm) <= EPS:
            return float("nan"), float("nan"), float("nan")
        v = dh / dh_norm
        t_len = int(h_clean_f32.shape[0])
        slot_t_emb = expand_time_condition(t_emb_f32, t_len)
        slot_adaln = expand_time_condition(adaln_f32, t_len)

        def f_output(h_in: torch.Tensor) -> torch.Tensor:
            out = final_layer_forward_f32(h_in, slot_t_emb, slot_adaln, fl)
            return out.reshape(-1)

        f0 = f_output(h_clean_f32).detach()
        denom = float(torch.dot(f0, f0)) + EPS

        _, jvp_shift = torch.autograd.functional.jvp(f_output, h_clean_f32, (v,))
        ray_shift = float(torch.dot(jvp_shift, jvp_shift) / denom)

        random_vals = []
        for _ in range(args.hessian_random):
            r = torch.randn_like(h_clean_f32)
            r = r / (torch.linalg.vector_norm(r) + EPS)
            _, jvp_rand = torch.autograd.functional.jvp(f_output, h_clean_f32, (r,))
            random_vals.append(float(torch.dot(jvp_rand, jvp_rand) / denom))
        ray_rand = float(np.mean(random_vals)) if random_vals else float("nan")
        return ray_shift, ray_rand, ray_shift / (ray_rand + EPS)

    for idx, row in enumerate(rows, 1):
        pair_dir = phase2_pair_dir(Path(args.results_dir), row["condition"], int(row["episode"]))
        print(f"[{idx}/{len(rows)}] Hessian-free proxy {row['condition']}/ep{int(row['episode']):02d} {row['group']}", flush=True)
        action_shift, action_rand, action_ratio = rayleigh_for_pair(pair_dir, "action")
        video_shift, video_rand, video_ratio = rayleigh_for_pair(pair_dir, "video")
        row["hessian_rayleigh_shift"] = action_shift
        row["hessian_rayleigh_random_mean"] = action_rand
        row["hessian_rayleigh_ratio"] = action_ratio
        row["hessian_action_rayleigh_shift"] = action_shift
        row["hessian_action_rayleigh_random_mean"] = action_rand
        row["hessian_action_rayleigh_ratio"] = action_ratio
        row["hessian_video_rayleigh_shift"] = video_shift
        row["hessian_video_rayleigh_random_mean"] = video_rand
        row["hessian_video_rayleigh_ratio"] = video_ratio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_stats = load_dataset_stats(str(args.policy_dir / "libero_dataset_statistics.json"))

    items = build_endpoint_items(args, dataset_stats)
    rows = summarize_by_condition(items, args)
    geometry_rows = add_direction_geometry(rows, items, centroid_leave_one_out=bool(args.centroid_leave_one_out))

    if args.hessian_final_layer:
        add_final_layer_hessian_proxy(rows, args)

    summary_fields = [
        "obs_norm", "obs_cos_weak", "obs_energy_weak",
        "obs_cos_flipped_centroid", "obs_cos_preserved_centroid", "obs_centroid_margin_fp",
        "action_error", "action_norm", "action_cos_weak", "action_energy_weak",
        "action_cos_flipped_centroid", "action_cos_preserved_centroid", "action_centroid_margin_fp",
        "hidden_action_norm", "hidden_action_cos_weak", "hidden_action_energy_weak",
        "hidden_action_cos_flipped_centroid", "hidden_action_cos_preserved_centroid",
        "hidden_action_centroid_margin_fp",
        "hidden_video_norm", "hidden_video_cos_weak", "hidden_video_energy_weak",
        "hidden_video_cos_flipped_centroid", "hidden_video_cos_preserved_centroid",
        "hidden_video_centroid_margin_fp",
        "hessian_rayleigh_shift", "hessian_rayleigh_random_mean", "hessian_rayleigh_ratio",
        "hessian_action_rayleigh_shift", "hessian_action_rayleigh_random_mean", "hessian_action_rayleigh_ratio",
        "hessian_video_rayleigh_shift", "hessian_video_rayleigh_random_mean", "hessian_video_rayleigh_ratio",
    ]
    summary_rows = group_summary(rows, summary_fields)
    threshold_rows = []
    for condition in sorted({row["condition"] for row in rows}):
        cond_rows = [row for row in rows if row["condition"] == condition]
        for field in summary_fields:
            rec = best_balanced_accuracy(cond_rows, field)
            if rec is not None:
                rec["condition"] = condition
                threshold_rows.append(rec)

    write_csv(out_dir / "endpoint_weakspace_detail.csv", DETAIL_FIELDS, rows)
    write_csv(
        out_dir / "endpoint_weakspace_summary.csv",
        ["condition", "group", "n"]
        + [f"{field}_mean" for field in summary_fields]
        + [f"{field}_std" for field in summary_fields],
        summary_rows,
    )
    write_csv(
        out_dir / "endpoint_feature_thresholds.csv",
        ["condition", "field", "balanced_acc", "accuracy", "rule", "threshold", "tp", "fp", "tn", "fn"],
        threshold_rows,
    )
    write_csv(out_dir / "endpoint_direction_geometry.csv", GEOMETRY_FIELDS, geometry_rows)

    payload = {
        "summary": str(args.summary),
        "results_dir": str(args.results_dir),
        "output_dir": str(out_dir),
        "num_pairs": len(rows),
        "leave_one_out": bool(args.leave_one_out),
        "centroid_leave_one_out": bool(args.centroid_leave_one_out),
        "hessian_final_layer": bool(args.hessian_final_layer),
        "elapsed_s": time.time() - start,
        "method": {
            "endpoint_shift": "delta = pert_endpoint - clean_endpoint; no obs(alpha) probes",
            "centroid_geometry": (
                "per-condition preserved/flipped unit-shift centroids; reports centroid angle, "
                "within/between pairwise cosine, and row-level cos(delta, flipped_centroid) "
                "- cos(delta, preserved_centroid) margins. With --centroid-leave-one-out, "
                "row-level centroid features exclude the current sample from its own group centroid."
            ),
            "weak_direction": (
                "legacy per-condition mean of flipped unit endpoint shifts; leave-one-out for flipped "
                "rows by default. Kept for compatibility, not the primary preserved-vs-flipped metric."
            ),
            "hessian_free_proxy": (
                "optional --hessian-final-layer: JVP Rayleigh proxy ||J v_shift||^2 / ||f(h_clean)||^2 "
                "for final action-slot hidden->action output and video-slot hidden->video output. "
                "The video proxy is not video-to-action sensitivity at the final layer."
            ),
        },
        "group_summary": summary_rows,
        "direction_geometry": geometry_rows,
        "feature_thresholds": threshold_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Done. Output: {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    default_summary = (
        ROOT
        / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/"
        / "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json"
    )
    default_results = ROOT / "experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_perturb_preserved_flipped_last"
    default_output = ROOT / "experiments/phase3_endpoint_weakspace/kitchen_scene4_seed7_all_perturb_20case"
    default_policy = ROOT.parent.parent / "checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"
    default_t5_extra = ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(default_summary))
    parser.add_argument("--results-dir", default=str(default_results))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--policy-dir", type=Path, default=default_policy)
    parser.add_argument("--t5-extra-embeddings", default=str(default_t5_extra))
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-seed", type=int, default=0)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-denoising-steps", type=int, default=5)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--centroid-leave-one-out", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hessian-final-layer", action="store_true")
    parser.add_argument("--hessian-random", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main()
