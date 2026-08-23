"""Shared recovery retrieval/clustering helpers.

This module extracts the duplicated logic from
``bin/execute/libero_pose_recovery.py`` and
``bin/execute/libero_feasible_recovery.py``.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - scipy is expected in eval
    Rotation = None


def arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def select_targets(
    targets: Iterable[dict[str, Any]],
    task_name: str,
    planner_step_id: int,
    skill: str,
    arguments: dict[str, str],
) -> list[dict[str, Any]]:
    items = list(targets)
    expected = arguments_key(arguments)
    exact = [
        item for item in items
        if item["task_name"] == task_name
        and int(item["planner_step_id"]) == int(planner_step_id)
        and item["skill"] == skill
        and arguments_key(item.get("arguments", {})) == expected
    ]
    if exact:
        return exact
    return [
        item for item in items
        if item["task_name"] == task_name
        and item["skill"] == skill
        and arguments_key(item.get("arguments", {})) == expected
    ]


def token(vae: Any) -> torch.Tensor:
    tensor = torch.as_tensor(vae)
    if tensor.dim() == 4 and tensor.shape[1] == 2:
        tensor = tensor[:, 1:2, :, :]
    flat = tensor.float().reshape(1, -1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return flat / norm


def combined_token(vae_main: Any, vae_wrist: Any | None = None) -> torch.Tensor:
    main = token(vae_main)
    if vae_wrist is None:
        return main
    wrist = token(vae_wrist)
    combined = torch.cat([main, wrist], dim=1)
    return combined / combined.norm(dim=1, keepdim=True).clamp_min(1e-8)


def similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left @ right.T).reshape(-1)[0])


def close(
    left: np.ndarray,
    right: np.ndarray,
    position_radius: float,
    rotation_radius: float,
) -> bool:
    left = np.asarray(left, dtype=np.float32).reshape(6)
    right = np.asarray(right, dtype=np.float32).reshape(6)
    if float(np.linalg.norm(left[:3] - right[:3])) > position_radius:
        return False
    rot_dist = float(np.linalg.norm(left[3:] - right[3:]))
    return rot_dist <= rotation_radius


def mean_ee_states(items: list[dict[str, Any]]) -> np.ndarray:
    positions = np.mean(
        [np.asarray(item["ee_states"], dtype=np.float64)[:3] for item in items],
        axis=0,
    )
    if Rotation is None:
        rotvecs = np.mean(
            [np.asarray(item["ee_states"], dtype=np.float64)[3:] for item in items],
            axis=0,
        )
        return np.concatenate([positions, rotvecs]).astype(np.float32)

    quats = []
    for item in items:
        q = Rotation.from_rotvec(
            np.asarray(item["ee_states"], dtype=np.float64)[3:]
        ).as_quat()
        if quats and np.dot(quats[0], q) < 0:
            q = -q
        quats.append(q)
    avg_q = np.mean(np.asarray(quats, dtype=np.float64), axis=0)
    avg_q = avg_q / np.linalg.norm(avg_q)
    avg_rotvec = Rotation.from_quat(avg_q).as_rotvec()
    return np.concatenate([positions, avg_rotvec]).astype(np.float32)


def retrieve_cluster(
    candidates: list[dict[str, Any]],
    current_token: torch.Tensor,
    item_token: Callable[[dict[str, Any]], torch.Tensor],
    similarity_threshold: float,
    position_radius: float,
    rotation_radius: float,
    min_demo_votes: int,
    target_average_count: int,
) -> tuple[list[dict[str, Any]], float, str] | None:
    """Return (target_items, best_similarity, best_frame) or None."""
    best_by_demo: dict[str, tuple[float, dict[str, Any]]] = {}
    for item in candidates:
        try:
            item_tok = item_token(item)
        except Exception:
            continue
        if item_tok is None:
            continue
        sim = similarity(current_token, item_tok)
        demo = str(item["demo_id"])
        if demo not in best_by_demo or sim > best_by_demo[demo][0]:
            best_by_demo[demo] = (sim, item)

    ranked = sorted(best_by_demo.values(), key=lambda pair: pair[0], reverse=True)
    if len(ranked) < min_demo_votes:
        return None
    if ranked[0][0] < similarity_threshold:
        return None

    best_item = ranked[0][1]
    cluster = [
        item for sim, item in ranked
        if sim >= similarity_threshold
        and close(
            item["ee_states"], best_item["ee_states"],
            position_radius, rotation_radius,
        )
    ]
    if len(cluster) < min_demo_votes:
        return None
    target_items = cluster[:target_average_count]
    frame_key = "recovery_frame" if "recovery_frame" in best_item else "ready_frame"
    return target_items, float(ranked[0][0]), str(best_item.get(frame_key, 0))
