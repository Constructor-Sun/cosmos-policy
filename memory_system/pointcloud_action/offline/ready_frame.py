"""PointCloud Action Memory ready-frame selection.

This is a pointcloud_action-local replacement for the shared
memory_system.offline ready selector. It selects a frame at roughly the
desired EE-to-object distance while requiring that the subsequent EE motion
continues to approach the target (i.e. the frame lies on the grasp approach
path, not just at an arbitrary 14 cm from the object).
"""
from __future__ import annotations

import numpy as np

# Reasonable defaults for LIBERO-90 Pick local skills.
DEFAULT_WINDOW = 40
DEFAULT_APPROACH_STEPS = 3
DEFAULT_MIN_APPROACH_M = 0.0
DEFAULT_TOLERANCE_M = 0.035


def _ee_position(ee_states, frame: int) -> np.ndarray:
    return np.asarray(ee_states[frame], dtype=np.float64).reshape(-1)[:3]


def select_ready_frame(
    segment: dict,
    ee_states: np.ndarray,
    target_xyz: np.ndarray,
    desired: float,
    window: int = DEFAULT_WINDOW,
    approach_steps: int = DEFAULT_APPROACH_STEPS,
    min_approach_m: float = DEFAULT_MIN_APPROACH_M,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> int | None:
    """Return a ready frame near ``desired`` metres on the grasp approach path.

    The frame is selected from the trailing part of the segment (before the
    segment end / grasp success). A candidate is only accepted if the EE is
    still moving toward ``target_xyz`` over the next ``approach_steps`` frames.
    """
    start = int(segment.get("start", 0))
    end = int(
        segment.get("end")
        or segment.get("success_start")
        or len(ee_states) - 1
    )
    start = min(max(start, 0), len(ee_states) - 1)
    end = min(max(end, start + 1), len(ee_states))

    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    search_start = max(start, end - window)
    # We need future frames to evaluate the approach direction.
    last_candidate = min(end - 1, len(ee_states) - 1 - approach_steps)
    if last_candidate < search_start:
        # Not enough future frames; fall back to the closest distance in window.
        last_candidate = end - 1

    best_frame = None
    best_score = float("inf")
    fallback_frame = None
    fallback_score = float("inf")

    for frame in range(search_start, last_candidate + 1):
        ee = _ee_position(ee_states, frame)
        dist = float(np.linalg.norm(ee - target))

        # Track the closest-distance fallback regardless of approach direction.
        fallback_score_now = abs(dist - desired)
        if fallback_score_now < fallback_score:
            fallback_score = fallback_score_now
            fallback_frame = frame

        if frame + approach_steps >= len(ee_states):
            continue

        future_ee = _ee_position(ee_states, frame + approach_steps)
        future_dist = float(np.linalg.norm(future_ee - target))
        approach = dist - future_dist
        if approach < min_approach_m:
            continue

        score = abs(dist - desired)
        if score < best_score:
            best_score = score
            best_frame = frame

    if best_frame is not None:
        return best_frame

    # Fallback: the closest distance in the search window, matching old behavior.
    return fallback_frame
