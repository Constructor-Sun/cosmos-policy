#!/usr/bin/env python3
"""Detect the earliest intervention point in one or more rollout HDF5 files.

The detector uses five calibrated thresholds:

* ``empty_width_max`` and ``retry_distance_max`` detect two nearby empty grasps.
* ``active_command_min``, ``stuck_response_max`` and
  ``stuck_displacement_max`` detect two consecutive stuck action chunks.

Examples::

    python bin/detect_intervention_point.py episode.hdf5
    python bin/detect_intervention_point.py trajectories/*.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "Cosmos-Policy-LIBERO-Predict2-2B"
    / "config.json"
)

# Structural choices, not calibration parameters.
_STUCK_CHUNKS = 2


@dataclass(frozen=True)
class Thresholds:
    """All numeric thresholds used by the two detectors."""

    empty_width_max: float = 0.020
    retry_distance_max: float = 0.030
    active_command_min: float = 1.000
    stuck_response_max: float = 0.020
    stuck_displacement_max: float = 0.005

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class ChunkMetrics:
    step_start: int
    step_end: int
    commanded_motion: float
    response_ratio: float
    displacement: float


@dataclass(frozen=True)
class Detection:
    t_star: int
    trigger_type: str
    details: dict


def score_early_manifold(
    actions: np.ndarray,
    chunk_size: int,
    current_delta: np.ndarray,
    memory_deltas: np.ndarray,
    memory_chunk_indices: np.ndarray,
) -> dict | None:
    """Score the first completed action chunk against early demo memory."""
    if len(actions) != chunk_size:
        return None

    current = np.asarray(current_delta, dtype=np.float32).reshape(-1)
    memory = np.asarray(memory_deltas, dtype=np.float32)
    chunk_indices = np.asarray(memory_chunk_indices, dtype=np.int64)
    if memory.ndim != 2 or memory.shape[1] != current.size:
        raise ValueError("current and memory VAE deltas have incompatible shapes")
    if chunk_indices.shape != (len(memory),):
        raise ValueError("memory chunk indices do not align with memory deltas")

    candidates = np.flatnonzero(chunk_indices == 1)
    current_norm = np.linalg.norm(current)
    if candidates.size == 0 or current_norm <= 1e-8:
        return None

    candidate_deltas = memory[candidates]
    denominators = np.linalg.norm(candidate_deltas, axis=1) * current_norm
    valid = denominators > 1e-8
    if not np.any(valid):
        return None

    similarities = np.full(candidates.size, -np.inf, dtype=np.float32)
    similarities[valid] = candidate_deltas[valid] @ current / denominators[valid]
    winner_position = int(np.argmax(similarities))
    winner = int(candidates[winner_position])
    return {
        "best_similarity": float(similarities[winner_position]),
        "best_memory_index": winner,
        "best_chunk_index": int(chunk_indices[winner]),
        "candidate_count": int(candidates.size),
    }


def detect_early_manifold_deviation(
    actions: np.ndarray,
    chunk_size: int,
    current_delta: np.ndarray,
    memory_deltas: np.ndarray,
    memory_chunk_indices: np.ndarray,
    similarity_threshold: float = 0.1,
) -> Detection | None:
    """Trigger once when the first action chunk falls outside early memory."""
    score = score_early_manifold(
        actions, chunk_size, current_delta, memory_deltas, memory_chunk_indices
    )
    if score is None or score["best_similarity"] >= similarity_threshold:
        return None
    return Detection(
        t_star=chunk_size,
        trigger_type="early_manifold_deviation",
        details={**score, "similarity_threshold": float(similarity_threshold)},
    )


def load_chunk_size(config_path: Path) -> int:
    """Read the model action horizon from ``config.json``."""
    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    try:
        chunk_size = int(config["output_spec"]["actions"]["horizon"])
    except KeyError:
        chunk_size = int(config["training"]["action_chunk_size"])

    if chunk_size <= 0:
        raise ValueError(f"action chunk size must be positive, got {chunk_size}")
    return chunk_size


def load_episode(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, bool | None]:
    """Load and validate the state and action arrays used by the detectors."""
    with h5py.File(hdf5_path, "r") as episode:
        if "proprio" not in episode or "actions" not in episode:
            raise ValueError("HDF5 file must contain 'proprio' and 'actions'")
        proprio = episode["proprio"][:]
        actions = episode["actions"][:]
        raw_success = episode.attrs.get("success")

    if proprio.ndim != 2 or proprio.shape[1] < 5:
        raise ValueError(f"expected proprio shape (T, >=5), got {proprio.shape}")
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"expected actions shape (T, >=7), got {actions.shape}")
    if len(proprio) < len(actions):
        raise ValueError(
            "proprio must contain at least one state per action "
            f"(got {len(proprio)} states and {len(actions)} actions)"
        )

    success = None if raw_success is None else bool(raw_success)
    return proprio, actions, success


def gripper_width(proprio: np.ndarray) -> np.ndarray:
    """Return Franka gripper opening width (finger joint 0 minus joint 1)."""
    return (
        proprio[:, 0].astype(np.float64)
        - proprio[:, 1].astype(np.float64)
    )


def compute_chunk_metrics(
    proprio: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
) -> list[ChunkMetrics]:
    """Measure commanded motion and aligned end-effector response per chunk."""
    metrics: list[ChunkMetrics] = []

    # Computing an actual delta for action t requires states t and t+1.
    # Rollouts store actions and states with equal length, so the final action
    # cannot contribute to a full chunk.
    usable_steps = min(len(actions), len(proprio) - 1)
    for step_start in range(0, usable_steps - chunk_size + 1, chunk_size):
        step_end = step_start + chunk_size
        commands = actions[step_start:step_end, :3].astype(np.float64)
        positions = proprio[step_start:step_end + 1, 2:5].astype(np.float64)
        actual_steps = np.diff(positions, axis=0)

        command_norms = np.linalg.norm(commands, axis=1)
        commanded_motion = float(command_norms.sum())

        directions = np.divide(
            commands,
            command_norms[:, None],
            out=np.zeros_like(commands),
            where=command_norms[:, None] > 1e-8,
        )
        aligned_progress = np.maximum(
            np.sum(directions * actual_steps, axis=1), 0.0
        )
        response_ratio = float(
            aligned_progress.sum() / (commanded_motion + 1e-8)
        )
        displacement = float(
            np.max(np.linalg.norm(positions - positions[0], axis=1))
        )

        metrics.append(
            ChunkMetrics(
                step_start=step_start,
                step_end=step_end,
                commanded_motion=commanded_motion,
                response_ratio=response_ratio,
                displacement=displacement,
            )
        )

    return metrics


def detect_stagnation(
    metrics: list[ChunkMetrics],
    thresholds: Thresholds,
) -> Detection | None:
    """Detect two consecutive active chunks with negligible response."""
    run: list[ChunkMetrics] = []

    for metric in metrics:
        stuck = (
            metric.commanded_motion >= thresholds.active_command_min
            and metric.response_ratio <= thresholds.stuck_response_max
            and metric.displacement <= thresholds.stuck_displacement_max
        )

        if not stuck:
            run.clear()
            continue

        run.append(metric)
        if len(run) == _STUCK_CHUNKS:
            first = run[0]
            return Detection(
                t_star=first.step_start,
                trigger_type="stagnation",
                details={
                    "step_start": first.step_start,
                    "step_end": metric.step_end,
                    "commanded_motion": first.commanded_motion,
                    "response_ratio": first.response_ratio,
                    "displacement": first.displacement,
                    "consecutive_chunks": _STUCK_CHUNKS,
                },
            )

    return None


def detect_double_empty_grasp(
    proprio: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
    thresholds: Thresholds,
) -> Detection | None:
    """Detect two consecutive nearby chunks dominated by empty-close actions.

    A pair triggers when both chunks command closing for a strict majority of
    their steps, the first chunk ends empty-closed, the second chunk also
    reaches the empty width, and their starting EE positions are nearby.
    ``t*`` is the boundary immediately after the second chunk.
    """
    widths = gripper_width(proprio)
    usable_steps = min(len(actions), len(proprio) - 1)
    last_pair_start = usable_steps - 2 * chunk_size

    for first_start in range(0, last_pair_start + 1, chunk_size):
        first_end = first_start + chunk_size
        second_start = first_end
        second_end = second_start + chunk_size

        first_close_fraction = float(
            np.mean(actions[first_start:first_end, 6] > 0.0)
        )
        second_close_fraction = float(
            np.mean(actions[second_start:second_end, 6] > 0.0)
        )
        if first_close_fraction <= 0.5 or second_close_fraction <= 0.5:
            continue

        first_end_width = float(widths[first_end])
        second_min_width = float(np.min(widths[second_start:second_end + 1]))
        if (
            first_end_width > thresholds.empty_width_max
            or second_min_width > thresholds.empty_width_max
        ):
            continue

        retry_distance = float(np.linalg.norm(
            proprio[second_start, 2:5].astype(np.float64)
            - proprio[first_start, 2:5].astype(np.float64)
        ))
        if retry_distance > thresholds.retry_distance_max:
            continue

        return Detection(
            t_star=second_end,
            trigger_type="double_empty_grasp",
            details={
                "first_chunk_start": first_start,
                "second_chunk_start": second_start,
                "first_close_fraction": first_close_fraction,
                "second_close_fraction": second_close_fraction,
                "first_end_width": first_end_width,
                "second_min_width": second_min_width,
                "retry_distance": retry_distance,
            },
        )

    return None


def detect_intervention(
    proprio: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
    thresholds: Thresholds,
) -> Detection | None:
    """Run both detectors and return the earliest intervention point."""
    stagnation = detect_stagnation(
        compute_chunk_metrics(proprio, actions, chunk_size), thresholds
    )
    empty_grasp = detect_double_empty_grasp(
        proprio, actions, chunk_size, thresholds
    )
    detections = [item for item in (stagnation, empty_grasp) if item is not None]

    if not detections:
        return None

    return min(detections, key=lambda item: item.t_star)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect intervention points using five calibrated thresholds."
    )
    parser.add_argument(
        "hdf5_paths",
        type=Path,
        nargs="+",
        help="One or more rollout HDF5 files (shell globs are supported).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Cosmos-Policy model config.json used to obtain the chunk size.",
    )
    parser.add_argument("--empty-width-max", type=float, default=0.020)
    parser.add_argument("--retry-distance-max", type=float, default=0.030)
    parser.add_argument("--active-command-min", type=float, default=1.000)
    parser.add_argument("--stuck-response-max", type=float, default=0.020)
    parser.add_argument("--stuck-displacement-max", type=float, default=0.005)
    return parser


def process_episode(
    hdf5_path: Path,
    chunk_size: int,
    thresholds: Thresholds,
) -> Detection | None:
    proprio, actions, success = load_episode(hdf5_path)
    result = detect_intervention(proprio, actions, chunk_size, thresholds)

    success_text = "?" if success is None else str(success)
    if result is None:
        print(f"{hdf5_path}: success={success_text}  t*=--")
        return None

    print(
        f"{hdf5_path}: success={success_text}  "
        f"t*={result.t_star}  [{result.trigger_type}]"
    )
    print(json.dumps(result.details, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    thresholds = Thresholds(
        empty_width_max=args.empty_width_max,
        retry_distance_max=args.retry_distance_max,
        active_command_min=args.active_command_min,
        stuck_response_max=args.stuck_response_max,
        stuck_displacement_max=args.stuck_displacement_max,
    )
    try:
        thresholds.validate()
        chunk_size = load_chunk_size(args.config.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"chunk_size={chunk_size}")
    print(f"thresholds={json.dumps(asdict(thresholds), sort_keys=True)}")

    results: list[Detection | None] = []
    had_error = False
    for path in args.hdf5_paths:
        try:
            results.append(process_episode(path.resolve(), chunk_size, thresholds))
        except (OSError, ValueError) as exc:
            had_error = True
            results.append(None)
            print(f"ERROR {path}: {exc}", file=sys.stderr)

    if len(args.hdf5_paths) == 1 and results[0] is not None:
        print(f"\nexport COSMOS_OFFSET_START_T={results[0].t_star}")

    if had_error:
        return 2
    if len(args.hdf5_paths) == 1 and results[0] is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
