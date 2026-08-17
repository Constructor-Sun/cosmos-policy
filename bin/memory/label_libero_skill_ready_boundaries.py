#!/usr/bin/env python3
"""Add action-derived ready/terminal boundaries to a LIBERO segment manifest.

The simulator segment labeler supplies the semantic skill interval and its success
endpoint. Auto mode (the default) uses a sustained gripper edge when available,
otherwise it selects a same-task terminal motif from action change-point candidates.
Fixed mode optionally uses the final action horizon. No simulator state or object
pose is read.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EVENT_SKILLS = {
    "Pick": ("close", 1.0),
    "Open": ("close", 1.0),
    "Close": ("close", 1.0),
    "PlaceIn": ("open", -1.0),
    "PlaceOn": ("open", -1.0),
}


@dataclass(frozen=True)
class Boundary:
    terminal_start: int | None
    method: str
    confidence: float


@dataclass
class SegmentRef:
    segment: dict[str, Any]
    actions: np.ndarray
    start: int
    end: int
    group: tuple[str, int, str, str]


def sustained_event_start(
    gripper: np.ndarray,
    target_value: float,
    min_run: int,
) -> int | None:
    """Return the final false->true target run, excluding an initial target run."""
    target = np.asarray(gripper) * target_value > 0.0
    starts = np.flatnonzero(target[1:] & ~target[:-1]) + 1
    for start in starts[::-1]:
        stop = start
        while stop < len(target) and target[stop]:
            stop += 1
        if stop - start >= min_run:
            return int(start)
    return None


def fixed_terminal_boundary(length: int, horizon: int) -> Boundary:
    return Boundary(max(0, length - horizon), "fixed_last_chunk", 0.0)


def action_features(actions: np.ndarray) -> np.ndarray:
    """Represent values, local changes, motion energy, and direction persistence."""
    actions = np.asarray(actions, dtype=np.float64)
    delta = np.vstack([np.zeros((1, actions.shape[1])), np.diff(actions, axis=0)])
    translation, rotation = actions[:, :3], actions[:, 3:6]

    def persistence(values: np.ndarray) -> np.ndarray:
        previous = np.vstack([values[:1], values[:-1]])
        denom = np.linalg.norm(values, axis=1) * np.linalg.norm(previous, axis=1)
        return np.divide(
            np.sum(values * previous, axis=1),
            denom,
            out=np.ones(len(values), dtype=np.float64),
            where=denom > 1e-8,
        )

    extras = np.column_stack(
        [
            np.linalg.norm(translation, axis=1),
            np.linalg.norm(rotation, axis=1),
            persistence(translation),
            persistence(rotation),
        ]
    )
    return np.concatenate([actions, delta, extras], axis=1)


def change_point_candidates(
    features: np.ndarray,
    fallback_length: int,
    max_suffix: int,
    min_terminal: int,
    count: int = 8,
) -> tuple[list[int], dict[int, float]]:
    """Find terminal change-point candidates using adjacent-window statistics."""
    length = len(features)
    lower = max(1, length - max_suffix)
    upper = length - min_terminal
    if upper < lower:
        return [max(0, length - min_terminal)], {}
    window = min(5, max(2, length // 12))
    raw_scores: dict[int, float] = {}
    for point in range(max(lower, window), min(upper, length - window) + 1):
        before, after = features[point - window : point], features[point : point + window]
        mean_change = np.linalg.norm(before.mean(0) - after.mean(0))
        scale_change = np.linalg.norm(before.std(0) - after.std(0))
        raw_scores[point] = float(mean_change + 0.5 * scale_change)
    maximum = max(raw_scores.values(), default=0.0)
    scores = {
        point: score / maximum if maximum > 0 else 0.0
        for point, score in raw_scores.items()
    }
    ranked = sorted(scores, key=scores.get, reverse=True)[:count]
    fallback = min(max(length - fallback_length, lower), upper)
    return sorted(set(ranked + [fallback, lower, upper])), scores


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Normalized multivariate DTW distance."""
    rows, cols = len(first), len(second)
    cost = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    steps = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            choices = (
                (cost[row - 1, col], steps[row - 1, col]),
                (cost[row, col - 1], steps[row, col - 1]),
                (cost[row - 1, col - 1], steps[row - 1, col - 1]),
            )
            previous_cost, previous_steps = min(choices, key=lambda value: value[0])
            cost[row, col] = previous_cost + np.linalg.norm(first[row - 1] - second[col - 1])
            steps[row, col] = previous_steps + 1
    return float(cost[rows, cols] / max(int(steps[rows, cols]), 1))


def medoid_index(sequences: list[np.ndarray]) -> int:
    if len(sequences) == 1:
        return 0
    totals = np.zeros(len(sequences), dtype=np.float64)
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            distance = dtw_distance(sequences[left], sequences[right])
            totals[left] += distance
            totals[right] += distance
    return int(np.argmin(totals))


def shared_terminal_boundaries(
    sequences: list[np.ndarray],
    fallback_length: int = 16,
    max_suffix: int = 64,
    min_terminal: int = 4,
) -> list[Boundary]:
    """Infer one terminal suffix boundary per same-task skill demonstration."""
    if len(sequences) < 2:
        return [
            Boundary(max(0, len(sequence) - fallback_length), "fallback_last_chunk", 0.0)
            for sequence in sequences
        ]
    features = [action_features(sequence) for sequence in sequences]
    stacked = np.concatenate(features, axis=0)
    scale = np.maximum(stacked.std(axis=0), 0.1)
    features = [(value - stacked.mean(axis=0)) / scale for value in features]
    candidates, strengths = zip(
        *[
            change_point_candidates(
                value, fallback_length, max_suffix, min_terminal
            )
            for value in features
        ]
    )
    starts = [max(0, len(value) - fallback_length) for value in features]
    initial_suffixes = [value[start:] for value, start in zip(features, starts)]
    template = initial_suffixes[medoid_index(initial_suffixes)]
    margins = [0.0] * len(features)
    for _ in range(3):
        selected: list[np.ndarray] = []
        for index, value in enumerate(features):
            scored: list[tuple[float, int]] = []
            for point in candidates[index]:
                suffix = value[point:]
                duration = abs(np.log(max(len(suffix), 1) / max(len(template), 1)))
                score = (
                    dtw_distance(suffix, template)
                    + 0.30 * duration
                    - 0.08 * strengths[index].get(point, 0.0)
                )
                scored.append((score, point))
            scored.sort()
            starts[index] = scored[0][1]
            margins[index] = scored[1][0] - scored[0][0] if len(scored) > 1 else 0.0
            selected.append(value[starts[index] :])
        template = selected[medoid_index(selected)]
    boundaries = []
    for value, start, margin in zip(features, starts, margins):
        if len(value) - start > 2 * fallback_length:
            boundaries.append(
                Boundary(
                    max(0, len(value) - fallback_length),
                    "fallback_last_chunk",
                    0.0,
                )
            )
        else:
            boundaries.append(
                Boundary(
                    start,
                    "same_task_shared_suffix",
                    float(np.clip(1.0 - np.exp(-max(margin, 0.0)), 0.0, 0.9)),
                )
            )
    return boundaries


def assign_boundary(segment_ref: SegmentRef, boundary: Boundary) -> None:
    segment = segment_ref.segment
    if boundary.terminal_start is None:
        terminal_start = None
    else:
        terminal_start = segment_ref.start + boundary.terminal_start
    segment.update(
        {
            "terminal_start": terminal_start,
            "ready_frame": terminal_start,
            "terminal_length": (
                segment_ref.end - terminal_start if terminal_start is not None else 0
            ),
            "boundary_method": boundary.method,
            "boundary_confidence": round(float(boundary.confidence), 6),
        }
    )


def enrich_manifest(
    manifest: dict[str, Any],
    input_dir: Path,
    boundary_mode: str,
    fallback_length: int,
    min_run: int,
    max_suffix: int,
) -> dict[str, int]:
    import h5py

    groups: dict[tuple[str, int, str, str], list[SegmentRef]] = {}
    counts = {"fixed": 0, "event": 0, "shared": 0, "fallback": 0, "skipped": 0}
    for record in manifest["records"]:
        if not record.get("valid"):
            continue
        path = input_dir / f"{record['task_name']}_demo.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            actions = handle["data"][record["demo_id"]]["actions"][:]
        for segment in record["segments"]:
            start = min(max(int(segment["start"]), 0), len(actions))
            end = min(max(int(segment["end"]), start), len(actions))
            if segment.get("status") == "already_satisfied" or end <= start:
                ref = SegmentRef(segment, actions[start:end], start, end, ("", 0, "", ""))
                assign_boundary(ref, Boundary(None, "already_satisfied", 1.0))
                counts["skipped"] += 1
                continue
            local = actions[start:end]
            if boundary_mode == "fixed":
                ref = SegmentRef(segment, local, start, end, ("", 0, "", ""))
                assign_boundary(
                    ref, fixed_terminal_boundary(len(local), fallback_length)
                )
                counts["fixed"] += 1
                continue
            event = EVENT_SKILLS.get(segment["skill"])
            event_start = (
                sustained_event_start(local[:, -1], event[1], min_run) if event else None
            )
            key = (
                record["task_name"],
                int(segment["planner_step_id"]),
                segment["skill"],
                json.dumps(segment.get("arguments", {}), sort_keys=True),
            )
            ref = SegmentRef(segment, local, start, end, key)
            if event_start is not None:
                assign_boundary(
                    ref,
                    Boundary(event_start, f"final_{event[0]}_edge", 0.95),
                )
                counts["event"] += 1
            else:
                groups.setdefault(key, []).append(ref)

    for refs in groups.values():
        boundaries = shared_terminal_boundaries(
            [ref.actions for ref in refs],
            fallback_length=fallback_length,
            max_suffix=max_suffix,
        )
        for ref, boundary in zip(refs, boundaries):
            assign_boundary(ref, boundary)
            bucket = "fallback" if boundary.method.startswith("fallback") else "shared"
            counts[bucket] += 1
    manifest["version"] = max(int(manifest.get("version", 1)), 2)
    manifest["ready_boundary_labeling"] = {
        "boundary_mode": boundary_mode,
        "source": (
            "fixed_action_horizon" if boundary_mode == "fixed" else "recorded_actions_only"
        ),
        "fallback_length": fallback_length,
        "minimum_event_run": min_run,
        "maximum_shared_suffix": max_suffix,
    }
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boundary-mode", choices=("fixed", "auto"), default="auto"
    )
    parser.add_argument("--fallback-length", type=int, default=16)
    parser.add_argument("--min-run", type=int, default=3)
    parser.add_argument("--max-suffix", type=int, default=64)
    args = parser.parse_args()
    if min(args.fallback_length, args.min_run, args.max_suffix) <= 0:
        parser.error("fallback-length, min-run, and max-suffix must be positive")
    manifest = json.loads(args.segments_manifest.read_text())
    counts = enrich_manifest(
        manifest,
        args.input_dir.resolve(),
        args.boundary_mode,
        args.fallback_length,
        args.min_run,
        args.max_suffix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote action-ready boundaries to {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
