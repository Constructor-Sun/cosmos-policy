#!/usr/bin/env python3
"""Leave-one-demo-out geometry evaluation for the feasible-region verifier."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXECUTE_BIN = ROOT / "bin" / "execute"
if str(EXECUTE_BIN) not in sys.path:
    sys.path.insert(0, str(EXECUTE_BIN))

from libero_feasible_region_verifier import (  # noqa: E402
    FEASIBLE,
    NOT_FEASIBLE,
    LiberoFeasibleRegionVerifier,
    ReadyDistanceMemory,
)


def arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def segment_key(task_name: str, demo_id: str, segment: dict[str, Any]):
    return (
        str(task_name),
        str(demo_id),
        int(segment["planner_step_id"]),
        str(segment["skill"]),
        arguments_key(segment.get("arguments", {})),
    )


def template_key(template: dict[str, Any]):
    return (
        str(template["task_name"]),
        str(template["demo_id"]),
        int(template["planner_step_id"]),
        str(template["skill"]),
        arguments_key(template.get("arguments", {})),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    memory = ReadyDistanceMemory(args.phase_targets, args.segments_manifest)
    verifier = LiberoFeasibleRegionVerifier(
        memory,
        min_demo_votes=args.min_demo_votes,
        progress_tolerance_px=args.progress_tolerance_px,
        evidence_updates=args.evidence_updates,
    )
    payload = torch.load(args.phase_targets, map_location="cpu", weights_only=False)
    templates_by_segment = defaultdict(list)
    for template in payload.get("templates", []):
        templates_by_segment[template_key(template)].append(template)

    manifest = json.loads(args.segments_manifest.read_text())
    counts = Counter()
    by_skill = defaultdict(Counter)
    ready_distances = defaultdict(list)
    for record in manifest.get("records", []):
        if not record.get("valid"):
            continue
        task_name, demo_id = str(record["task_name"]), str(record["demo_id"])
        for segment in record.get("segments", []):
            ready_frame = segment.get("ready_frame")
            if ready_frame is None:
                counts["segments_without_ready"] += 1
                continue
            skill = str(segment["skill"])
            templates = templates_by_segment.get(
                segment_key(task_name, demo_id, segment), ()
            )
            ready = [item for item in templates if int(item["frame"]) == int(ready_frame)]
            prefix = [item for item in templates if int(item["frame"]) < int(ready_frame)]
            if not ready:
                counts["segments_missing_ready_template"] += 1
                continue

            verifier.reset(
                task_name,
                int(segment["planner_step_id"]),
                skill,
                segment.get("arguments", {}),
                exclude_demo_ids=(demo_id,),
            )
            false_correction = False
            for sample in sorted((*prefix, *ready), key=lambda item: int(item["frame"])):
                label = "ready" if int(sample["frame"]) == int(ready_frame) else "prefix"
                result = verifier.update(
                    sample["target_center_xy"],
                    sample["bbox_xyxy"],
                    sample["gripper_xy"],
                )
                counts[f"{label}_samples"] += 1
                counts[f"{label}_{result.status}"] += 1
                counts[f"reason_{result.reason}"] += 1
                by_skill[skill][f"{label}_samples"] += 1
                by_skill[skill][f"{label}_{result.status}"] += 1
                by_skill[skill][f"reason_{result.reason}"] += 1
                false_correction = false_correction or result.status == NOT_FEASIBLE
                if label == "ready" and result.current_distance is not None:
                    ready_distances[skill].append(result.current_distance)
            counts["success_segments"] += 1
            by_skill[skill]["success_segments"] += 1
            if false_correction:
                counts["success_false_corrections"] += 1
                by_skill[skill]["success_false_corrections"] += 1

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "counts": dict(sorted(counts.items())),
        "success_false_correction_rate": ratio(
            counts["success_false_corrections"], counts["success_segments"]
        ),
        "ready_recall": ratio(counts[f"ready_{FEASIBLE}"], counts["ready_samples"]),
        "prefix_early_entry_rate": ratio(
            counts[f"prefix_{FEASIBLE}"], counts["prefix_samples"]
        ),
        "by_skill": {
            skill: {
                **dict(sorted(values.items())),
                "success_false_correction_rate": ratio(
                    values["success_false_corrections"], values["success_segments"]
                ),
                "ready_recall": ratio(
                    values[f"ready_{FEASIBLE}"], values["ready_samples"]
                ),
                "prefix_early_entry_rate": ratio(
                    values[f"prefix_{FEASIBLE}"], values["prefix_samples"]
                ),
                "ready_distance_min_mean_max": (
                    [
                        float(np.min(ready_distances[skill])),
                        float(np.mean(ready_distances[skill])),
                        float(np.max(ready_distances[skill])),
                    ]
                    if ready_distances[skill]
                    else []
                ),
            }
            for skill, values in sorted(by_skill.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-targets",
        type=Path,
        default=ROOT / "skill_memory/libero_10/phase_targets.pt",
    )
    parser.add_argument(
        "--segments-manifest",
        type=Path,
        default=ROOT / "skill_memory/libero_10/segments_ready_fixed16.json",
    )
    parser.add_argument("--min-demo-votes", type=int, default=2)
    parser.add_argument("--progress-tolerance-px", type=float, default=5.0)
    parser.add_argument("--evidence-updates", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
