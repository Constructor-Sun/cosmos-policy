"""Safely update only Pick ready memory to a 9 cm target distance.

The existing active artifacts are left untouched while temporary output files
are built and validated.  On success, the original artifacts are renamed to
timestamped backups and the validated files become active.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from memory_system.offline import update_memory_ready_by_distance as updater


DEFAULT_PICK_DISTANCE_M = 0.09
DEFAULT_PICK_WINDOW = 48


def artifact_key(item: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(item["task_name"]),
        str(item["demo_id"]),
        int(item["planner_step_id"]),
    )


def validate_outputs(
    old_ready_path: Path,
    old_feasible_path: Path,
    new_ready_path: Path,
    new_feasible_path: Path,
    pick_distance_m: float,
    tolerance_m: float,
) -> dict[str, Any]:
    old_ready = torch.load(old_ready_path, map_location="cpu", weights_only=False)
    old_feasible = torch.load(
        old_feasible_path, map_location="cpu", weights_only=False
    )
    new_ready = torch.load(new_ready_path, map_location="cpu", weights_only=False)
    new_feasible = torch.load(
        new_feasible_path, map_location="cpu", weights_only=False
    )

    old_prototypes = old_ready.get("prototypes", [])
    new_prototypes = new_ready.get("prototypes", [])
    old_targets = old_feasible.get("targets", [])
    new_targets = new_feasible.get("targets", [])
    if len(old_prototypes) != len(new_prototypes):
        raise RuntimeError("ready3d prototype count changed")
    if len(old_targets) != len(new_targets):
        raise RuntimeError("feasible target count changed")

    old_ready_by_key = {artifact_key(item): item for item in old_prototypes}
    new_ready_by_key = {artifact_key(item): item for item in new_prototypes}
    old_feasible_by_key = {artifact_key(item): item for item in old_targets}
    new_feasible_by_key = {artifact_key(item): item for item in new_targets}
    if old_ready_by_key.keys() != new_ready_by_key.keys():
        raise RuntimeError("ready3d keys changed")
    if old_feasible_by_key.keys() != new_feasible_by_key.keys():
        raise RuntimeError("feasible target keys changed")

    changed_pick_frames = 0
    unchanged_pick_frames = 0
    pick_distances = []
    non_pick_frame_changes = []
    cross_artifact_mismatches = []

    for key, new_item in new_ready_by_key.items():
        old_item = old_ready_by_key[key]
        old_frame = int(old_item["ready_frame"])
        new_frame = int(new_item["ready_frame"])
        if str(new_item["skill"]) == "Pick":
            pick_distances.append(float(new_item["distance_m"]))
            if new_frame == old_frame:
                unchanged_pick_frames += 1
            else:
                changed_pick_frames += 1
        elif new_frame != old_frame:
            non_pick_frame_changes.append((key, old_frame, new_frame))

        feasible_item = new_feasible_by_key.get(key)
        if feasible_item is not None and int(feasible_item["ready_frame"]) != new_frame:
            cross_artifact_mismatches.append(
                (key, new_frame, int(feasible_item["ready_frame"]))
            )

    for key, new_item in new_feasible_by_key.items():
        old_item = old_feasible_by_key[key]
        if (
            str(new_item["skill"]) != "Pick"
            and int(new_item["ready_frame"]) != int(old_item["ready_frame"])
        ):
            non_pick_frame_changes.append(
                (key, int(old_item["ready_frame"]), int(new_item["ready_frame"]))
            )

    if non_pick_frame_changes:
        raise RuntimeError(
            f"non-Pick ready frames changed: {non_pick_frame_changes[:5]}"
        )
    if cross_artifact_mismatches:
        raise RuntimeError(
            "ready3d/feasible ready-frame mismatch: "
            f"{cross_artifact_mismatches[:5]}"
        )
    if not pick_distances:
        raise RuntimeError("no Pick prototypes found")

    distances = np.asarray(pick_distances, dtype=np.float64)
    outside_tolerance = int(
        np.sum(np.abs(distances - float(pick_distance_m)) > float(tolerance_m))
    )
    return {
        "pick_count": int(len(distances)),
        "changed_pick_frames": int(changed_pick_frames),
        "unchanged_pick_frames": int(unchanged_pick_frames),
        "outside_tolerance_count": outside_tolerance,
        "mean_cm": float(np.mean(distances) * 100.0),
        "median_cm": float(np.median(distances) * 100.0),
        "p5_cm": float(np.percentile(distances, 5) * 100.0),
        "p95_cm": float(np.percentile(distances, 95) * 100.0),
        "min_cm": float(np.min(distances) * 100.0),
        "max_cm": float(np.max(distances) * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-dir", type=Path, default=updater.DEFAULT_MEM_DIR)
    parser.add_argument("--manifest", type=Path, default=updater.DEFAULT_MANIFEST)
    parser.add_argument("--demo-dir", type=Path, default=updater.DEFAULT_DEMO_DIR)
    parser.add_argument("--pick-distance-m", type=float, default=DEFAULT_PICK_DISTANCE_M)
    parser.add_argument("--window", type=int, default=DEFAULT_PICK_WINDOW)
    parser.add_argument("--tolerance-m", type=float, default=updater.TOLERANCE)
    parser.add_argument("--fallback-offset", type=int, default=updater.FALLBACK_OFFSET)
    parser.add_argument("--backup-label", default="pick7cm_before_pick9cm")
    args = parser.parse_args()

    memory_dir = args.memory_dir.resolve()
    ready_path = memory_dir / "ready3d_targets.pt"
    feasible_path = memory_dir / "feasible_recovery_targets.pt"
    manifest_path = args.manifest.resolve()
    for path in (ready_path, feasible_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.backup_label}_{stamp}"
    backup_ready = memory_dir / f"ready3d_targets.{run_id}.pt"
    backup_feasible = memory_dir / f"feasible_recovery_targets.{run_id}.pt"
    backup_manifest = memory_dir / f"segments_ready_fixed16.{run_id}.json"
    temp_ready = memory_dir / f".ready3d_targets.pick9.{stamp}.{os.getpid()}.tmp.pt"
    temp_feasible = (
        memory_dir / f".feasible_recovery_targets.pick9.{stamp}.{os.getpid()}.tmp.pt"
    )
    receipt_path = memory_dir / f"update_memory_pick9.{stamp}.json"

    for path in (backup_ready, backup_feasible, backup_manifest, receipt_path):
        if path.exists():
            raise FileExistsError(path)

    shutil.copy2(ready_path, temp_ready)
    shutil.copy2(feasible_path, temp_feasible)

    updater.SKILL_DISTANCE = {"Pick": float(args.pick_distance_m)}
    updater.WINDOW = int(args.window)
    updater.TOLERANCE = float(args.tolerance_m)
    updater.FALLBACK_OFFSET = int(args.fallback_offset)

    print(
        f"Building Pick-only update: distance={args.pick_distance_m:.3f} m, "
        f"window={args.window}, tolerance={args.tolerance_m:.3f} m",
        flush=True,
    )
    try:
        updater.update_files(
            manifest_path=manifest_path,
            ready3d_path=temp_ready,
            feasible_path=temp_feasible,
            demo_dir=args.demo_dir.resolve(),
            update_vae=True,
        )
        stats = validate_outputs(
            old_ready_path=ready_path,
            old_feasible_path=feasible_path,
            new_ready_path=temp_ready,
            new_feasible_path=temp_feasible,
            pick_distance_m=args.pick_distance_m,
            tolerance_m=args.tolerance_m,
        )

        # Commit only after both temporary artifacts have passed validation.
        # If any rename fails, restore active files from the retained backups.
        try:
            shutil.copy2(manifest_path, backup_manifest)
            ready_path.rename(backup_ready)
            feasible_path.rename(backup_feasible)
            temp_ready.replace(ready_path)
            temp_feasible.replace(feasible_path)
        except BaseException:
            if backup_ready.is_file():
                shutil.copy2(backup_ready, ready_path)
            if backup_feasible.is_file():
                shutil.copy2(backup_feasible, feasible_path)
            raise

        receipt = {
            "run_id": run_id,
            "pick_distance_m": float(args.pick_distance_m),
            "window": int(args.window),
            "tolerance_m": float(args.tolerance_m),
            "fallback_offset": int(args.fallback_offset),
            "active_ready3d": str(ready_path),
            "active_feasible": str(feasible_path),
            "backup_ready3d": str(backup_ready),
            "backup_feasible": str(backup_feasible),
            "backup_manifest": str(backup_manifest),
            "stats": stats,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
    except BaseException:
        # These files were created by this run and are never user artifacts.
        temp_ready.unlink(missing_ok=True)
        temp_feasible.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
