"""Generate a full LIBERO-90 segment manifest for PointCloud Action Memory.

This script lives under memory_system/pointcloud_action and only *calls* the
existing LIBERO manifest labelers; it does not modify other memory_system logic.

Pipeline:
1. Build a full LIBERO-90 task config (task name + language) from BDDL files.
2. Call memory_system.offline.label_segments.label_manifest to produce base segments.
3. Call memory_system.offline.label_boundaries.enrich_manifest to add ready/terminal
   boundaries (same style as libero90_segments_ready.json).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from memory_system.pointcloud_action.config import DEFAULT_DEMO_DIR

ROOT = Path(__file__).resolve().parents[3]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"


def _bddl_dir(suite: str = "libero_90") -> Path:
    try:
        from libero.libero import get_libero_path
        installed = Path(get_libero_path("bddl_files")) / suite
        if installed.is_dir():
            return installed
    except Exception:
        pass
    return LIBERO_PLUS / "libero/libero/bddl_files" / suite


def _language_from_bddl(path: Path) -> str:
    text = path.read_text(errors="ignore")
    match = re.search(r"\(:language\s+([^)]+)\)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"No :language found in {path}")
    return " ".join(match.group(1).split())


def build_task_config(suite: str = "libero_90") -> dict:
    bddl_dir = _bddl_dir(suite)
    if not bddl_dir.is_dir():
        raise FileNotFoundError(f"BDDL directory not found: {bddl_dir}")
    tasks = []
    for path in sorted(bddl_dir.glob("*.bddl")):
        name = path.stem
        language = _language_from_bddl(path)
        tasks.append({"name": name, "language": language})
    if not tasks:
        raise RuntimeError(f"No BDDL tasks found under {bddl_dir}")
    return {"suite": suite, "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a full LIBERO-90 segment manifest for pointcloud_action memory."
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=DEFAULT_DEMO_DIR,
        help="Directory containing <task>_demo.hdf5 files.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="If set, only label this single LIBERO-90 task (useful for smoke tests).",
    )
    parser.add_argument(
        "--task-config-output",
        type=Path,
        default=ROOT
        / "memory_system/pointcloud_action/libero90_experiment_tasks_full.json",
        help="Output path for the generated task config JSON.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT
        / "memory_system/pointcloud_action/libero90_segments_ready_full.json",
        help="Output path for the final enriched segment manifest.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=10,
        help="Number of demos to label per task (0 means all).",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--lift-threshold",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--boundary-mode",
        choices=["event", "fixed"],
        default="event",
    )
    parser.add_argument(
        "--fallback-length",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    # 1. Build full task config from BDDL.
    task_config = build_task_config("libero_90")
    args.task_config_output.parent.mkdir(parents=True, exist_ok=True)
    args.task_config_output.write_text(
        json.dumps(task_config, indent=2) + "\n"
    )
    print(f"Wrote task config with {len(task_config['tasks'])} tasks: {args.task_config_output}")

    # 2. Label base semantic segments.
    #    This is the expensive step: it creates a MuJoCo env per task and checks
    #    simulator predicates over the requested demos.
    from memory_system.offline.label_segments import label_manifest

    base_manifest = args.manifest_output.with_name(
        args.manifest_output.stem + ".base.json"
    )
    label_manifest(
        input_dir=args.demo_dir,
        output=base_manifest,
        suite="libero_90",
        task=args.task,
        task_config=args.task_config_output,
        max_per_task=args.max_demos,
        stable_frames=args.stable_frames,
        lift_threshold=args.lift_threshold,
    )

    # 3. Add ready/terminal boundaries.
    from memory_system.offline.label_boundaries import enrich_manifest

    manifest = json.loads(base_manifest.read_text())
    counts = enrich_manifest(
        manifest,
        input_dir=args.demo_dir,
        boundary_mode=args.boundary_mode,
        fallback_length=args.fallback_length,
        min_run=args.stable_frames,
        max_suffix=64,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote enriched manifest: {args.manifest_output}")
    print(f"Boundary counts: {counts}")


if __name__ == "__main__":
    main()
