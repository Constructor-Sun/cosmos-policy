"""Check ready3d_targets.pt against the existing 2D ready memory.

The 2D and 3D artifacts are produced from different image sources, so their
RGB pixels are not expected to be identical.  The key metadata fields
(task / step / skill / arguments / demo / ready_frame) must match.

This script prints coverage and mismatch summaries and optionally writes
side-by-side visualizations: 2D template crop vs regenerated 3D observation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
if LIBERO_PLUS.is_dir() and str(LIBERO_PLUS) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS))

from memory_system.offline.build_ready3d import (  # noqa: E402
    create_env,
    instance_mask,
    resolve_instance,
)
from memory_system.offline.build_targets import patch_numpy2_segmentation  # noqa: E402

DEFAULT_MANIFEST = ROOT / "skill_memory_test/libero_10/segments_ready_fixed16.json"
DEFAULT_PHASE_TARGETS = ROOT / "skill_memory_test/libero_10/phase_targets.pt"
DEFAULT_READY3D = ROOT / "skill_memory_test/libero_10/ready3d_targets.pt"
DEFAULT_DEMO_DIR = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"


def arguments_key(arguments: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in arguments.items()))


def entry_key(entry: dict) -> tuple:
    return (
        entry["task_name"],
        int(entry["planner_step_id"]),
        entry["skill"],
        arguments_key(entry["arguments"]),
        entry["demo_id"],
    )


def load_2d_entries(phase_targets: Path, manifest_path: Path) -> list[dict]:
    payload = torch.load(phase_targets, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(manifest_path).read_text())

    ready_frames = {}
    for record in manifest.get("records", []):
        if not record.get("valid"):
            continue
        for segment in record.get("segments", []):
            ready_frame = segment.get("ready_frame")
            if ready_frame is None:
                continue
            key = (
                record["task_name"],
                record["demo_id"],
                int(segment["planner_step_id"]),
                segment["skill"],
                arguments_key(segment.get("arguments", {})),
            )
            ready_frames[key] = int(ready_frame)

    entries = []
    for template in payload.get("templates", []):
        key = (
            template["task_name"],
            template["demo_id"],
            int(template["planner_step_id"]),
            template["skill"],
            arguments_key(template.get("arguments", {})),
        )
        ready_frame = ready_frames.get(key)
        if ready_frame is None or int(template["frame"]) != ready_frame:
            continue
        entries.append({
            "task_name": template["task_name"],
            "planner_step_id": int(template["planner_step_id"]),
            "skill": template["skill"],
            "arguments": dict(template.get("arguments", {})),
            "demo_id": template["demo_id"],
            "ready_frame": ready_frame,
            "crop_rgb": np.asarray(template.get("crop_rgb")),
            "crop_mask": np.asarray(template.get("crop_mask")),
        })
    return entries


def load_3d_entries(path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload.get("prototypes", []))


def visualize_entries(
    entries: list[dict],
    demo_dir: Path,
    output_dir: Path,
    max_visualize: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task = {}
    for entry in entries:
        by_task.setdefault(entry["task_name"], []).append(entry)

    for task_name, task_entries in by_task.items():
        env = create_env(task_name, 256)
        env.reset()
        try:
            import h5py

            h5_path = demo_dir / f"{task_name}_demo.hdf5"
            with h5py.File(h5_path, "r") as handle:
                for entry in task_entries[:max_visualize]:
                    group = handle["data"][entry["demo_id"]]
                    states = group["states"][:]
                    frame = min(max(int(entry["ready_frame"]), 0), len(states) - 1)
                    obs = env.regenerate_obs_from_state(states[frame])
                    inst = resolve_instance(env, entry["arguments"], entry["skill"])
                    if inst is None:
                        continue
                    mask = instance_mask(env, obs, inst)
                    rgb = np.asarray(obs["agentview_image"])
                    rgb = np.flipud(rgb) if True else rgb
                    overlay = rgb.copy()
                    if mask is not None and mask.any():
                        ys, xs = np.nonzero(mask)
                        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                        cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                    crop2d = entry.get("crop_rgb")
                    if crop2d is None or not hasattr(crop2d, "size") or crop2d.size == 0:
                        crop2d = np.zeros((128, 128, 3), dtype=np.uint8)
                    crop2d = cv2.resize(crop2d, (256, 256))
                    canvas = np.hstack([crop2d, overlay])
                    title = (
                        f"{entry['skill']} {entry['planner_step_id']} "
                        f"{entry['demo_id']} frame={entry['ready_frame']}"
                    )
                    cv2.putText(
                        canvas, title, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )
                    out = output_dir / (
                        f"{task_name[:20]}__{entry['demo_id']}__"
                        f"{entry['planner_step_id']}.png"
                    )
                    cv2.imwrite(str(out), canvas)
                    print("wrote", out)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-targets", type=Path, default=DEFAULT_PHASE_TARGETS)
    parser.add_argument("--segments-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ready3d", type=Path, default=DEFAULT_READY3D)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--visualize-dir", type=Path)
    parser.add_argument("--max-visualize", type=int, default=5)
    args = parser.parse_args()
    patch_numpy2_segmentation()

    two_d = load_2d_entries(args.phase_targets, args.segments_manifest)
    three_d = load_3d_entries(args.ready3d)

    keys_2d = {entry_key(e): e for e in two_d}
    keys_3d = {entry_key(e): e for e in three_d}

    missing = sorted(set(keys_2d) - set(keys_3d))
    extra = sorted(set(keys_3d) - set(keys_2d))
    common = sorted(set(keys_2d) & set(keys_3d))
    frame_mismatch = [
        key for key in common
        if keys_2d[key]["ready_frame"] != keys_3d[key]["ready_frame"]
    ]

    print("=== summary ===")
    print(f"2D ready entries      : {len(two_d)}")
    print(f"3D ready prototypes   : {len(three_d)}")
    print(f"common keys           : {len(common)}")
    print(f"missing in 3D         : {len(missing)}")
    print(f"extra in 3D           : {len(extra)}")
    print(f"ready_frame mismatch  : {len(frame_mismatch)}")

    print("\n=== missing by skill ===")
    for skill, n in Counter(keys_2d[k]["skill"] for k in missing).most_common():
        print(f"{n:4d}  {skill}")

    print("\n=== extra by skill ===")
    for skill, n in Counter(keys_3d[k]["skill"] for k in extra).most_common():
        print(f"{n:4d}  {skill}")

    print("\n=== common by skill ===")
    for skill, n in Counter(keys_2d[k]["skill"] for k in common).most_common():
        print(f"{n:4d}  {skill}")

    if missing:
        print("\n=== missing examples ===")
        for key in missing[:10]:
            print(key)

    if extra:
        print("\n=== extra examples ===")
        for key in extra[:10]:
            print(key)

    if frame_mismatch:
        print("\n=== frame mismatches ===")
        for key in frame_mismatch[:10]:
            e2 = keys_2d[key]
            e3 = keys_3d[key]
            print(key, "2d=", e2["ready_frame"], "3d=", e3["ready_frame"])

    print("\n=== 3D distance_m by skill ===")
    for skill in sorted({e["skill"] for e in three_d}):
        ds = np.asarray([e["distance_m"] for e in three_d if e["skill"] == skill])
        if len(ds) == 0:
            continue
        print(
            f"{skill:10s} n={len(ds):3d} "
            f"median={np.median(ds):.3f} "
            f"p10={np.percentile(ds, 10):.3f} "
            f"p90={np.percentile(ds, 90):.3f} "
            f"min={ds.min():.3f} max={ds.max():.3f}"
        )

    def phase_key(e: dict) -> tuple:
        return (
            e["task_name"],
            e["planner_step_id"],
            e["skill"],
            arguments_key(e["arguments"]),
        )

    phase_counts_3d = Counter(phase_key(e) for e in three_d)
    low_3d = {k: v for k, v in phase_counts_3d.items() if v < 2}
    print("\n=== 3D phases with <2 demos ===")
    print(f"total 3D phases: {len(phase_counts_3d)}")
    print(f"phases with <2 demos: {len(low_3d)}")
    for k, v in sorted(low_3d.items())[:20]:
        print(f"n={v} {k}")

    phase_counts_2d = Counter(phase_key(e) for e in two_d)
    low_2d = {k: v for k, v in phase_counts_2d.items() if v < 2}
    print("\n=== 2D phases with <2 demos ===")
    print(f"total 2D phases: {len(phase_counts_2d)}")
    print(f"phases with <2 demos: {len(low_2d)}")
    for k, v in sorted(low_2d.items())[:20]:
        print(f"n={v} {k}")

    dup_keys = [k for k, v in Counter(entry_key(e) for e in three_d).items() if v > 1]
    print("\n=== duplicate 3D keys ===")
    print(f"duplicates: {len(dup_keys)}")
    for k in dup_keys[:10]:
        print(k)

    if args.visualize_dir is not None:
        print("\n=== visualize common examples ===")
        visualize_entries(
            [keys_2d[k] for k in common],
            args.demo_dir,
            args.visualize_dir,
            args.max_visualize,
        )


if __name__ == "__main__":
    main()
