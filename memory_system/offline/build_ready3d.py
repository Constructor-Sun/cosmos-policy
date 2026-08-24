"""Offline builder for the 3D ready-distance memory used by Feasible3D.

This module belongs to the formal offline data-construction path.  It must not
depend on ``tests/``.  It reuses shared helpers from ``build_targets.py`` and
``memory_system.geometry`` to compute, for each training ``ready_frame``:

    target_xyz_world = median of RGB-D back-projected visible-surface points
    distance_m       = ||eef_pos_ready - target_xyz_world||
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
if LIBERO_PLUS.is_dir() and str(LIBERO_PLUS) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS))

from memory_system.geometry import (  # noqa: E402
    camera_params,
    depth_to_metric,
    flip_depth,
    pixel_to_world,
)
from memory_system.offline.build_targets import (  # noqa: E402
    match_instance,
    match_region_anchor,
    patch_numpy2_segmentation,
    resolve_bddl,
)

READY3D_FORMAT = "libero_ready3d_targets_v1"
DEFAULT_DEMO_DIR = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
DEFAULT_MANIFEST = ROOT / "skill_memory_test/libero_10/segments_ready_fixed16.json"
DEFAULT_OUTPUT = ROOT / "skill_memory_test/libero_10/ready3d_targets.pt"


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def create_env(task_name: str, resolution: int):
    from libero.libero.envs import SegmentationRenderEnv

    return SegmentationRenderEnv(
        bddl_file_name=str(resolve_bddl(task_name)),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=[True, False],
    )


def resolve_instance(env, arguments: dict, skill: str) -> str | None:
    role = "item" if skill == "Pick" else "target"
    argument = str(arguments.get(role, ""))
    if not argument:
        return None
    instance = match_instance(argument, env.instance_to_id)
    if instance is not None:
        return instance
    return match_region_anchor(argument, env.instance_to_id)


def instance_mask(env, obs, instance_name: str) -> np.ndarray:
    seg = np.asarray(obs["agentview_segmentation_instance"])[..., 0]
    instance_id = env.instance_to_id[instance_name]
    return np.flipud(seg == instance_id)


def target_xyz_from_obs(env, obs, instance_name: str, resolution: int):
    mask = instance_mask(env, obs, instance_name)
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None
    cam = camera_params(env.env.sim, "agentview", resolution, resolution)
    metric = depth_to_metric(obs["agentview_depth"], cam.near, cam.far)
    depth = flip_depth(metric)
    pts = pixel_to_world(np.stack([ys, xs], axis=-1), depth, cam)
    valid = np.isfinite(pts).all(axis=1)
    if int(valid.sum()) < 4:
        return None
    return np.median(pts[valid], axis=0)


def build_ready3d(
    manifest_path: Path,
    output_path: Path,
    demo_dir: Path,
    resolution: int = 256,
    max_demos: int = 0,
    tasks: tuple[str, ...] = (),
) -> dict:
    manifest = load_manifest(manifest_path)
    patch_numpy2_segmentation()
    all_protos = []
    warnings = []

    records = [
        record for record in manifest.get("records", [])
        if record.get("valid") and (not tasks or record["task_name"] in tasks)
    ]
    task_names = sorted({record["task_name"] for record in records})
    for task_name in task_names:
        env = create_env(task_name, resolution)
        env.reset()
        try:
            task_records = [r for r in records if r["task_name"] == task_name]
            if max_demos > 0:
                task_records = task_records[:max_demos]
            h5_path = demo_dir / f"{task_name}_demo.hdf5"
            with h5py.File(h5_path, "r") as handle:
                for record in task_records:
                    group = handle["data"][record["demo_id"]]
                    states = group["states"][:]
                    for segment in record.get("segments", []):
                        if segment.get("status") == "already_satisfied":
                            continue
                        ready_frame = segment.get("ready_frame")
                        if ready_frame is None:
                            continue
                        skill = str(segment["skill"])
                        arguments = dict(segment.get("arguments", {}))
                        instance = resolve_instance(env, arguments, skill)
                        if instance is None:
                            warnings.append({
                                "demo_id": record["demo_id"],
                                "planner_step_id": segment.get("planner_step_id"),
                                "skill": skill,
                                "warning": "instance_not_found",
                            })
                            continue
                        frame = min(max(int(ready_frame), 0), len(states) - 1)
                        obs = env.regenerate_obs_from_state(states[frame])
                        target_xyz = target_xyz_from_obs(
                            env, obs, instance, resolution
                        )
                        if target_xyz is None:
                            warnings.append({
                                "demo_id": record["demo_id"],
                                "planner_step_id": segment.get("planner_step_id"),
                                "skill": skill,
                                "warning": "target_3d_unavailable",
                            })
                            continue
                        eef = np.asarray(
                            obs["robot0_eef_pos"], dtype=np.float64
                        ).reshape(3)
                        all_protos.append({
                            "task_name": task_name,
                            "planner_step_id": int(segment["planner_step_id"]),
                            "skill": skill,
                            "arguments": arguments,
                            "demo_id": record["demo_id"],
                            "ready_frame": frame,
                            "target_xyz_world": np.asarray(
                                target_xyz, dtype=np.float64
                            ),
                            "eef_pos_ready": eef,
                            "distance_m": float(np.linalg.norm(eef - target_xyz)),
                        })
        finally:
            env.close()
        n_task = sum(1 for p in all_protos if p["task_name"] == task_name)
        print(f"{task_name}: {n_task} ready3d prototypes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": READY3D_FORMAT,
            "prototypes": all_protos,
            "warnings": warnings,
        },
        output_path,
    )
    print(f"Wrote {len(all_protos)} prototypes to {output_path}")
    return {"prototypes": len(all_protos), "warnings": len(warnings)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 3D ready-distance memory for Feasible3D."
    )
    parser.add_argument("--segments-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--tasks", nargs="*", default=())
    args = parser.parse_args()
    build_ready3d(
        args.segments_manifest,
        args.output,
        args.demo_dir,
        resolution=args.resolution,
        max_demos=args.max_demos,
        tasks=tuple(args.tasks),
    )


if __name__ == "__main__":
    main()
