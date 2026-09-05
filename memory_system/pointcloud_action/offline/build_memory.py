"""Build the PointCloud Action Memory artifact from LIBERO-90 HDF5 + BDDL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from memory_system.pointcloud_action.config import (
    ACTION_SCALE,
    CONTROLLER_CONFIG,
    DEFAULT_DEMO_DIR,
    DEFAULT_MAX_DEMOS,
    DEFAULT_OUTPUT,
    DEFAULT_RESOLUTION,
    READY_DISTANCE_M,
)
from memory_system.pointcloud_action.offline.action_utils import (
    build_action_sequences,
    build_ee_sequences,
)
from memory_system.pointcloud_action.offline.extraction import (
    complete_point_cloud,
    create_env,
    object_frame,
    visible_point_cloud,
)
from memory_system.pointcloud_action.offline.ready_frame import select_ready_frame
from memory_system.pointcloud_action.schema import MEMORY_FORMAT, SUITE
from memory_system.offline.build_ready3d import resolve_instance
from memory_system.offline.build_targets import patch_numpy2_segmentation


def _make_record(
    task_name: str,
    demo_id: str,
    segment: dict,
    ready_frame: int,
    segment_end: int,
    instance: str,
    visible: np.ndarray,
    complete: np.ndarray,
    T_world_object: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    actions: np.ndarray,
    ee_states: np.ndarray,
) -> dict | None:
    ready_frame = int(ready_frame)
    segment_end = int(segment_end)
    if (
        segment_end <= ready_frame
        or segment_end >= len(ee_states)
        or ready_frame >= len(actions)
        or len(visible) < 4
    ):
        return None

    raw, world_physical, object_physical = build_action_sequences(
        actions, ready_frame, segment_end, rotation
    )
    ready_ee_states, T_object_ee_ready, ee_world, ee_object = build_ee_sequences(
        ee_states, ready_frame, segment_end, T_world_object
    )

    return {
        "memory_id": f"{task_name}::{demo_id}::step{int(segment['planner_step_id'])}",
        "source_task": task_name,
        "source_demo": demo_id,
        "planner_step_id": int(segment["planner_step_id"]),
        "skill": "Pick",
        "arguments": dict(segment.get("arguments", {})),
        "target_points_world": visible.astype(np.float32),
        "target_points_object": (
            (visible - translation) @ rotation
        ).astype(np.float32),
        "target_xyz_world": np.median(visible, axis=0).astype(np.float32),
        "complete_points_world": complete.astype(np.float32),
        "complete_points_object": (
            (complete - translation) @ rotation
        ).astype(np.float32),
        "T_world_object_anchor": T_world_object.astype(np.float32),
        "object_frame_translation": translation.astype(np.float32),
        "object_frame_rotation": rotation.astype(np.float32),
        "frame_source": "simulator_body_xmat",
        "frame_convention": "T_AB_maps_B_to_A",
        "ready_frame": ready_frame,
        "segment_end": segment_end,
        "sequence_length": segment_end - ready_frame,
        "ready_ee_states": ready_ee_states.astype(np.float32),
        "T_object_ee_ready": T_object_ee_ready.astype(np.float32),
        "action_sequence_raw": raw.astype(np.float32),
        "action_sequence_world_physical": world_physical.astype(np.float32),
        "action_sequence_object_physical": object_physical.astype(np.float32),
        "action_scale": ACTION_SCALE.copy(),
        "gripper_sequence": raw[:, -1].astype(np.float32).copy(),
        "ee_pose_world_sequence": ee_world.astype(np.float32),
        "ee_pose_object_sequence": ee_object.astype(np.float32),
    }


def build_memory(
    manifest_path: str | Path,
    demo_dir: str | Path,
    output_path: str | Path,
    resolution: int = DEFAULT_RESOLUTION,
    max_demos: int = DEFAULT_MAX_DEMOS,
    tasks: tuple[str, ...] = (),
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    demo_dir = Path(demo_dir)
    patch_numpy2_segmentation()

    records = [
        record
        for record in manifest.get("records", [])
        if record.get("valid") and (not tasks or record["task_name"] in tasks)
    ]
    task_names = sorted({record["task_name"] for record in records})
    all_records = []
    warnings = []

    for task_name in task_names:
        env = create_env(task_name, resolution)
        env.reset()
        h5_path = demo_dir / f"{task_name}_demo.hdf5"
        try:
            with h5py.File(h5_path, "r") as handle:
                task_records = [
                    record for record in records if record["task_name"] == task_name
                ]
                if max_demos > 0:
                    task_records = task_records[:max_demos]
                for record in task_records:
                    group = handle["data"][record["demo_id"]]
                    states = group["states"][:]
                    actions = group["actions"][:]
                    ee_states = group["obs"]["ee_states"][:]
                    for segment in record.get("segments", []):
                        if segment.get("status") == "already_satisfied":
                            continue
                        if segment.get("skill") != "Pick":
                            continue
                        ref_ready = segment.get("ready_frame")
                        segment_end = segment.get("success_end") or segment.get("end")
                        if ref_ready is None or segment_end is None:
                            continue
                        instance = resolve_instance(
                            env, segment.get("arguments", {}), "Pick"
                        )
                        if instance is None:
                            warnings.append(
                                f"{record['demo_id']} step {segment.get('planner_step_id')}: no instance"
                            )
                            continue
                        ref_frame = min(max(int(ref_ready), 0), len(states) - 1)
                        obs_ref = env.regenerate_obs_from_state(states[ref_frame])
                        ref_visible = visible_point_cloud(
                            env, obs_ref, instance, resolution
                        )
                        if len(ref_visible) < 4:
                            warnings.append(
                                f"{record['demo_id']} step {segment.get('planner_step_id')}: no reference cloud"
                            )
                            continue
                        target_xyz = np.median(ref_visible, axis=0)
                        ready_frame = select_ready_frame(
                            segment, ee_states, target_xyz, READY_DISTANCE_M
                        )
                        if ready_frame is None:
                            warnings.append(
                                f"{record['demo_id']} step {segment.get('planner_step_id')}: no ready frame at {READY_DISTANCE_M}m"
                            )
                            continue
                        ready_frame = int(ready_frame)
                        frame = min(max(ready_frame, 0), len(states) - 1)
                        obs = env.regenerate_obs_from_state(states[frame])
                        visible = visible_point_cloud(env, obs, instance, resolution)
                        complete = complete_point_cloud(env, instance)
                        T_world_object, translation, rotation = object_frame(
                            env, instance
                        )
                        memory = _make_record(
                            task_name,
                            record["demo_id"],
                            segment,
                            ready_frame,
                            segment_end,
                            instance,
                            visible,
                            complete,
                            T_world_object,
                            translation,
                            rotation,
                            actions,
                            ee_states,
                        )
                        if memory is None:
                            warnings.append(
                                f"{record['demo_id']} step {segment.get('planner_step_id')}: empty visible cloud"
                            )
                            continue
                        all_records.append(memory)
        finally:
            env.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": MEMORY_FORMAT,
            "suite": SUITE,
            "controller_config": CONTROLLER_CONFIG,
            "records": all_records,
        },
        output_path,
    )
    print(f"Wrote {len(all_records)} records to {output_path}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  {warning}")
    return {"records": len(all_records), "warnings": len(warnings)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PointCloud Action Memory from LIBERO-90 demos."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--max-demos", type=int, default=DEFAULT_MAX_DEMOS)
    parser.add_argument("--tasks", nargs="*", default=())
    args = parser.parse_args()
    build_memory(
        args.manifest,
        args.demo_dir,
        args.output,
        resolution=args.resolution,
        max_demos=args.max_demos,
        tasks=tuple(args.tasks),
    )


if __name__ == "__main__":
    main()
