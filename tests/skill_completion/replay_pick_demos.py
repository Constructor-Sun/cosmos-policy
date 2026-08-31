"""Batch replay LIBERO demonstrations against the test-time Pick checker."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
for path in (ROOT, LIBERO_PLUS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MUJOCO_GL", "egl")

from memory_system.execute.skill_completion.pick import PickCompletionChecker  # noqa: E402
from memory_system.geometry import (  # noqa: E402
    camera_params,
    depth_to_metric,
    flip_depth,
    pixel_to_world,
)
from memory_system.offline.build_ready3d import (  # noqa: E402
    create_env,
    instance_mask,
    resolve_instance,
)
from memory_system.offline.build_targets import patch_numpy2_segmentation  # noqa: E402
from memory_system.offline.label_segments import (  # noqa: E402
    is_grasping,
    object_position,
)

DEFAULT_MANIFEST = ROOT / "skill_memory_test/libero_10/segments_ready_fixed16.json"
DEFAULT_DEMOS = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"


def target_points(env: Any, obs: dict[str, Any], instance: str, resolution: int):
    mask = instance_mask(env, obs, instance)
    pixels = np.stack(np.nonzero(mask), axis=-1)
    if len(pixels) < 4:
        return None
    cam = camera_params(env.env.sim, "agentview", resolution, resolution)
    metric = depth_to_metric(obs["agentview_depth"], cam.near, cam.far)
    points = pixel_to_world(pixels, flip_depth(metric), cam)
    points = points[np.isfinite(points).all(axis=1)]
    return points if len(points) >= 4 else None


def frame_limit(segments: list[dict[str, Any]], index: int, total: int) -> int:
    if index + 1 >= len(segments):
        return total
    following = segments[index + 1]
    return min(total, int(following.get("success_end", following["success_start"])) + 1)


def replay_pick(
    env: Any,
    states: np.ndarray,
    actions: np.ndarray,
    segment: dict[str, Any],
    stop: int,
    resolution: int,
) -> dict[str, Any]:
    checker = PickCompletionChecker()
    item = str(segment["arguments"]["item"])
    instance = resolve_instance(env, segment["arguments"], "Pick")
    if instance is None:
        return {"status": "instance_missing", "item": item}

    start = max(0, int(segment["start"]))
    predicted = None
    oracle = None
    vertical_oracle = None
    object_origin = None
    grasp_origin_z = None
    oracle_run = 0
    vertical_run = 0
    contact_run = 0
    max_contact_run = 0
    first_object_z = None
    max_object_lift = 0.0
    max_contact_lift = 0.0
    missing = 0
    point_counts: list[int] = []
    prediction_state: dict[str, float | None] = {}

    for frame in range(start, stop):
        obs = env.regenerate_obs_from_state(states[frame])
        points = target_points(env, obs, instance, resolution)
        if points is None:
            missing += 1
        else:
            point_counts.append(len(points))

        closed = frame > 0 and bool(actions[frame - 1, -1] > 0)
        done = checker.update(
            target_points=points,
            eef_pos=obs.get("robot0_eef_pos"),
            eef_quat=obs.get("robot0_eef_quat"),
            gripper_closed=closed,
            gripper_qpos=obs.get("robot0_gripper_qpos"),
        )
        if done and predicted is None:
            predicted = frame

        grasped = is_grasping(env.env, item)
        object_pos = object_position(env.env, item)
        object_z = float(object_pos[2])
        if object_origin is None:
            object_origin = object_pos.copy()
        if first_object_z is None:
            first_object_z = object_z
        max_object_lift = max(max_object_lift, object_z - first_object_z)
        if not grasped:
            grasp_origin_z = None
            contact_run = 0
        else:
            contact_run += 1
            max_contact_run = max(max_contact_run, contact_run)
            if grasp_origin_z is None:
                grasp_origin_z = object_z
            contact_lift = object_z - grasp_origin_z
            max_contact_lift = max(max_contact_lift, contact_lift)

        displacement = object_pos - object_origin
        simulator_pick = bool(
            (grasped and np.linalg.norm(displacement) >= checker.min_lift_distance)
            or displacement[2] >= checker.min_lift_distance
        )
        oracle_run = oracle_run + 1 if simulator_pick else 0
        if oracle_run >= checker.stable_frames and oracle is None:
            oracle = frame
        vertical_pick = bool(displacement[2] >= checker.min_lift_distance)
        vertical_run = vertical_run + 1 if vertical_pick else 0
        if vertical_run >= checker.stable_frames and vertical_oracle is None:
            vertical_oracle = frame

        if done and not prediction_state:
            prediction_state = {
                "prediction_gap": checker.last_gripper_gap,
                "prediction_progress": checker.vertical_progress,
                "prediction_rigid_error": checker.last_rigid_error,
            }

        if predicted is not None and oracle is not None and vertical_oracle is not None:
            break

    if predicted is None and oracle is None:
        status = "both_missing"
    elif predicted is None:
        status = "false_negative"
    elif oracle is None:
        status = "false_positive"
    else:
        status = "matched"
    if predicted is None and vertical_oracle is None:
        vertical_status = "both_missing"
    elif predicted is None:
        vertical_status = "false_negative"
    elif vertical_oracle is None:
        vertical_status = "false_positive"
    else:
        vertical_status = "matched"
    return {
        "status": status,
        "vertical_status": vertical_status,
        "item": item,
        "predicted": predicted,
        "oracle": oracle,
        "vertical_oracle": vertical_oracle,
        "delta": None if predicted is None or oracle is None else predicted - oracle,
        "vertical_delta": (
            None
            if predicted is None or vertical_oracle is None
            else predicted - vertical_oracle
        ),
        "missing_cloud_frames": missing,
        "point_count_min": min(point_counts) if point_counts else 0,
        "point_count_median": int(np.median(point_counts)) if point_counts else 0,
        "max_contact_run": max_contact_run,
        "max_contact_lift": max_contact_lift,
        "max_object_lift": max_object_lift,
        **prediction_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--task")
    parser.add_argument("--demo")
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["records"] if record.get("valid")]
    if args.task:
        records = [record for record in records if record["task_name"] == args.task]
    if args.demo:
        records = [record for record in records if record["demo_id"] == args.demo]
    if args.max_demos:
        records = records[: args.max_demos]

    patch_numpy2_segmentation()
    results: list[dict[str, Any]] = []
    completed_demos = 0
    for task_name in sorted({record["task_name"] for record in records}):
        env = create_env(task_name, args.resolution)
        env.reset()
        h5_path = args.demo_dir / f"{task_name}_demo.hdf5"
        try:
            with h5py.File(h5_path, "r") as handle:
                task_records = [r for r in records if r["task_name"] == task_name]
                for record in task_records:
                    group = handle["data"][record["demo_id"]]
                    states = group["states"][:]
                    actions = group["actions"][:]
                    segments = record["segments"]
                    for index, segment in enumerate(segments):
                        if segment["skill"] != "Pick":
                            continue
                        result = replay_pick(
                            env,
                            states,
                            actions,
                            segment,
                            frame_limit(segments, index, len(states)),
                            args.resolution,
                        )
                        result.update(
                            task=task_name,
                            demo=record["demo_id"],
                            step=segment["planner_step_id"],
                            manifest_success=segment["success_start"],
                        )
                        results.append(result)
                    completed_demos += 1
                    if completed_demos % 10 == 0:
                        print(f"progress demos={completed_demos} picks={len(results)}", flush=True)
        finally:
            env.close()

    counts = Counter(result["status"] for result in results)
    vertical_counts = Counter(result["vertical_status"] for result in results)
    deltas = [result["delta"] for result in results if result["delta"] is not None]
    vertical_deltas = [
        result["vertical_delta"]
        for result in results
        if result["vertical_delta"] is not None
    ]
    missing = sum(result["missing_cloud_frames"] for result in results)
    print("SUMMARY")
    print(f"demos={completed_demos} picks={len(results)} resolution={args.resolution}")
    print("statuses=" + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    print(
        "vertical_statuses="
        + json.dumps(dict(sorted(vertical_counts.items())), sort_keys=True)
    )
    print(f"missing_cloud_frames={missing}")
    if deltas:
        exact = sum(delta == 0 for delta in deltas)
        within_two = sum(abs(delta) <= 2 for delta in deltas)
        print(
            f"delta_frames min={min(deltas)} median={statistics.median(deltas)} "
            f"max={max(deltas)} mean={statistics.mean(deltas):.3f} "
            f"exact={exact}/{len(deltas)} within_2={within_two}/{len(deltas)}"
        )
    if vertical_deltas:
        exact = sum(delta == 0 for delta in vertical_deltas)
        within_two = sum(abs(delta) <= 2 for delta in vertical_deltas)
        print(
            f"vertical_delta_frames min={min(vertical_deltas)} "
            f"median={statistics.median(vertical_deltas)} "
            f"max={max(vertical_deltas)} "
            f"mean={statistics.mean(vertical_deltas):.3f} "
            f"exact={exact}/{len(vertical_deltas)} "
            f"within_2={within_two}/{len(vertical_deltas)}"
        )
    failures = [result for result in results if result["status"] != "matched"]
    print(f"failures={len(failures)}")
    for result in failures:
        print("FAIL " + json.dumps(result, sort_keys=True))
    if args.show_all:
        for result in results:
            print("RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
