"""Minimal Place executor built from the existing Pick primitives."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.offline.build_ready3d import resolve_instance
from memory_system.offline.label_segments import PREDICATE_SKILLS, is_grasping
from memory_system.pointcloud_action.eval.eval_pointcloud_pick import _move_to_ready
from memory_system.pointcloud_action.eval.local_pick_core import (
    _current_cloud_key,
    _ee_from_obs,
    _replay_waypoint_timegrip,
)
from memory_system.pointcloud_action.offline.extraction import (
    complete_point_cloud,
    object_frame,
    visible_point_cloud,
)
from memory_system.pointcloud_action.offline.geometry_utils import (
    ee_states_to_matrix,
    matrix_to_ee_states,
)


def _fit_grasp(env, obs: dict, item_instance: str, target_instance: str, hit: dict):
    """Map a demo path so the currently held item follows the demo item path."""
    record = hit["record"]
    item_ready = record.get("item_pose_anchor_ready")
    if item_ready is None:
        return float("inf"), hit
    ee_target = np.asarray(record["ee_pose_object_sequence"], dtype=np.float64)
    T_target, _, _ = object_frame(env, target_instance)
    T_item, _, _ = object_frame(env, item_instance)
    T_ee = ee_states_to_matrix(_ee_from_obs(obs))
    current_grasp = np.linalg.inv(T_ee) @ T_item
    demo_grasp = np.linalg.inv(ee_target[0]) @ np.asarray(item_ready)
    correction = demo_grasp @ np.linalg.inv(current_grasp)
    fitted = dict(hit)
    fitted["ee_pose_world_sequence"] = np.stack(
        [T_target @ pose @ correction for pose in ee_target]
    )
    cost = Rotation.from_matrix(correction[:3, :3]).magnitude()
    cost += np.linalg.norm(correction[:3, 3])
    return float(cost), fitted


def run_local_place(
    env,
    obs: dict,
    item: str,
    target: str,
    skill: str,
    selector,
    resolution: int = 256,
    top_k: int = 1,
    max_steps: int = 200,
    post_release_hold_steps: int = 40,
    stable_frames: int = 5,
) -> dict[str, Any]:
    """Retrieve an exact-pair Place memory, move closed, replay, and verify."""
    target_instance = resolve_instance(env, {"target": target}, skill)
    item_instance = resolve_instance(env, {"item": item}, "Pick")
    if target_instance is None or item_instance is None:
        return {"success": False, "error": "item or target instance not found"}

    if _current_cloud_key() == "complete_points_object":
        points = complete_point_cloud(env, target_instance)
    else:
        points = visible_point_cloud(env, obs, target_instance, resolution)
    if len(points) < 4:
        return {"success": False, "error": "empty destination point cloud"}

    T_target, _, _ = object_frame(env, target_instance)
    hits = selector.select(
        points,
        T_target,
        skill=skill,
        top_k=max(int(top_k), 16),
        item=item,
        target=target,
    )
    if not hits:
        return {"success": False, "error": "no exact-pair memory"}
    _, best = min(
        (_fit_grasp(env, obs, item_instance, target_instance, hit) for hit in hits),
        key=lambda value: value[0],
    )

    try:
        ready = matrix_to_ee_states(best["ee_pose_world_sequence"][0])
        obs = _move_to_ready(
            env, obs, ready, resolution, max_steps, gripper_command=1.0
        )
    except Exception as exc:
        return {"success": False, "error": f"ready_motion: {exc}"}
    obs, replay = _replay_waypoint_timegrip(env, obs, item, best, max_steps)

    predicate = PREDICATE_SKILLS[skill]
    hold = np.zeros(7, dtype=np.float32)
    hold[-1] = -1.0
    stable = 0
    for _ in range(int(post_release_hold_steps)):
        step = env.step(hold.tolist())
        obs = step[0] if isinstance(step, tuple) else step
        relation = bool(env.env._eval_predicate([predicate, item, target]))
        stable = stable + 1 if relation and not is_grasping(env.env, item) else 0
        if stable >= int(stable_frames):
            break
    return {
        "success": stable >= int(stable_frames),
        "memory_id": best["record"]["memory_id"],
        "key_stage": best.get("key_stage"),
        **replay,
    }
