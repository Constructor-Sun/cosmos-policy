"""Shared local-Pick execution core for pointcloud_action evaluations.

This module expects an environment that is already placed in the desired initial
state (HDF5 frame, benchmark init state, or random reset).  It runs:

    object point cloud -> memory retrieval -> ready motion -> Pick replay -> success

and returns a JSON-friendly result dictionary.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.pointcloud_action import config as pc_config
from memory_system.pointcloud_action.eval.eval_pointcloud_pick import (
    _execute_controller,
    _main_image,
    _move_to_ready,
    _replay_actions,
    _sync_controller,
    _write_video,
)
from memory_system.pointcloud_action.execute.mujoco_ik import solve_ee_ik
from memory_system.pointcloud_action.execute.pointcloud_controller import (
    PointCloudPickController,
)
from memory_system.pointcloud_action.offline.extraction import (
    complete_point_cloud,
    object_frame,
    visible_point_cloud,
)
from memory_system.offline.build_ready3d import resolve_instance
from memory_system.offline.label_segments import is_grasping, object_position


def _current_cloud_key() -> str:
    if pc_config.POINT_CLOUD_SOURCE == "complete":
        return "complete_points_object"
    return "target_points_object"


def _ee_from_obs(obs: dict) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
    rotvec = Rotation.from_quat(quat).as_rotvec()
    return np.concatenate([pos, rotvec])


def run_local_pick(
    env,
    obs: dict,
    item: str,
    selector,
    resolution: int = 256,
    top_k: int = 1,
    max_steps: int = 200,
    stable_hold_steps: int = 20,
    move_to_ready: bool = True,
    init_at_ready: bool = False,
    save_video: str | None = None,
    task: str = "local_pick",
    demo: str = "random",
    frame: int = 0,
) -> dict[str, Any]:
    """Run one local Pick evaluation from an already-initialized env.

    Args:
        env: LIBERO SegmentationRenderEnv (or compatible wrapper).
        obs: Observation after reset / state restore.
        item: Target object instance name, e.g. ``moka_pot_1``.
        selector: A ``PointCloudSelector`` instance.
        resolution: Rendering resolution.
        top_k: Number of memory candidates.
        max_steps: Maximum execution steps.
        stable_hold_steps: After Pick replay, keep the gripper closed for this
            many steps before checking stable Pick success.
        move_to_ready: If True, run the ready-motion planner before replay.
        init_at_ready: If True, directly IK to the mapped ready pose instead of
            running the ready-motion planner.  Useful only as a diagnostic.
        save_video: Optional video path/directory.
    """
    frames: list | None = [_main_image(obs)] if save_video is not None else None

    instance = resolve_instance(env, {"item": item}, "Pick")
    if instance is None:
        return {"success": False, "error": f"instance not found: {item}"}

    cloud_key = _current_cloud_key()
    if cloud_key == "complete_points_object":
        points = complete_point_cloud(env, instance)
    else:
        points = visible_point_cloud(env, obs, instance, resolution)
    if len(points) < 4:
        return {"success": False, "error": "empty point cloud"}

    T_world_object, _, _ = object_frame(env, instance)
    candidates = selector.select(
        points, T_world_object, skill="Pick", top_k=top_k
    )
    if not candidates:
        return {"success": False, "error": "no retrieved memory"}
    best = candidates[0]

    ready_motion_error: str | None = None
    ready_pos_error: float | None = None
    ready_reached: bool | None = None

    if init_at_ready:
        ready_ee = np.asarray(best["ready_ee_states"], dtype=np.float64).reshape(6)
        target_pos = ready_ee[:3]
        target_rot = Rotation.from_rotvec(ready_ee[3:]).as_matrix()
        init_q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(7)
        q_ready = solve_ee_ik(env.env.sim, target_pos, target_rot, init_q=init_q)
        if q_ready is None:
            return {
                "success": False,
                "error": "IK failed to reach ready pose",
                "memory_id": best["record"]["memory_id"],
                "distance": best["distance"],
            }
        obs = env.regenerate_obs_from_state(env.env.sim.get_state().flatten())
        _sync_controller(env)
    elif move_to_ready:
        try:
            obs = _move_to_ready(
                env,
                obs,
                best["ready_ee_states"],
                resolution,
                max_steps,
                frames=frames,
            )
        except Exception as exc:  # noqa: BLE001 - report any planner/controller failure
            ready_motion_error = f"{type(exc).__name__}: {exc}"
            result = {
                "success": False,
                "error": f"ready_motion: {ready_motion_error}",
                "memory_id": best["record"]["memory_id"],
                "distance": best["distance"],
                "ready_motion_error": ready_motion_error,
                "ready_pos_error": None,
                "ready_reached": False,
                "controller_status": None,
            }
            if save_video is not None and frames is not None:
                _write_video(frames, save_video, task, demo, frame, False)
            return result

    # Measure whether the arm actually reached the mapped ready EE pose.
    ready_ee = np.asarray(best["ready_ee_states"], dtype=np.float64).reshape(6)
    current_ee = _ee_from_obs(obs)
    ready_pos_error = float(np.linalg.norm(current_ee[:3] - ready_ee[:3]))
    # Use 3 cm as a practical threshold for "entered ready pose".
    ready_reached = bool(ready_pos_error <= 0.03)

    start_pos = object_position(env.env, item)
    _, _, R_cur = object_frame(env, instance)
    replay_actions = _replay_actions(best["record"], R_cur)
    pick_controller = PointCloudPickController(actions=replay_actions)
    obs = _execute_controller(
        env, pick_controller, obs, max_steps, frames=frames
    )

    pos_after_replay = object_position(env.env, item)
    lift_after_replay = float(pos_after_replay[2] - start_pos[2])
    grasping_after_replay = bool(is_grasping(env.env, item))

    # Stable Pick criterion: after the Pick replay, keep the gripper closed for
    # a fixed number of steps. Success requires that the object is still grasped
    # and lifted at least 2 cm above the pre-replay position.
    # In LIBERO raw action space, gripper > 0 means close, < 0 means open.
    hold_action = np.zeros(7, dtype=np.float32)
    hold_action[-1] = 1.0
    for _ in range(max(0, int(stable_hold_steps))):
        step_result = env.step(hold_action.tolist())
        if isinstance(step_result, tuple):
            obs = step_result[0]
        else:
            obs = step_result
        if frames is not None:
            frames.append(_main_image(obs))

    final_pos = object_position(env.env, item)
    lift = float(final_pos[2] - start_pos[2])
    success = bool(is_grasping(env.env, item) and lift >= 0.02)

    video_path = None
    if save_video is not None and frames is not None:
        video_path = _write_video(
            frames, save_video, task, demo, frame, success
        )

    return {
        "success": success,
        "memory_id": best["record"]["memory_id"],
        "distance": best["distance"],
        "controller_status": pick_controller.status,
        "ready_motion_error": ready_motion_error,
        "ready_reached": ready_reached,
        "ready_pos_error": ready_pos_error,
        "grasping_after_replay": grasping_after_replay,
        "lift_after_replay": lift_after_replay,
        "stable_hold_steps": int(stable_hold_steps),
        "lift": lift,
        "video_path": video_path,
        "frame": frame,
    }
