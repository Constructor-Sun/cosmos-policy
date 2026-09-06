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
from memory_system.pointcloud_action.offline.geometry_utils import matrix_to_ee_states
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


def _replay_closed_loop(
    env,
    obs: dict,
    item: str,
    candidate: dict,
    max_steps: int,
    frames: list | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Closed-loop replay: track the recorded realized EE path with a
    convergence-guaranteed waypoint controller; gripper closes by path
    progress and the grasp is confirmed by finger contact before lifting.

    Returns (obs, info) with instrumentation fields.
    """
    from memory_system.execute.curobo_trajectory import WaypointPoseController

    ref = np.asarray(candidate["ee_pose_world_sequence"], dtype=np.float64)
    ref6 = np.stack([matrix_to_ee_states(m) for m in ref], axis=0)
    grip_seq = np.asarray(
        candidate["record"]["gripper_sequence"], dtype=np.float64
    ).reshape(-1)
    close_idx = (
        int(np.argmax(grip_seq > 0)) if (grip_seq > 0).any() else len(grip_seq)
    )

    info: dict[str, Any] = {
        "replay_mode": "closed_loop",
        "close_time_ee_error_mm": None,
        "grasp_confirmed": False,
        "grasp_confirmed_step": None,
        "waypoint_max_final_error_mm": None,
        "forced_advances": 0,
    }
    # Close the gripper when the hand comes within close_radius of the recorded
    # grasp pose (metric-space trigger, latched) and let the finger closure
    # complete by contact — reproducing the demo's close-while-moving semantics.
    # A waypoint-count lead is wrong here: waypoint spacing varies per chunk
    # (3 waypoints can be 8mm on one chunk and 27mm on another).
    close_radius = 0.012
    closed_fired = False
    ref_grasp = ref6[min(close_idx + 1, len(ref6) - 1)]
    wp = WaypointPoseController(ref6[1:], dwell_timeout=40)
    confirm_run = 0
    steps_since_close = 0
    for step in range(max_steps):
        if wp.finished:
            break
        cur = _ee_from_obs(obs)
        dist_grasp = float(np.linalg.norm(cur[:3] - ref_grasp[:3]))
        if not closed_fired and dist_grasp <= close_radius:
            closed_fired = True
            info["close_time_ee_error_mm"] = round(dist_grasp * 1000, 2)
        g = 1.0 if closed_fired else -1.0
        a6 = wp.step(cur)
        step_result = env.step(
            np.concatenate([a6, [g]]).astype(np.float32).tolist()
        )
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        if frames is not None:
            frames.append(_main_image(obs))
        if not closed_fired:
            continue
        steps_since_close += 1
        if is_grasping(env.env, item):
            confirm_run += 1
        else:
            confirm_run = 0
        if confirm_run >= 5:
            info["grasp_confirmed"] = True
            info["grasp_confirmed_step"] = step
            info["close_time_ee_error_mm"] = round(dist_grasp * 1000, 2)
            break
    perr = np.asarray(wp.waypoint_pos_errors, float)
    if len(perr):
        info["waypoint_max_final_error_mm"] = round(float(perr.max()) * 1000, 2)
    info["forced_advances"] = int(wp.forced_advances)
    info["wp_status"] = wp.status
    return obs, info


def _replay_waypoint_timegrip(
    env,
    obs: dict,
    item: str,
    candidate: dict,
    max_steps: int,
    frames: list | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Waypoint EE tracking with convergence waiting; gripper follows the
    source time index (waypoint progress) instead of the position/contact
    trigger of closed_loop.  P0-1 2x2 ablation (2026-09-06): the waypoint
    tracking alone fixes the open-loop realization error (moka 3/3), while
    the position-triggered gripper alone does not (0/3).
    """
    from memory_system.execute.curobo_trajectory import WaypointPoseController

    ref = np.asarray(candidate["ee_pose_world_sequence"], dtype=np.float64)
    ref6 = np.stack([matrix_to_ee_states(m) for m in ref], axis=0)
    grip_seq = np.asarray(
        candidate["record"]["gripper_sequence"], dtype=np.float64
    ).reshape(-1)

    info: dict[str, Any] = {
        "replay_mode": "wp_time",
        "close_time_ee_error_mm": None,
        "grasp_confirmed": None,
        "grasp_confirmed_step": None,
        "waypoint_max_final_error_mm": None,
        "forced_advances": 0,
    }
    wp = WaypointPoseController(ref6[1:], dwell_timeout=40)
    # NOTE: no `done` check here -- in the single-object coverage envs the
    # task-success predicate can fire mid-replay (e.g. the open gripper
    # nudges the object into the success region), and aborting on it cuts
    # off a replay that the open-loop path would have completed.  Episode
    # termination errors are caught per-case by the evaluation harness.
    for step in range(max_steps):
        if wp.finished:
            break
        cur = _ee_from_obs(obs)
        a6 = wp.step(cur)
        index = min(wp.index, len(grip_seq) - 1)
        g = 1.0 if grip_seq[index] > 0 else -1.0
        step_result = env.step(
            np.concatenate([a6, [g]]).astype(np.float32).tolist()
        )
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        if frames is not None:
            frames.append(_main_image(obs))
    info["wp_finished"] = bool(wp.finished)
    if not wp.finished:
        info["wp_status"] = f"TIMEOUT({wp.status})"
    else:
        info["wp_status"] = wp.status
    perr = np.asarray(wp.waypoint_pos_errors, float)
    if len(perr):
        info["waypoint_max_final_error_mm"] = round(float(perr.max()) * 1000, 2)
    info["forced_advances"] = int(wp.forced_advances)
    return obs, info


def _remap_object_sequence(
    T_world_object: np.ndarray, ee_pose_object_sequence: np.ndarray
) -> np.ndarray:
    """Re-map an object-frame EE pose sequence to a new object pose:
    world[t] = T_world_object @ object[t] (rigid -- orientation and offset
    travel with the object frame)."""
    seq = np.asarray(ee_pose_object_sequence, dtype=np.float64)
    return np.stack([T_world_object @ m for m in seq], axis=0)


def _reanchor_after_ready(
    env,
    obs: dict,
    instance: str,
    best: dict,
    T_world_object: np.ndarray,
    resolution: int,
    max_steps: int,
    frames: list | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, float]:
    """Re-anchor the pick trajectory to the object pose measured after the
    ready motion, then realign the gripper before replay.

    The first anchor is taken before the object has settled, so the mapped
    ready pose can be tens of mm off by replay time (P0-2: waiting without
    re-anchoring never fixes this).  Waypoint-based replay modes also need the
    tracked reference path re-mapped, or the controller servo-es onto the
    stale pre-settle trajectory.

    Returns (obs, ready_target, reanchored_world_seq, anchor_to_ready_disp_mm).
    """
    T_ready, _, _ = object_frame(env, instance)
    ref_obj = np.asarray(
        best["record"]["ee_pose_object_sequence"], dtype=np.float64
    )
    reanchored_world_seq = _remap_object_sequence(T_ready, ref_obj)
    ready_target = matrix_to_ee_states(reanchored_world_seq[0])
    anchor_to_ready_disp_mm = round(
        float(np.linalg.norm(T_ready[:3, 3] - T_world_object[:3, 3]) * 1000), 2
    )
    obs = _move_to_ready(
        env, obs, ready_target, resolution, max_steps, frames=frames
    )
    return obs, ready_target, reanchored_world_seq, anchor_to_ready_disp_mm


def run_local_pick(
    env,
    obs: dict,
    item: str,
    selector,
    resolution: int = 256,
    top_k: int = 1,
    max_steps: int = 200,
    stable_hold_steps: int = 20,
    post_replay_lift_steps: int = 12,
    replay_mode: str | None = None,
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
        post_replay_lift_steps: After Pick replay, execute this many steps of
            vertical +5 cm/step (commanded) lift with the gripper closed.
            Compensates memory chunks that end at the grasp without a lift.
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
    reanchored = False
    reanchored_world_seq: np.ndarray | None = None
    anchor_to_ready_disp_mm: float | None = None
    ready_target = np.asarray(best["ready_ee_states"], dtype=np.float64).reshape(6)

    if init_at_ready:
        ready_ee = np.asarray(best["ready_ee_states"], dtype=np.float64).reshape(6)
        ready_target = ready_ee
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
            if pc_config.REANCHOR_AFTER_READY:
                (
                    obs,
                    ready_target,
                    reanchored_world_seq,
                    anchor_to_ready_disp_mm,
                ) = _reanchor_after_ready(
                    env,
                    obs,
                    instance,
                    best,
                    T_world_object,
                    resolution,
                    max_steps,
                    frames=frames,
                )
                reanchored = True
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

    # Measure whether the arm actually reached the ready EE pose that the
    # replay starts from (re-anchored target when REANCHOR_AFTER_READY is on).
    ready_ee = ready_target
    current_ee = _ee_from_obs(obs)
    ready_pos_error = float(np.linalg.norm(current_ee[:3] - ready_ee[:3]))
    # Use 3 cm as a practical threshold for "entered ready pose".
    ready_reached = bool(ready_pos_error <= 0.03)

    start_pos = object_position(env.env, item)
    replay_mode_eff = replay_mode or pc_config.REPLAY_MODE
    cl_info: dict[str, Any] = {
        "replay_mode": replay_mode_eff,
        "close_time_ee_error_mm": None,
        "grasp_confirmed": None,
        "grasp_confirmed_step": None,
        "waypoint_max_final_error_mm": None,
        "forced_advances": None,
    }
    pick_controller = None
    replay_candidate = best
    if reanchored_world_seq is not None and replay_mode_eff in (
        "closed_loop",
        "wp_time",
    ):
        replay_candidate = dict(best)
        replay_candidate["ee_pose_world_sequence"] = reanchored_world_seq
    if replay_mode_eff == "closed_loop":
        obs, cl_info = _replay_closed_loop(
            env, obs, item, replay_candidate, max_steps, frames=frames
        )
    elif replay_mode_eff == "wp_time":
        obs, cl_info = _replay_waypoint_timegrip(
            env, obs, item, replay_candidate, max_steps, frames=frames
        )
    else:
        _, _, R_cur = object_frame(env, instance)
        replay_actions = _replay_actions(best["record"], R_cur)
        pick_controller = PointCloudPickController(actions=replay_actions)
        obs = _execute_controller(
            env, pick_controller, obs, max_steps, frames=frames
        )

    pos_after_replay = object_position(env.env, item)
    lift_after_replay = float(pos_after_replay[2] - start_pos[2])
    grasping_after_replay = bool(is_grasping(env.env, item))

    # Scripted post-replay lift: some memory chunks end at the grasp without a
    # lift segment, so raise the gripper vertically with the gripper closed
    # before the stability hold.  The +1.0 z command maps to the OSC +5 cm
    # per-step delta; realization is partial, hence the step count.
    if post_replay_lift_steps > 0:
        lift_action = np.zeros(7, dtype=np.float32)
        lift_action[2] = 1.0
        lift_action[-1] = 1.0
        for _ in range(int(post_replay_lift_steps)):
            step_result = env.step(lift_action.tolist())
            if isinstance(step_result, tuple):
                obs = step_result[0]
            else:
                obs = step_result
            if frames is not None:
                frames.append(_main_image(obs))

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

    replay_status = "CLOSED_LOOP"
    if pick_controller is not None:
        replay_status = pick_controller.status
    elif replay_mode_eff == "wp_time":
        replay_status = str(cl_info.get("wp_status", "WP_TIME"))

    return {
        "success": success,
        "memory_id": best["record"]["memory_id"],
        "distance": best["distance"],
        "controller_status": replay_status,
        **cl_info,
        "ready_motion_error": ready_motion_error,
        "ready_reached": ready_reached,
        "ready_pos_error": ready_pos_error,
        "reanchored": reanchored,
        "anchor_to_ready_disp_mm": anchor_to_ready_disp_mm,
        "grasping_after_replay": grasping_after_replay,
        "lift_after_replay": lift_after_replay,
        "stable_hold_steps": int(stable_hold_steps),
        "lift": lift,
        "video_path": video_path,
        "frame": frame,
    }
