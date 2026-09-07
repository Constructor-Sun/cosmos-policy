"""Standalone local Pick evaluation for PointCloud Action Memory."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import h5py
import imageio
import numpy as np
from scipy.spatial.transform import Rotation
from memory_system.geometry import camera_params
from memory_system.offline.build_ready3d import resolve_instance
from memory_system.offline.label_segments import is_grasping, object_position
from memory_system.pointcloud_action.config import (
    ACTION_POS_SCALE,
    ACTION_ROT_SCALE,
    POINT_CLOUD_SOURCE,
    READY_MOTION_MODE,
)
from memory_system.pointcloud_action.execute.mujoco_ik import solve_ee_ik
from memory_system.pointcloud_action.execute.pointcloud_controller import (
    PointCloudPickController,
)
from memory_system.pointcloud_action.execute.pointcloud_selector import PointCloudSelector
from memory_system.pointcloud_action.offline.extraction import (
    complete_point_cloud,
    create_env,
    object_frame,
    visible_point_cloud,
)
from memory_system.pointcloud_action.offline.geometry_utils import (
    rotate_action_object_to_world,
)
LIFT_THRESHOLD = 0.02
DEFAULT_ROLLOUT_DIR = Path(
    "/data1/liu/exp/counterfactual/external/cosmos-policy/rollouts"
)
def _ee_from_obs(obs) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
    rotvec = Rotation.from_quat(quat).as_rotvec()
    return np.concatenate([pos, rotvec])
def _main_image(obs) -> np.ndarray:
    """Return the main agentview image in the same orientation as memory_system."""
    return np.flipud(np.asarray(obs["agentview_image"]))
def _replay_actions(record: dict, R_world_object: np.ndarray) -> np.ndarray:
    """Map object-frame physical actions back to raw LIBERO actions."""
    object_physical = np.asarray(
        record["action_sequence_object_physical"], dtype=np.float64
    )
    world_physical = rotate_action_object_to_world(object_physical, R_world_object)
    raw = world_physical.copy()
    action_scale = record.get("action_scale")
    if action_scale is not None:
        scale = np.asarray(action_scale, dtype=np.float64).reshape(6)
        raw[..., :3] /= scale[:3]
        raw[..., 3:6] /= scale[3:6]
    else:
        raw[..., :3] /= ACTION_POS_SCALE
        raw[..., 3:6] /= ACTION_ROT_SCALE
    return raw.astype(np.float32)
def _set_state(env, state: np.ndarray) -> None:
    env.set_state(state)
    env.sim.forward()

def _sync_controller(env) -> None:
    """Refresh cached robot controller state after restoring a MuJoCo state."""
    env.env.robots[0].controller.update(force=True)

def _execute_controller(
    env, controller, obs, max_steps: int, frames: list | None = None
) -> dict:
    try:
        for _ in range(max_steps):
            if getattr(controller, "finished", False) or getattr(
                controller, "converged", False
            ):
                break
            requires_observation = bool(
                getattr(controller, "requires_observation", False)
            )
            if requires_observation:
                action = controller.step(obs)
            else:
                action = controller.step(_ee_from_obs(obs))
            if np.asarray(action).shape[0] == 6:
                action = np.concatenate([action, [-1.0]])
            step_result = env.step(np.asarray(action).tolist())
            if isinstance(step_result, tuple):
                obs = step_result[0]
            else:
                obs = step_result
            if requires_observation:
                controller.observe(obs)
            if frames is not None:
                frames.append(_main_image(obs))
        return obs
    finally:
        close = getattr(controller, "close", None)
        if close is not None:
            close()
def _pick_succeeded(env, item: str, start_pos: np.ndarray) -> bool:
    pos = object_position(env.env, item)
    return bool(
        (is_grasping(env.env, item) and np.linalg.norm(pos - start_pos) >= LIFT_THRESHOLD)
        or (pos[2] - start_pos[2] >= LIFT_THRESHOLD)
    )
def _write_video(
    frames: list,
    save_video: str | None,
    task: str,
    demo: str,
    frame: int,
    success: bool,
) -> str | None:
    if save_video is None:
        return None
    if save_video == "__default__":
        output = DEFAULT_ROLLOUT_DIR
    else:
        output = Path(save_video)
    if Path(save_video).suffix == ".mp4":
        path = Path(save_video)
    else:
        output.mkdir(parents=True, exist_ok=True)
        name = (
            f"{task}__{demo}__frame{frame}__success{success}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        path = output / name
    writer = imageio.get_writer(path, fps=30)
    for image in frames:
        writer.append_data(image)
    writer.close()
    print(f"Saved rollout video: {path}")
    return str(path)
def _move_to_ready(
    env,
    obs,
    target_ee: np.ndarray,
    resolution: int,
    max_steps: int,
    frames: list | None = None,
    gripper_command: float = -1.0,
) -> dict:
    current_ee = _ee_from_obs(obs)
    if READY_MOTION_MODE == "legacy":
        controller = None
        try:
            from memory_system.execute.curobo_planner import CuroboPlanner
            planner = CuroboPlanner(device="cuda:0")
            cam = camera_params(env.env.sim, "agentview", resolution, resolution)
            plan = planner.plan(
                current_ee_states=current_ee,
                target_ee_states=target_ee,
                depth=obs["agentview_depth"],
                camera_params=cam,
                joint_positions=obs["robot0_joint_pos"],
                gripper_joint_positions=obs["robot0_gripper_qpos"],
            )
            if plan is not None and plan.controller is not None:
                controller = plan.controller
        except Exception:
            controller = None
        if controller is None:
            from memory_system.execute.recovery.controller import PoseController
            controller = PoseController(target_ee_states=target_ee)
        return _execute_controller(env, controller, obs, max_steps, frames=frames)

    # Lightweight waypoint mode.
    from memory_system.pointcloud_action.execute.motion_primitives import (
        MotionPrimitives,
    )
    from memory_system.pointcloud_action.execute.ready_motion_planner import (
        ReadyMotionPlanner,
    )

    planner = ReadyMotionPlanner(current_ee, target_ee)
    plan = planner.plan()
    # gripper_command: -1.0 keeps the gripper open during Pick ready motion;
    # +1.0 keeps it closed so a held object is not dropped (Place).
    motion = MotionPrimitives(env, gripper_command=gripper_command)
    # The joint-space controller can declare a plan infeasible if its internal
    # deadline is too tight.  Give the ready motion a more generous budget than
    # the later open-loop Pick replay, and fail loudly if it cannot even start.
    motion_max_steps = max(int(max_steps), 400)
    controller = motion.build_controller(
        plan, obs, max_steps=motion_max_steps
    )
    if controller is None:
        raise RuntimeError("ReadyMotionPlanner/MotionPrimitives failed to build a valid joint controller")
    if getattr(controller, "finished", False):
        raise RuntimeError(
            "Ready motion controller failed to start: "
            f"{getattr(controller, 'status', 'unknown')}"
        )
    obs = _execute_controller(
        env, controller, obs, motion_max_steps, frames=frames
    )
    # The joint-space controller temporarily switches away from OSC_POSE.
    # Force the restored OSC controller to refresh its internal state before
    # the subsequent Pick action replay uses it.
    _sync_controller(env)
    return obs

def evaluate(
    memory_path: str,
    task: str,
    demo: str,
    frame: int,
    item: str | None,
    resolution: int = 256,
    top_k: int = 1,
    max_steps: int = 200,
    save_video: str | None = None,
    suite: str = "libero_10",
    demo_dir: str | Path = (
        Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
        / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
    ),
    move_to_ready: bool = True,
    init_at_ready: bool = False,
) -> dict:
    cloud_key = (
        "complete_points_object"
        if POINT_CLOUD_SOURCE == "complete"
        else "target_points_object"
    )
    selector = PointCloudSelector(memory_path, cloud_key=cloud_key)
    if item is None and selector.memory.records:
        item = str(selector.memory.records[0]["arguments"]["item"])
    env = create_env(task, resolution, suite=suite)
    try:
        env.reset()
        h5_path = Path(demo_dir) / f"{task}_demo.hdf5"
        with h5py.File(h5_path, "r") as handle:
            states = handle["data"][demo]["states"][:]
        frame = min(max(int(frame), 0), len(states) - 1)
        _set_state(env, states[frame])
        obs = env.regenerate_obs_from_state(states[frame])
        _sync_controller(env)
        frames = [_main_image(obs)]
        instance = resolve_instance(env, {"item": item}, "Pick")
        if instance is None:
            return {"success": False, "error": f"instance not found: {item}"}
        if POINT_CLOUD_SOURCE == "complete":
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
        if init_at_ready:
            ready_ee = np.asarray(best["ready_ee_states"], dtype=np.float64).reshape(6)
            target_pos = ready_ee[:3]
            target_rot = Rotation.from_rotvec(ready_ee[3:]).as_matrix()
            init_q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(7)
            q_ready = solve_ee_ik(
                env.env.sim, target_pos, target_rot, init_q=init_q
            )
            if q_ready is None:
                return {"success": False, "error": "IK failed to reach ready pose"}
            obs = env.regenerate_obs_from_state(env.env.sim.get_state().flatten())
            _sync_controller(env)
        elif move_to_ready:
            obs = _move_to_ready(
                env,
                obs,
                best["ready_ee_states"],
                resolution,
                max_steps,
                frames=frames,
            )
        start_pos = object_position(env.env, item)
        _, _, R_cur = object_frame(env, instance)
        replay_actions = _replay_actions(best["record"], R_cur)
        pick_controller = PointCloudPickController(actions=replay_actions)
        obs = _execute_controller(env, pick_controller, obs, max_steps, frames=frames)
        success = _pick_succeeded(env, item, start_pos)
        video_path = _write_video(
            frames, save_video, task, demo, frame, success
        )
        return {
            "success": success,
            "memory_id": best["record"]["memory_id"],
            "distance": best["distance"],
            "frame": frame,
            "controller_status": pick_controller.status,
            "video_path": video_path,
        }
    finally:
        env.close()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PointCloud Action Memory on a local Pick."
    )
    parser.add_argument("--memory", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--frame", type=int, default=70)
    parser.add_argument("--item", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--suite", type=str, default="libero_10")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path(
            "/data1/liu/exp/counterfactual/external/cosmos-policy"
            "/LIBERO-Cosmos-Policy/success_only/libero_10_regen"
        ),
    )
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH_OR_DIR",
        help="Save main-view MP4. Use --save-video without a value to save to the default rollouts dir.",
    )
    parser.add_argument(
        "--no-move-to-ready",
        action="store_false",
        dest="move_to_ready",
        help="Skip moving to ready pose and directly replay actions.",
    )
    parser.add_argument(
        "--init-at-ready",
        action="store_true",
        help="Use MuJoCo IK to reset the arm to the mapped ready pose before replay.",
    )
    args = parser.parse_args()
    result = evaluate(
        args.memory,
        args.task,
        args.demo,
        args.frame,
        args.item,
        resolution=args.resolution,
        top_k=args.top_k,
        max_steps=args.max_steps,
        save_video=args.save_video,
        suite=args.suite,
        demo_dir=args.demo_dir,
        move_to_ready=args.move_to_ready,
        init_at_ready=args.init_at_ready,
    )
    print(result)
if __name__ == "__main__":
    main()
