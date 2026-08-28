#!/usr/bin/env python
"""Test whether cuRobo can route episode 2 through episode 1's over-bottle pose.

This is an offline diagnostic.  It creates an empty external collision world,
while retaining cuRobo's joint limits and self-collision model, and plans two
joint-space segments:

    episode-2 start -> episode-1 over-bottle pose -> memory target

If both segments succeed, episode 2's initial joint configuration is not, by
itself, a kinematic reason that the short over-bottle route cannot be used.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logging.disable(logging.CRITICAL)

from libero.libero import benchmark

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from memory_system.execute.curobo_planner import CuroboPlanner
from memory_system.execute.curobo_trajectory import (
    JointTrajectoryPlan,
    retarget_tool_pose,
)


DEFAULT_TASK = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "_view_0_0_100_0_0_initstate_274"
)
DEFAULT_TARGET = np.array(
    [
        -0.05882002040743828,
        -0.08573316782712936,
        0.9660174250602722,
        2.6380581855773926,
        -1.9302700757980347,
        0.28322678804397583,
    ],
    dtype=np.float64,
)
DEFAULT_LOG = (
    "scripts/experiments/libero10_robotinit_single/robotinit/logs/"
    "ENV_EVAL-libero_10-cosmos-2026_08_26-22_58_52--paired-"
    "robot_initial_states-20pair.txt"
)


def parse_logged_waypoints(log_path: Path, episode_index: int) -> np.ndarray:
    """Return the unique visited waypoints for one zero-based episode."""
    waypoint_by_index: dict[int, np.ndarray] = {}
    marker = "[INIT_ALIGN_DEBUG] "
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if marker not in line:
            continue
        try:
            record = json.loads(line.split(marker, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if int(record.get("episode", -1)) != episode_index:
            continue
        index = record.get("waypoint_index")
        waypoint = record.get("waypoint")
        if index is not None and waypoint is not None:
            waypoint_by_index[int(index)] = np.asarray(waypoint, dtype=np.float64)
    if not waypoint_by_index:
        raise RuntimeError(
            f"no logged waypoints for episode index {episode_index} in {log_path}"
        )
    return np.stack([waypoint_by_index[i] for i in sorted(waypoint_by_index)])


def body_position(env, body_name: str) -> np.ndarray:
    body_names = list(env.sim.model.body_names)
    if body_name not in body_names:
        raise RuntimeError(f"body not found: {body_name}")
    body_id = env.sim.model.body_name2id(body_name)
    return env.sim.data.body_xpos[body_id].copy().astype(np.float64)


def set_state_and_settle(env, state) -> tuple[dict, np.ndarray]:
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
    bottle = body_position(env, "wine_bottle_1_main")
    return obs, bottle


def make_goal_tool_pose(
    goal_ee: np.ndarray,
    *,
    calibration_ee: np.ndarray,
    calibration_tool_position: np.ndarray,
    calibration_tool_rotation: np.ndarray,
    robot_base_pose: np.ndarray,
    tool_frames,
    device: str,
):
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    position, rotvec = retarget_tool_pose(
        calibration_ee,
        goal_ee,
        calibration_tool_position,
        calibration_tool_rotation,
        robot_base_pose,
    )
    quaternion = Rotation.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]]
    pose = Pose(
        position=torch.as_tensor(
            position, device=device, dtype=torch.float32
        ).reshape(1, 1, 3),
        quaternion=torch.as_tensor(
            quaternion, device=device, dtype=torch.float32
        ).reshape(1, 1, 4),
    )
    goal_tools = GoalToolPose.from_poses(
        {tool_frames[0]: pose.unsqueeze(1)},
        ordered_tool_frames=tool_frames,
    )
    return goal_tools, position, rotvec


def outcome_success(outcome) -> bool:
    if outcome is None:
        return False
    return bool(torch.as_tensor(outcome.success).any().item())


def plan_segment(
    motion_planner,
    start_state,
    goal_tools,
    *,
    label: str,
    max_attempts: int,
) -> JointTrajectoryPlan | None:
    outcome = motion_planner.plan_pose(
        current_state=start_state,
        goal_tool_poses=goal_tools,
        max_attempts=max_attempts,
    )
    success = outcome_success(outcome)
    status = None if outcome is None else getattr(outcome, "status", None)
    print(f"{label}: success={success} status={status}")
    if not success:
        return None
    interpolated = outcome.get_interpolated_plan()
    trajectory = JointTrajectoryPlan.from_curobo(
        interpolated,
        joint_names=motion_planner.joint_names,
    )
    joint_length = float(
        np.linalg.norm(np.diff(trajectory.position, axis=0), axis=1).sum()
    )
    print(
        f"{label}: trajectory_points={len(trajectory.position)} "
        f"joint_path_length={joint_length:.4f} "
        f"motion_time={trajectory.motion_time:.4f}s"
    )
    return trajectory


def endpoint_errors(
    motion_planner,
    joint_position: np.ndarray,
    desired_position: np.ndarray,
    desired_rotvec: np.ndarray,
    *,
    device: str,
) -> tuple[float, float]:
    from curobo._src.state.state_joint import JointState

    state = JointState.from_position(
        torch.as_tensor(
            joint_position, device=device,
            dtype=torch.float32,
        ).reshape(1, -1),
        joint_names=motion_planner.joint_names,
    )
    tool_pose = motion_planner.compute_kinematics(state).tool_poses.get_link_pose(
        motion_planner.tool_frames[0]
    )
    actual_position = tool_pose.position.detach().cpu().numpy().reshape(3)
    quaternion_wxyz = tool_pose.quaternion.detach().cpu().numpy().reshape(4)
    actual_rotation = Rotation.from_quat(
        quaternion_wxyz[[1, 2, 3, 0]]
    )
    desired_rotation = Rotation.from_rotvec(desired_rotvec)
    return (
        float(np.linalg.norm(actual_position - desired_position)),
        float((actual_rotation.inv() * desired_rotation).magnitude()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--log-path", default=DEFAULT_LOG)
    parser.add_argument(
        "--reference-episode-index", type=int, default=0,
        help="zero-based episode whose short over-bottle path supplies the via pose",
    )
    parser.add_argument("--reference-init-state-index", type=int, default=0)
    parser.add_argument("--test-init-state-index", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    reference_waypoints = parse_logged_waypoints(
        log_path, args.reference_episode_index
    )

    suite = benchmark.get_benchmark_dict()["libero_10"](
        category_value="Robot Initial States"
    )
    matches = [
        (index, task)
        for index, task in enumerate(suite.tasks)
        if task.name == args.task_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"task not found or ambiguous: {args.task_name}")
    task_index, task = matches[0]
    states = suite.get_task_init_states(task_index)

    env, _ = get_libero_env(
        task,
        "cosmos",
        resolution=256,
        camera_depths=[True, False],
    )
    try:
        _, reference_bottle = set_state_and_settle(
            env, states[args.reference_init_state_index]
        )
        obs, test_bottle = set_state_and_settle(
            env, states[args.test_init_state_index]
        )

        closest_index = int(
            np.argmin(
                np.linalg.norm(
                    reference_waypoints[:, :2] - reference_bottle[:2], axis=1
                )
            )
        )
        reference_via = reference_waypoints[closest_index].copy()
        via = reference_via.copy()
        via[:3] += test_bottle - reference_bottle

        current_ee = np.concatenate(
            [
                obs["robot0_eef_pos"],
                Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
            ]
        ).astype(np.float64)
        joint_positions = np.asarray(
            obs["robot0_joint_pos"], dtype=np.float32
        )
        robot_base_pose = np.concatenate(
            [env.robots[0].base_pos, env.robots[0].base_ori]
        ).astype(np.float64)

        print(f"reference_bottle={np.round(reference_bottle, 5).tolist()}")
        print(f"test_bottle={np.round(test_bottle, 5).tolist()}")
        print(f"reference_via={np.round(reference_via, 5).tolist()}")
        print(f"shifted_test_via={np.round(via, 5).tolist()}")
        print(
            "via_relative_to_test_bottle="
            f"{np.round(via[:3] - test_bottle, 5).tolist()}"
        )
        print(f"episode2_joint_start={np.round(joint_positions, 5).tolist()}")

        from curobo._src.geom.types import SceneCfg
        from curobo._src.state.state_joint import JointState

        backend = CuroboPlanner(
            device=args.device,
            joint_execution=True,
        )
        # Empty SceneCfg removes external RGB-D obstacles but preserves the
        # robot model's joint limits and self-collision constraints.
        motion_planner = backend._ensure_planner(SceneCfg())
        start = JointState.from_position(
            torch.as_tensor(
                joint_positions, device=args.device, dtype=torch.float32
            ).reshape(1, -1),
            joint_names=motion_planner.joint_names,
        )

        start_kinematics = motion_planner.compute_kinematics(start)
        start_tool = start_kinematics.tool_poses.get_link_pose(
            motion_planner.tool_frames[0]
        )
        calibration_tool_position = (
            start_tool.position.detach().cpu().numpy().reshape(3)
        )
        quaternion_wxyz = (
            start_tool.quaternion.detach().cpu().numpy().reshape(4)
        )
        calibration_tool_rotation = Rotation.from_quat(
            quaternion_wxyz[[1, 2, 3, 0]]
        ).as_matrix()

        via_goal, via_tool_position, via_tool_rotvec = make_goal_tool_pose(
            via,
            calibration_ee=current_ee,
            calibration_tool_position=calibration_tool_position,
            calibration_tool_rotation=calibration_tool_rotation,
            robot_base_pose=robot_base_pose,
            tool_frames=motion_planner.tool_frames,
            device=args.device,
        )
        first = plan_segment(
            motion_planner,
            start,
            via_goal,
            label="segment_1_start_to_above_bottle",
            max_attempts=args.max_attempts,
        )
        if first is None:
            print(
                "RESULT: INCONCLUSIVE_OR_INFEASIBLE - cuRobo did not find "
                "episode2 start -> above-bottle via in the empty external world"
            )
            return 2

        via_pos_error, via_rot_error = endpoint_errors(
            motion_planner,
            first.position[-1],
            via_tool_position,
            via_tool_rotvec,
            device=args.device,
        )
        print(
            f"segment_1_endpoint: position_error={via_pos_error:.6f}m "
            f"orientation_error={via_rot_error:.6f}rad"
        )

        second_start = JointState.from_position(
            torch.as_tensor(
                first.position[-1], device=args.device, dtype=torch.float32
            ).reshape(1, -1),
            joint_names=motion_planner.joint_names,
        )
        target_goal, target_tool_position, target_tool_rotvec = make_goal_tool_pose(
            DEFAULT_TARGET,
            calibration_ee=current_ee,
            calibration_tool_position=calibration_tool_position,
            calibration_tool_rotation=calibration_tool_rotation,
            robot_base_pose=robot_base_pose,
            tool_frames=motion_planner.tool_frames,
            device=args.device,
        )
        second = plan_segment(
            motion_planner,
            second_start,
            target_goal,
            label="segment_2_above_bottle_to_target",
            max_attempts=args.max_attempts,
        )
        if second is None:
            print(
                "RESULT: PARTIALLY_FEASIBLE - cuRobo reached the above-bottle "
                "via but did not find via -> target in the empty external world"
            )
            return 3

        target_pos_error, target_rot_error = endpoint_errors(
            motion_planner,
            second.position[-1],
            target_tool_position,
            target_tool_rotvec,
            device=args.device,
        )
        print(
            f"segment_2_endpoint: position_error={target_pos_error:.6f}m "
            f"orientation_error={target_rot_error:.6f}rad"
        )
        print(
            "RESULT: KINEMATICALLY_FEASIBLE - cuRobo found episode2 start -> "
            "above-bottle via -> target with joint limits and self-collision enabled"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
