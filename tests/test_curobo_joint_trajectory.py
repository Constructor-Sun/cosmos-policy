from types import SimpleNamespace

import numpy as np

from memory_system.execute.curobo_trajectory import (
    JointTrajectoryPlan,
    backoff_pose,
    find_minimum_feasible_backoff,
    retarget_tool_pose,
)
from memory_system.execute.curobo_planner import CuroboPlanner


def _plan(dof: int = 7) -> JointTrajectoryPlan:
    position = np.stack(
        [np.zeros(dof), np.full(dof, 0.02), np.full(dof, 0.04)]
    )
    return JointTrajectoryPlan(
        joint_names=tuple(f"panda_joint{i + 1}" for i in range(dof)),
        position=position,
        velocity=np.zeros_like(position),
        acceleration=np.zeros_like(position),
        dt=0.1,
    )


def test_from_curobo_preserves_timing_and_derivatives():
    source = _plan()
    padded = np.pad(source.position, ((0, 0), (0, 2)))
    state = SimpleNamespace(
        position=padded[None],
        velocity=np.zeros_like(padded)[None],
        acceleration=np.zeros_like(padded)[None],
        dt=np.array([source.dt]),
        joint_names=list(source.joint_names),
    )
    result = JointTrajectoryPlan.from_curobo(state)
    assert result.position.shape == (3, 7)
    assert result.motion_time == 0.2


def test_resampling_uses_control_time_not_waypoint_count():
    plan = _plan()
    sampled = plan.sample_positions(0.05)
    assert sampled.shape == (5, 7)
    np.testing.assert_allclose(sampled[0], plan.position[0])
    np.testing.assert_allclose(sampled[-1], plan.position[-1])
    assert plan.required_steps(0.05, settle_steps=2) == 6


def test_retarget_tool_pose_rotates_fixed_tool_offset():
    current = np.zeros(6)
    target = np.array([1.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0])
    tool_position, tool_rotation = retarget_tool_pose(
        current,
        target,
        current_tool_position=np.array([0.0, 0.0, -0.1]),
        current_tool_rotation=np.eye(3),
        robot_base_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(tool_position, [1.0, 0.1, 0.0], atol=1e-8)
    np.testing.assert_allclose(tool_rotation, target[3:], atol=1e-8)


def test_backoff_pose_retreats_opposite_local_tool_z():
    target = np.array([1.0, 2.0, 3.0, 0.0, np.pi / 2.0, 0.0])
    result = backoff_pose(target, 0.01)
    np.testing.assert_allclose(result[:3], [0.99, 2.0, 3.0], atol=1e-8)
    np.testing.assert_allclose(result[3:], target[3:])


def test_backoff_search_keeps_direct_goal_when_feasible():
    calls = []
    result = find_minimum_feasible_backoff(
        lambda distance: calls.append(distance) or "plan"
    )
    assert result == (0.0, "plan")
    assert calls == [0.0]


def test_backoff_search_finds_and_refines_first_feasible_interval():
    result = find_minimum_feasible_backoff(
        lambda distance: distance if distance >= 0.0133 else None
    )
    assert result is not None
    distance, plan = result
    assert 0.0133 <= distance <= 0.0143
    assert plan == distance


def test_backoff_search_returns_none_at_bounded_failure():
    calls = []
    result = find_minimum_feasible_backoff(
        lambda distance: calls.append(distance), max_backoff=0.01
    )
    assert result is None
    assert max(calls) == 0.01


def test_depth_self_filter_uses_actual_robot_spheres():
    planner = CuroboPlanner(robot_padding=0.005)
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    spheres = np.array([[[0.0, 0.0, 0.0, 0.1]]])
    np.testing.assert_allclose(planner._filter_robot(points, spheres), points[1:])


def test_surface_sampling_preserves_target_and_global_points():
    planner = CuroboPlanner(max_spheres=4)
    points = np.array(
        [[0.0, 0.0, value] for value in (0.01, 0.02, 0.1, 0.2, 0.3, 0.4)]
    )
    sampled = planner._sample_points(points, np.zeros(3))
    assert len(sampled) == 4
    assert 0.01 in sampled[:, 2]
    assert 0.4 in sampled[:, 2]


def test_libero_adapter_switches_action_dimension_and_restores(monkeypatch):
    from cosmos_policy.experiments.robot.libero import libero_joint_control as module

    class Controller:
        def __init__(self, name, control_dim):
            self.name = name
            self.control_dim = control_dim

        def update_base_pose(self, *_):
            pass

        def update(self):
            pass

        def reset_goal(self):
            pass

    class Robot:
        base_pos = np.zeros(3)
        base_ori = np.eye(3)

        def __init__(self):
            self.controller = Controller("OSC_POSE", 6)
            self.controller_config = {"type": "OSC_POSE"}

        @property
        def action_dim(self):
            return self.controller.control_dim + 1

        def _load_controller(self):
            self.controller = Controller("JOINT_POSITION", 7)

    robot = Robot()
    base_env = SimpleNamespace(action_dim=7, _action_dim=7, control_timestep=0.05)
    env = SimpleNamespace(env=base_env, robots=[robot])
    monkeypatch.setattr(
        module.suite,
        "load_controller_config",
        lambda **_: {"type": "JOINT_POSITION"},
    )
    controller = module.LiberoJointTrajectoryController(
        env,
        _plan(),
        target_ee_states=np.zeros(6),
        gripper_command=-1.0,
    )
    observation = {
        "robot0_joint_pos": np.zeros(7),
        "robot0_eef_pos": np.ones(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    assert base_env._action_dim == 8
    assert controller.step(observation).shape == (8,)
    controller.close()
    assert robot.controller.name == "OSC_POSE"
    assert base_env._action_dim == 7


def test_deadline_accepts_final_reference_inside_goal_tolerance():
    from cosmos_policy.experiments.robot.libero.libero_joint_control import (
        LiberoJointTrajectoryController,
    )

    controller = LiberoJointTrajectoryController.__new__(
        LiberoJointTrajectoryController
    )
    controller._status = "ACTIVE"
    controller.last_reference = np.zeros(7)
    controller.tracking_tolerance = 0.08
    controller.target = np.zeros(6)
    controller.references = np.zeros((2, 7))
    controller.reference_index = 1
    controller.position_tolerance = 0.005
    controller.rotation_tolerance = 0.02
    controller.stable_steps = 2
    controller._stable_count = 0
    controller.step_count = 64
    controller.max_steps = 64
    observation = {
        "robot0_joint_pos": np.zeros(7),
        "robot0_eef_pos": np.array([0.001, 0.0, 0.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    controller.observe(observation)
    assert controller.status == "CONVERGED"
