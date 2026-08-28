import numpy as np

from memory_system.execute.curobo_trajectory import WaypointPoseController


def test_waypoint_execution_budget_is_independent_of_waypoint_count():
    waypoints = np.array([[0, 0, 0, 0, 0, 0], [0.03, 0, 0, 0, 0, 0]], dtype=float)
    controller = WaypointPoseController(waypoints, max_steps=20)
    current = np.zeros(6, dtype=float)

    for _ in range(controller.max_steps):
        action = controller.step(current)
        current[:3] += action[:3] * 0.01
        if controller.finished:
            break

    assert controller.converged
    assert controller.status == "CONVERGED"
    assert controller.step_count > len(waypoints)


def test_waypoint_controller_reports_unconverged_deadline():
    target = np.array([[1, 0, 0, 0, 0, 0]], dtype=float)
    controller = WaypointPoseController(target, max_steps=2)
    current = np.zeros(6, dtype=float)

    controller.step(current)
    controller.step(current)

    assert controller.finished
    assert not controller.converged
    assert controller.status == "GOAL_NOT_CONVERGED"


def test_empty_waypoint_sequence_is_already_converged():
    controller = WaypointPoseController(np.empty((0, 6)), max_steps=1)

    assert controller.finished
    assert controller.converged
    assert controller.status == "CONVERGED"
