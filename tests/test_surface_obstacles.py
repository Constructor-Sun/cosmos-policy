import numpy as np

from memory_system.execute.surface_obstacles import (
    densify_joint_path,
    filter_robot_points,
    find_trajectory_conflict,
    select_surface_points,
    voxel_representatives,
)


def test_voxel_representatives_keep_each_occupied_surface_cell():
    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.009, 0.009, 0.009],
            [0.021, 0.001, 0.001],
            [0.001, 0.021, 0.001],
        ]
    )
    result = voxel_representatives(points, 0.02)
    assert len(result) == 3
    np.testing.assert_allclose(result[0], points[0])


def test_surface_selection_is_deterministic_and_target_dense():
    target_surface = np.stack(
        [np.linspace(0.005, 0.05, 20), np.zeros(20), np.zeros(20)], axis=1
    )
    surrounding = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points = np.concatenate([target_surface, surrounding], axis=0)
    first = select_surface_points(
        points, np.zeros(3), max_points=8, voxel_size=0.02,
        target_fraction=0.25,
    )
    second = select_surface_points(
        points, np.zeros(3), max_points=8, voxel_size=0.02,
        target_fraction=0.25,
    )
    np.testing.assert_allclose(first.points, second.points)
    assert first.target_count == 2
    assert first.global_count == 6
    assert np.count_nonzero(np.linalg.norm(first.points, axis=1) < 0.06) >= 2
    assert np.max(np.linalg.norm(first.points, axis=1)) == 1.0


def test_path_conflict_points_take_priority_over_target_refinement():
    points = np.array(
        [[value, 0.0, 0.0] for value in np.linspace(-1.0, 1.0, 101)]
    )
    mandatory = np.array([[0.75, 0.0, 0.0], [0.80, 0.0, 0.0]])
    result = select_surface_points(
        points,
        target=np.zeros(3),
        max_points=8,
        target_fraction=0.25,
        mandatory_points=mandatory,
    )
    assert result.mandatory_count == 2
    assert result.target_count == 0
    assert any(np.allclose(point, mandatory[0]) for point in result.points)
    assert any(np.allclose(point, mandatory[1]) for point in result.points)


def test_robot_filter_uses_collision_sphere_surface_and_padding():
    points = np.array([[0.0, 0.0, 0.0], [0.106, 0.0, 0.0]])
    spheres = np.array([[[0.0, 0.0, 0.0, 0.1]]])
    filtered = filter_robot_points(points, spheres, padding=0.005)
    np.testing.assert_allclose(filtered, points[1:])


def test_dense_surface_validation_finds_intermediate_collision():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    spheres = np.array(
        [
            [[-0.2, 0.0, 0.0, 0.05]],
            [[0.04, 0.0, 0.0, 0.05]],
            [[0.2, 0.0, 0.0, 0.05]],
        ]
    )
    conflict = find_trajectory_conflict(points, spheres, safety_margin=0.0)
    assert conflict is not None
    assert conflict.first_step == 1
    assert conflict.last_step == 1
    assert conflict.point_indices.tolist() == [0]
    assert conflict.min_clearance < 0.0


def test_dense_surface_validation_accepts_safe_path():
    points = np.array([[0.0, 0.0, 0.0]])
    spheres = np.array([[[0.2, 0.0, 0.0, 0.05]]])
    assert find_trajectory_conflict(points, spheres, safety_margin=0.01) is None


def test_joint_path_densification_bounds_each_step():
    path = np.array([[0.0, 0.0], [0.05, -0.03]])
    dense = densify_joint_path(path, max_joint_step=0.02)
    assert len(dense) == 4
    assert np.max(np.abs(np.diff(dense, axis=0))) <= 0.02
    np.testing.assert_allclose(dense[[0, -1]], path)
