"""Round-trip checks for object-centric SE(3) and action transforms."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.pointcloud_action.offline.action_utils import (
    build_action_sequences,
    build_ee_sequences,
)
from memory_system.pointcloud_action.offline.geometry_utils import (
    ee_states_to_matrix,
    inverse_transform,
    make_transform,
    matrix_to_ee_states,
    object_to_world_pose,
    rotate_action_object_to_world,
    rotate_action_world_to_object,
    world_to_object_points,
    world_to_object_pose,
)


def _random_transform(rng: np.random.Generator) -> np.ndarray:
    return make_transform(
        Rotation.random(random_state=rng).as_matrix(),
        rng.normal(size=3),
    )


def test_point_transform() -> None:
    rng = np.random.default_rng(0)
    transform = _random_transform(rng)
    points = rng.normal(size=(20, 3))
    local = world_to_object_points(points, transform)
    restored = (local @ transform[:3, :3].T) + transform[:3, 3]
    np.testing.assert_allclose(restored, points, atol=1e-10)


def test_pose_round_trip() -> None:
    rng = np.random.default_rng(1)
    T_WO = _random_transform(rng)
    T_WE = _random_transform(rng)
    T_OE = world_to_object_pose(T_WE, T_WO)
    np.testing.assert_allclose(object_to_world_pose(T_OE, T_WO), T_WE, atol=1e-10)
    np.testing.assert_allclose(T_WO @ inverse_transform(T_WO), np.eye(4), atol=1e-10)


def test_ee_state_round_trip() -> None:
    rng = np.random.default_rng(2)
    ee = np.concatenate([rng.normal(size=3), Rotation.random(random_state=rng).as_rotvec()])
    np.testing.assert_allclose(matrix_to_ee_states(ee_states_to_matrix(ee)), ee, atol=1e-10)


def test_action_round_trip() -> None:
    rng = np.random.default_rng(3)
    rotation = Rotation.random(random_state=rng).as_matrix()
    # Use small rotation vectors so rotvec representation round-trips uniquely.
    action = np.concatenate([rng.normal(size=3), rng.normal(scale=0.1, size=3), [1.0]])
    object_action = rotate_action_world_to_object(action, rotation)
    restored = rotate_action_object_to_world(object_action, rotation)
    np.testing.assert_allclose(restored, action, atol=1e-10)


def test_sequence_shapes() -> None:
    rng = np.random.default_rng(4)
    actions = rng.normal(size=(20, 7))
    ee_states = np.stack(
        [
            np.concatenate([rng.normal(size=3), Rotation.random(random_state=rng).as_rotvec()])
            for _ in range(21)
        ]
    )
    rotation = Rotation.random(random_state=rng).as_matrix()
    transform = make_transform(rotation, rng.normal(size=3))
    raw, world_physical, object_physical = build_action_sequences(
        actions, 5, 15, rotation
    )
    assert raw.shape == (10, 7)
    assert world_physical.shape == (10, 7)
    assert object_physical.shape == (10, 7)
    _, _, ee_world, ee_object = build_ee_sequences(ee_states, 5, 15, transform)
    assert ee_world.shape == (11, 4, 4)
    assert ee_object.shape == (11, 4, 4)


def main() -> None:
    test_point_transform()
    test_pose_round_trip()
    test_ee_state_round_trip()
    test_action_round_trip()
    test_sequence_shapes()
    print("ROUND_TRIP_OK")


if __name__ == "__main__":
    main()
