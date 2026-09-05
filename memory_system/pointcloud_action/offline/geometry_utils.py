"""SE(3) helpers for object-centric point cloud and action memory."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from a 3x3 rotation and 3D translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid 4x4 transform."""
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3].T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ transform[:3, 3]
    return result


def world_to_object_points(points: np.ndarray, T_world_object: np.ndarray) -> np.ndarray:
    """Transform world points to the object frame defined by T_world_object."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(T_world_object, dtype=np.float64)[:3, :3]
    translation = np.asarray(T_world_object, dtype=np.float64)[:3, 3]
    return (points - translation) @ rotation


def world_to_object_pose(T_WE: np.ndarray, T_WO: np.ndarray) -> np.ndarray:
    """Map a world EE pose to object frame: T_OE = inv(T_WO) @ T_WE."""
    return inverse_transform(T_WO) @ T_WE


def object_to_world_pose(T_OE: np.ndarray, T_WO: np.ndarray) -> np.ndarray:
    """Map an object-frame EE pose back to world: T_WE = T_WO @ T_OE."""
    return T_WO @ T_OE


def ee_states_to_matrix(ee_states: np.ndarray) -> np.ndarray:
    """Convert 6D EE state (position, rotvec) to a 4x4 matrix."""
    ee = np.asarray(ee_states, dtype=np.float64).reshape(6)
    return make_transform(
        Rotation.from_rotvec(ee[3:6]).as_matrix(),
        ee[:3],
    )


def matrix_to_ee_states(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 EE matrix to 6D EE state (position, rotvec)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate(
        [
            matrix[:3, 3],
            Rotation.from_matrix(matrix[:3, :3]).as_rotvec(),
        ]
    )


def rotate_action_world_to_object(
    action_world: np.ndarray,
    R_world_object: np.ndarray,
) -> np.ndarray:
    """Transform a physical world delta action to object frame.

    action_world is (..., 7) with [delta_p, omega, gripper].
    """
    action_world = np.asarray(action_world, dtype=np.float64)
    rotation = np.asarray(R_world_object, dtype=np.float64).reshape(3, 3)
    delta_p_world = action_world[..., :3]
    omega_world = action_world[..., 3:6]
    delta_r_world = Rotation.from_rotvec(omega_world).as_matrix()
    delta_r_object = rotation.T @ delta_r_world @ rotation
    omega_object = Rotation.from_matrix(delta_r_object).as_rotvec()
    result = np.concatenate(
        [delta_p_world @ rotation, omega_object, action_world[..., 6:]],
        axis=-1,
    )
    return result


def rotate_action_object_to_world(
    action_object: np.ndarray,
    R_world_object: np.ndarray,
) -> np.ndarray:
    """Transform an object-frame physical delta action back to world."""
    action_object = np.asarray(action_object, dtype=np.float64)
    rotation = np.asarray(R_world_object, dtype=np.float64).reshape(3, 3)
    delta_p_object = action_object[..., :3]
    omega_object = action_object[..., 3:6]
    delta_r_object = Rotation.from_rotvec(omega_object).as_matrix()
    delta_r_world = rotation @ delta_r_object @ rotation.T
    omega_world = Rotation.from_matrix(delta_r_world).as_rotvec()
    result = np.concatenate(
        [delta_p_object @ rotation.T, omega_world, action_object[..., 6:]],
        axis=-1,
    )
    return result
