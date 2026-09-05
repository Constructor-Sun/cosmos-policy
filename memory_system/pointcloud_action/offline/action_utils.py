"""Build object-centric action and EE pose sequences from LIBERO HDF5 data."""
from __future__ import annotations

import numpy as np

from memory_system.pointcloud_action.config import ACTION_POS_SCALE, ACTION_ROT_SCALE
from memory_system.pointcloud_action.offline.geometry_utils import (
    ee_states_to_matrix,
    rotate_action_world_to_object,
    world_to_object_pose,
)


def physical_world_actions(raw_actions: np.ndarray) -> np.ndarray:
    """Convert raw OSC_POSE actions to physical world-frame delta actions."""
    raw = np.asarray(raw_actions, dtype=np.float64)
    physical = raw.copy()
    physical[..., :3] *= ACTION_POS_SCALE
    physical[..., 3:6] *= ACTION_ROT_SCALE
    return physical


def object_actions(world_physical: np.ndarray, R_world_object: np.ndarray) -> np.ndarray:
    """Convert world physical delta actions to object frame."""
    return rotate_action_world_to_object(world_physical, R_world_object)


def build_action_sequences(
    actions: np.ndarray,
    ready_frame: int,
    segment_end: int,
    R_world_object: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw, world_physical, object_physical) action sequences."""
    raw = np.asarray(actions[ready_frame:segment_end], dtype=np.float64)
    world_physical = physical_world_actions(raw)
    object_physical = object_actions(world_physical, R_world_object)
    return raw, world_physical, object_physical


def build_ee_sequences(
    ee_states: np.ndarray,
    ready_frame: int,
    segment_end: int,
    T_world_object: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build EE pose sequences.

    ee_states[i] is treated as the EE pose before action i.  Therefore L actions
    from ready_frame to segment_end-1 correspond to L+1 EE poses at indices
    ready_frame ... segment_end.
    """
    start = int(ready_frame)
    end = int(segment_end)
    world_matrices = np.stack(
        [ee_states_to_matrix(ee_states[index]) for index in range(start, end + 1)],
        axis=0,
    )
    object_matrices = np.stack(
        [world_to_object_pose(matrix, T_world_object) for matrix in world_matrices],
        axis=0,
    )
    ready_ee_states = np.asarray(ee_states[start], dtype=np.float64).reshape(6).copy()
    T_object_ee_ready = object_matrices[0].copy()
    return ready_ee_states, T_object_ee_ready, world_matrices, object_matrices
