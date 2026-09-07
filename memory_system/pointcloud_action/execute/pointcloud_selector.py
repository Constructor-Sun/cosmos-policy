"""Select memory candidates and map object-centric trajectories to a scene."""
from __future__ import annotations

from typing import Any

import numpy as np

from memory_system.pointcloud_action.offline.geometry_utils import (
    matrix_to_ee_states,
    object_to_world_pose,
)
from memory_system.pointcloud_action.retrieval.pointcloud_action_memory import (
    PointCloudActionMemory,
)


class PointCloudSelector:
    """Retrieve top-k Pick memories and map them to the current object frame."""

    def __init__(
        self,
        memory_path: str,
        cloud_key: str = "target_points_object",
    ):
        self.cloud_key = cloud_key
        self.memory = PointCloudActionMemory(memory_path, cloud_key=cloud_key)

    def select(
        self,
        points: np.ndarray,
        T_world_object_current: np.ndarray,
        skill: str = "Pick",
        top_k: int = 1,
        **retrieve_kwargs: Any,
    ) -> list[dict[str, Any]]:
        hits = self.memory.retrieve(
            points, skill=skill, top_k=top_k, **retrieve_kwargs
        )
        results = []
        for hit in hits:
            record = hit["record"]
            ee_object = np.asarray(record["ee_pose_object_sequence"], dtype=np.float64)
            ee_world = np.stack(
                [object_to_world_pose(matrix, T_world_object_current) for matrix in ee_object],
                axis=0,
            )
            ee_states = np.stack(
                [matrix_to_ee_states(matrix) for matrix in ee_world],
                axis=0,
            )
            results.append(
                {
                    "record": record,
                    "distance": float(hit["distance"]),
                    "key_stage": hit.get("key_stage", "shape"),
                    "ready_ee_states": ee_states[0].copy(),
                    "ee_pose_world_sequence": ee_world,
                    "ee_states_sequence": ee_states,
                    "gripper_sequence": np.asarray(
                        record.get("gripper_sequence", record["action_sequence_raw"][:, -1]),
                        dtype=np.float64,
                    ),
                }
            )
        return results
