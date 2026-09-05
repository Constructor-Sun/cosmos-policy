"""Artifact schema for PointCloud Action Memory v1."""
from __future__ import annotations

MEMORY_FORMAT = "libero_pointcloud_action_memory_v1"
SUITE = "libero_90"
RETRIEVAL_KEY = "target_points_object"

# The artifact is a dict:
#   {
#       "format": MEMORY_FORMAT,
#       "suite": SUITE,
#       "controller_config": CONTROLLER_CONFIG,
#       "records": [record, ...],
#   }
# Each record stores visible point cloud under target_points_* (retrieval key),
# and the complete object point cloud under complete_points_*.
TOP_LEVEL_FIELDS = (
    "format",
    "suite",
    "controller_config",
    "records",
)

RECORD_FIELDS = (
    "memory_id",
    "source_task",
    "source_demo",
    "planner_step_id",
    "skill",
    "arguments",
    "target_points_world",
    "target_points_object",
    "target_xyz_world",
    "complete_points_world",
    "complete_points_object",
    "T_world_object_anchor",
    "object_frame_translation",
    "object_frame_rotation",
    "frame_source",
    "frame_convention",
    "ready_frame",
    "segment_end",
    "sequence_length",
    "ready_ee_states",
    "T_object_ee_ready",
    "action_sequence_raw",
    "action_sequence_world_physical",
    "action_sequence_object_physical",
    "action_scale",
    "gripper_sequence",
    "ee_pose_world_sequence",
    "ee_pose_object_sequence",
)
