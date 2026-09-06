"""Configuration for PointCloud Action Memory."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_DIR = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_90"
DEFAULT_OUTPUT = ROOT / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
DEFAULT_RESOLUTION = 256
DEFAULT_MAX_DEMOS = 0

# LIBERO-90 OSC_POSE controller scales used to convert raw actions to physical
# world-frame delta commands.
ACTION_POS_SCALE = 0.05
ACTION_ROT_SCALE = 0.5

# Per-dimension scale vector compatible with older action_scale fields:
# [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
ACTION_SCALE = np.array(
    [ACTION_POS_SCALE] * 3 + [ACTION_ROT_SCALE] * 3,
    dtype=np.float32,
)

# Serializable subset of the OSC_POSE controller config used by LIBERO.
CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [ACTION_POS_SCALE] * 3 + [ACTION_ROT_SCALE] * 3,
    "output_min": [-ACTION_POS_SCALE] * 3 + [-ACTION_ROT_SCALE] * 3,
    "control_delta": True,
    "impedance_mode": "fixed",
    "interpolation": None,
    "ramp_ratio": 0.2,
}

# Desired EE-to-target distance used to select Pick ready_frame.
READY_DISTANCE_M = 0.14

# Point cloud source used for retrieval.
# "visible" uses the single-view visible cloud.
# "complete" uses the oracle complete object cloud from the simulator.
POINT_CLOUD_SOURCE = "complete"

# Pick replay mode.
# "open_loop" replays the recorded action commands (original behaviour).
# "closed_loop" tracks the recorded realized EE path with a
#   convergence-guaranteed waypoint controller and closes the gripper by path
#   progress with grasp confirmation.  Opt-in; validated in
#   CLOSED_LOOP_REPLAY_PLAN.md (V1-V5).
# "wp_time" tracks the recorded realized EE path with the same waypoint
#   controller but keeps the source time-indexed gripper.  P0-1 2x2 ablation
#   (2026-09-06, moka only): the waypoint tracking is the component that fixes
#   the open-loop realization error there.  The V4 regression attribution
#   between waypoint tracking and the position trigger is NOT yet isolated --
#   full-coverage runs are in PICK_MEMORY_REUSE.md (§2.2/§4).
REPLAY_MODE = "open_loop"

# Re-anchor the Pick trajectory to the object pose measured after the ready
# motion completes (re-map the ready target and realign before replay).
# P0-2 ablation (2026-09-06): equivalent to settling before the first anchor
# (6/6 cases), while waiting without re-anchoring never fixes them (0/6) --
# the first anchor happens before the object has settled.
REANCHOR_AFTER_READY = True

# Ready-motion mode.
# "legacy" keeps the original CuroboPlanner path.
# "waypoint" uses the ASPIRE-inspired lightweight waypoint path.
READY_MOTION_MODE = "waypoint"

# Parameters for the lightweight ready-motion path (Plan 1).
READY_MOTION_LIFT_HEIGHT = 0.15
READY_MOTION_CLEARANCE = 0.05
READY_MOTION_APPROACH_DISTANCE = 0.05
READY_MOTION_MAX_WAYPOINTS = 8
READY_MOTION_JOINT_INTERP_STEPS = 8
READY_MOTION_IK_SEEDS = 12
READY_MOTION_MAX_JOINT_STEP = 0.8
READY_MOTION_MAX_MID_POINTS = 4
READY_MOTION_CONTROL_DT = 0.05
