"""Pick controller: closed-loop EE waypoints or open-loop action replay."""
from __future__ import annotations

import numpy as np

from memory_system.execute.curobo_trajectory import WaypointPoseController


class PointCloudPickController:
    """Execute a Pick chunk by EE waypoints or by replaying mapped actions."""

    def __init__(
        self,
        ee_states_sequence: np.ndarray | None = None,
        gripper_sequence: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        **waypoint_kwargs,
    ):
        if actions is not None:
            self.actions = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
            self.index = 0
            self.waypoint_controller = None
            return
        if ee_states_sequence is None:
            raise ValueError("either ee_states_sequence or actions must be provided")
        self.actions = None
        self.index = 0
        self.waypoint_controller = WaypointPoseController(
            np.asarray(ee_states_sequence, dtype=np.float64).reshape(-1, 6),
            **waypoint_kwargs,
        )
        self.gripper_sequence = (
            np.asarray(gripper_sequence, dtype=np.float64).reshape(-1)
            if gripper_sequence is not None
            else np.ones(len(ee_states_sequence) - 1)
        )

    def step(self, current_ee_states: np.ndarray | None = None) -> np.ndarray:
        if self.actions is not None:
            if self.index >= len(self.actions):
                return np.zeros(7, dtype=np.float32)
            action = self.actions[self.index].copy()
            self.index += 1
            return action
        ee_action = self.waypoint_controller.step(current_ee_states)
        index = min(self.waypoint_controller.index, len(self.gripper_sequence) - 1)
        gripper = 1.0 if self.gripper_sequence[index] > 0 else -1.0
        return np.concatenate([ee_action, [gripper]]).astype(np.float32)

    @property
    def finished(self) -> bool:
        if self.actions is not None:
            return self.index >= len(self.actions)
        return self.waypoint_controller.finished

    @property
    def converged(self) -> bool:
        return self.finished

    @property
    def status(self) -> str:
        if self.actions is not None:
            return "CONVERGED" if self.finished else "ACTIVE"
        return self.waypoint_controller.status
