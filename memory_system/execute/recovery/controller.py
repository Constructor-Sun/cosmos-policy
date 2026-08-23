"""Closed-loop pose controller for memory_system recovery.

This is the memory_system replacement for
``bin/execute/closed_loop.py``.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - scipy is expected in eval
    Rotation = None

POS_ACTION_SCALE = 0.01  # meters per action unit
ROT_ACTION_SCALE = 0.10  # radians per action unit


class PoseController:
    """P controller that drives the EE from its current pose to a target pose."""

    def __init__(
        self,
        target_ee_states,
        k: float = 0.2,
        s_pos: float = POS_ACTION_SCALE,
        s_rot: float = ROT_ACTION_SCALE,
        action_clip: float = 0.5,
        z_lift: float = 0.0,
        eps_pos: float = 0.005,
        eps_rot: float = 0.02,
        eps_z_lift: float = 0.003,
    ):
        self.target = np.asarray(target_ee_states, dtype=np.float64).reshape(6)
        self.k = float(k)
        self.scale = np.array(
            [float(s_pos)] * 3 + [float(s_rot)] * 3, dtype=np.float64
        )
        self.action_clip = float(action_clip)
        self.z_lift = float(z_lift)
        self.eps_pos = float(eps_pos)
        self.eps_rot = float(eps_rot)
        self.eps_z_lift = float(eps_z_lift)
        self._lift_target_z: float | None = None
        self._lift_done = False
        self._converged = False
        self.step_count = 0

    def _error(self, current: np.ndarray) -> np.ndarray:
        e = np.zeros(6, dtype=np.float64)
        e[:3] = self.target[:3] - current[:3]
        if Rotation is not None:
            rot = (
                Rotation.from_rotvec(self.target[3:])
                * Rotation.from_rotvec(current[3:]).inv()
            ).as_rotvec()
            e[3:] = np.asarray(rot, dtype=np.float64)
        else:
            e[3:] = self.target[3:] - current[3:]
        return e

    def _clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, -self.action_clip, self.action_clip).astype(np.float32)

    def preview_action(self, current_ee_states) -> np.ndarray:
        current = np.asarray(current_ee_states, dtype=np.float64).reshape(6)
        if not self._lift_done and self.z_lift > 0.0:
            z_goal = (
                current[2] + self.z_lift
                if self._lift_target_z is None
                else self._lift_target_z
            )
            e_z = z_goal - current[2]
            if abs(e_z) > self.eps_z_lift:
                action = np.zeros(6, dtype=np.float64)
                action[2] = self.k / self.scale[2] * e_z
                return self._clip(action)
        e = self._error(current)
        if (
            float(np.linalg.norm(e[:3])) <= self.eps_pos
            and float(np.linalg.norm(e[3:])) <= self.eps_rot
        ):
            return np.zeros(6, dtype=np.float32)
        return self._clip((self.k / self.scale) * e)

    def step(self, current_ee_states) -> np.ndarray:
        current = np.asarray(current_ee_states, dtype=np.float64).reshape(6)
        self.step_count += 1
        if not self._lift_done and self.z_lift > 0.0:
            if self._lift_target_z is None:
                self._lift_target_z = current[2] + self.z_lift
            e_z = self._lift_target_z - current[2]
            if abs(e_z) > self.eps_z_lift:
                action = np.zeros(6, dtype=np.float64)
                action[2] = self.k / self.scale[2] * e_z
                return self._clip(action)
            self._lift_done = True
        e = self._error(current)
        if (
            float(np.linalg.norm(e[:3])) <= self.eps_pos
            and float(np.linalg.norm(e[3:])) <= self.eps_rot
        ):
            self._converged = True
            return np.zeros(6, dtype=np.float32)
        return self._clip((self.k / self.scale) * e)

    @property
    def converged(self) -> bool:
        return self._converged
