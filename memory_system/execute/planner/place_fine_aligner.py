"""Optional lightweight Place fine-alignment mode.

This module implements the ``simple`` Place mode:
  - Let VLA approach the memory ready pose.
  - When the end-effector is close enough, switch to a short PoseController
    correction.
  - If the correction fails or never triggers, the caller can simply resume
    VLA.  There is no cuRobo fallback by design.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.execute.recovery.controller import PoseController


class PoseControllerAdapter:
    """Minimal adapter to expose ``finished``/``status`` for PoseController.

    The existing eval loop checks ``controller.finished`` to detect early
    termination.  PoseController only exposes ``converged``, so this adapter
    mirrors the WaypointPoseController state machine without modifying
    PoseController itself.
    """

    def __init__(self, controller: PoseController, max_steps: int) -> None:
        self.controller = controller
        self.max_steps = int(max_steps)
        self.step_count = 0

    def step(self, current_ee_states: np.ndarray) -> np.ndarray:
        self.step_count += 1
        return self.controller.step(current_ee_states)

    @property
    def converged(self) -> bool:
        return self.controller.converged

    @property
    def status(self) -> str:
        if self.converged:
            return "CONVERGED"
        if self.step_count >= self.max_steps:
            return "GOAL_NOT_CONVERGED"
        return "ACTIVE"

    @property
    def finished(self) -> bool:
        return self.status != "ACTIVE"

    def close(self) -> None:
        pass


class PlaceFineAligner:
    """Check VLA progress and create a short PoseController correction.

    The aligner is intentionally stateless across episodes; the caller should
    create a fresh instance for each Pick->Place transition.
    """

    def __init__(
        self,
        ready_pose: np.ndarray,
        *,
        trigger_distance: float = 0.10,
        check_interval: int = 4,
        max_steps: int = 48,
        controller_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.ready_pose = np.asarray(ready_pose, dtype=np.float64).reshape(6)
        self.trigger_distance = float(trigger_distance)
        self.check_interval = max(1, int(check_interval))
        self.max_steps = max(1, int(max_steps))
        self.controller_kwargs = dict(controller_kwargs or {})
        self._check_counter = 0
        self.last_pos_error: float | None = None
        self.last_rot_error: float | None = None
        self.triggered = False

    def reset(self) -> None:
        self._check_counter = 0
        self.last_pos_error = None
        self.last_rot_error = None
        self.triggered = False

    def _current_ee(self, obs: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3),
                Rotation.from_quat(
                    np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
                ).as_rotvec(),
            ]
        ).astype(np.float64)

    def maybe_controller(
        self,
        obs: dict[str, Any],
        frame: int,
        active_phase: Any,
    ) -> PoseControllerAdapter | None:
        """Return a controller if the EE is close enough to the ready pose.

        ``frame`` is only used for diagnostics/logging; the internal counter
        controls the check interval.
        """
        del frame
        if self.triggered:
            return None
        if active_phase is None or active_phase.skill not in {"PlaceIn", "PlaceOn"}:
            return None

        self._check_counter += 1
        if self._check_counter % self.check_interval != 0:
            return None

        current = self._current_ee(obs)
        pos_error = float(np.linalg.norm(current[:3] - self.ready_pose[:3]))
        rot_error = float(
            (
                Rotation.from_rotvec(self.ready_pose[3:])
                * Rotation.from_rotvec(current[3:]).inv()
            ).magnitude()
        )
        self.last_pos_error = pos_error
        self.last_rot_error = rot_error

        if pos_error > self.trigger_distance:
            return None

        self.triggered = True
        controller = PoseController(
            target_ee_states=self.ready_pose,
            **self.controller_kwargs,
        )
        return PoseControllerAdapter(controller, max_steps=self.max_steps)
