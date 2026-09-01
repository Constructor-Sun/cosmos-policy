"""Environment-neutral representation of a time-parameterized cuRobo plan."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def backoff_pose(target_ee: np.ndarray, distance: float) -> np.ndarray:
    """Retreat a pose opposite its local tool-z approach direction."""
    target = np.asarray(target_ee, dtype=np.float64).reshape(6).copy()
    approach = Rotation.from_rotvec(target[3:]).as_matrix()[:, 2]
    target[:3] -= float(distance) * approach
    return target


def find_minimum_feasible_backoff(
    probe: Any,
    max_backoff: float = 0.08,
    initial_step: float = 0.005,
    resolution: float = 0.005,
    min_backoff: float | None = None,
) -> tuple[float, Any] | None:
    """Find the smallest feasible signed backoff around the direct target.

    ``min_backoff=None`` preserves the historical one-sided ``[0, max]``
    search. Passing a negative value enables a two-sided search; each half is
    bracketed and refined independently, then the closest feasible result is
    returned.
    """
    direct = probe(0.0)
    if direct is not None:
        return 0.0, direct
    lower = 0.0 if min_backoff is None else float(min_backoff)
    upper = float(max_backoff)

    def search(sign: float, bound: float) -> tuple[float, Any] | None:
        bound = abs(float(bound))
        if bound <= 0.0:
            return None
        low = 0.0
        high = min(abs(float(initial_step)), bound)
        feasible = None
        while True:
            feasible = probe(sign * high)
            if feasible is not None:
                break
            low = high
            if high >= bound - 1e-12:
                return None
            high = min(2.0 * high, bound)
        while high - low > resolution:
            middle = 0.5 * (low + high)
            candidate = probe(sign * middle)
            if candidate is None:
                low = middle
            else:
                high, feasible = middle, candidate
        return sign * high, feasible

    candidates = []
    if lower < 0.0:
        result = search(-1.0, lower)
        if result is not None:
            candidates.append(result)
    if upper > 0.0:
        result = search(1.0, upper)
        if result is not None:
            candidates.append(result)
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item[0]))


def _matrix(value: Any, name: str) -> np.ndarray:
    """Convert a single cuRobo trajectory tensor to a ``[time, dof]`` array."""
    if value is None:
        raise ValueError(f"cuRobo trajectory has no {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > 2:
        if array.shape[0] != 1:
            raise ValueError(f"expected one trajectory for {name}, got {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected [time, dof] {name}, got {array.shape}")
    return array


def _optional_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.float64)
    result = _matrix(value, "derivative")
    if result.shape[0] != shape[0] or result.shape[1] < shape[1]:
        raise ValueError(f"trajectory derivative shape {result.shape} != {shape}")
    return result[:, : shape[1]]


def retarget_tool_pose(
    current_ee: np.ndarray,
    target_ee: np.ndarray,
    current_tool_position: np.ndarray,
    current_tool_rotation: np.ndarray,
    robot_base_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a LIBERO grip-site goal to cuRobo's tool frame in base coordinates."""
    current_ee = np.asarray(current_ee, dtype=np.float64).reshape(6)
    target_ee = np.asarray(target_ee, dtype=np.float64).reshape(6)
    base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
    base_rotation = Rotation.from_quat(base[3:]).as_matrix()
    current_ee_rotation = Rotation.from_rotvec(current_ee[3:]).as_matrix()
    current_ee_base_rotation = base_rotation.T @ current_ee_rotation
    current_ee_base_position = base_rotation.T @ (current_ee[:3] - base[:3])
    ee_to_tool_rotation = current_ee_base_rotation.T @ current_tool_rotation
    ee_to_tool_position = current_ee_base_rotation.T @ (
        np.asarray(current_tool_position) - current_ee_base_position
    )
    target_ee_base_rotation = (
        base_rotation.T @ Rotation.from_rotvec(target_ee[3:]).as_matrix()
    )
    target_ee_base_position = base_rotation.T @ (target_ee[:3] - base[:3])
    target_tool_position = (
        target_ee_base_position + target_ee_base_rotation @ ee_to_tool_position
    )
    target_tool_rotation = target_ee_base_rotation @ ee_to_tool_rotation
    return target_tool_position, Rotation.from_matrix(target_tool_rotation).as_rotvec()


class WaypointPoseController:
    """Closed-loop P controller that tracks a sequence of EE waypoints."""

    def __init__(
        self,
        waypoints: np.ndarray,
        k: float = 0.5,
        s_pos: float = 0.01,
        s_rot: float = 0.10,
        action_clip: float = 0.5,
        eps_pos: float = 0.005,
        eps_rot: float = 0.02,
        max_steps: int = 96,
    ) -> None:
        self.waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 6)
        self.k = float(k)
        self.scale = np.array([s_pos] * 3 + [s_rot] * 3, dtype=np.float64)
        self.action_clip = float(action_clip)
        self.eps_pos = float(eps_pos)
        self.eps_rot = float(eps_rot)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.index = 0
        self.step_count = 0
        self._converged = len(self.waypoints) == 0
        self._status = "CONVERGED" if self._converged else "ACTIVE"

    def _target(self) -> np.ndarray | None:
        return None if self.index >= len(self.waypoints) else self.waypoints[self.index]

    def _error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.zeros(6, dtype=np.float64)
        error[:3] = target[:3] - current[:3]
        error[3:] = (
            Rotation.from_rotvec(target[3:])
            * Rotation.from_rotvec(current[3:]).inv()
        ).as_rotvec()
        return error

    def _action(self, current: np.ndarray) -> np.ndarray:
        while True:
            target = self._target()
            if target is None:
                self._converged = True
                return np.zeros(6, dtype=np.float32)
            error = self._error(current, target)
            if not (
                np.linalg.norm(error[:3]) <= self.eps_pos
                and np.linalg.norm(error[3:]) <= self.eps_rot
            ):
                break
            self.index += 1
        action = np.clip(
            (self.k / self.scale) * error, -self.action_clip, self.action_clip
        )
        return action.astype(np.float32)

    def step(self, current_ee_states: np.ndarray) -> np.ndarray:
        if self.finished:
            return np.zeros(6, dtype=np.float32)
        self.step_count += 1
        action = self._action(
            np.asarray(current_ee_states, dtype=np.float64).reshape(6)
        )
        if self._converged:
            self._status = "CONVERGED"
        elif self.step_count >= self.max_steps:
            self._status = "GOAL_NOT_CONVERGED"
        return action

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def status(self) -> str:
        return self._status

    @property
    def finished(self) -> bool:
        return self._status != "ACTIVE"


@dataclass(frozen=True)
class JointTrajectoryPlan:
    """CPU copy of a cuRobo trajectory, including its physical timing."""

    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        if position.ndim != 2 or len(position) < 2:
            raise ValueError(f"invalid position trajectory shape: {position.shape}")
        if position.shape[1] != len(self.joint_names):
            raise ValueError("joint_names do not match trajectory DOF")
        if not np.isfinite(position).all() or self.dt <= 0.0:
            raise ValueError("trajectory position/dt must be finite and positive")
        for name in ("velocity", "acceleration"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != position.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {name} trajectory")

    @classmethod
    def from_curobo(
        cls,
        state: Any,
        joint_names: Sequence[str] | None = None,
    ) -> "JointTrajectoryPlan":
        names = tuple(joint_names or state.joint_names or ())
        raw_position = _matrix(state.position, "position")
        if not names or raw_position.shape[1] < len(names):
            raise ValueError(f"trajectory has {raw_position.shape[1]} DOF for {len(names)} names")
        position = raw_position[:, : len(names)]
        raw_dt = getattr(state, "dt", None)
        if raw_dt is None:
            raise ValueError("cuRobo interpolated trajectory has no dt")
        if hasattr(raw_dt, "detach"):
            raw_dt = raw_dt.detach().cpu().numpy()
        dt = float(np.asarray(raw_dt).reshape(-1)[0])
        return cls(
            joint_names=names,
            position=position,
            velocity=_optional_matrix(getattr(state, "velocity", None), position.shape),
            acceleration=_optional_matrix(
                getattr(state, "acceleration", None), position.shape
            ),
            dt=dt,
        )

    @property
    def motion_time(self) -> float:
        return float((len(self.position) - 1) * self.dt)

    def sample_positions(self, control_dt: float) -> np.ndarray:
        """Linearly sample joint references at the environment control period."""
        control_dt = float(control_dt)
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        source_t = np.arange(len(self.position), dtype=np.float64) * self.dt
        target_t = np.arange(0.0, self.motion_time + 1e-12, control_dt)
        if target_t.size == 0 or target_t[-1] < self.motion_time - 1e-9:
            target_t = np.append(target_t, self.motion_time)
        sampled = np.empty((len(target_t), self.position.shape[1]), dtype=np.float64)
        for joint_index in range(self.position.shape[1]):
            sampled[:, joint_index] = np.interp(
                target_t, source_t, self.position[:, joint_index]
            )
        sampled[0] = self.position[0]
        sampled[-1] = self.position[-1]
        return sampled

    def required_steps(self, control_dt: float, settle_steps: int = 2) -> int:
        return len(self.sample_positions(control_dt)) - 1 + max(0, int(settle_steps))
