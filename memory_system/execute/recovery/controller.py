"""Closed-loop pose controller for memory_system recovery.

This is the memory_system replacement for
``bin/execute/closed_loop.py``.
"""
from __future__ import annotations

import logging

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - scipy is expected in eval
    Rotation = None

POS_ACTION_SCALE = 0.01  # meters per action unit
ROT_ACTION_SCALE = 0.10  # radians per action unit

logger = logging.getLogger(__name__)


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


class LiftTranslateDescendController:
    """三段接近：先在原地竖直上升，再水平平移到目标上方，最后下降到目标。

    每段是一个 ``PoseController``，串起来用；没有新机制、没有规划器。中间高度取
    ``max(起点 z, 目标 z) + LIFT``，所以水平位移发生在高一点的地方，不贴地横扫。
    对外接口与其它 approach 控制器一致：``step(current_ee_states)`` /
    ``converged`` / ``finished`` / ``status``。

    ``PoseController`` 自身没有步数上限（只有 5mm/0.02rad 的收敛判据），所以每段
    带一个步数上限 ``leg_caps``：到点或超限都切下一段，避免永久停在一段。竖直两段
    很快（~10mm/步，十几步就够），所以 ``leg_caps`` 把预算主要留给水平段——三段
    平分的话（3 × budget/2）会被调用方的 correction 预算截断在水平段。
    """

    LIFT = 0.04  # 中间高度比两端最高者再高出的量

    def __init__(self, current_ee_states, target_ee_states, step_budget: int):
        cur = np.asarray(current_ee_states, dtype=np.float64).reshape(6)
        target = np.asarray(target_ee_states, dtype=np.float64).reshape(6)
        h = max(float(cur[2]), float(target[2])) + self.LIFT
        up_here = cur.copy()
        up_here[2] = h      # 段 1：原地竖直上升
        up_there = target.copy()
        up_there[2] = h     # 段 2：水平平移到目标上方（仍在高处）

        self.targets = (up_here, up_there, target)   # 段 3：下降到目标
        budget = max(3, int(step_budget))
        vertical = max(16, budget // 8)
        self.leg_caps = (vertical, max(32, budget - 2 * vertical), vertical)
        self.index = 0
        self._steps = 0
        self._controller = PoseController(target_ee_states=self.targets[0])
        logger.info(
            "lift-translate-descend: start_z=%.4f h=%.4f target_z=%.4f (+%.2fm) "
            "leg_caps=%s",
            cur[2], h, target[2], self.LIFT, self.leg_caps,
        )

    def step(self, current_ee_states):
        action = self._controller.step(current_ee_states)
        self._steps += 1
        last = self.index == len(self.targets) - 1
        if not last and (
            self._controller.converged or self._steps >= self.leg_caps[self.index]
        ):
            self.index += 1
            self._steps = 0
            self._controller = PoseController(target_ee_states=self.targets[self.index])
            logger.info("lift-translate-descend: leg %d/%d", self.index,
                        len(self.targets) - 1)
        return action

    @property
    def converged(self):
        return bool(self._controller.converged) and self.index == len(self.targets) - 1

    @property
    def finished(self):
        if self.index < len(self.targets) - 1:
            return False
        return (
            bool(self._controller.converged)
            or self._steps >= self.leg_caps[self.index]
        )

    @property
    def status(self):
        return "LIFT_TRANSLATE_DESCEND %d/%d" % (self.index, len(self.targets) - 1)
