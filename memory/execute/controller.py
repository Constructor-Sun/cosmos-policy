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


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """6D pose error: position difference and rotation from current to target."""
    error = np.zeros(6, dtype=np.float64)
    error[:3] = target[:3] - current[:3]
    if Rotation is not None:
        error[3:] = np.asarray(
            (
                Rotation.from_rotvec(target[3:])
                * Rotation.from_rotvec(current[3:]).inv()
            ).as_rotvec(),
            dtype=np.float64,
        )
    else:
        error[3:] = target[3:] - current[3:]
    return error


def pose_action(
    error: np.ndarray, scale: np.ndarray, k: float, action_clip: float
) -> np.ndarray:
    """One P step toward the target, clipped to the per-axis action bound."""
    return np.clip((k / scale) * error, -action_clip, action_clip).astype(np.float32)


class PoseController:
    """P controller that drives the EE from its current pose to a target pose."""

    def __init__(
        self,
        target_ee_states,
        k: float = 0.2,
        s_pos: float = POS_ACTION_SCALE,
        s_rot: float = ROT_ACTION_SCALE,
        action_clip: float = 0.6,
        eps_pos: float = 0.006,
        eps_rot: float = 0.03,
    ):
        self.target = np.asarray(target_ee_states, dtype=np.float64).reshape(6)
        self.k = float(k)
        self.scale = np.array(
            [float(s_pos)] * 3 + [float(s_rot)] * 3, dtype=np.float64
        )
        self.action_clip = float(action_clip)
        self.eps_pos = float(eps_pos)
        self.eps_rot = float(eps_rot)
        self._converged = False

    def step(self, current_ee_states) -> np.ndarray:
        current = np.asarray(current_ee_states, dtype=np.float64).reshape(6)
        error = pose_error(current, self.target)
        if (
            float(np.linalg.norm(error[:3])) <= self.eps_pos
            and float(np.linalg.norm(error[3:])) <= self.eps_rot
        ):
            self._converged = True
            return np.zeros(6, dtype=np.float32)
        return pose_action(error, self.scale, self.k, self.action_clip)

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
    按预算的 1/4 取份额（构造处注释记了标定依据），水平段拿剩余的大头。
    """

    LIFT = 0.08  # 中间高度比两端最高者再高出的量

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
        # 垂直段份额 ÷4：大高差接手时爬升可达 30~44cm（~1cm/步，swap100 实测最大
        # 上升 315mm/下降 439mm），÷8 的 16~29 步会在半路烧断（29/128 次，表现为
        # 未到高度就斜着横移）。水平段拿剩余，仍远超实际横移需要。
        vertical = max(16, budget // 4)
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
