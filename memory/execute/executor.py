"""Approach the ready pose, then run the memory segment.

One executor drives one intervention.  ``prepare`` maps the candidate's ready
pose into the current scene, plans the approach and arms the segment; ``step``
returns the action for the current frame (the caller runs ``env.step``);
``observe`` reports the intervention's internal hand-overs -- arrived, finished,
timeout -- without deciding whether the skill is complete; ``close`` restores
the control mode.

Controllers are driven through one minimal interface: ``step``, ``observe``,
``close`` and an arrival flag.  The adapters accept both the observation-based
and the legacy EE-state controllers, and both property-style and method-style
arrival flags.

The segment keeps the recorded policy: ``timegrip`` tracks the recorded
waypoints and plays the demo's gripper timing, ``rawactions`` replays the
object-frame actions frame by frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from memory.types import ExecutionResult, MemoryCandidate


@dataclass(frozen=True)
class InterventionConfig:
    """Calibrated values in one place; the defaults are the current ones."""

    # Segment budget: waypoints are stepped at this many frames each, with a
    # floor as the padding around the segment.
    segment_steps_per_frame: int = 4
    min_segment_pad: int = 64
    # A waypoint never times out on its own: the budget is the only bound, so a
    # stuck waypoint cannot fast-forward the index (and the gripper timing
    # bound to it).
    dwell_timeout: int = 0
    # Place phases always go lift-translate-descend instead of cuRobo.
    force_lift_place: bool = True
    # TurnOn replays raw object-frame actions: waypoint tracking freezes when
    # a waypoint is out of reach while the knob is grasped.
    turnon_segment: str = "rawactions"
    joint_execution: bool = True
    # Arrival tolerance shared by the intervention controllers (m, rad).
    position_tolerance: float = 0.006
    rotation_tolerance: float = 0.03
    # ik mode retries the IK with a larger iteration budget before giving up.
    teleport_retry_iterations: int = 2000
    # cuRobo 接近段计划的参考步数上限 = 执行预算 / approach_lag_factor。
    # 为什么是 1.5：机械臂跟踪参考点慢于计划（swap100 日志、127 次接近尝试实测，
    # 每次执行都记录"计划参考步数/实际步数/结局"）：成功案例里最慢的一次用满
    # 64 步预算才走完 41 个参考点（≈1.56 倍）；≥46 参考点的计划 0/37 全部没走完。
    # 取 1.5 带余量。这是执行系统的属性（机器人+控制器+控制周期），与场景无关；
    # 预算变了上限自动跟随，换执行系统才需按日志重标。
    approach_lag_factor: float = 1.5


def _ee_states(observation: dict[str, Any]) -> np.ndarray:
    """Current end-effector pose as the 6D state the controllers use."""
    return np.concatenate(
        [
            observation["robot0_eef_pos"],
            Rotation.from_quat(observation["robot0_eef_quat"]).as_rotvec(),
        ]
    ).astype(np.float32)


def _ee_states_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """A 4x4 EE pose as the 6D state the controllers use (xyz + rotvec).

    Replaces the old-tree ``matrix_to_ee_states`` (same computation; that one
    returns float64, this one follows this file's float32 convention).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate(
        [
            matrix[:3, 3],
            Rotation.from_matrix(matrix[:3, :3]).as_rotvec(),
        ]
    ).astype(np.float32)


def _arrived(controller: Any) -> bool:
    """Arrival flag of any controller, property or method."""
    for name in ("finished", "converged"):
        value = getattr(controller, name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value:
            return True
    return False


def _step_controller(controller: Any, observation: dict[str, Any]) -> np.ndarray:
    """Step a controller with whichever input it declares."""
    if getattr(controller, "requires_observation", False):
        return np.asarray(controller.step(observation), dtype=np.float32)
    return np.asarray(controller.step(_ee_states(observation)), dtype=np.float32)


def _observe_controller(controller: Any, observation: dict[str, Any]) -> None:
    observe = getattr(controller, "observe", None)
    if callable(observe):
        observe(observation)


def _close_controller(controller: Any) -> None:
    close = getattr(controller, "close", None)
    if callable(close):
        close()


class _IdleController:
    """ik mode: the robot is already teleported; one frame is enough."""

    requires_observation = True
    converged = True
    finished = True
    status = "TELEPORT"

    def step(self, observation: dict[str, Any]) -> np.ndarray:  # noqa: ARG002
        return np.zeros(7, dtype=np.float32)

    def observe(self, observation: dict[str, Any]) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass


class InterventionExecutor:
    """Drive one approach-plus-segment intervention."""

    # The caller runs env.step and must not overwrite the gripper dimension:
    # this executor owns the per-frame gripper timing inside the segment.
    requires_observation = True
    gripper_authority = True

    def __init__(self, config: InterventionConfig | None = None) -> None:
        self.config = InterventionConfig() if config is None else config
        self._reset()

    def prepare(
        self,
        candidate: MemoryCandidate,
        observation: dict[str, Any],
        env: Any,
        mode: str,
    ) -> bool:
        """Map the candidate into the scene and arm the intervention.

        ``mode`` is ``ik`` (teleport, approach only), ``align`` (approach and
        stop), or a segment policy: ``timegrip`` or ``rawactions``.  ``main_depth``
        and ``camera_params`` come from the observation when present; they are
        prepared by the caller today.  Returns False when nothing could be armed.
        """
        from memory_system.pointcloud_action.offline.extraction import object_frame

        self._reset()
        record = candidate.record
        skill = str(record.get("skill"))
        self._approach_gripper_cmd = _approach_gripper(skill)
        reference = _resolve_reference(env, _reference_argument(skill, record))
        T_world_reference, _, _ = object_frame(env, reference)
        T_world_ee = T_world_reference @ np.asarray(candidate.target_ee, dtype=np.float64)
        ee_states = _ee_states_from_matrix(T_world_ee)

        if mode == "ik":
            if not self._teleport(env, T_world_ee):
                return False
            self._approach = _IdleController()
            self._budget = 1
            return True

        current_ee_states = _ee_states(observation)
        segment_length = int(record.get("sequence_length", 64))
        segment = self.config.turnon_segment if skill == "TurnOn" else mode
        replays = mode != "align"
        force_lift = self.config.force_lift_place and skill in {"PlaceIn", "PlaceOn"}

        plan = None
        if not force_lift:
            from memory.execute.curobo_planner import CuroboPlanner

            planner = CuroboPlanner(joint_execution=self.config.joint_execution)
            plan = planner.plan(
                current_ee_states=current_ee_states,
                target_ee_states=ee_states,
                depth=observation.get("main_depth"),
                camera_params=observation.get("camera_params"),
                joint_positions=observation.get("robot0_joint_pos"),
                gripper_joint_positions=observation.get("robot0_gripper_qpos"),
                robot_base_pose=_robot_base_pose(env),
            )
        # 出发前验计划：参考步数超出"预算÷滞后系数"（64/1.5≈42）的轨迹在预算内
        # 走不完，连同构造即终态的（DEADLINE_INFEASIBLE 等）一并丢弃，当作没拿到
        # 计划走下面的三段兜底。丢弃前必须 close：构造成功的候选已经切到关节控制
        # （环境动作维度 7→8），不还原就把控制模式留给了后面的三段动作。
        approach = None if plan is None else plan.controller
        if plan is not None and approach is None and plan.joint_trajectory is not None:
            from cosmos_policy.experiments.robot.libero.libero_joint_control import (
                LiberoJointTrajectoryController,
            )

            approach = LiberoJointTrajectoryController(
                env,
                plan.joint_trajectory,
                ee_states,
                _approach_gripper(skill),
                position_tolerance=self.config.position_tolerance,
                rotation_tolerance=self.config.rotation_tolerance,
            )
            if getattr(approach, "status", "ACTIVE") != "ACTIVE" or getattr(
                approach, "planned_steps", 0
            ) > int(plan.correction_steps / self.config.approach_lag_factor):
                approach.close()
                approach = None
                plan = None
        if plan is None and not replays:
            return False

        if plan is None:
            # 兜底仍属"接近段"：换成不考虑碰撞的 lift-translate-descend，
            # replay 依旧只回放记忆段，不承担走向 ready pose 的工作。
            from memory.execute.controller import LiftTranslateDescendController

            steps = self.config.min_segment_pad + max(
                self.config.min_segment_pad,
                self.config.segment_steps_per_frame * segment_length,
            )
            self._approach = LiftTranslateDescendController(
                current_ee_states, ee_states, step_budget=steps
            )
            self._budget = steps
        else:
            self._approach = approach
            self._budget = plan.correction_steps

        if replays:
            self._arm_segment(record, T_world_reference, segment, segment_length)
        return self._approach is not None

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        """Return this frame's action; the caller runs env.step."""
        if self._stage == "done":
            # Nothing left to run: hold still.  The caller hands control back
            # once observe() has reported the outcome.
            return np.zeros(7, dtype=np.float32)
        self._steps_used += 1
        if self._stage == "approach":
            action = _step_controller(self._approach, observation)
            if action.reshape(-1).shape[0] == 6:
                # 无夹爪维的接近控制器（三段式）由 executor 补上该技能的夹爪命令
                action = np.append(action, self._approach_gripper_cmd).astype(np.float32)
            if _arrived(self._approach):
                # Arrival is reported by the next observe(); the stage flips
                # here so this frame still returns the approach action.
                self._handover = True
                self._stage = "replay" if self._segment is not None else "done"
            # 本帧仍是关节模式（可能 8 维）；close() 推迟到下一帧段执行前，
            # 否则先恢复 OSC（7 维）再发 8 维动作会让 env 断言失败。
            return action

        if not self._approach_closed:
            _close_controller(self._approach)
            self._approach_closed = True

        if self._raw is not None:
            action = self._raw[min(self._segment_step, len(self._raw) - 1)]
            self._segment_step += 1
            return np.asarray(action, dtype=np.float32).reshape(-1)[:7]

        current = _ee_states(observation)
        action = self._waypoints.step(current)
        index = min(self._waypoints.index, len(self._gripper) - 1)
        gripper = 1.0 if self._gripper[index] > 0 else -1.0
        return np.concatenate([action, [gripper]]).astype(np.float32)

    def observe(self, observation: dict[str, Any]) -> ExecutionResult | None:
        """Report a hand-over inside the intervention, or None while running."""
        if self._handover:
            self._handover = False
            return self._result("arrived", "approach", observation)
        if self._approach is None or self._stage == "done":
            return None
        if self._stage == "approach":
            _observe_controller(self._approach, observation)
            if self._steps_used >= self._budget:
                self._stage = "done"
                return self._result("timeout", "approach", observation)
            return None
        if self._segment_finished():
            self._stage = "done"
            return self._result("finished", "replay", observation)
        if self._steps_used >= self._budget:
            self._stage = "done"
            return self._result("timeout", "replay", observation)
        return None

    def close(self) -> None:
        """Drop the current intervention and restore the control mode."""
        if self._approach is not None and not self._approach_closed and self._stage == "replay":
            _close_controller(self._approach)
            self._approach_closed = True
        if self._waypoints is not None:
            _close_controller(self._waypoints)
        self._stage = "done"

    @property
    def status(self) -> str:
        if self._approach is None:
            return "idle"
        if self._stage == "approach":
            return f"approach:{getattr(self._approach, 'status', '?')}"
        return "REPLAY_DONE" if self._segment_finished() else "REPLAY_ACTIVE"

    @property
    def waypoints(self) -> np.ndarray | None:
        """Recorded segment EE poses (x,y,z,rx,ry,rz), for telemetry."""
        return None if self._waypoints is None else self._waypoints.waypoints

    # ---- internals

    def _reset(self) -> None:
        self._approach: Any = None
        self._stage = "approach"
        self._segment: str | None = None
        self._raw: np.ndarray | None = None
        self._waypoints: Any = None
        self._gripper: np.ndarray | None = None
        self._segment_step = 0
        self._steps_used = 0
        self._budget = 0
        self._approach_closed = False
        self._handover = False
        self._approach_gripper_cmd = 0.0

    def _arm_segment(
        self,
        record: dict,
        T_world_reference: np.ndarray,
        segment: str,
        segment_length: int,
    ) -> None:
        self._segment = segment
        self._gripper = np.asarray(record["gripper_sequence"], dtype=np.float64).reshape(-1)
        self._budget += max(
            self.config.min_segment_pad,
            self.config.segment_steps_per_frame * segment_length,
        )
        if segment == "rawactions":
            from memory_system.pointcloud_action.eval.eval_pointcloud_pick import (
                _replay_actions,
            )

            self._raw = _replay_actions(record, T_world_reference[:3, :3])
            return
        from memory.execute.curobo_trajectory import WaypointPoseController

        seq_world = np.stack(
            [
                T_world_reference @ matrix
                for matrix in np.asarray(record["ee_pose_object_sequence"], dtype=np.float64)
            ]
        )
        reference6 = np.stack([_ee_states_from_matrix(matrix) for matrix in seq_world])
        self._waypoints = WaypointPoseController(
            reference6[1:],
            dwell_timeout=self.config.dwell_timeout,
            max_steps=self._budget,
        )

    def _segment_finished(self) -> bool:
        if self._raw is not None:
            return self._segment_step >= len(self._raw)
        if self._waypoints is None:
            return True
        return bool(_arrived(self._waypoints))

    def _teleport(self, env: Any, T_world_ee: np.ndarray) -> bool:
        """ik mode: solve the joints and write them straight into the sim."""
        from memory_system.pointcloud_action.execute.mujoco_ik import solve_ee_ik

        sim = env.env.sim
        saved_state = sim.get_state().flatten()
        q_sol = solve_ee_ik(sim, T_world_ee[:3, 3], T_world_ee[:3, :3])
        if q_sol is None:
            sim.set_state_from_flattened(saved_state)
            sim.forward()
            q_sol = solve_ee_ik(
                sim,
                T_world_ee[:3, 3],
                T_world_ee[:3, :3],
                max_iter=self.config.teleport_retry_iterations,
            )
        if q_sol is None:
            sim.set_state_from_flattened(saved_state)
            sim.forward()
            return False
        env.regenerate_obs_from_state(sim.get_state().flatten())
        robots = getattr(env, "robots", None) or env.env.robots
        robots[0].controller.update(force=True)
        return True

    def _result(
        self, outcome: str, stage: str, observation: dict[str, Any]
    ) -> ExecutionResult:
        return ExecutionResult(
            outcome=outcome,
            stage=stage,
            steps_used=self._steps_used,
            final_ee=_ee_states(observation),
        )


def _reference_argument(skill: str, record: dict) -> str:
    """The scene body the candidate's ready pose is expressed against."""
    arguments = record.get("arguments") or {}
    return str(arguments.get("item" if skill == "Pick" else "target", ""))


def _approach_gripper(skill: str) -> float:
    """Approach-phase gripper command: holding through a Place, open otherwise."""
    return 1.0 if skill in {"PlaceIn", "PlaceOn"} else -1.0


def _robot_base_pose(env: Any) -> np.ndarray:
    robot = env.robots[0]
    return np.concatenate([robot.base_pos, robot.base_ori])


def _resolve_reference(env: Any, argument: str) -> str:
    """参照体 body 名：把 Place 的 region 名解析成锚物体 body。

    Moves to EnvironmentAdapter with §2.8; kept here so the runtime does not
    read it out of scripts/.
    """
    from memory_system.offline.build_targets import match_region_anchor

    sim = env.env.sim
    body_names: dict[str, None] = {}
    for index in range(sim.model.nbody):
        name = sim.model.body_id2name(index)
        if not name:
            continue
        body_names[name] = None
        if name.endswith("_main"):
            body_names[name[: -len("_main")]] = None
    if argument in body_names:
        return argument
    anchor = match_region_anchor(argument, body_names)
    return argument if anchor is None else anchor
