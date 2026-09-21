"""Oracle ready-pose alignment experiment for LIBERO-PRO.

同时是修复路径的 selector 来源：COSMOS_SKILL_READY_MEMORY=1 时
run_libero_eval._create_initial_alignment_selector 构造本模块的 OracleReadySelector
（controller 模式），因此本文件不是可整份删除的一次性实验。

目的
    验证「把机器人放到物体系 ready pose，再交回 VLA」能否救回 swap / task 的失败。
    物体系 ready 位姿取自 pointcloud_action memory（LIBERO-90 构建，不是 libero_10
    memory），经 T_world_object 映射到当前场景，因此不依赖物体在场景中的绝对位置。
    参照体按技能取自当前阶段：Pick→arguments["item"]，Place/TurnOn→arguments["target"]。
    阶段来源：修复路径已传入 phase；开局没有 phase，就用当前任务 BDDL 计划的第一阶段。
    不再依赖命令行 --item。

模式（--mode）
    none        不干预，跑原生 baseline。用来量「VLA 对当前指令的裸能力」= oracle 天花板。
    ik          解 IK 后直接把关节角写进 sim（teleport）。零执行误差，是上界，
                但真机不可用——只用来判死/判活干预范式。
    controller  cuRobo 从单帧 RGB-D 规划避障轨迹，交给 LiberoJointTrajectoryController
                闭环执行。机器人是"走过去"的，这条执行路径真机要复用。

语言（--language-from-bddl）
    LIBERO 的 benchmark 用 grab_language_from_filename() 从【文件名】推指令，不读 BDDL
    的 (:language ...)。而 libero_10_task 的 BDDL 文件名与 base 完全相同，于是 VLA
    永远收到 base 的旧指令（"put the moka pot on it"），尽管 BDDL 里写的是新指令
    （"put the pan on it"）—— 任务被判新 goal，却被下了旧命令。
    本开关 patch get_libero_env，改用 BDDL 里真实的 (:language)。
    对 swap / base 是 no-op（两者本来就一致）；对 _task / _lan 变体是必须的。

隔离
    不修改仓库里的任何文件。只在 __main__ 里 monkey-patch run_libero_eval 模块的全局名
    （get_libero_env / _create_initial_alignment_selector），patch 仅存在于本进程内存；
    其他入口各自起进程，行为逐位不变。（validate_config 不再 patch：PRO suites 已在
    仓库白名单内。）

用法
    见同目录 run_swap_oracle.sh；单任务 smoke test 见该脚本头部注释。
    对照基线 experiments/liberopro/libero_10_swap_seed7/summary.txt = 0/200。

诊断
    <out>/oracle_debug.log 记录 selector 是否被构造、模式、目标位姿、IK/规划结果。
    跑完先看这个文件：出现 "skipping intervention" 说明该 case 没干预，等同 baseline。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

import cosmos_policy.experiments.robot.libero.run_libero_eval as R
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    PolicyEvalConfig,
    eval_libero,
)
from memory_system.execute.initial_alignment import InitialAlignmentResult
from memory_system.execute.plan import drop_presatisfied_turnon, plan_from_bddl
from memory_system.types import RecoveryTarget

DEFAULT_MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
DEFAULT_MODEL = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"

_DEBUG_LOG = None


def _diag(message: str) -> None:
    """诊断输出：stdout 且必定落盘，避免被上层 except 吞掉后查无对证。"""
    line = f"[ORACLE] {message}"
    print(line, flush=True)
    if _DEBUG_LOG is not None:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ------------------------------------------------- 任务钩子（阶段计划 + 语言修正）

# get_libero_env 拦截时缓存的当前任务阶段计划。开局 select() 没有 phase 参数，
# 就从这里取第一阶段；每换任务重建一次。
_TASK_PLAN: dict = {}


def _bddl_path(task) -> Path:
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _bddl_language(task) -> str:
    """读 BDDL 里真正的 (:language ...)；失败则回退到 benchmark 从文件名推的值。"""
    try:
        match = re.search(r"\(:language\s+(.+?)\)", _bddl_path(task).read_text())
        if match:
            return match.group(1).strip()
        _diag(f"no (:language) block in {task.bddl_file}; keeping benchmark language")
    except Exception as exc:  # noqa: BLE001 - 回退即可，不阻断评测
        _diag(f"bddl language read failed for {task.name}: {exc}")
    return task.language


def _patch_get_libero_env(language_from_bddl: bool, capture_plan: bool) -> None:
    """拦截 get_libero_env：缓存当前任务的 BDDL 阶段计划；可选换成 BDDL 真实指令。"""
    original = R.get_libero_env

    def get_libero_env(task, *args, **kwargs):
        env, language = original(task, *args, **kwargs)
        if capture_plan:
            _TASK_PLAN.clear()
            _TASK_PLAN["phases"] = plan_from_bddl(_bddl_path(task))
            _diag(
                f"plan task={task.name} "
                f"phases={[(p.skill, dict(p.arguments)) for p in _TASK_PLAN['phases']]}"
            )
        if language_from_bddl:
            language = _bddl_language(task)
            _diag(f"language task={task.name} -> {language!r}")
        return env, language

    R.get_libero_env = get_libero_env


# --------------------------------------------------------------------- memory


def _object_type(name: str) -> str:
    """moka_pot_2 -> moka_pot。

    memory 存的是【物体系】ready 位姿，实例编号不参与匹配；真系统的检索按点云
    形状走，同型物体天然命中。所以 Pick 也用同型回退，而不是要求实例名精确一致
    （否则 KITCHEN_SCENE8 的 moka_pot_2 在 memory 里找不到，直接崩）。
    """
    return re.sub(r"_\d+$", "", str(name))


# 已加载 memory 记录的缓存：place 库 ~100MB，select() 每个 episode 都会调，不能反复读盘。
_RECORD_CACHE: dict[Path, list] = {}


def _records(memory_path: Path) -> list:
    import torch

    if memory_path not in _RECORD_CACHE:
        _RECORD_CACHE[memory_path] = torch.load(
            memory_path, map_location="cpu", weights_only=False
        )["records"]
    return _RECORD_CACHE[memory_path]


def _memory_paths(pick: Path) -> dict[str, Path]:
    """每个技能一条库：Pick 用主库，Place 共用 place 库，TurnOn 用 turnon 库。"""
    return {
        "Pick": pick,
        "PlaceIn": pick.with_name("pointcloud_action_memory_place.pt"),
        "PlaceOn": pick.with_name("pointcloud_action_memory_place.pt"),
        "TurnOn": pick.with_name("pointcloud_action_memory_turnon.pt"),
    }


def _resolve_reference(env, argument: str) -> str:
    """参照体 body 名。Place 的 target 常是 region 名（flat_stove_1_cook_region），
    其中嵌着锚物体名——用构库时 match_region_anchor 的同一规则对 sim body 名回退，
    保证运行时锚与库内 T_object_ee_ready 的锚是同一个 body。"""
    from memory_system.offline.build_targets import match_region_anchor

    sim = env.env.sim
    # root body 名带 _main 后缀（flat_stove_1_main）；剥掉后才是构库时
    # instance_to_id 里的实例级键，match_region_anchor 才能命中。
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
    if anchor is not None:
        _diag(f"region {argument!r} -> anchor body {anchor!r}")
        return anchor
    return argument  # 解析不到就原样传给 object_frame 抛错（fail loud）


def _load_ready_pose(
    skill: str, arguments: dict, memory_paths: dict[str, Path]
) -> tuple[np.ndarray, str, dict]:
    """按 (skill, arguments) 精确匹配取参照物系的 ready 位姿 T_reference_ee_ready (4x4)
    及整条记录（回放要用 ee_pose_object_sequence / gripper_sequence）。

    Pick 保留同型回退（memory 来自 LIBERO-90，实例名与当前场景不保证一致）；
    Place/TurnOn 只做精确匹配，缺记录直接抛错。
    """
    memory_path = memory_paths[skill]
    want = {str(key): str(value) for key, value in arguments.items()}
    same_type: tuple[np.ndarray, str] | None = None
    want_type = ""
    for record in _records(memory_path):
        if record.get("skill") != skill:
            continue
        got = {str(key): str(value) for key, value in record.get("arguments", {}).items()}
        matrix = np.asarray(record["T_object_ee_ready"], dtype=np.float64).reshape(4, 4)
        memory_id = str(record["memory_id"])
        if got == want:
            return matrix, memory_id, record
        if skill == "Pick":
            if not want_type:
                want_type = _object_type(want.get("item", ""))
            if same_type is None and _object_type(got.get("item", "")) == want_type:
                same_type = (matrix, memory_id, record)
    if same_type is not None:
        _diag(f"{skill} {want} 无精确记录，回退到同型 {want_type!r}: {same_type[1]}")
        return same_type
    raise KeyError(f"no {skill} record for {want} in {memory_path}")


class _IdleController:
    """ik 模式：机器人已被 teleport 到位，只需让对齐跑一步后交回 VLA。"""

    requires_observation = True
    converged = True
    finished = True
    status = "ORACLE_TELEPORT"

    def step(self, observation):  # noqa: ARG002 - pose already reached
        return np.zeros(7, dtype=np.float32)

    def observe(self, observation):  # noqa: ARG002
        pass

    def close(self):
        pass


# ==== 已停用：把接近段第二段换成 cuRobo 的实现（见 keep/oracle_ready_eval.py.seg2curobo）====
# 现状：调用处已还原为 LiftTranslateDescendController。要重新启用，取消下面整块的注释，
# 并把调用处的 LiftTranslateDescendController 换回 _LiftCuRoboDescendController。
# class _LiftCuRoboDescendController:
#     """三段接近：竖直抬升（P 控制）→ cuRobo 关节轨迹（第二段）→ 竖直下降（P 控制）。
#
#     三段目标、预算和夹爪约定与 LiftTranslateDescendController 相同，区别只在第二段：
#     不再由 P 控制器一路保持目标姿态（手腕会被顶在关节限位上磨），而是交给 Joint space
#     规划，由它自己决定中途怎么走。规划不出来，或关节控制器因为预算不够起不来（它会在
#     构造函数里直接判不可行、且不切换控制器），就什么都不做：第二段继续用 P 控制走完，
#     并记一行 curobo_plan_failed。第二段是无环境障碍规划，因此不需要深度图。
#     """
#
#     requires_observation = True
#     LIFT = 0.04
#
#     def __init__(
#         self,
#         env,
#         planner,
#         current_ee_states,
#         target_ee_states,
#         step_budget,
#         gripper_command=1.0,
#     ):
#         cur = np.asarray(current_ee_states, dtype=np.float64).reshape(6)
#         target = np.asarray(target_ee_states, dtype=np.float64).reshape(6)
#         h = max(float(cur[2]), float(target[2])) + self.LIFT
#         up_here = cur.copy()
#         up_here[2] = h          # 段 1：原地竖直上升
#         up_there = target.copy()
#         up_there[2] = h         # 段 2 的终点：目标位姿抬到 h
#         self.env = env
#         self.planner = planner
#         self.gripper_command = float(gripper_command)
#         self.targets = (up_here, up_there, target)
#         self.index = 0
#         self._steps = 0
#         self._controller = PoseController(target_ee_states=self.targets[0])
#         self._joint = None      # 第二段的关节控制器；规划失败时一直是 None
#         self._joint_tried = False
#         budget = max(3, int(step_budget))
#         vertical = max(16, budget // 8)
#         self.leg_caps = (vertical, max(32, budget - 2 * vertical), vertical)
#
#     @staticmethod
#     def _ee_state(obs):
#         return np.concatenate([
#             obs["robot0_eef_pos"],
#             Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
#         ]).astype(np.float64)
#
#     def _start_joint_segment(self, obs):
#         """进入第二段：先问 cuRobo；失败就保持 P 控制。"""
#         self._joint_tried = True
#         from cosmos_policy.experiments.robot.libero.libero_joint_control import (
#             LiberoJointTrajectoryController,
#         )
#
#         base = np.concatenate([
#             self.env.robots[0].base_pos, self.env.robots[0].base_ori
#         ])
#         plan = self.planner.plan(
#             current_ee_states=self._ee_state(obs),
#             target_ee_states=self.targets[1],
#             joint_positions=obs["robot0_joint_pos"],
#             gripper_joint_positions=obs.get("robot0_gripper_qpos"),
#             robot_base_pose=base,
#             obstacle_free=True,
#         )
#         if plan is None or plan.joint_trajectory is None:
#             _diag("curobo_plan_failed: segment 2 stays on the PoseController")
#             try:
#                 import json as _json
#                 from pathlib import Path as _Path
#                 _dump = {
#                     "current_ee": np.asarray(self._ee_state(obs), dtype=float).tolist(),
#                     "joints": np.asarray(obs["robot0_joint_pos"], dtype=float).tolist(),
#                     "gripper": np.asarray(
#                         obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=float
#                     ).tolist(),
#                     "base": np.asarray(base, dtype=float).tolist(),
#                     "up_there": np.asarray(self.targets[1], dtype=float).tolist(),
#                     "ready": np.asarray(self.targets[2], dtype=float).tolist(),
#                 }
#                 with _Path("/tmp/seg2_dump.jsonl").open("a") as _fh:
#                     _fh.write(_json.dumps(_dump) + "\n")
#             except Exception:
#                 pass
#             return
#         controller = LiberoJointTrajectoryController(
#             self.env, plan.joint_trajectory, self.targets[1], self.gripper_command,
#             max_steps=self.leg_caps[1],  # 与 LTLD 第二段的步数预算一致
#         )
#         if controller.finished:  # 预算不够等：构造函数判不可行，且没切控制器
#             _diag(
#                 "curobo_plan_failed: joint controller not startable "
#                 f"({controller.status})"
#             )
#             try:
#                 import json as _json
#                 from pathlib import Path as _Path
#                 _traj = plan.joint_trajectory
#                 _dump = {
#                     "why": "not_startable",
#                     "status": str(controller.status),
#                     "planned_steps": int(controller.planned_steps),
#                     "max_steps": int(self.leg_caps[1]),
#                     "traj_points": int(len(_traj.position)),
#                     "traj_dt": float(_traj.dt),
#                     "motion_time": float(_traj.motion_time),
#                     "current_ee": np.asarray(self._ee_state(obs), dtype=float).tolist(),
#                     "joints": np.asarray(obs["robot0_joint_pos"], dtype=float).tolist(),
#                     "gripper": np.asarray(
#                         obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=float
#                     ).tolist(),
#                     "base": np.asarray(base, dtype=float).tolist(),
#                     "up_there": np.asarray(self.targets[1], dtype=float).tolist(),
#                     "ready": np.asarray(self.targets[2], dtype=float).tolist(),
#                     "traj_first": np.asarray(_traj.position[0][:7], dtype=float).tolist(),
#                     "traj_last": np.asarray(_traj.position[-1][:7], dtype=float).tolist(),
#                 }
#                 with _Path("/tmp/seg2_dump.jsonl").open("a") as _fh:
#                     _fh.write(_json.dumps(_dump) + "\n")
#             except Exception:
#                 pass
#             return
#         self._joint = controller
#
#     def _advance(self):
#         self.index += 1
#         self._steps = 0
#         self._controller = PoseController(target_ee_states=self.targets[self.index])
#
#     def step(self, obs):
#         if self._joint is not None:
#             if not self._joint.finished:
#                 return self._joint.step(obs)
#             self._joint.close()  # 第二段走完：切回 OSC，第三段继续用 P 控制
#             self._joint = None
#             self._advance()
#         action = self._controller.step(self._ee_state(obs))
#         self._steps += 1
#         last = self.index == len(self.targets) - 1
#         if not last and (
#             self._controller.converged or self._steps >= self.leg_caps[self.index]
#         ):
#             if self.index == 0 and not self._joint_tried:
#                 self._start_joint_segment(obs)
#                 if self._joint is not None:
#                     self._advance()
#                     return self._joint.step(obs)
#             self._advance()
#         return action
#
#     def observe(self, obs):
#         if self._joint is not None:
#             self._joint.observe(obs)
#
#     @property
#     def finished(self):
#         if self._joint is not None:
#             return False
#         last = self.index == len(self.targets) - 1
#         return last and (
#             self._controller.converged or self._steps >= self.leg_caps[self.index]
#         )
#
#     @property
#     def converged(self):
#         return self.finished
#
#     @property
#     def status(self):
#         if self._joint is not None:
#             return f"CUROBO_SEGMENT2 {self._joint.status}"
#         return "LIFT_CUROBO_DESCEND %d/%d" % (self.index, len(self.targets) - 1)
#
#     def close(self):
#         if self._joint is not None:
#             self._joint.close()
#             self._joint = None
#
#
class _ReplayController:
    """对齐（cuRobo 走到 ready）之后按记忆段回放，段尾交回 VLA。

    ready 位就是记录段的第 0 帧（构库时同一帧），两段无缝相接。timegrip：
    waypoint 跟踪 + demo 夹爪信号按 waypoint 进度原样播放；rawactions：把
    物体系动作序列旋到当前世界系逐帧执行。接近段夹爪沿用【段首夹爪命令】：Pick 段
    为开、Place/TurnOn 段为闭（见 _via_planner 的 segment_gripper）。"""

    requires_observation = True
    # 逐帧夹爪时序由本 controller 决定，runtime 不得用常量覆写动作末维。
    gripper_authority = True

    def __init__(self, approach, record: dict, T_world_ref: np.ndarray,
                 segment: str = "timegrip", wp_max_steps: int | None = None):
        self._approach = approach
        self._segment = segment
        self._mode = "approach"
        self._approach_closed = False
        self._raw = None
        self._wp = None
        self._t = 0
        self._grip = np.asarray(
            record["gripper_sequence"], dtype=np.float64
        ).reshape(-1)
        seq_world = np.stack(
            [T_world_ref @ m
             for m in np.asarray(record["ee_pose_object_sequence"], dtype=np.float64)]
        )
        if segment == "rawactions":
            from memory_system.pointcloud_action.eval.eval_pointcloud_pick import (
                _replay_actions,
            )
            self._raw = _replay_actions(record, T_world_ref[:3, :3])
        else:
            from memory_system.execute.curobo_trajectory import WaypointPoseController
            from memory_system.pointcloud_action.offline.geometry_utils import (
                matrix_to_ee_states,
            )
            ref6 = np.stack([matrix_to_ee_states(m) for m in seq_world])
            # dwell_timeout=0：取消"单个 waypoint 驻留超 40 步就强制推进"这条限制
            # —— 它会在手臂离目标还很远时把 waypoint 索引（以及绑在索引上的夹爪
            # 时序）一起快进。代价是卡住的 waypoint 不再被跳过，只能靠 correction
            # 的步数预算收尾，所以这里把控制器的 max_steps 显式对齐到该预算，
            # 让"预算"成为唯一的界（否则会退回库里的 len(waypoints)×8）。
            self._wp = WaypointPoseController(
                ref6[1:], dwell_timeout=0, max_steps=wp_max_steps
            )

    def step(self, obs):
        if self._mode == "approach":
            # 复用 runtime 的分发约定：requires_observation=True 的控制器收 obs，
            # 传统的 EE 控制器（如 PoseController）收 6 维 EE 状态。
            from cosmos_policy.experiments.robot.libero.libero_joint_control import (
                step_correction_controller,
            )

            action = step_correction_controller(self._approach, obs)
            # PoseController 只有 converged、没有 finished。
            if getattr(self._approach, "finished", False) or getattr(
                self._approach, "converged", False
            ):
                self._mode = "segment"
                _diag("replay: approach done, starting memory segment")
            # 本帧仍属关节模式（可能 8 维）；close() 推迟到下一帧段执行前，
            # 否则先恢复 OSC(7 维) 再发 8 维动作会让 env 断言失败。
            return action
        if not self._approach_closed:
            if hasattr(self._approach, "close"):
                self._approach.close()
            self._approach_closed = True
        if self._raw is not None:
            action = np.asarray(self._raw[min(self._t, len(self._raw) - 1)],
                                dtype=np.float32).reshape(-1)[:7]
            self._t += 1
            return action
        cur = np.concatenate(
            [obs["robot0_eef_pos"], Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec()]
        ).astype(np.float32)
        a6 = self._wp.step(cur)
        index = min(self._wp.index, len(self._grip) - 1)
        g = 1.0 if self._grip[index] > 0 else -1.0
        return np.concatenate([a6, [g]]).astype(np.float32)

    def observe(self, obs):
        if self._mode == "approach" and hasattr(self._approach, "observe"):
            self._approach.observe(obs)

    def close(self):
        if not self._approach_closed:
            if hasattr(self._approach, "close"):
                self._approach.close()
            self._approach_closed = True

    @property
    def finished(self):
        if self._mode == "approach":
            return False
        if self._raw is not None:
            return self._t >= len(self._raw)
        return bool(self._wp.finished)

    @property
    def waypoints(self):
        """记录段的 EE 参考位姿（x,y,z,rx,ry,rz）；供 INIT_ALIGN_FULL_PLAN 遥测。"""
        return None if self._wp is None else self._wp.waypoints

    @property
    def converged(self):
        return self.finished

    @property
    def status(self):
        if self._mode == "approach":
            return f"approach:{getattr(self._approach, 'status', '?')}"
        return "REPLAY_DONE" if self.finished else "REPLAY_ACTIVE"


class OracleReadySelector:
    """用物体系 ready 位姿替换 Initial Alignment 的检索结果；参照体按技能取自当前阶段。"""

    # run_episode 在 select() 返回后会读这个属性。
    last_spatial_match = None

    def __init__(self, memory_paths: dict[str, Path], mode: str = "ik", replay: str = ""):
        self.memory_paths = memory_paths
        self.mode = mode
        self.replay = replay
        # Set when select() gives up so the caller can tell "no memory for this
        # phase" (per-phase loop: hand back to VLA) from a planning failure.
        self.last_skip_reason = None
        self.planner = None
        if mode == "controller":
            from memory_system.execute.curobo_planner import CuroboPlanner

            # joint_execution=True：走带时间信息的关节轨迹 + 闭环跟踪，即真机路径。
            self.planner = CuroboPlanner(joint_execution=True)
        _diag(
            f"selector constructed mode={mode} "
            f"memories={ {skill: path.name for skill, path in memory_paths.items()} }"
        )

    def resolve_task_name(self, task_name):  # noqa: ARG002
        return task_name

    def sequence_for_demo(self, *args, **kwargs):  # noqa: ARG002
        return None  # 不启用 skill-completion 遥测，与 baseline 保持一致

    @staticmethod
    def _phase(provided, env):
        """阶段来源：修复路径传入的 phase 优先；开局没有 phase 就用 BDDL 计划第一阶段，
        先剔除当前状态已满足的 TurnOn/TurnOff（K8 的灶台初始就是开的）。"""
        if provided is not None:
            return provided
        phases = _TASK_PLAN.get("phases")
        if not phases:
            raise RuntimeError("no phase given and no cached BDDL plan (get_libero_env hook)")
        return drop_presatisfied_turnon(phases, env)[0]

    def select(self, task_name, current_vae_main, current_ee_states, **kwargs):  # noqa: ARG002
        from memory_system.pointcloud_action.offline.extraction import object_frame

        env = kwargs["env"]
        phase = self._phase(kwargs.get("phase"), env)
        # BDDL 参数一般是场景 body 名；Place 的 region 名经 _resolve_reference
        # 回退到锚物体 body。评测用无分割的 OffScreenRenderEnv，instance_to_id
        # 恒空，没有实例解析这一步；body 找不到在 object_frame 抛错。
        raw_reference = str(
            phase.arguments.get("item" if phase.skill == "Pick" else "target", "")
        )
        if not raw_reference:
            raise KeyError(
                f"no reference argument for {phase.skill} {dict(phase.arguments)}"
            )
        reference = _resolve_reference(env, raw_reference)
        try:
            T_object_ee, memory_id, record = _load_ready_pose(
                phase.skill, phase.arguments, self.memory_paths
            )
        except KeyError as exc:
            # No memory for this phase (Open/Close have none).  A skip, not a
            # failure: the phase runs on the VLA.
            self.last_skip_reason = f"no_memory:{phase.skill}"
            _diag(f"no memory record for {phase.skill} {dict(phase.arguments)}: {exc}")
            return None
        T_world_object, _, _ = object_frame(env, reference)
        T_world_ee = T_world_object @ T_object_ee
        ee_states = np.concatenate(
            [T_world_ee[:3, 3], Rotation.from_matrix(T_world_ee[:3, :3]).as_rotvec()]
        ).astype(np.float32)
        _diag(
            f"mode={self.mode} phase={phase.skill} {dict(phase.arguments)} "
            f"reference={reference} memory={memory_id} "
            f"object_pos={np.round(T_world_object[:3, 3], 4).tolist()} "
            f"target_pos={np.round(T_world_ee[:3, 3], 4).tolist()}"
        )

        self.last_skip_reason = None
        if self.mode == "controller":
            return self._via_planner(
                env, ee_states, current_ee_states, kwargs, memory_id,
                record, T_world_object,
            )
        return self._via_teleport(env, T_world_ee, ee_states, memory_id)

    # ---- ik：直接写关节角 ----

    def _via_teleport(self, env, T_world_ee, ee_states, memory_id):
        from memory_system.pointcloud_action.execute.mujoco_ik import solve_ee_ik

        # solve_ee_ik 在失败时不会还原 sim（least_squares 迭代已经改写了 qpos），
        # 所以每次尝试前先存档、失败后还原；否则"跳过对齐"其实已把机器人挪走了。
        sim = env.env.sim
        saved_state = sim.get_state().flatten()
        # init_q 省略：solve_ee_ik 在 None 时自动从 sim 当前关节角取初值。
        q_sol = solve_ee_ik(sim, T_world_ee[:3, 3], T_world_ee[:3, :3])
        if q_sol is None:
            sim.set_state_from_flattened(saved_state)
            sim.forward()
            q_sol = solve_ee_ik(sim, T_world_ee[:3, 3], T_world_ee[:3, :3], max_iter=2000)
            _diag(f"IK retry(max_iter=2000) -> {q_sol is not None}")
        if q_sol is None:
            sim.set_state_from_flattened(saved_state)
            sim.forward()
            _diag("IK failed; sim state restored; skipping intervention")
            return None

        env.regenerate_obs_from_state(sim.get_state().flatten())
        # 复位 OSC 控制器缓存，否则下次 env.step 可能把机械臂拉回旧目标。
        robots = getattr(env, "robots", None) or env.env.robots
        robots[0].controller.update(force=True)
        _diag("teleported; handing back to VLA")
        return self._result(ee_states, 1, _IdleController(), None, memory_id)

    # ---- controller：cuRobo 规划 + 关节轨迹闭环执行 ----

    def _via_planner(self, env, ee_states, current_ee_states, kwargs, memory_id,
                     record, T_world_ref):  # noqa: ARG002
        depth = kwargs.get("main_depth")
        camera = kwargs.get("camera_params")
        joints = kwargs.get("joint_positions")
        if depth is None or camera is None or joints is None:
            _diag("controller mode: missing depth/camera/joint_positions; skipping")
            return None

        # Place 阶段固定走 lift-then-descend，不调 cuRobo：cuRobo 对放置目标既可能
        # 直接拒绝（stove: goal in collision），也可能规划成功但执行到不了（汤任务）。
        # 只有开了 replay 才这样做——非 replay 的调用方行为不变。
        force_lift = self.replay and str(record.get("skill")) in {"PlaceIn", "PlaceOn"}
        plan = None
        if not force_lift:
            plan = self.planner.plan(
                current_ee_states=current_ee_states,
                target_ee_states=ee_states,
                depth=depth,
                camera_params=camera,
                joint_positions=joints,
                gripper_joint_positions=kwargs.get("gripper_joint_positions"),
                robot_base_pose=kwargs.get("robot_base_pose"),
            )
        if plan is None and not self.replay:
            _diag("controller mode: cuRobo plan FAILED; skipping intervention")
            return None
        if plan is None:
            # 兜底仍然属于"接近段"的职责，只是换成不考虑碰撞的
            # LiftTranslateDescendController（三段：竖直→水平→竖直；curobo_planner
            # 的文档就写明调用方回退到 PoseController 那一类），因此 replay 依旧只
            # 回放记忆段，不承担任何走向 ready pose 的工作。代价：没有碰撞检查。
            from memory_system.execute.recovery import LiftTranslateDescendController

            seg_len = int(record.get("sequence_length", 64))
            steps = 64 + max(64, 2 * seg_len)
            _diag(
                ("place: forced lift-translate-descend" if force_lift
                 else "cuRobo plan FAILED; fallback approach")
                + f" (no collision check) segment_len={seg_len}"
            )
            return self._result(
                np.asarray(ee_states, dtype=np.float32),
                steps,
                _ReplayController(
                    LiftTranslateDescendController(
                        current_ee_states, ee_states, step_budget=steps
                    ),
                    record, T_world_ref, segment=self.replay, wp_max_steps=steps,
                ),
                None,
                memory_id,
            )

        planned = getattr(plan, "target_ee_states", None)
        target_ee = ee_states if planned is None else np.asarray(planned, dtype=np.float32)
        controller = plan.controller
        steps = plan.correction_steps
        joint_trajectory = plan.joint_trajectory
        if self.replay:
            # 回放模式：走完 ready 位的关节轨迹后继续执行记忆段；joint_trajectory
            # 必须置 None，否则 runtime 会自己包 LiberoJointTrajectoryController，
            # 我们的复合 controller 就不会被使用。
            # 接近段的夹爪命令由技能语义决定，不能照抄 gripper_sequence[0]：
            # 那是该记忆段的【第一个夹爪命令】而非起始状态。TurnOn 段的第一步
            # 就是"合上旋钮"，照抄会让接近段全程夹紧、手指跨不到旋钮两侧；
            # Pick 段第一步是"合上物体"，同理。只有 Place 类阶段开始时手上确实
            # 有东西，必须一直夹着——否则接近途中就把物体丢在路上了。
            segment_gripper = (
                1.0 if str(record.get("skill")) in {"PlaceIn", "PlaceOn"} else -1.0
            )
            approach = plan.controller
            if approach is None and plan.joint_trajectory is not None:
                # joint_execution=True 时 plan 只有轨迹没有执行器，这里复刻
                # runtime 的包装（run_libero_eval._maybe_start_repair_alignment）。
                from cosmos_policy.experiments.robot.libero.libero_joint_control import (
                    LiberoJointTrajectoryController,
                )

                approach = LiberoJointTrajectoryController(
                    env, plan.joint_trajectory, target_ee, segment_gripper
                )
            seg_len = int(record.get("sequence_length", 64))
            steps = plan.correction_steps + max(64, 2 * seg_len)
            controller = _ReplayController(
                approach, record, T_world_ref, segment=self.replay,
                wp_max_steps=steps,
            )
            joint_trajectory = None
            _diag(
                f"replay mode={self.replay} segment_len={seg_len} "
                f"approach_gripper={segment_gripper:+.0f} steps<= {steps}"
            )
        _diag(
            f"controller mode: planned steps={plan.correction_steps} "
            f"joint_trajectory={plan.joint_trajectory is not None}"
        )
        return self._result(
            target_ee, steps, controller, joint_trajectory,
            memory_id,
        )

    def _result(self, ee_states, correction_steps, controller, joint_trajectory, memory_id):
        return InitialAlignmentResult(
            target=RecoveryTarget(
                demo_ids=(memory_id,),
                target_ee_states=np.asarray(ee_states, dtype=np.float32),
                similarity=1.0,
                frame=0,
                z_lift=0.0,
            ),
            correction_steps=correction_steps,
            controller=controller,
            joint_trajectory=joint_trajectory,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("none", "ik", "controller"),
        default="ik",
        help="none=不干预（裸基线）；ik=直接设关节角（上界）；"
        "controller=cuRobo规划+关节轨迹（可迁移路径）",
    )
    parser.add_argument(
        "--language-from-bddl",
        action="store_true",
        help="用 BDDL 的 (:language) 作为指令。_task/_lan 变体必须开，swap/base 是 no-op",
    )
    parser.add_argument("--suite", default="libero_10_swap")
    parser.add_argument("--task", default="", help="task_filter；留空跑整套件")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY))
    parser.add_argument("--out", default=str(REPO / "experiments/liberopro/oracle"))
    args = parser.parse_args()

    # _task / _lan 变体不改文件名，必须靠 BDDL 取指令；忘了开会静默跑错任务。
    if args.suite.endswith(("_task", "_lan")) and not args.language_from_bddl:
        _diag(
            f"WARNING: suite={args.suite} 但未开 --language-from-bddl；"
            "VLA 将收到 base 的旧指令，结果不可解读"
        )

    out_dir = Path(args.out)
    (out_dir / "local_logs").mkdir(parents=True, exist_ok=True)
    global _DEBUG_LOG
    _DEBUG_LOG = out_dir / "oracle_debug.log"
    _diag(
        f"=== run start mode={args.mode} "
        f"language_from_bddl={args.language_from_bddl} "
        f"task={args.task or '<all>'} trials={args.trials} ==="
    )

    # 干预需要阶段计划（开局对齐第一阶段）；mode none 只在显式要求语言修正时打补丁。
    if args.mode != "none" or args.language_from_bddl:
        _patch_get_libero_env(args.language_from_bddl, capture_plan=args.mode != "none")

    model_dir = Path(args.model)
    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(model_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        dataset_stats_path=str(model_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(model_dir / "libero_t5_embeddings.pkl"),
        task_suite_name=args.suite,
        unnorm_key="libero_10",
        task_filter=args.task,
        num_trials_per_task=args.trials,
        seed=args.seed,
        local_log_dir=str(out_dir / "local_logs"),
        run_id_note=f"oracle-{args.mode}",
        # none 模式走原生 baseline 路径，不建 selector、不触发任何对齐代码。
        enable_initial_alignment=(args.mode != "none"),
    )

    if args.mode != "none":
        # ---- 打补丁（仅本进程，见模块 docstring 的隔离说明）----
        R._create_initial_alignment_selector = lambda _cfg: OracleReadySelector(
            _memory_paths(Path(args.memory)), args.mode
        )
    eval_libero.__wrapped__(cfg)


if __name__ == "__main__":
    main()
