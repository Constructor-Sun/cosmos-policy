"""Oracle ready-pose alignment experiment for LIBERO-PRO (一次性实验，可整份删除).

目的
    验证「把机器人放到物体系 ready pose，再交回 VLA」能否救回 swap / task 的失败。
    物体系 ready 位姿取自 pointcloud_action memory（LIBERO-90 构建，不是 libero_10
    memory），经 T_world_object 映射到当前场景，因此不依赖物体在场景中的绝对位置。

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
    （get_libero_env / validate_config / _create_initial_alignment_selector），
    patch 仅存在于本进程内存；其他入口各自起进程，行为逐位不变。

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


# --------------------------------------------------------------------- 语言修正


def _bddl_language(task) -> str:
    """读 BDDL 里真正的 (:language ...)；失败则回退到 benchmark 从文件名推的值。"""
    from libero.libero import get_libero_path

    try:
        path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        match = re.search(r"\(:language\s+(.+?)\)", path.read_text())
        if match:
            return match.group(1).strip()
        _diag(f"no (:language) block in {path.name}; keeping benchmark language")
    except Exception as exc:  # noqa: BLE001 - 回退即可，不阻断评测
        _diag(f"bddl language read failed for {task.name}: {exc}")
    return task.language


def _patch_language() -> None:
    """把 get_libero_env 返回的指令换成 BDDL 里真实的 (:language)。"""
    original = R.get_libero_env

    def get_libero_env(task, *args, **kwargs):
        env, _benchmark_language = original(task, *args, **kwargs)
        language = _bddl_language(task)
        _diag(f"language task={task.name} -> {language!r}")
        return env, language

    R.get_libero_env = get_libero_env


# --------------------------------------------------------------------- memory


def _object_type(name: str) -> str:
    """moka_pot_2 -> moka_pot。

    memory 存的是【物体系】ready 位姿，实例编号不参与匹配；真系统的检索按点云
    形状走，同型物体天然命中。所以 oracle 也用同型回退，而不是要求实例名精确一致
    （否则 KITCHEN_SCENE8 的 moka_pot_2 在 memory 里找不到，直接崩）。
    """
    return re.sub(r"_\d+$", "", str(name))


def _load_object_ready_pose(item: str, memory_path: Path) -> tuple[np.ndarray, str]:
    """取该物体的物体系 ready 位姿 T_object_ee_ready (4x4)：实例名精确优先，否则同型。"""
    import torch

    records = torch.load(memory_path, map_location="cpu", weights_only=False)["records"]
    want_type = _object_type(item)
    same_type: tuple[np.ndarray, str] | None = None
    for record in records:
        if record.get("skill") != "Pick":
            continue
        got = record.get("arguments", {}).get("item")
        if got is None:
            continue
        matrix = np.asarray(record["T_object_ee_ready"], dtype=np.float64).reshape(4, 4)
        memory_id = str(record["memory_id"])
        if got == item:
            return matrix, memory_id
        if same_type is None and _object_type(got) == want_type:
            same_type = (matrix, memory_id)
    if same_type is not None:
        _diag(f"{item!r} 无精确记录，回退到同型 {want_type!r}: {same_type[1]}")
        return same_type
    raise KeyError(f"no Pick record for item={item!r} (type {want_type!r}) in {memory_path}")


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


class OracleReadySelector:
    """用物体系 ready 位姿替换 Initial Alignment 的检索结果。"""

    # run_episode 在 select() 返回后会读这个属性。
    last_spatial_match = None

    def __init__(self, item: str, memory_path: Path, mode: str = "ik"):
        self.item = item
        self.mode = mode
        self.T_object_ee, self.memory_id = _load_object_ready_pose(item, memory_path)
        self.planner = None
        if mode == "controller":
            from memory_system.execute.curobo_planner import CuroboPlanner

            # joint_execution=True：走带时间信息的关节轨迹 + 闭环跟踪，即真机路径。
            self.planner = CuroboPlanner(joint_execution=True)
        _diag(f"selector constructed item={item} memory={self.memory_id} mode={mode}")

    def resolve_task_name(self, task_name):  # noqa: ARG002
        return task_name

    def sequence_for_demo(self, *args, **kwargs):  # noqa: ARG002
        return None  # 不启用 skill-completion 遥测，与 baseline 保持一致

    def select(self, task_name, current_vae_main, current_ee_states, **kwargs):  # noqa: ARG002
        from memory_system.pointcloud_action.offline.extraction import object_frame

        env = kwargs["env"]
        T_world_object, _, _ = object_frame(env, self.item)
        T_world_ee = T_world_object @ self.T_object_ee
        ee_states = np.concatenate(
            [T_world_ee[:3, 3], Rotation.from_matrix(T_world_ee[:3, :3]).as_rotvec()]
        ).astype(np.float32)
        _diag(
            f"mode={self.mode} object_pos={np.round(T_world_object[:3, 3], 4).tolist()} "
            f"target_pos={np.round(T_world_ee[:3, 3], 4).tolist()}"
        )

        if self.mode == "controller":
            return self._via_planner(env, ee_states, current_ee_states, kwargs)
        return self._via_teleport(env, T_world_ee, ee_states)

    # ---- ik：直接写关节角 ----

    def _via_teleport(self, env, T_world_ee, ee_states):
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
        return self._result(ee_states, 1, _IdleController(), None)

    # ---- controller：cuRobo 规划 + 关节轨迹闭环执行 ----

    def _via_planner(self, env, ee_states, current_ee_states, kwargs):  # noqa: ARG002
        depth = kwargs.get("main_depth")
        camera = kwargs.get("camera_params")
        joints = kwargs.get("joint_positions")
        if depth is None or camera is None or joints is None:
            _diag("controller mode: missing depth/camera/joint_positions; skipping")
            return None

        plan = self.planner.plan(
            current_ee_states=current_ee_states,
            target_ee_states=ee_states,
            depth=depth,
            camera_params=camera,
            joint_positions=joints,
            gripper_joint_positions=kwargs.get("gripper_joint_positions"),
            robot_base_pose=kwargs.get("robot_base_pose"),
        )
        if plan is None:
            _diag("controller mode: cuRobo plan FAILED; skipping intervention")
            return None

        planned = getattr(plan, "target_ee_states", None)
        target_ee = ee_states if planned is None else np.asarray(planned, dtype=np.float32)
        _diag(
            f"controller mode: planned steps={plan.correction_steps} "
            f"joint_trajectory={plan.joint_trajectory is not None}"
        )
        return self._result(
            target_ee, plan.correction_steps, plan.controller, plan.joint_trajectory
        )

    def _result(self, ee_states, correction_steps, controller, joint_trajectory):
        return InitialAlignmentResult(
            target=RecoveryTarget(
                demo_ids=(self.memory_id,),
                target_ee_states=np.asarray(ee_states, dtype=np.float32),
                similarity=1.0,
                frame=0,
                z_lift=0.0,
            ),
            correction_steps=correction_steps,
            controller=controller,
            joint_trajectory=joint_trajectory,
        )


def _relax_suite_guard(original_validate):
    """保留原校验，只在调用时把 suite 名伪装成 libero_10（memory 仅支持该套件）。"""

    def validate_config(cfg):
        real = cfg.task_suite_name
        cfg.task_suite_name = R.TaskSuite.LIBERO_10
        try:
            original_validate(cfg)
        finally:
            cfg.task_suite_name = real

    return validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", default="", help="目标物体实例名；--mode none 时可省略")
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

    if args.mode != "none" and not args.item:
        parser.error("--item is required unless --mode none")

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
        f"=== run start item={args.item or '<none>'} mode={args.mode} "
        f"language_from_bddl={args.language_from_bddl} "
        f"task={args.task or '<all>'} trials={args.trials} ==="
    )

    if args.language_from_bddl:
        _patch_language()

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
        run_id_note=f"oracle-{args.mode}-{args.item or 'none'}",
        # none 模式走原生 baseline 路径，不建 selector、不触发任何对齐代码。
        enable_initial_alignment=(args.mode != "none"),
    )

    if args.mode != "none":
        # ---- 打补丁（仅本进程，见模块 docstring 的隔离说明）----
        R.validate_config = _relax_suite_guard(R.validate_config)
        R._create_initial_alignment_selector = lambda _cfg: OracleReadySelector(
            args.item, Path(args.memory), args.mode
        )
    eval_libero.__wrapped__(cfg)


if __name__ == "__main__":
    main()
