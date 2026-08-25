# Initial Alignment 独立设计说明

本文说明当前 `initial alignment` 是如何与原有 `ExecutionMonitor` / phase / feasible recovery 逻辑解耦，独立运行的。

## 目标

在 episode 最开始、第一个 policy action 执行之前，直接根据 main-camera VAE 相似度，从第一个 phase 的 ready-pose memory 中选一个目标，并把机械臂移动到该 ready pose 上方 2cm 的位置。

这个机制只做一次，不做在线 recovery，不进入 verifier 状态机。

## 核心原则

- 判断逻辑独立：不依赖 `PHASE_ERROR`、`FEASIBLE_ERROR` 或 `ExecutionMonitor`。
- 底层能力复用：复用 `PoseController` 和现有 correction 执行通道。
- 严格分离：开启 `enable_initial_alignment` 后，强制关闭所有 verifier / recovery。

## 如何跳过之前的逻辑

1. `PolicyEvalConfig` 新增 `enable_initial_alignment`。
2. `validate_config()` 中，如果 `enable_initial_alignment=True`，会强制把以下开关全部置为 `False`：
   - `enable_phase_verifier`
   - `enable_phase_3d`
   - `enable_feasible_3d`
   - `enable_phase_recovery`
   - `enable_feasible_recovery`

   因此不会创建 `ExecutionMonitor`，也不会进入 phase / feasible recovery 分支。

3. `run_episode()` 中，`_phase_check_and_recover()` 只有在 `episode_execution_monitor is not None` 时才会执行；在 initial alignment 模式下该对象为 `None`，所以不会进入原有检测逻辑。

4. initial alignment 使用独立的 `_maybe_start_initial_alignment()` one-shot gate：
   - 在第一次 model query 之后、第一个 policy action 执行之前触发；
   - 只触发一次；
   - 不读取/写入 `intervention_reason`；
   - 不修改 `phase_index`；
   - 不调用 `mark_feasible_after_recovery()`；
   - 不消耗在线 correction budget。

## 执行流程

```text
env.reset()
-> 10 步 warmup
-> 第一次 model query，得到初始 main VAE
-> 触发 initial alignment
   -> InitialAlignmentSelector 选择第一个 phase 的 ready pose
   -> 目标 z + 0.02m（2cm）
   -> 清空第一个 action chunk
   -> 通过现有 _correction_active 通道执行 PoseController 移动
-> 移动完成
-> 恢复正常 policy rollout
```

## cuRobo 无碰撞规划（可选，默认开启）

- 新增配置 `enable_collision_aware_initial_alignment`，默认 `True`。
- 当 `enable_initial_alignment=True` 且该配置为 `True` 时：
  - `run_libero_eval.py` 会开启 `agentview` 深度图；
  - 构建 `phase_camera_params`；
  - 将 `main_depth` / `camera_params` / `joint_positions` 传给 `InitialAlignmentSelector.select()`。
- `InitialAlignmentSelector` 会先尝试 `CuroboPlanner`：
  - 从 RGB-D 生成点云；
  - 简单剔除机械臂附近点；
  - 用球体近似障碍物并调用 cuRobo 规划无碰撞轨迹；
  - 规划成功时使用 `WaypointPoseController` 闭环执行；
  - 规划失败或缺少输入时回退到原 `PoseController`，并记录 warning 日志。
- 新增文件：`memory_system/execute/curobo_planner.py`

## 关键文件

- `memory_system/execute/initial_alignment.py`
  - `InitialAlignmentSelector`
  - 自己加载 phase plan 和 feasible/ready memory
  - 不依赖 `ExecutionMonitor`
  - 使用 main-camera VAE 相似度
  - 使用 demo 执行顺序中的第一个 phase，而不是简单按 `planner_step_id` 排序

- `cosmos_policy/experiments/robot/libero/run_libero_eval.py`
  - 配置项 `enable_initial_alignment`
  - `validate_config()` 强制关闭其他 verifier/recovery
  - `_create_initial_alignment_selector()`
  - `_maybe_start_initial_alignment()`
  - 复用现有 `_correction_active` 执行通道

- `scripts/run_libero_smoke_test.py`
  - 读取 `COSMOS_INITIAL_ALIGNMENT` 环境变量

- `scripts/run_libero10_robotinit_20.sh`
- `scripts/run_libero10_other9_robotinit_20.sh`
- `scripts/run_libero10_robotinit_low_success_20.sh`
  - 支持 `COSMOS_INITIAL_ALIGNMENT`
  - 开启时在 launcher 层也关闭其他 verifier/recovery

## 已修复的问题

1. **z 累加 bug**
   - 之前直接修改 memory 中的 `ee_states`，导致每个 episode 目标 z 不断增加 2cm。
   - 现在先 copy 再修改，保证每次都是“原始 ready pose + 2cm”。

2. **第一个 phase 顺序 bug**
   - 之前使用 `load_phase_plans()` 按 `planner_step_id` 排序，但某些任务（如 `KITCHEN_SCENE8`）的 demo 执行顺序与 step id 不一致。
   - 现在 `InitialAlignmentSelector` 直接从 segment manifest 中统计 demo 执行顺序里的第一个 phase，确保匹配训练轨迹。
