# Initial Alignment Only 在线清理计划

## 0. 执行结果（2026-08-29）

本计划已经执行完成：

- 旧 Phase / Feasible / Completion / `ExecutionMonitor` runtime 与专用 recovery selector 已删除；
- Eval、smoke runner 和 launcher 的旧 feature flag、对象构造和运行分支已清除；
- Initial Alignment 保持第一次 policy query 后、第一条 policy action 前触发；
- `scripts/run_libero10_robotinit_20.sh` 明确保留 `Initial Alignment=1`、`joint execution=0`、`URDF filter=0` 的原默认行为；
- `memory_system/offline/` 的 Git diff 为空；
- Python 编译检查、6 个 launcher 的 `sh -n` 检查和核心回归测试通过；
- 核心回归测试结果为 `29 passed, 3 warnings`，warnings 均为 robosuite 已有弃用提示；
- 已正式运行独立目录下的 1-case robot-init smoke：选择 `demo_13`，similarity `0.125`，执行 48 步 waypoint/OSC alignment，状态 `completed`，episode 成功率 `1/1`；
- smoke 日志没有 Phase、Feasible、Completion、Verifier 或 wrong-grasp 记录。

本次 smoke 输出位于：

```text
experiments/libero10_robotinit_cleanup_smoke/
rollouts/cleanup_smoke_20260829/
```

该单例回归验证了 launcher 和 Initial Alignment 主通路可以运行，不代表 20-case 成功率结论。

## 1. 背景与目标

当前在线执行链同时包含 Initial Alignment、Phase、Feasible、Skill Completion 和多类 Recovery，导致 `memory_system/execute` 与 `run_libero_eval.py` 的状态和分支过多。

后续路线将以 Initial Alignment 为当前唯一正式干预通路，并在此基础上重新实现：

- 按具体物体位置生成或修正对齐目标；
- 面向每种 skill 的结束判定；
- skill 之间的 planner continuation；
- 必要时重新对齐、重新规划或恢复。

本阶段只清理已经放弃的旧在线判断通路，不实现新的 Skill Check，也不改变 Initial Alignment 的任何运行行为。

## 2. 已确认的决策

### 2.1 删除整个旧在线状态机

以下逻辑不再保留为正式路径：

- 2D/3D Phase 判断；
- 2D/3D Feasible 判断；
- `ExecutionMonitor` 的 `PHASE_CHECK -> FEASIBLE_CHECK -> COMPLETION_CHECK` 状态机；
- 当前 `skill_completion` 实现；
- Phase/Feasible 专用 recovery selector；
- wrong-grasp、phase-error、feasible-error 等依赖旧状态机的恢复编排。

旧版本已经提交到 Git，需要时从历史恢复，不在当前工作树保留 legacy runtime 分支。

### 2.2 严格保持当前 Initial Alignment 行为

本阶段不改变：

- 第一次 policy query 后、第一条 policy action 执行前触发；
- 使用 main-camera VAE 选择第一阶段 ready pose；
- 没有 similarity threshold，存在候选时选择最高相似度项；
- 对 memory ready pose 的世界坐标 z 增加 `0.02 m`；
- 当前 RGB-D、相机参数、关节位置、夹爪关节位置和机器人 base pose 输入；
- collision-aware cuRobo 规划参数；
- waypoint/OSC 和 joint-trajectory 两种现有执行能力；
- planner 失败时当前非 joint 模式下的 `PoseController` fallback；
- alignment 开始时清空 action queue；
- alignment 结束后重新 query policy；
- gripper command 保持和 Initial Alignment debug logging。

当前 `scripts/run_libero10_robotinit_20.sh` 默认没有启用 `COSMOS_CUROBO_JOINT_EXECUTION`，因此默认行为是：

```text
cuRobo planning
-> EE waypoints
-> WaypointPoseController
-> OSC action
```

本阶段不得顺便切换为 strict joint execution。

### 2.3 暂不移动或重命名文件

为减少同时变化的维度，本阶段保留现有文件位置和公开名称，例如：

- `execute/initial_alignment.py`
- `execute/curobo_planner.py`
- `execute/curobo_trajectory.py`
- `execute/surface_obstacles.py`
- `execute/urdf_depth_filter.py`
- `execute/recovery/controller.py`
- `execute/recovery/retrieval.py`

`InitialAlignmentSelector`、`FeasibleRecoveryMemory`、`PhaseSpec` 等名称本阶段也不修改。目录重组和语义重命名在行为回归通过后另行进行。

## 3. 明确不修改的范围

### 3.1 `offline/` 完全不动

本阶段不修改、删除或重命名 `memory_system/offline/` 下的任何文件，包括：

- planner 和 predicate planner；
- demo segment/boundary 标注；
- phase/wrist/recovery/ready3d artifact builder；
- ready-frame 更新脚本。

### 3.2 Artifact 与数据格式不动

不修改：

- 现有 `.pt`、`.json` 文件；
- artifact format 字符串；
- `feasible_recovery_targets.pt` 的字段和内容；
- Initial Alignment 当前使用的 memory 检索结果；
- `artifacts.py` 中现有 loader 的数据兼容性。

旧在线 verifier 不再使用某些 artifact，并不等于本阶段删除这些离线数据。未来 Skill Check 的 memory schema 确定后，再单独清理数据层。

### 3.3 不实现未来功能

本阶段不新增：

- `SkillAlignmentSelector`；
- `SkillCheck3D`；
- Pick/Place/Open/Close/TurnOn 的新结束规则；
- 新 planner orchestrator；
- 基于目标相对坐标系的 ready-pose memory；
- 新 recovery 策略。

## 4. `memory_system` 清理范围

### 4.1 删除文件

删除：

```text
memory_system/execute/phase.py
memory_system/execute/phase3d.py
memory_system/execute/feasible.py
memory_system/execute/feasible3d.py
memory_system/execute/execution_monitor.py
memory_system/execute/skill_completion/
memory_system/execute/recovery/selectors.py
```

删除对应的 `__pycache__` 不属于源码修改要求；运行环境可以自然重新生成或由后续环境维护清理。

### 4.2 精简 `execute/plan.py`

保留：

- `PhaseSpec`；
- `_argument_key()`；
- `load_phase_plans()`；
- 当前按 `planner_step_id` 排序和输入校验行为。

删除：

- `PhaseMonitorResult`；
- `PhaseMonitor`；
- `PhaseVerifier` 和 Phase status 依赖；
- current/next target 同时比较和 switch-evidence 逻辑。

这样 `InitialAlignmentSelector` 仍可按当前方式加载计划，同时 `plan.py` 不再依赖已删除的 Phase verifier。

### 4.3 精简 `execute/recovery/__init__.py`

删除：

- `PhaseRecoverySelector`；
- `FeasibleRecoverySelector`；
- 对 `recovery/selectors.py` 的 import/export。

保留：

- `PoseController`；
- `token()`；
- `similarity()`；
- `InitialAlignmentSelector` 当前直接依赖的通用 retrieval 能力。

`recovery/controller.py` 和 `recovery/retrieval.py` 本阶段不移动。若其中存在不影响 Initial Alignment 的未使用 helper，可在目录重组阶段再处理。

### 4.4 精简 `execute/__init__.py`

移除所有已删除模块的 import 和 `__all__` 项，包括：

- Phase/Feasible status 与 verifier；
- `ExecutionMonitor`；
- completion verifier；
- Phase/Feasible recovery selector；
- `PhaseMonitor`。

保留 Initial Alignment 当前需要的公开入口：

- `InitialAlignmentSelector`、`InitialAlignmentResult`；
- `CuroboPlanner`、`PlanResult`；
- `JointTrajectoryPlan`；
- `PhaseSpec`、`load_phase_plans()`；
- `PoseController`。

### 4.5 共享类型暂不做语义重构

`types.py` 中的旧 result dataclass 可以在引用审计后删除，但本阶段不做名称替换或未来接口设计。

必须保留：

- `CameraParams`；
- `RecoveryTarget`；
- Initial Alignment 及 cuRobo 当前使用的类型；
- 未来 RGB-D 几何仍可能复用的 `VerifierObservation` 和 `TargetGeometry`。

若 `PhaseResult`、`FeasibleResult`、`CompletionResult`、`RecoveryRequest` 在删除旧通路后确认没有任何源码引用，可以作为纯 dead-code cleanup 删除；该删除不得引入新的替代抽象。

## 5. 与 Eval 接线清理的边界

删除 `memory_system` 模块前，必须同步确保调用层不再 import 或构造这些对象。相关文件为：

```text
cosmos_policy/experiments/robot/libero/run_libero_eval.py
scripts/run_libero_smoke_test.py
scripts/run_libero10_robotinit_20.sh
```

该调用层清理属于配套工作，但不允许改变 Initial Alignment 行为。

调用层必须保留：

- 首次 model forward 的 VAE capture；
- `_maybe_start_initial_alignment()` 的触发边界；
- main metric depth 和 camera params 构造；
- alignment controller 的逐步执行、`observe()`、`finished` 和 `close()`；
- waypoint/OSC 与 joint-trajectory 分支；
- alignment 期间固定 gripper command；
- action queue 清空与结束后的 policy re-query。

不得把服务 Initial Alignment 的 `_correction_*` 状态与旧 recovery 分支一起整段删除。应先抽取或保留 Initial Alignment 所需状态，再删除 Phase/Feasible/wrong-grasp 分支。

## 6. 推荐实施顺序

### Step 0：记录行为基线

在修改前记录至少一个固定 robot-init episode 的：

- 选中的 `demo_id`；
- VAE similarity；
- `target_ee_states`；
- controller 类型；
- correction steps；
- cuRobo waypoints 或 joint trajectory；
- 第一个非 dummy action；
- alignment 完成状态和最终 EEF residual；
- alignment 结束后的第一次 policy query 时刻。

### Step 1：解除内部依赖

1. 精简 `execute/plan.py`，使其不再 import `phase.py`；
2. 精简 `execute/recovery/__init__.py`，使其不再 import selectors；
3. 精简 `execute/__init__.py`；
4. 保证 `initial_alignment.py` 单独 import 时仍可工作。

### Step 2：清理调用层引用

1. 从 Eval 配置中删除 Phase/Feasible/Recovery 字段；
2. 从 smoke runner 和 launcher 删除对应环境变量；
3. 删除 ExecutionMonitor 和 selector 构造；
4. 将共享 correction 状态缩减为 Initial Alignment 所需部分；
5. 保持当前 Initial Alignment 输入、动作和结束行为。

### Step 3：删除旧模块

完成全仓引用检查后，删除第 4.1 节列出的文件和目录。

### Step 4：静态验证

- 全仓不再 import 已删除模块；
- `memory_system`、`memory_system.execute` 和 Initial Alignment 相关模块可以正常 import；
- Python 编译检查通过；
- Initial Alignment、cuRobo trajectory、surface obstacle、URDF filter 相关测试通过；
- 不存在为了通过 import 而加入的 legacy stub。

### Step 5：运行时回归

先运行单个固定 case，再运行原 20-case launcher。

## 7. 验收标准

### 7.1 结构验收

- `memory_system/offline/` 没有任何修改；
- 旧 Phase/Feasible/Completion/ExecutionMonitor 源码已删除；
- Initial Alignment 不再依赖这些模块；
- 当前工作树没有 legacy verifier feature flag 或运行时分支；
- 尚未引入新的 Skill Check 或 orchestrator 空壳。

### 7.2 行为验收

对相同输入和固定 episode，清理前后必须满足：

- 选择相同 ready-pose memory/demo；
- similarity 和 `target_ee_states` 一致；
- controller 类型和执行模式一致；
- waypoint/joint trajectory 在允许的数值误差内一致；
- 第一个非 dummy action 仍是 Initial Alignment action；
- alignment 仍在第一条 policy action 前开始；
- alignment 完成后 action queue 为空并重新 query policy；
- planner 失败时保持当前 fallback/失败语义；
- 不出现 Phase、Feasible、Completion 或旧 Recovery 日志；
- `scripts/run_libero10_robotinit_20.sh` 可以继续执行 Initial Alignment baseline。

### 7.3 明确不作为本阶段验收内容

- 不要求提高成功率；
- 不调 cuRobo 或 controller 参数；
- 不修复新的物体相对对齐能力；
- 不实现任何 skill 完成判定；
- 不切换默认 joint execution 模式。

## 8. 完成后的预期状态

清理完成后，当前在线系统应只有一条正式干预路径：

```text
reset / warmup
-> first policy query and main VAE capture
-> InitialAlignmentSelector
-> cuRobo plan or current fallback
-> alignment execution
-> clear/requery policy
-> normal rollout
```

该状态将作为后续逐步加入以下能力的干净基线：

```text
current SkillStep
-> object-relative alignment
-> policy execution
-> skill-specific 3D completion check
-> next skill / re-align / replan / recovery
```

后续目录重组、类重命名和新接口设计必须在本阶段行为回归通过后单独进行。
