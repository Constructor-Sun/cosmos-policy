# Initial Alignment 当前实现

本文是清理旧 Phase / Feasible / Completion 在线状态机之后，Initial Alignment 的主说明文档。

## 1. 当前定位

Initial Alignment 是当前唯一正式的在线干预通路。它在每个 episode 的第一条 policy action 执行前，根据首次策略推理捕获的 main-camera VAE 表征，从第一段 ready-pose memory 中选择目标，并将机器人移动到该目标附近。

它每个 episode 最多尝试一次，不依赖 Phase、Feasible、Completion 或 `ExecutionMonitor`，也不承担 skill 结束判断和 planner continuation。

## 2. 不变的触发与交接行为

```text
env.reset()
-> 10 个 dummy step，使场景稳定
-> 第一次 policy query，同时捕获 main-camera VAE
-> InitialAlignmentSelector 选择 ready pose
-> 清空尚未执行的 policy action queue
-> 当前 timestep 立即执行第一条 alignment action
-> alignment 完成或停止
-> 重新 query policy
-> 正常 rollout
```

关键约束：

- `_policy_step_count == 0` 时才允许触发；
- `_initial_align_attempted` 保证每个 episode 只尝试一次；
- 第一次 query 只用于取得视觉表征，其 action chunk 不先作用于环境；
- alignment 开始时调用 `action_queue.clear()`；
- alignment 期间保持开始前捕获的 gripper command；
- controller 的 `observe()`、`finished`、`close()` 语义保持不变；
- alignment 结束后 queue 为空，因此 policy 会重新推理。

## 3. 目标选择

`InitialAlignmentSelector` 读取：

- `skill_memory_test/libero_10/segments_ready_fixed16.json`；
- `skill_memory_test/libero_10/feasible_recovery_targets.pt`。

第二个文件名是历史 artifact 名称，目前仍作为 Initial Alignment 的 ready-pose memory 使用；本次清理不修改 artifact 名称、格式或内容。

选择流程：

1. 从 segment manifest 统计 demo 实际执行顺序中的第一段，而不是仅按 `planner_step_id` 猜测第一段；
2. 用 main-camera VAE cosine similarity 检索同一 task/skill/arguments 下的候选；
3. 当前没有 similarity threshold，存在候选时选择最高相似度项；
4. 对选中 memory pose 做 copy，再把世界坐标 z 增加 `0.02 m`，避免跨 episode 累加；
5. 返回目标 pose、demo id、similarity 和相应 controller/trajectory。

## 4. cuRobo 规划

当 `enable_collision_aware_initial_alignment=True` 时，Eval 会开启 agentview depth，并向 selector 传入 metric depth、相机参数、当前关节位置、夹爪关节位置和机器人 base pose。

`CuroboPlanner` 的主要过程是：

1. 将单帧 RGB-D 反投影到机器人基座坐标系；
2. 用机器人 collision spheres 移除自身深度；
3. 在 512 个 cuboid 预算内，组合全场景覆盖点、目标附近高密度点和冲突 mandatory points；
4. 从当前 7 维关节状态规划到 memory EE target；
5. 对候选关节轨迹加密，并用完整过滤后点云复核整条机器人扫掠轨迹；
6. 冲突时补充 mandatory obstacles，对同一个目标重新规划；
7. 同目标仍不可行时，搜索不超过 `80 mm` 的最小目标回退。

当前安全边界仍是单帧 RGB-D 的可见空间；不可见障碍、标定误差和机器人模型误差不在保证范围内。

## 5. 两种现有执行模式

### 默认 waypoint/OSC 模式

`enable_curobo_joint_execution=False` 时：

```text
cuRobo planning
-> EE waypoints
-> WaypointPoseController
-> OSC action
```

如果规划失败或缺少规划输入，当前行为允许回退到直接 `PoseController`。`scripts/run_libero10_robotinit_20.sh` 明确保持该模式为默认：

```text
COSMOS_INITIAL_ALIGNMENT=1
COSMOS_CUROBO_JOINT_EXECUTION=0
COSMOS_URDF_ROBOT_FILTER=0
```

### strict joint-trajectory 模式

`enable_curobo_joint_execution=True` 时，保留带时间信息的 cuRobo joint trajectory，并在 LIBERO 中临时切换到 `JOINT_POSITION` controller 闭环跟踪；结束后恢复 OSC controller。该模式下不接受无碰撞保证的直接 pose fallback。

`run_libero10_other9_robotinit_20.sh` 和 `run_libero10_robotinit_low_success_20.sh` 当前仍默认启用该模式。

## 6. 当前代码边界

- `memory_system/execute/initial_alignment.py`：memory 加载、首段统计、VAE 检索和目标生成；
- `memory_system/execute/curobo_planner.py`：点云障碍、cuRobo 求解、轨迹复核和最小回退；
- `memory_system/execute/surface_obstacles.py`：表面采样与完整点云碰撞检查；
- `memory_system/execute/curobo_trajectory.py`：时间化 joint trajectory 和回退搜索；
- `memory_system/execute/urdf_depth_filter.py`：可选 URDF 机器人深度过滤；
- `memory_system/execute/recovery/controller.py`：现有 pose controller；
- `memory_system/execute/recovery/retrieval.py`：Initial Alignment 使用的通用检索 helper；
- `cosmos_policy/experiments/robot/libero/run_libero_eval.py`：首次 query、一次性接管、动作执行和 policy 交接；
- `cosmos_policy/experiments/robot/libero/libero_joint_control.py`：joint trajectory 闭环执行和 controller 恢复。

旧 Phase / Feasible / Completion / ExecutionMonitor runtime 已删除。`recovery/` 目录名和部分历史类名本阶段暂不移动，避免把结构重命名与行为清理混在一起。

## 7. 后续扩展边界

下一阶段可以在这条稳定通路旁新增明确接口，但不应重新把多阶段判断塞回 Initial Alignment：

```text
current SkillStep
-> 根据具体物体位置做 object-relative alignment
-> policy execution
-> skill-specific 3D Skill Check
-> planner 选择 next skill
-> 必要时 re-align / replan / recovery
```

Initial Alignment 只负责 episode 开始前的一次对齐；未来的 skill check、planner continuation 和物体相对判断应拥有独立状态与测试。
