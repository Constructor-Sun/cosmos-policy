# cuRobo Initial Alignment 关节轨迹执行计划

## 1. 目标与结论

本计划用于替换当前的执行链路：

```text
cuRobo 关节轨迹
-> FK 转成 EE waypoints
-> WaypointPoseController 逐点 P 控制
-> waypoint 数量被当作 correction step 预算
```

新的主链路为：

```text
单帧 RGB-D 碰撞世界
-> cuRobo MotionGen 生成带时间信息的无碰撞关节轨迹
-> 按环境控制周期重采样
-> 关节空间闭环执行
-> 基于实际状态检查碰撞、跟踪误差和最终 5 mm 收敛
```

分两阶段实施：

1. **LIBERO 验证阶段**：直接执行 MotionGen 的时间参数化关节轨迹，先验证 48–64 个环境 step 内能否安全收敛。
2. **真机阶段**：使用同一条 MotionGen 轨迹初始化 cuRobo MPC，由 MPC 根据实时关节状态闭环跟踪；RGB-D 地图接口保持不变。

这不是两套独立方案，而是同一规划链路的两个执行层级。当前 EE waypoint P-controller 不再作为最终执行方案。

## 2. 验收要求

按优先级排序：

1. **碰撞约束**
   - cuRobo 规划结果必须通过整条轨迹的环境碰撞和自碰撞检查。
   - 执行过程中一旦安全距离或跟踪误差越界，立即停止，不得继续追赶轨迹。
   - 在安全模式下，规划失败不得回退到直线 `PoseController`。
   - “无碰撞”指相对于当前 RGB-D 可见场景、机器人模型及设定安全裕量；LIBERO 当前场景暂不要求主动换视角。

2. **收敛时间**
   - 一个 action chunk 为 16 个环境 step。
   - 正常目标：3 个 chunk，即不超过 48 step。
   - 硬上限：4 个 chunk，即不超过 64 step。
   - 预算依据是轨迹运动时间和环境控制周期，不再等于 waypoint 数量。

3. **最终精度**
   - 实际末端位置误差 `<= 0.005 m`。
   - 实际末端姿态误差建议 `<= 0.02 rad`。
   - 建议连续 2 个控制周期满足阈值后才判定收敛，避免瞬时越过阈值。

如果 cuRobo 在速度、加速度、碰撞约束下无法生成能在 64 step 内完成的轨迹，应明确返回 `infeasible/timeout`，不得通过提高速度绕过碰撞或动力学约束。

## 3. 当前问题基线

当前 `memory_system/execute/curobo_planner.py`：

1. 从一帧 RGB-D 生成点云，并用最多 512 个球表示障碍物。
2. cuRobo 规划出关节轨迹。
3. 对关节轨迹做 FK，转换为最多 48 个 EE waypoint。
4. `WaypointPoseController` 使用 EE delta action 逐点追踪。
5. `correction_steps=len(waypoints)`，把轨迹采样点数量错误地当成环境执行预算。

已观测到的基线结果：

- 规划得到 41 个 waypoint，initial alignment 从环境 `t=10` 执行到 `t=50`。
- 取消中间 zero action 且设置 `k=0.5` 后，41 step 结束时只执行到约第 27/41 个 waypoint。
- 末端位置误差约 `0.0629 m`，姿态误差约 `0.3276 rad`。
- 停止原因是 41 step 预算耗尽，不是 controller 收敛。

因此当前瓶颈不是 RGB-D 看不到必经空间，也不是 cuRobo 没有生成路径，而是规划轨迹的时间与动力学信息在执行前被丢弃了。

## 4. 目标架构

### 4.1 RGB-D 碰撞世界

LIBERO 第一阶段保留单帧 RGB-D 输入，以减少变量：

- 使用相机内参、外参和深度图构建 cuRobo 碰撞世界。
- 保留确定性的点云下采样。
- 对深度噪声、相机标定误差和执行误差加入安全膨胀距离。
- 自碰撞继续使用 cuRobo 的机器人碰撞球模型。
- 当前 LIBERO 工作空间基本可见，暂不实现主动感知或换视角。

后续真机可把碰撞世界替换为持续更新的 voxel/ESDF 地图，但不得改变规划器和执行器之间的轨迹接口。

### 4.2 MotionGen 输出必须完整保留

`CuroboPlanner.plan()` 不再只返回 EE waypoints。新的结果至少包含：

- `joint_names`
- `position`
- `velocity`
- `acceleration`
- `dt`
- `motion_time`
- cuRobo success/feasibility/collision metrics
- goal pose 与容差

应直接保留 `result.get_interpolated_plan()` 中的 `JointState` 信息。禁止先做 FK 再把它退化为没有时间信息的 EE waypoint 列表。

### 4.3 轨迹时间与 48–64 step 预算

设环境控制周期为 `dt_env`，则：

```text
required_steps = ceil(motion_time / dt_env) + settle_steps
```

实现要求：

- 先测出 LIBERO 实际 `dt_env`，不能把视频帧率当作控制频率。
- 按 `dt_env` 对 cuRobo 关节轨迹进行时间重采样。
- `required_steps <= 48`：正常接受。
- `48 < required_steps <= 64`：允许执行但记录 slow-path。
- `required_steps > 64`：在执行前返回 deadline infeasible，重新规划或终止 alignment。
- `interpolation_dt` 只决定轨迹采样密度，不能通过增大它伪造更短的运动时间。
- 速度提升应通过机器人允许的 velocity/acceleration/jerk 约束和 cuRobo 时间优化完成。

### 4.4 LIBERO：时间索引的关节轨迹执行

每个环境 step：

1. 读取实际关节位置和速度。
2. 根据已执行时间取得当前关节参考状态，而不是等待到达某个 waypoint 才切换。
3. 通过关节位置/速度控制接口跟踪该参考状态。
4. 检查实际状态与参考轨迹的跟踪误差。
5. 检查当前状态及下一小段轨迹的碰撞安全距离。
6. 轨迹结束后，以实际 EE pose 判断是否达到 5 mm，而不是以数组索引结束判断成功。

关键前置检查：确认当前 LIBERO/robosuite 环境是否能在 initial alignment 期间接收关节 position/velocity command，并在结束后无缝恢复原 policy 使用的 OSC action 接口。不得用修改模拟器 joint state 的方式“瞬移”。

如果同一 episode 中不能安全切换控制器，需要在环境创建阶段接入支持两种命令模式的控制层；在完成这一接口前，不宣称已经实现 cuRobo 关节轨迹执行。

### 4.5 真机：MotionGen + cuRobo MPC

LIBERO 直接执行验证通过后，将执行器替换为 cuRobo MPC：

1. MotionGen 负责全局无碰撞轨迹和可达性。
2. 将 MotionGen 轨迹作为 MPC seed/reference。
3. 每个控制周期用真实关节状态调用 MPC 优化。
4. MPC 输出关节 position/velocity/acceleration command。
5. RGB-D 地图更新后，重新检查后续轨迹；安全距离不足时减速、停止或重新规划。

cuRobo 的 5 mm 目标是相对于机器人运动学模型。真机要达到物理 5 mm，还需要相机内外参、手眼标定和机器人运动学标定满足精度要求；必要时在末端阶段增加视觉反馈，但这不属于 LIBERO 第一阶段。

## 5. 代码改造计划

### 阶段 A：接口审计与基线固化

- 找到 LIBERO 环境的控制周期、当前 OSC controller 配置和 joint command 接口。
- 保存当前失败样例的轨迹长度、运动时间、关节约束、末端误差和最小碰撞距离。
- 增加日志，区分 `plan_success`、`deadline_feasible`、`tracking_success` 和 `goal_converged`。

交付条件：能够回答当前 41 点轨迹真实需要多少秒、按 `dt_env` 应该是多少 step。

### 阶段 B：重构规划结果

- 将 `PlanResult` 改造成包含完整时间参数化 `JointState` 的结果。
- 保留 EE waypoint 仅用于可视化和诊断，不再驱动 controller。
- MotionGen 配置显式设置 5 mm position tolerance、姿态 tolerance、自碰撞和环境碰撞安全距离。
- 安全模式下删除规划失败到直线 `PoseController` 的 fallback。

交付条件：规划结果序列化后仍保留 position/velocity/acceleration/dt，且 cuRobo 轨迹验证通过。

### 阶段 C：LIBERO 关节轨迹执行器

- 新增独立的 `CuroboJointTrajectoryExecutor`。
- 按 `dt_env` 重采样并按绝对时间推进参考轨迹。
- 增加 tracking error、collision margin、deadline 和 convergence gate。
- 修改 initial alignment correction 通道：预算来自 `motion_time`，结束条件来自实际收敛。
- alignment 结束后清理 joint executor 状态并恢复 policy 控制。

交付条件：指定 episode 在不超过 64 step 内达到 5 mm，且没有碰撞或安全监控越界。

### 阶段 D：cuRobo MPC 执行器

- 创建 `MPCSolverCfg/MPCSolver`，控制周期与真机一致。
- 使用 MotionGen 轨迹调用 `update_seed_trajectory()`。
- 每周期更新 current joint state 并调用 `optimize_action_sequence()`。
- 接入动态 RGB-D voxel/ESDF 世界更新和 stop/replan gate。

交付条件：在执行扰动和地图变化测试中仍能安全停止或收敛，不依赖 EE waypoint P-controller。

## 6. 测试与记录

### 单元测试

- 轨迹按 `dt_env` 重采样后起点、终点和时间一致。
- 预算由 `motion_time` 计算，与 waypoint 数量无关。
- 超过 64 step 的轨迹在执行前被拒绝。
- 规划失败不会进入 direct PoseController fallback。
- 只有实际 EE position/rotation 同时满足容差才返回 converged。

### LIBERO 集成测试

先固定当前分析的 episode 和 robot initial state，记录：

- planning time
- motion time / required env steps
- 实际执行 step 数
- 每 step joint tracking error
- 每 step EE position/orientation error
- 规划与执行时的最小 collision distance
- 最终是否在 48/64 step 内收敛
- 是否发生 controller switch discontinuity

再扩展到 `libero_10_robotinit_low_success` 全集，分别统计 48-step 和 64-step 成功率，不以 task 最终 success 代替 initial-alignment success。

### 必须保留的失败类型

- `PLANNING_FAILED`
- `COLLISION_INFEASIBLE`
- `DEADLINE_INFEASIBLE`
- `TRACKING_ERROR`
- `SAFETY_STOP`
- `GOAL_NOT_CONVERGED`

## 7. 明确不做的事情

- 不再通过单独调大 `k` 或 `action_clip` 解决轨迹执行时间问题。
- 不把减少 waypoint 数量当作加速方法。
- 不以 waypoint index 到达末尾代替 5 mm 收敛检查。
- 不在安全模式下静默回退到没有碰撞约束的直接 EE 修正。
- LIBERO 第一阶段不加入主动感知、换视角或多相机融合。

## 8. 最小实施顺序

1. 审计 LIBERO joint command/controller-switch 能力和真实 `dt_env`。
2. 让 `CuroboPlanner` 返回完整的时间参数化关节轨迹。
3. 实现 LIBERO joint trajectory executor 和安全/收敛 gate。
4. 在固定 episode 上验证 `<=64 step + 5 mm + no collision`。
5. 扩大到低成功率任务集。
6. 最后接入 cuRobo MPC 和动态 RGB-D 地图，迁移到真机。

只有第 4 步通过后，才进入 MPC/真机阶段，避免同时修改感知、规划和控制而无法定位误差来源。
