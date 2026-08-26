# cuRobo Initial Alignment 当前实现

## 1. 目的

Initial alignment 在策略执行任务动作之前，把机器人从随机初始关节状态移动到记忆中第一阶段的 ready pose。当前实现遵循两个原则：

1. **从第一个策略动作之前开始干预**，不等任务执行失败后再纠正。
2. **由 RGB-D 建立障碍物表面，再由 cuRobo 规划和执行关节轨迹**，不使用末端直线移动穿过场景。

要启用完整链路，需要同时启用：

```text
enable_initial_alignment = True
enable_collision_aware_initial_alignment = True
enable_curobo_joint_execution = True
```

其中 `enable_curobo_joint_execution=True` 也使规划失败成为安全失败：不会退回没有碰撞保证的直接 `PoseController`。

## 2. 一开始就干预

### 2.1 触发时机

LIBERO reset 后先执行 10 个 dummy step，让物体和机器人状态稳定。这些 step 不属于 policy action。进入第一个可执行策略动作的时刻后：

1. 构造第一次策略观测，并获得当前主相机 VAE 表征。
2. `_maybe_start_initial_alignment()` 只在 `_policy_step_count == 0` 时触发。
3. 从第一阶段 ready-pose memory 中选取主相机表征最相似的目标。
4. 目标默认在记忆 pose 的世界坐标 z 方向上增加 `0.02 m`。
5. 使用当前 RGB-D、关节位置和机器人基座位姿调用 cuRobo。
6. 规划成功后清空尚未执行的 policy action queue，并在当前 timestep 立即输出第一条 alignment action。

因此，模型可以先完成一次推理以提供目标选择所需的视觉表征，但它生成的第一批 policy action 不会先作用于环境。环境收到的第一条非 dummy action 是 initial alignment action。

### 2.2 一次性接管

Initial alignment 每个 episode 最多尝试一次，并与 phase verifier、在线 recovery 分离。执行期间由关节轨迹控制器接管；收敛、失败或 64-step 预算结束后恢复原来的 OSC controller，然后才继续执行 policy action。

```text
reset
-> 10 个稳定环境的 dummy step
-> 首次观测与 ready pose 选择
-> RGB-D/cuRobo initial alignment
-> 恢复 OSC
-> 执行第一个 policy action chunk
```

## 3. 如何实现避障移动

### 3.1 从 RGB-D 构造可见表面

规划器把主相机每个有效深度像素反投影为三维点，并转换到 cuRobo 使用的机器人基座坐标系。它不需要 MuJoCo 的物体几何或语义标签，因此同一感知接口可以迁移到真机。

当前机器人碰撞球用于移除点云中属于机器人自身的点。`robot_padding=0.015 m` 是机器人自过滤带，不是删除目标附近障碍，也不是给杯子或微波炉设置特殊规则。过滤以后，所有可见的非机器人表面都作为障碍来源。

### 3.2 有限障碍预算下覆盖整个场景

cuRobo 的障碍缓存最多放入 512 个小立方体，所以不能直接把全部深度点都送入优化器。当前采用确定性分层采样：

- 约 384 个点用于覆盖完整可见场景：先按 `20 mm` voxel 合并，再用最远点顺序均匀覆盖各表面。
- 约 128 个点用于目标附近 `150 mm` 范围：使用更密的 `8 mm` voxel，优先保留离目标最近的表面。
- 每个选中点转换为边长 `10 mm` 的 cuRobo cuboid。
- 如果后续完整轨迹检查发现冲突，冲突表面点成为 mandatory points，优先占用目标附近预算，再次参与规划。

这只是在固定的 512 障碍预算内提高目标和冲突区域的分辨率，并没有删除目标周围的点。杯子、微波炉黑框、桌面和其他物体使用相同的几何规则。

### 3.3 cuRobo 生成关节轨迹

规划从当前 7 维关节状态开始。记忆中的 LIBERO 末端目标会被转换为 cuRobo tool frame 下的目标，MotionGen 同时处理：

- 机器人运动学和自碰撞；
- RGB-D 表面障碍；
- `5 mm` 位置容差与 `0.02 rad` 姿态容差；
- `10 mm` optimizer collision activation distance。

规划结果保留为带时间信息的关节轨迹。EE waypoints 只保留给旧执行模式，完整模式不再用末端 P-controller 逐点追踪。

### 3.4 用完整点云复核整条轨迹

512 个采样障碍只用于让 cuRobo 高效求解。候选轨迹生成后，还必须通过一次更严格的完整点云检查：

1. 关节轨迹被加密到相邻状态最大关节变化不超过 `0.02 rad`。
2. 对每个加密状态计算整台机器人的 cuRobo collision spheres。
3. 使用 KD-tree，将所有 collision spheres 与完整的、仅移除了机器人自身的 RGB-D 点云比较。
4. 任一轨迹状态距表面小于等于 `5 mm` safety margin，就拒绝该候选轨迹。
5. 把导致冲突的完整点云点加入 mandatory obstacles，并对**同一个目标**重新规划，最多细化两次。

因此，稀疏的 512 点不会成为最终安全判据；它们只是规划表示，最终判据覆盖整条机器人扫掠轨迹和完整可见表面。

### 3.5 只有同目标不可行时才最小回退

系统首先尝试到达原目标。只有原目标经过冲突点补充和重新规划后仍不可行，才沿末端局部接近方向的反方向回退：

1. 从 `2 mm` 开始试探，并按倍增方式寻找第一个可行区间。
2. 最大允许回退 `80 mm`。
3. 找到可行区间后做二分细化，分辨率为 `1 mm`。
4. 对每个回退候选仍执行同样的完整点云轨迹检查和冲突点重规划。

最终选择的是满足碰撞约束的最小回退量，而不是预先固定回退 6 cm，也不会为了保持目标误差强行执行一条有碰撞的轨迹。

### 3.6 闭环执行与结束条件

LIBERO 执行时临时切换到 `JOINT_POSITION` controller，按照环境控制周期重采样并跟踪 cuRobo 关节轨迹。每一步使用实际关节位置计算动作；跟踪误差超过阈值时停止，不能继续追赶。

到达轨迹末端后，以实际 EE pose 判断是否收敛：位置误差不超过 `5 mm`、姿态误差不超过 `0.02 rad`。正常情况下要求连续两个观测周期满足条件；如果第 64 步已经位于最终参考且满足误差，则在 deadline 直接接受收敛。随后恢复原 OSC controller。

## 4. 安全边界

- 当前避障覆盖单帧 RGB-D 可见空间；看不到的区域没有几何证据，不能承诺避开不可见障碍。
- 当前 LIBERO 场景在 initial alignment 期间近似静态，所以使用一次性点云。真机若环境会移动，应在执行期间更新 RGB-D 地图并重新验证剩余轨迹。
- “不碰撞”是相对于 RGB-D 表面、相机标定、cuRobo 机器人模型和配置裕量而言；真机精度仍依赖深度质量、手眼标定和机器人模型精度。

## 5. 主要实现位置

- `cosmos_policy/experiments/robot/libero/run_libero_eval.py`：首个 policy action 前触发并接管 action。
- `memory_system/execute/initial_alignment.py`：选择第一阶段 ready pose，并在严格关节执行模式下拒绝不安全 fallback。
- `memory_system/execute/curobo_planner.py`：构造场景、调用 cuRobo、冲突重规划和最小回退。
- `memory_system/execute/surface_obstacles.py`：RGB-D 表面采样、轨迹加密和完整点云碰撞检查。
- `memory_system/execute/curobo_trajectory.py`：时间化关节轨迹与最小可行回退搜索。
- `cosmos_policy/experiments/robot/libero/libero_joint_control.py`：关节轨迹闭环执行、误差门限和 OSC 恢复。
