# Initial Alignment 偏移问题调查与修正记录

## 1. 问题概述

任务：

```text
put the black bowl in the bottom drawer of the cabinet and close it
```

原始运行：

```text
scripts/rollouts/08-26/2026_08_26-16_19_03--with_future_img--episode=1--success=True--task=put_the_black_bowl_in_the_bottom_dr.mp4
scripts/rollouts/08-26/2026_08_26-16_19_03--with_future_img--episode=2--success=False--task=put_the_black_bowl_in_the_bottom_dr.mp4
scripts/rollouts/08-26/2026_08_26-16_19_03--with_future_img--episode=10--success=False--task=put_the_black_bowl_in_the_bottom_dr.mp4
```

主要现象：

- episode 1 在视频 `t≈50` 时位于黑碗附近，之后成功完成任务。
- episode 2、episode 10 在相近时刻却停在柜子附近，出现明显的空间偏差。
- 这种差异不是小幅跟踪误差，而是不同 episode 的实际运动轨迹和到达区域明显不同。

## 2. 当前执行链路

本次配置为：

```text
enable_initial_alignment=True
enable_collision_aware_initial_alignment=True
enable_curobo_joint_execution=False
```

因此当前不是直接通过 `PoseController` 移动到 memory target，也不是直接执行 cuRobo joint trajectory。实际链路是：

```text
memory ready pose
    ↓
CuroboPlanner 规划 joint trajectory
    ↓
cuRobo FK 计算 panda_hand tool poses
    ↓
转换为 LIBERO robot0_eef waypoints
    ↓
WaypointPoseController
    ↓
六维 OSC action
    ↓
LIBERO / MuJoCo
```

相关代码：

```text
memory_system/execute/initial_alignment.py
memory_system/execute/curobo_planner.py
memory_system/execute/curobo_trajectory.py
memory_system/execute/recovery/controller.py
cosmos_policy/experiments/robot/libero/run_libero_eval.py
cosmos_policy/experiments/robot/libero/libero_joint_control.py
```

## 3. 已确认的事实

### 3.1 Memory 检索结果不是 episode 差异的来源

原始 20 个 episode 全部选中了同一个 memory：

```text
demo=('demo_13',)
```

并且记录的 target 完全一致：

```text
[-0.0588200204,
 -0.0857331678,
  0.9660174251,
  2.6380581856,
 -1.9302700758,
  0.2832267880]
```

各 episode 的 VAE similarity 略有差异，但最终都选中了同一个 demo 和同一个 target。因此不能用“选错 memory”解释 episode 1 与 episode 2/10 的巨大差异。

### 3.2 原始运行的 correction 长度相同

原始运行中各 episode 均为：

```text
steps=48
```

因此不能仅凭固定步数就断言失败原因是“步数不足”。如果认为步数或 waypoint 推进存在问题，必须先证明失败 episode 的 controller 停留在较早 waypoint，或 correction 结束时存在显著残差。

### 3.3 初始腕姿/关节构型是重要差异

episode 1 与 episode 2 的主相机 EE 投影位置接近，但 wrist 图像和机械臂构型明显不同。两条轨迹在 initial alignment 阶段就开始分叉，而不是在 policy 恢复后才分叉。

这表明问题与 robot initial state 下的姿态/构型相关。

### 3.4 Memory 使用非规范化 axis-angle 表示

Memory 中的 `ee_states` 来自 Robosuite：

```python
T.quat2axisangle(obs["robot0_eef_quat"])
```

该实现保留 quaternion 符号，输出的旋转角可以大于 π。SciPy 的 `Rotation.as_rotvec()` 则会规范化到旋转角不超过 π。

例如：

```text
memory rotvec:    [ 2.638058, -1.930270,  0.283227], norm≈3.281
canonical rotvec: [-2.413751,  1.766144, -0.259145], norm≈3.002
```

两者的旋转轴方向相反，旋转角之和约为 `2π`，因此表示同一个物理旋转。raw 六维坐标不同，但 SO(3) pose 等价。

### 3.5 cuRobo panda_hand 与 LIBERO EEF 之间确实存在固定偏移

cuRobo 的 `franka.yml` 使用：

```text
tool_frames: [panda_hand]
```

LIBERO observation 使用 `robot0_eef` grip-site。真实数值诊断显示二者的相对位置约为：

```text
T_eef_tool translation ≈ [0, 0, -0.09654] m
```

episode 1 与 episode 2 估计出的外参差异仅为：

```text
position delta ≈ 0.0000199 m
rotation delta ≈ 0.000125 rad
```

所以 `T_eef_tool` 近似固定，两套运动学模型之间不存在足以解释大偏差的 episode 级外参变化。

## 4. 对原始 Rw/tw 转换的重新认识

原始代码根据当前姿态计算：

```text
T_fake = T_curobo_current · inverse(T_libero_current)
```

然后正向映射目标、反向映射 waypoint：

```text
T_curobo_goal = T_fake · T_libero_goal
T_libero_waypoint = inverse(T_fake) · T_curobo_waypoint
```

对于最后一个 waypoint：

```text
T_libero_final
= inverse(T_fake) · T_curobo_goal
= inverse(T_fake) · T_fake · T_libero_goal
= T_libero_goal
```

因此，只要正向和反向使用同一个 `T_fake`，它不会直接改变最终 LIBERO target。

此前将 `T_fake` 本身认定为 episode 级终点偏差根因，是错误的。它可能影响中间路径在 cuRobo 坐标系中的形状、碰撞环境表达和规划结果，但不能单独解释为什么日志中的相同终点会变成不同的实际到达点。

## 5. 尝试过但已撤销的修改

曾尝试把 `T_fake` 替换为显式固定的 `T_eef_tool` 外参组合：

```text
T_base_tool = T_base_eef · T_eef_tool
T_base_eef  = T_base_tool · inverse(T_eef_tool)
```

修改内容包括：

- 新增 `tool_waypoints_to_eef_poses()`。
- 非 joint-execution 分支也使用 `retarget_tool_pose()`。
- 将 cuRobo tool waypoints 逐个转换为 LIBERO EEF waypoints。
- 让 `PlanResult.target_ee_states` 记录转换后的最后 waypoint。
- 新增不同初始腕姿下的固定外参回归测试。

单元测试结果：

```text
20 passed
```

但是实际 rollout 明显退化：

```text
scripts/rollouts/08-26/2026_08_26-17_28_50--with_future_img--episode=1--success=False--task=put_the_black_bowl_in_the_bottom_dr.mp4
```

该运行中规划 waypoint 数由 48 变为 41，整条中间 trajectory 也发生变化。单元测试只验证了人为构造的刚体变换，没有覆盖真实 cuRobo 规划、OSC waypoint 跟踪和 robot-init 差异，所以不能证明运行时修复正确。

上述修改已全部撤销，以下文件目前已恢复到修改前状态：

```text
memory_system/execute/curobo_planner.py
memory_system/execute/curobo_trajectory.py
tests/test_curobo_joint_trajectory.py
```

## 6. 当前最可能的问题层级

现有证据更支持问题发生在：

```text
EEF waypoint pose
    ↓
WaypointPoseController
    ↓
OSC action
    ↓
实际 EE pose change
```

而不是：

```text
memory selection
或 memory target 本身
或简单的最终坐标刚体变换
```

`WaypointPoseController` 当前使用：

```python
error[:3] = target[:3] - current[:3]
error[3:] = Log(R_target @ inverse(R_current))
action = clip((k / scale) * error, -0.5, 0.5)
```

重点风险如下。

### 6.1 旋转增量的坐标系约定可能与 OSC 不一致

controller 计算的是 world/spatial-frame 旋转误差。如果 Robosuite OSC action 使用 body/tool-frame 或采用不同的左乘/右乘约定，那么不同初始腕姿会得到不同的实际旋转方向。

初始姿态接近目标时该错误可能不明显；姿态差异较大的 robot-init case 中则可能被放大。

### 6.2 对六维 action 逐分量裁剪会改变旋转轴

当前：

```python
np.clip(action, -0.5, 0.5)
```

会分别裁剪旋转向量的三个分量。当旋转误差较大时，这不只是限制角速度，还会改变旋转轴方向。

因此某些初始姿态可能正常，某些姿态可能持续沿错误旋转方向运动。

### 6.3 action scale 是硬编码的

当前使用：

```text
s_pos = 0.01
s_rot = 0.10
```

这些值没有从实际 Robosuite OSC controller 的 `output_min/output_max` 或 action scaling 配置中读取。如果真实 scale 不一致，P controller 的实际增益和饱和区间都会错误。

### 6.4 OSC 位置和姿态控制存在 Jacobian 耦合

错误或饱和的旋转 action 不一定只影响朝向。OSC 通过同一机器人 Jacobian 同时控制位置和姿态；在不同关节构型、接近奇异位形或存在接触时，姿态控制错误可能表现为明显的位置偏移。

### 6.5 waypoint 推进/固定 correction budget 可能是放大因素，但尚未证实

`WaypointPoseController` 只有在当前位置和姿态同时进入 tolerance 后才推进到下一 waypoint，但执行循环使用固定 correction step budget。

这可能导致 controller 停留在中间 waypoint，但目前没有逐步的 controller index 和最终残差日志，因此不能把它当成已确认根因。原始多数 episode 可以成功，也说明仅仅“一个 waypoint 对应一个环境步”不是充分解释。

## 7. 下一步建议的最小诊断

在修改 planner、memory 或坐标变换之前，应只增加观测日志，对成功 episode 1 和失败 episode 2 记录每个 correction step：

```text
current_eef_pose
target_waypoint_pose
waypoint_index
position_error
rotation_error
unclipped_action
clipped_action
obs_next_eef_pose - obs_current_eef_pose
```

需要回答以下问题：

1. episode 2 是否停留在某个中间 waypoint，而 episode 1 正常推进？
2. episode 2 的 rotation action 是否长期处于逐分量饱和？
3. 输入的平移/旋转 action 与实际 `ΔEEF` 的方向是否一致？
4. OSC 实际使用的 action scale 与 `0.01/0.10` 是否一致？
5. correction 结束时，实际 EE 与最终 waypoint 的 position/rotation residual 分别是多少？

只有回答这些问题后，才能决定修复应该是：

- 更正 spatial/body rotation convention；
- 使用保持旋转轴方向的 norm clipping；
- 从 OSC 配置读取真实 scale；
- 或根据 convergence 而不是 waypoint 数结束 correction。

## 8. 当前结论

目前最可靠的结论是：

1. Memory 检索和 memory target 在各 episode 中相同，不是 episode 差异来源。
2. 原始 `T_fake` 的正向/反向映射在终点处会抵消，不能直接解释不同实际终点。
3. 显式 `T_eef_tool` 修改改变了真实规划路径并导致回归，已撤销。
4. axis-angle 数值分支差异是真实的表示不一致，但对应物理旋转等价，不足以单独解释大幅位置偏移。
5. 当前最值得优先验证的是 `WaypointPoseController → OSC action → 实际 EEF 变化` 的执行转换，特别是旋转坐标约定、逐分量裁剪和 action scale。

在获得逐步 action/pose 数值证据前，不应继续修改 cuRobo tool/EEF 坐标变换。
