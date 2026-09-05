# ASPIRE 如何将夹爪移动到指定位置

## 基于 RGB-D、PyRoKi IK、LIBERO 关节控制器与 MuJoCo 接触物理的代码解析

> 本文分析 ASPIRE 官方仓库中 LIBERO 实验所使用的运动方式。重点不是重新设计一套控制器，而是解释作者发布的代码如何把“移动夹爪到一个三维目标位置”落到实际执行。
>
> 核对版本：NVlabs/ASPIRE commit `f4c8939aab0af9b97690c561bd80e282940f7886`（2026-08-31）。

---

## 1. 核心结论

在 ASPIRE 的 LIBERO 主实验路径中，夹爪移动由以下调用链完成：

```text
RGB-D 观测
  -> SAM3 分割物体或目标区域
  -> 深度反投影得到世界坐标点云
  -> 计算抓取位姿或放置位姿
  -> PyRoKi 求目标关节配置
  -> LIBERO blocking joint controller 反复执行关节增量
  -> MuJoCo 推进机器人与物体动力学
```

ASPIRE 没有要求 coding agent 为每次移动构建完整的 cuRobo collision world。程序生成少量具有明确语义的 waypoint：

```text
预抓取 -> 抓取 -> 抬升 -> 目标上方 -> 下降 -> 释放 -> 后退
```

其中：

- RGB-D 决定“目标在哪里”；
- GraspNet、点云几何和规则决定“夹爪应该以什么姿态接触目标”；
- PyRoKi 决定“机器人应该使用什么关节角到达该位姿”；
- LIBERO 控制器决定“如何逐步把关节驱动到目标角度”；
- MuJoCo 决定“物体是否被夹住、是否滑落、是否与目标表面发生接触”。

需要特别注意：这套方法保证的是每个 waypoint 的 IK 可解性和关节控制收敛，不是两个 waypoint 之间经过严格验证的连续无碰撞路径。

---

## 2. 系统中的四个坐标与状态层次

理解这套代码时，需要区分四类量。

### 2.1 图像坐标

SAM3 返回目标在 RGB 图像中的 mask：

```python
mask: np.ndarray  # H x W
```

这个 mask 只能说明哪些像素属于目标，不能直接控制机器人。

### 2.2 世界坐标中的三维点云

ASPIRE 使用深度、相机内参和相机位姿，将 mask 内像素反投影为世界坐标点：

```python
points_world = mask_to_world_points(mask, depth, K, T_world_camera)
```

得到：

```python
points_world.shape == (N, 3)
```

随后可从点云估计：

- 物体中心；
- 表面最高点；
- oriented bounding box；
- 抓取或释放位置。

### 2.3 末端目标位姿

末端目标由位置和四元数组成：

```python
position_xyz: np.ndarray       # shape (3,)
quaternion_wxyz: np.ndarray    # shape (4,)
```

数学上表示为：

\[
T^W_{EE,goal}=
\begin{bmatrix}
R^W_{EE,goal} & p^W_{EE,goal}\\
0 & 1
\end{bmatrix}
\]

### 2.4 机器人关节配置

Franka 的七个机械臂关节角记为：

\[
q=[q_1,q_2,\ldots,q_7]^T
\]

PyRoKi 将末端目标位姿转换成一个目标关节配置 `q_target`。LIBERO 控制器再将机器人从当前配置逐步驱动到 `q_target`。

---

## 3. 第一步：从 RGB-D 得到目标三维位置

### 3.1 目标物体分割

ASPIRE 的 coding agent 可以调用：

```python
masks = segment_sam3_text_prompt(rgb, "black bowl")
best_mask = max(masks, key=lambda item: item["score"])["mask"]
```

这里的文本 prompt 来自任务指令。例如任务是“把黑碗放在盘子上”，程序会分别定位：

- 被操作物：`black bowl`；
- 放置目标：`plate`。

### 3.2 深度反投影

对一个像素 \((u,v)\)，深度为 \(z\)，相机内参为：

\[
K=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
\]

相机坐标中的三维点为：

\[
x=(u-c_x)z/f_x,\qquad
y=(v-c_y)z/f_y
\]

再通过相机外参变换到世界坐标：

\[
p^W=T^W_Cp^C
\]

在 ASPIRE 代码中，这些计算封装在：

```python
points_world = mask_to_world_points(
    best_mask.astype(np.uint8),
    depth,
    intrinsics,
    camera_pose,
)
```

### 3.3 计算抓取或放置位置

对于简单目标区域，程序经常使用点云中心：

```python
target_center = points_world.mean(axis=0)
```

对于放置表面，代码通常取点云的最高 Z：

```python
surface_z = points_world[:, 2].max()
```

于是释放位置可以构造成：

```python
release_pos = np.array([
    target_center[0],
    target_center[1],
    surface_z + 0.03,
])
```

这里的 `0.03 m` 是模板中的经验释放余量，不是通用常数。ASPIRE 会在 debug seeds 上根据执行结果修改这类参数。

---

## 4. 第二步：确定夹爪的目标姿态

只有目标位置还不够。夹爪还必须有正确的方向，否则可能出现：

- 手指无法包围物体；
- 夹爪从错误方向撞向表面；
- 搬运途中旋转导致物体掉落；
- 抽屉拉动方向与把手方向不匹配。

### 4.1 使用 GraspNet 获取抓取姿态

ASPIRE 提供的典型调用为：

```python
grasp_poses, grasp_scores = plan_grasp(depth, K, object_mask)

grasp_world, _ = select_top_down_grasp(
    grasp_poses,
    grasp_scores,
    T_world_camera,
)

grasp_pos, grasp_quat = decompose_transform(grasp_world)
```

这里 `plan_grasp()` 根据目标深度和 mask 产生多个六自由度抓取候选；`select_top_down_grasp()` 再选取较适合桌面操作的候选。

### 4.2 使用规则化 top-down 姿态

对于碗、盒子或平放物体，skill 也允许使用规则化的 top-down quaternion。此时位置来自 RGB-D，方向则由模板指定。

这种做法限制了末端姿态的自由度，但会使：

- 抓取方式更一致；
- lift 和 transport 时不必频繁转腕；
- IK 更容易保持在同一局部分支；
- 程序更容易跨不同位置复用。

---

## 5. 第三步：`goto_pose()` 如何把目标位姿交给 PyRoKi

ASPIRE 的高级运动接口为：

```python
goto_pose(position, quaternion_wxyz, z_approach=0.0)
```

它不是全局路径规划器，而是“末端目标位姿 -> IK -> 关节执行”的便利封装。

### 5.1 TCP offset 修正

输入位置通常表示期望工具中心点或抓取接触位置。代码会根据夹爪姿态加入 TCP offset：

```python
offset_pos = pos + rotation.apply(TCP_OFFSET)
```

这一步用于处理：机器人运动学模型中的末端 link，与实际手指接触点之间存在固定空间偏移。

如果 TCP offset 不正确，即使 IK 精确到达目标，手指也会在目标上方或下方发生系统性偏差。

### 5.2 `z_approach` 的真实含义

当调用：

```python
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
```

ASPIRE 不是简单地给世界坐标 Z 加 `0.15`。官方代码执行：

```python
approach_pos = offset_pos + rotation.apply([0, 0, -z_approach])
```

因此预接近方向由夹爪局部坐标轴决定。对标准 top-down grasp，它表现为先到目标上方，再沿抓取方向到达目标。

### 5.3 第一次 IK 与连续 IK

第一次求解时使用：

```python
cfg = solve_ik(target_position, target_wxyz)
```

后续求解时使用：

```python
cfg = solve_ik_vel_cost(
    target_position=target_position,
    target_wxyz=target_wxyz,
    prev_cfg=previous_cfg,
)
```

`prev_cfg` 让优化器倾向于选择接近上一关节配置的解。它主要用于：

- 降低 IK 解在不同分支之间跳变；
- 减少突然翻腕；
- 使连续 waypoint 的动作更平滑；
- 避免无意义的大幅关节运动。

但它不等价于碰撞检测，也不能证明中间路径安全。

---

## 6. PyRoKi 在其中负责什么

PyRoKi 是一个模块化机器人运动学优化工具包。在 ASPIRE 的这条代码路径里，它主要作为 IK solver 使用。

其目标可概念化为：

\[
q^*=\arg\min_q
\left[
w_p\|p(q)-p_{goal}\|^2
+w_Rd(R(q),R_{goal})^2
+w_v\|q-q_{prev}\|^2
\right]
\]

其中：

- \(p(q)\)：正运动学得到的末端位置；
- \(R(q)\)：正运动学得到的末端方向；
- \(q_{prev}\)：前一次 IK 使用的关节配置；
- 位置和方向误差要求夹爪达到目标；
- velocity/continuity cost 让新解不要偏离上一配置过远。

PyRoKi 输出的是目标关节角，不直接驱动 MuJoCo，也不自动完成抓取或接触操作。

---

## 7. 第四步：LIBERO 如何真正执行目标关节角

PyRoKi 求得：

```python
q_target = np.ndarray(shape=(7,))
```

之后 ASPIRE 调用：

```python
move_to_joints_blocking(q_target)
```

### 7.1 Blocking controller 的实际循环

官方实现的核心逻辑可以概括为：

```python
target = q_target

for step in range(max_steps):
    current = read_current_joint_positions()
    error = target - current

    if norm(error) < tolerance and step > 0:
        break

    delta_action = error * control_frequency
    env.step(concat(delta_action, gripper_command))
```

默认参数为：

```python
tolerance = 0.01
max_steps = 120
```

### 7.2 为什么它能到达目标

每个仿真步都会重新读取当前关节位置，然后重新计算误差：

\[
e_k=q_{target}-q_k
\]

控制命令基于当前误差：

\[
a_k=f_{control}e_k
\]

MuJoCo/robosuite 的底层控制器执行该 action 后，机器人得到新的关节状态 \(q_{k+1}\)。外层循环继续修正，直到：

\[
\|q_{target}-q_k\|_2<0.01
\]

或者达到最大步数。

因此机器人不是瞬移到 IK 解，而是通过闭环关节误差逐步逼近。

### 7.3 夹爪命令与机械臂命令同时发送

LIBERO action 由七个机械臂关节命令和一个夹爪命令组成：

```text
[arm_action_1, ..., arm_action_7, gripper_action]
```

环境在每个仿真步同时维持当前夹爪开合状态。因此搬运时，机器人关节在变化，但夹爪继续保持闭合。

---

## 8. ASPIRE 的标准抓取—搬运—放置序列

### 8.1 抓取

```python
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
close_gripper()
```

`goto_pose(..., z_approach=0.15)` 在一次调用中先访问预接近位姿，再访问最终抓取位姿。夹爪随后闭合，MuJoCo 模拟手指与物体的接触和摩擦。

### 8.2 抬升

ASPIRE 的标准 skill 使用约 `0.15 m` 的抬升：

```python
lift_pos = np.array([
    grasp_pos[0],
    grasp_pos[1],
    grasp_pos[2] + 0.15,
])

q_lift = solve_ik(lift_pos, grasp_quat)
if q_lift is not None:
    move_to_joints(q_lift)
```

物体不是在 PyRoKi 中 attach 到机器人，而是因为手指仍然闭合，由 MuJoCo 接触力随夹爪一起移动。

### 8.3 定位放置目标

程序重新读取 RGB-D，分割目标表面并计算：

```python
target_center = target_points.mean(axis=0)
surface_z = target_points[:, 2].max()
```

重新观测可以适应：

- 不同 LIBERO seed 中目标位置变化；
- 机器人移动后相机视野变化；
- 上一步物理交互造成的场景变化。

不过官方 skill 也指出：机械臂抬起后可能遮挡固定相机，因此某些程序会在抓取前同时定位物体和放置目标。

### 8.4 移动到目标上方

```python
above_target = np.array([
    target_center[0],
    target_center[1],
    lift_pos[2],
])

q_above = solve_ik(above_target, grasp_quat)
if q_above is not None:
    move_to_joints(q_above)
```

这里保留搬运高度，只改变目标 XY。

### 8.5 三点圆弧搬运

ASPIRE 的 evolutionary-search skill 给出了更保守的三点形式：

```python
apex = (lift_pos + above_target) / 2
apex[2] += 0.05

move_to_joints(solve_ik(apex, grasp_quat))
move_to_joints(solve_ik(above_target, grasp_quat))
```

几何含义为：

```text
lift_pos -> 中点上方 5 cm -> above_target
```

它利用更高的中间 waypoint 增加 clearance，但仍然没有对连续路径进行完整碰撞验证。

### 8.6 下降与释放

```python
release_pos = np.array([
    target_center[0],
    target_center[1],
    surface_z + 0.03,
])

q_release = solve_ik(release_pos, grasp_quat)
if q_release is not None:
    move_to_joints(q_release)

open_gripper()
```

松开后，MuJoCo 负责：

- 物体在重力作用下落到目标表面；
- 物体与目标容器、桌面或其他物体发生接触；
- 物体反弹、倾斜或稳定。

`transport.md` 中还建议在释放后读取数次 observation，让物理继续推进，再检查最终任务 predicate：

```python
open_gripper()
for _ in range(3):
    get_observation()
```

---

## 9. 完整的等价伪代码

下面的代码是对 ASPIRE 官方 skill 调用链的结构化整理，用于说明整体流程；具体 prompt、偏移量和释放高度需要按任务调试。

```python
import numpy as np

# ---------- 1. 读取 RGB-D ----------
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
depth = obs["agentview"]["images"]["depth"]
depth = depth[:, :, 0] if depth.ndim == 3 else depth
K = obs["agentview"]["intrinsics"]
T_world_camera = obs["agentview"]["pose_mat"]

# ---------- 2. 定位被抓物 ----------
object_masks = segment_sam3_text_prompt(rgb, object_prompt)
object_mask = max(object_masks, key=lambda item: item["score"])["mask"]

# ---------- 3. 生成抓取位姿 ----------
grasp_poses, grasp_scores = plan_grasp(depth, K, object_mask)
grasp_world, _ = select_top_down_grasp(
    grasp_poses,
    grasp_scores,
    T_world_camera,
)
grasp_pos, grasp_quat = decompose_transform(grasp_world)

# ---------- 4. 抓取 ----------
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
close_gripper()

# ---------- 5. 抬升 ----------
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
q_lift = solve_ik(lift_pos.tolist(), grasp_quat.tolist())
if q_lift is None:
    raise RuntimeError("lift IK failed")
move_to_joints(q_lift)

# ---------- 6. 重新观测并定位目标 ----------
obs2 = get_observation()
rgb2 = obs2["agentview"]["images"]["rgb"]
depth2 = obs2["agentview"]["images"]["depth"]
depth2 = depth2[:, :, 0] if depth2.ndim == 3 else depth2
K2 = obs2["agentview"]["intrinsics"]
T2 = obs2["agentview"]["pose_mat"]

target_masks = segment_sam3_text_prompt(rgb2, target_prompt)
target_mask = max(target_masks, key=lambda item: item["score"])["mask"]
target_points = mask_to_world_points(target_mask, depth2, K2, T2)

target_center = target_points.mean(axis=0)
surface_z = target_points[:, 2].max()

# ---------- 7. 搬运到目标上方 ----------
above_target = np.array([
    target_center[0],
    target_center[1],
    lift_pos[2],
])

apex = 0.5 * (lift_pos + above_target)
apex[2] += 0.05

for waypoint in [apex, above_target]:
    q = solve_ik(waypoint.tolist(), grasp_quat.tolist())
    if q is None:
        raise RuntimeError("transport IK failed")
    move_to_joints(q)

# ---------- 8. 下降并释放 ----------
release_pos = np.array([
    target_center[0],
    target_center[1],
    surface_z + 0.03,
])

q_release = solve_ik(release_pos.tolist(), grasp_quat.tolist())
if q_release is None:
    raise RuntimeError("placement IK failed")
move_to_joints(q_release)
open_gripper()

# 推进少量物理步骤并获取结果观测
for _ in range(3):
    get_observation()
```

---

## 10. “正确到达”与“正确交互”是两件事

### 10.1 正确到达

正确到达主要由以下环节保证：

1. RGB-D 给出世界坐标目标；
2. TCP offset 将接触位置转换为末端 link 目标；
3. PyRoKi 使末端位置和方向逼近期望位姿；
4. `prev_cfg` 抑制 IK 分支跳变；
5. blocking controller 反复修正关节误差。

### 10.2 正确交互

正确交互来自 task-specific 动作模板：

| 任务 | 到达后的动作 |
|---|---|
| 抓取 | 进入 grasp pose 后闭合夹爪 |
| 放置 | 降到表面上方后打开夹爪 |
| 放入容器 | 到容器中心上方，下降到高于边缘的位置后释放 |
| 拉抽屉 | 抓住把手后沿抽屉滑轨方向移动末端 |
| 推动物体 | 接触后保持姿态并沿指定方向移动末端 |
| 按按钮 | 沿按钮法向从预接触位姿推进到按压位姿 |

PyRoKi 不理解“抓取”“拉抽屉”这些语义。程序必须明确提供接触前后的 waypoint、夹爪命令和移动方向。

---

## 11. ASPIRE 如何改进失败程序

ASPIRE 的核心贡献之一不是新的底层运动控制器，而是让 coding agent 根据执行 trace 修改程序。

系统可以记录：

- primitive API 及其输入输出；
- IK 是否成功；
- 关键动作前后的 RGB；
- segmentation overlay；
- grasp candidate；
- 最终 `taskcompleted_0/1`；
- 物体最终是否位于任务要求的位置。

典型修复包括：

| 失败现象 | 程序修改 |
|---|---|
| 手指位于物体上方但没有夹住 | 降低 grasp Z 或修改 TCP offset |
| 抓到物体边缘 | 修改抓取候选或 yaw |
| 抬升时掉落 | 减少腕部旋转，保持连续 lift–transport–descend |
| 搬运时撞到低矮障碍 | 增加 lift Z 或添加更高 apex |
| 撞到容器边缘 | 增加 above-target 高度，修正目标中心 |
| 释放后物体弹出 | 降低 release height |
| 低位放置 IK 失败 | 使用 pre-probe IK conditioning |
| 连续任务关节构型恶化 | 在未抓物的子任务之间回 home 并重新观测 |

因此 ASPIRE 的性能来自：

```text
动作模板 + 感知参数化 + IK/控制执行 + 多次试验修复
```

而不是一次运动调用天然具备完整鲁棒性。

---

## 12. Pre-Probe IK Conditioning

`transport.md` 还记录了一种特殊技巧：在抓取之前预先访问低位放置目标附近的多个高度。

```python
for dz in [0.08, 0.06, 0.04, 0.02]:
    goto_pose(
        np.array([target_x, target_y, target_z + dz]),
        top_down_quat,
    )

goto_home_joint_position()
```

其目的不是碰撞规避，而是利用有状态 IK 的局部分支连续性：先让求解器找到能够接近低位目标的关节分支，之后抓取和放置时更可能回到相似分支。

这一技巧体现了 ASPIRE 的工程风格：通过程序化 waypoint 和求解器状态 conditioning 提高成功率，而不一定求解完整的全局运动规划问题。

---

## 13. 这套方法能够保证什么

### 可以较好保证

- 输入目标在机器人工作空间内时，找到一个末端 IK 解；
- 关节控制器逐步收敛到目标关节配置；
- top-down 桌面抓取具有一致的动作结构；
- 通过 lift 和 above-target waypoint 避开大量低矮桌面障碍；
- 使用 RGB-D 适应不同 seed 中的目标位置；
- 通过 debug/evolutionary search 修正任务特定参数。

### 不能严格保证

- 两个 waypoint 之间连续无碰撞；
- 肘部、腕部或夹持物不会扫过障碍；
- 单视角 RGB-D 看不见的障碍不会被撞到；
- 物体一定被稳定夹住；
- 接触力满足某个约束；
- 插入、装配等精密接触任务一定成功；
- IK 有解就代表整个动作可安全执行。

特别是从 `q_lift` 到 `q_above_target` 时，LIBERO 控制器是在关节空间减小误差。末端轨迹通常不是严格直线，也没有 attached-object collision validation。

---

## 14. 与 cuRobo 方法的区别

| ASPIRE 默认 LIBERO 路径 | cuRobo 规划路径 |
|---|---|
| 少量语义 waypoint | 优化或搜索完整轨迹 |
| PyRoKi 求每个 waypoint 的 IK | collision-aware IK + graph/trajopt |
| 不构造完整 manipulation collision scene | 需要 WorldConfig 和碰撞模型 |
| 不在规划器内 attach 被抓物 | 被抓物转换为 attached collision geometry |
| 依靠 lift/arc 等规则增加余量 | 显式计算 robot/object/world collision |
| MuJoCo 负责真实接触与物体跟随 | MuJoCo 仍负责执行，cuRobo 负责规划侧碰撞 |
| 失败后修改程序和参数 | 规划失败后改变 seeds、世界模型或规划配置 |

ASPIRE 仓库中确实包含可选的 cuRobo 实验代码，但论文 LIBERO coding-agent 协议默认开放的核心运动 API 是：

```text
solve_ik
move_to_joints
goto_pose
open_gripper
close_gripper
```

---

## 15. 阅读官方代码的推荐顺序

1. [`aspire/sim/.claude/libero/skills/grasp.md`](https://github.com/NVlabs/ASPIRE/blob/main/aspire/sim/.claude/libero/skills/grasp.md)  
   查看完整 RGB-D、抓取、抬升、搬运和放置模板。

2. [`aspire/sim/.claude/libero/skills/transport.md`](https://github.com/NVlabs/ASPIRE/blob/main/aspire/sim/.claude/libero/skills/transport.md)  
   查看搬运 waypoint、pre-probe conditioning、release 后物理 settling 等策略。

3. [`aspire/sim/.claude/libero/evosearch/skills/motion-efficiency.md`](https://github.com/NVlabs/ASPIRE/blob/main/aspire/sim/.claude/libero/evosearch/skills/motion-efficiency.md)  
   查看三点圆弧 transport 以及作者对 waypoint 数量的限制。

4. [`aspire/sim/cap/integrations/franka/libero.py`](https://github.com/NVlabs/ASPIRE/blob/main/aspire/sim/cap/integrations/franka/libero.py)  
   查看 `goto_pose()`、TCP offset、PyRoKi 调用和连续 IK 分支。

5. [`aspire/sim/cap/envs/simulators/libero.py`](https://github.com/NVlabs/ASPIRE/blob/main/aspire/sim/cap/envs/simulators/libero.py)  
   查看 `move_to_joints_blocking()` 如何读取当前关节角、生成 delta action 并推进 MuJoCo。

6. [`aspire/sim/.claude/libero/inference-time-scaling/subagent-prompt.md`](https://github.com/NVlabs/ASPIRE/blob/main/aspire/sim/.claude/libero/inference-time-scaling/subagent-prompt.md)  
   查看论文复现实验允许 coding agent 使用的 LIBERO API。

7. [PyRoKi 项目页](https://pyroki-toolkit.github.io/) 和 [PyRoKi GitHub](https://github.com/chungmin99/pyroki)  
   查看运动学优化器本身的设计和示例。

---

## 16. 总结

ASPIRE 将夹爪移动到指定地点的过程，可以压缩成下面五步：

1. **看见目标**：RGB-D + SAM3 得到目标 mask 和三维位置；
2. **定义末端目标**：GraspNet 或规则模板给出位置、方向和接近偏移；
3. **求关节配置**：PyRoKi 根据目标位姿和上一配置求 IK；
4. **闭环执行**：LIBERO 控制器重复读取关节误差并推进 MuJoCo，直到接近目标配置；
5. **产生交互**：在目标位姿闭合或打开夹爪，或者沿任务方向继续移动，让 MuJoCo 接触动力学产生抓、放、推、拉行为。

其工程本质不是“一个通用 planner 自动解决全部问题”，而是：

\[
\boxed{
\text{RGB-D 参数化的规则动作模板}
+\text{PyRoKi IK}
+\text{关节闭环控制}
+\text{MuJoCo 接触物理}
+\text{LLM 失败修复}
}
\]

这解释了它为什么能在大量结构化 LIBERO 任务中有效，也解释了它在狭窄空间、精密接触和严格连续碰撞安全方面的边界。
