# Held Object 感知与携物规划

## 1. 文档范围

本文档统一说明以下两个已经分别验证的部分如何衔接：

1. Pick 完成后，从 RGB-D 中无 oracle 提取手持物点云。
2. 将手持物点云作为 attachment 交给 `HeldObjectPlanner`，规划到 Place 的 memory ready pose。

这里描述的是算法和模块边界。当前正式 runtime 尚未完成二者的在线接线。

## 2. 整体数据流

```text
Pick 完成后的当前观测
  -> metric depth
  -> GPU URDF depth filter
  -> cuRobo robot sphere filter
  -> EEF 周围 ROI
  -> 3-D DBSCAN
  -> 选择平均 EEF 距离最近的有效组件
  -> world-frame held-object points
  -> 转换到 hand/tool-local frame
  -> HeldObjectObservation
  -> attachment spheres
  -> HeldObjectPlanner
  -> 携物规划到 Place ready pose
  -> VLA 继续完成 Place
```

## 3. 无 oracle 手持物提取

### 3.1 输入

```text
实时 RGB-D
相机内外参
机械臂与夹爪关节角
EEF 位姿
robot base 位姿
与机器人一致的 URDF 和 mesh
```

提取过程中不使用：

```text
simulator instance segmentation
simulator geom segmentation
target object ID
robot geom ID
物体真值位姿
```

### 3.2 已验证流程

```text
GPU URDF depth rendering
  -> observed/rendered depth consistency
  -> collision-sphere residual filter
  -> 0.25 m EEF ROI
  -> DBSCAN(eps=0.025 m, min_samples=10)
  -> 忽略 noise 和小于 20 点的组件
  -> 选择平均 EEF 距离最近的组件
```

GPU URDF 使用 NVIDIA Warp 的 CUDA ray query。URDF filter 后继续使用 robot collision spheres，是为了清理少量 mesh/depth 边界残留。

### 3.3 验证范围

该流程已经在 LIBERO-10 的 10 个任务上各检查一个 Pick 后、Place 阶段的 case。输出位于：

```text
tests/held_object/libero10_gpu_urdf_no_oracle/
```

KITCHEN_SCENE6 的 yellow-and-white mug case 中，无 oracle 提取结果为 1439 个点。simulator segmentation 只在提取完成后用于离线评分和着色，不影响组件选择。

## 4. HeldObjectObservation

被选中的 world-frame 点云必须在 Pick 完成时转换为 hand/tool-local 点云：

```text
p_hand = R_hand_to_world^T * (p_world - t_hand_to_world)
```

然后构造：

```text
HeldObjectObservation(
    item=...,
    points_hand=...,
    source="gpu_urdf_connected_component",
)
```

`points_hand` 在后续机械臂运动中保持不变；物体的世界位置由当前 hand pose 决定。因此，连通体只在 Pick→Place 切换时提取一次。

## 5. HeldObjectPlanner

### 5.1 输入

```text
joint_positions
ee_states
gripper_joint_positions
depth
camera_params
robot_base_pose
HeldObjectObservation
Place ready_pose
```

### 5.2 规划过程

```text
points_hand
  -> attachment spheres
  -> 将 attachment 注册到 cuRobo
  -> 从当前 RGB-D 场景删除机器人点
  -> 从场景点云删除手持物点
  -> 构建静态障碍
  -> 携物规划到 ready pose
  -> WaypointPoseController
```

规划器只消费已经生成的 `HeldObjectObservation`，不在 Place 阶段重新匹配图像，也不重新选择连通体。

## 6. 模块职责

```text
connected-component extractor
    负责：机器人删除、ROI、聚类、组件选择、hand-local 转换

HeldObjectPlanner
    负责：attachment、场景障碍、cuRobo 规划、controller 输出

rollout runtime
    负责：检测 Pick→Place 切换、调用 extractor、取得 ready pose、执行 controller、恢复 VLA
```

不得把 simulator segmentation 放入 extractor 或 planner 的输入路径。

## 7. 当前代码状态

已存在：

```text
memory_system/execute/planner/held_object/
tests/held_object/visualize_held_object_after_robot_removal.py
tests/held_object/visualize_libero10_held_object_cases.py
```

其中 `HeldObjectPlanner` 已接入 runtime 的 `Pick→Place` 切换。当前每次只处理一个 held object：在 Pick→Place 时提取一次、建立一次 attachment，Place controller 结束后释放；不支持同时 attachment 多个物体，也不在 Place→Close 等其他转换中叠加调用。

Place planner 的目标搜索优先尝试直接目标 `distance=0`。若直接路径不可行，则沿工具局部 z 轴进行有符号 backoff 搜索，当前范围为 `[-0.04 m, +0.04 m]`，分别搜索两侧并选择绝对偏移最小的可行结果。该逻辑是坐标无关的，不针对某个具体 task 写世界 z 方向规则。
