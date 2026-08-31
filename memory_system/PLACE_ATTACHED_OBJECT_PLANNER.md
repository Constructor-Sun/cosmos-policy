# HeldObject 携物 cuRobo 规划

## 1. 目标与边界

- Pick 完成后，手上已经有物体。
- `HeldObjectPlanner` 使用 cuRobo 把机械臂从当前状态规划到 memory 中的 ready pose。
- 规划成功后交给 VLA 继续执行。
- 本模块不负责：Pick completion、最终放置、phase advance、判断物体是否真的在手里。

## 2. 输入输出

### 输入

- 当前机器人状态：joint_positions、ee_states、gripper_joint_positions
- 感知：depth、camera_params、robot_base_pose
- `HeldObjectObservation`：Pick 阶段保存的 hand-local 物体点云
- 目标 ready pose

### 输出

- 成功：waypoints + `WaypointPoseController`
- 失败：`None`，直接交给 VLA

## 3. 核心流程

```text
Pick 阶段：
    memory mask + depth
    -> 构建 hand-local 物体点云
    -> 保存 HeldObjectObservation

HeldObjectPlanner 阶段：
    加载已保存的 HeldObjectObservation
    -> 生成 attachment spheres
    -> 删除机械臂点
    -> 删除手持物点
    -> 构建静态场景障碍
    -> cuRobo 规划到 ready pose
```

关键点：

- **HeldObjectPlanner 不重新匹配 memory，也不重新做连通域提取。**
- 它直接使用 Pick 阶段保存的 hand-local 模型。

## 4. 机械臂删除与 residual robot

- 机械臂删除与 `CuroboPlanner` 验证路径保持一致：
  - URDF depth filter + sphere filter
- 当前仅 sphere filter 时仍有 residual robot。
- URDF 理论上更好，但当前 CPU 光线投射太慢，尚未在 held-object 流程中完整验证。

## 5. 当前已知问题

### 5.1 HeldObjectObservation 会包含环境点

- memory 模板的 `crop_mask` 不是 tight mask，会包含物体周围背景。
- 当前临时处理：保存前用 DBSCAN 保留最大 3D 连通域。
- 根本问题：mask 质量不够紧，需要更精确的物体 mask 或 Pick 阶段更可靠的点云分割。

### 5.2 不能在 HeldObjectPlanner 阶段重新做连通域提取

- 在机械臂靠近其他物体时，重新提取会合并到周围物体或选错。
- 结论：必须使用 Pick 阶段保存的 observation。

### 5.3 residual robot 仍需 URDF 验证

- sphere-only 删除后仍有 residual robot。
- 需要跑通 URDF + sphere 的完整删除路径并验证效果。

## 6. 待办

- 将 Pick 阶段保存 / 加载 `HeldObjectObservation` 接入实际 runtime。
- 跑通 URDF + sphere 的机械臂删除验证。
- 在多个 LIBERO-10 task 上验证 observation 的稳定性和 attachment 覆盖率。
