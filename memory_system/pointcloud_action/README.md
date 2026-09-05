# PointCloud Action Memory 设计文档

## 1. 背景与目标

当前 memory 系统主要面向 LIBERO-10，使用 VAE/ready3d/feasible recovery targets 做检索和对齐。

本子项目目标是：

- 使用官方 **LIBERO-90** 的 episode 数据；
- 构建一个**独立的、以物体点云为唯一检索 key、以可变长度 object-centric Pick sequence 为 value** 的 memory；
- 该 memory 不替换、不融合现有 LIBERO-10 memory；
- 在线先检索 memory，由记录中的相对 ready pose 生成当前场景 ready pose；再复用现有 CuroboPlanner 移动到 ready pose，并执行局部 Pick。

v1 只支持 Pick，目标是把物体拿起。其他 skill 不在 v1 范围内，出现问题后再按需要扩展。

## 2. 独立通路原则

- 新代码全部放在 `memory_system/pointcloud_action/` 下；
- 不修改现有 `memory_system/offline/`、`memory_system/execute/`、`memory_system/artifacts.py` 的默认行为；
- 新 artifact 使用独立文件名和 format；
- 新评估脚本独立于现有完整 episode eval。
- 已在旧系统中实现的功能必须直接复用旧定义，不在本模块中重新发明规则：
  - segment / boundary 使用现有 `label_segments`、`label_boundaries` 输出；
  - ready frame 使用现有 Pick ready 定义；
  - raw/model-normalized action 使用现有 `build_recovery` 约定；
  - RGB-D 几何使用现有 `memory_system.geometry`；
  - Curobo 和闭环 pose 控制使用现有执行接口。

## 3. 总体数据流

### 3.1 离线构建

```text
LIBERO-90 HDF5 + BDDL
        |
        v
Segment labeling（直接复用现有 label_segments / label_boundaries 定义和输出）
        |
        v
对每个 segment 提取：
  - ready_frame
  - 物体点云
  - object frame
  - simulator oracle object pose（包含方向）
  - 可变长度 object-centric action sequence
  - 可变长度 object-centric EE pose sequence
  - ee/ready pose 的相对表示
        |
        v
pointcloud_action_memory.pt
```

### 3.2 在线使用

```text
当前场景 RGB-D + instance mask + simulator oracle object pose
        |
        v
提取当前物体点云
        |
        v
将当前点云转换到 object frame
        |
        v
PointCloudActionMemory 检索 top-k
        |
        v
将 memory 的相对 ready pose 映射回当前场景
        |
        v
复用现有 CuroboPlanner 移动到 ready pose
        |
        v
将 object-centric EE/action sequence 映射回当前场景并执行 Pick
        |
        v
按现有 Pick 成功定义判断是否抓起
```

## 4. Memory Schema 草案

```python
{
    "format": "libero_pointcloud_action_memory_v1",
    "suite": "libero_90",
    "key": "target_points_object",
    "controller_config": ...,              # 直接保存 HDF5 env_args 中的配置
    "normalization_stats": ...,            # 复用现有 build_recovery 使用的 stats
    "records": [
        {
            "memory_id": "...",
            "source_task": "...",
            "source_demo": "...",
            "planner_step_id": 1,
            "skill": "Pick",
            "arguments": {"item": "moka_pot_1"},

            # 点云
            "target_points_world": ...,       # (N, 3)
            "target_points_object": ...,      # (N, 3) canonical/object-local
            "target_xyz_world": ...,          # (3,)

            # 物体坐标系
            "T_world_object_anchor": ...,   # (4, 4)，ready frame 时冻结
            "object_frame_translation": ...,  # (3,)
            "object_frame_rotation": ...,     # (3, 3)
            "frame_source": "simulator_body_xmat",
            "frame_convention": "T_AB_maps_B_to_A",

            # 动作
            "ready_frame": ...,
            "segment_end": ...,             # 直接来自现有 segment manifest
            "sequence_length": ...,          # L = segment_end - ready_frame
            "ready_ee_states": ...,           # (6,)
            "T_object_ee_ready": ...,         # (4, 4)
            "action_sequence_raw": ...,       # (L, 7)，HDF5 原始 action
            "action_sequence_normalized": ...,# (L, 7)，复用现有 normalize_actions
            "action_sequence_world_physical": ...,  # (L, 7)，m/rad
            "action_sequence_object_physical": ..., # (L, 7)，m/rad
            "ee_pose_world_sequence": ...,    # (L + 1, 4, 4)
            "ee_pose_object_sequence": ...,   # (L + 1, 4, 4)，主要 value
            "action_scale": ...,              # 兼容旧字段时按现有实现生成
        }
    ]
}
```

`L` 不固定为 16，artifact 不为固定长度做 padding。如果某个旧模型接口要求 16-step chunk，则继续使用现有 `make_chunk` / `valid_length` 行为，只在送入该模型时切块。

## 5. Object-Centric Pick Sequence

### 5.1 为什么需要

LIBERO 的 action 是末端执行器 delta command。v1 同时保存 action 和实际 EE pose sequence，但以 object-frame EE pose sequence 作为主要执行 value；action 表示用于复现、兼容和诊断。

### 5.2 转换方式

定义 `T_WO` 为 object frame 到 world frame 的变换，`T_WE` 为 EE frame 到 world frame 的变换。内部统一使用 `4 x 4` homogeneous matrix 和 `3 x 3` rotation matrix。

v1 的 source object frame 直接来自 simulator：

```python
t_obj = sim.data.body_xpos[body_id]
R_obj = sim.data.body_xmat[body_id].reshape(3, 3)
```

不使用 PCA / ICP，不使用未注明顺序的 quaternion。source anchor 固定为现有 `ready_frame` 对应 simulator state 中的 object pose。

世界坐标点转物体坐标：

```text
p_object = R_obj^T (p_world - t_obj)
```

实际 EE pose 转物体坐标：

```text
T_OE[k] = inverse(T_WO_anchor) @ T_WE[k]
```

HDF5 时序按已有数据生成方式处理：

```text
states[i] -- actions[i] --> ee_states[i]
```

因此 `[ready_frame, segment_end)` 中 `L` 个 actions 对应 `L + 1` 个 EE poses：起始 pose 从 `states[ready_frame]` 恢复，之后使用每个 action 执行后记录的 `ee_states`。

对 action 的几何变换先使用 HDF5 controller config 转为物理量。当前 LIBERO-90 数据为：

```text
delta_p_world = 0.05 * action[:3]
omega_world   = 0.5  * action[3:6]
DeltaR_world  = Exp(omega_world)
```

平移和旋转严格转换为：

```text
delta_p_object = R_obj^T @ delta_p_world
DeltaR_object  = R_obj^T @ DeltaR_world @ R_obj
omega_object   = Log(DeltaR_object)
```

gripper 不做坐标变换。

### 5.3 执行时逆变换

假设当前 simulator oracle object pose 为 `T_WO_current`，其旋转部分为 `R_cur`：

```text
T_WE_target[k] = T_WO_current @ T_OE[k]

delta_p_world = R_cur @ delta_p_object
DeltaR_world  = R_cur @ DeltaR_object @ R_cur^T
omega_world   = Log(DeltaR_world)
```

当前场景的第一个 `T_WE_target` 是 ready pose，先交给现有 CuroboPlanner。局部执行开始后冻结本次 `T_WO_current`，不能在物体被抓起后逐帧改变 anchor。

闭环 waypoint controller 的旋转误差继续使用现有 SO(3) 组合方式，不直接相减 rotvec：

```text
R_error = R_target @ R_current^T
omega_error = Log(R_error)
```

### 5.4 Normalization

本模块不改变已有 action normalization：

- `action_sequence_raw`：HDF5 原始 OSC_POSE command；
- `action_sequence_normalized`：直接复用现有 `build_recovery.normalize_actions(actions, stats)`；
- `action_sequence_*_physical`：只用于上述几何变换，尺度来自 HDF5 controller config；
- `action_scale`：如果保留兼容字段，按现有实现生成，不在本模块中重新定义。

`action_sequence_normalized` 不参与坐标旋转，也不直接发送给 controller。点云只转换到 object frame 并降采样，保留米制尺度，不做 unit-sphere scaling。

### 5.5 v1 参考坐标系

| Skill | 参考坐标系 |
|---|---|
| Pick | 被抓物体 frame |

v1 只实现 Pick。其他 skill 不在本文档中预先设计。

## 6. 代码目录结构

```text
memory_system/
  pointcloud_action/
    __init__.py
    README.md
    schema.py
    config.py

    offline/
      __init__.py
      build_memory.py
      extraction.py
      geometry_utils.py
      action_utils.py

    retrieval/
      __init__.py
      pointcloud_action_memory.py

    execute/
      __init__.py
      pointcloud_selector.py
      pointcloud_controller.py   # 可选

    eval/
      __init__.py
      eval_pointcloud_pick.py
```

## 7. 评估方案

不跑完整 LIBERO episode，而是：

```text
1. 使用当前物体点云作为唯一 key 检索 top-k
2. 将候选的 object-relative ready pose 映射回当前场景
3. 复用现有 CuroboPlanner 移动到 ready pose
4. 映射并执行完整的可变长度 Pick sequence
5. 按现有 Pick 成功定义判断是否抓起
```

评估指标：

- 检索 top-k 中是否存在合理动作；
- 动作迁移后执行成功率；
- 最终抓取成功率。

## 8. 待定参数

- 每个 LIBERO-90 task 使用多少 demos（数据已下载完成，每个 task 有 50 demos）；
- 点云降采样点数（建议 256 / 512）；
- top-k 数量；

v1 已确定：

- 唯一检索 key 是 `target_points_object`；skill、task、object name 只作为 metadata，不参与相似度；
- object frame 使用 simulator oracle `body_xpos/body_xmat`；
- 点云保留物理尺度；
- 先使用 Chamfer distance；
- 只支持 Pick；
- sequence 使用现有 `ready_frame` 到现有 segment `end` 的真实可变长度。

## 9. 里程碑建议

1. 定义 schema 和 artifact 格式；
2. 实现 LIBERO-90 离线构建，生成小规模测试 artifact；
3. 实现 pointcloud_action_memory 检索；
4. 实现严格的 object-centric SE(3) pose/action 坐标变换和 round-trip 测试；
5. 实现独立 Pick 局部评估脚本；
6. oracle object-frame 版本稳定后，再决定是否扩展范围。
