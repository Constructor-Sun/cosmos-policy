# Place Fine-Align 并行方案（simple mode）

## 1. 背景

当前 Pick→Place 的携物规划依赖 `HeldObjectPlanner`（cuRobo）从 Pick 结束位姿规划到 Place ready pose。

实际 K6 验证中发现：

- cuRobo 规划失败率不低，主要失败为 `no feasible backoff to ready pose`；
- cuRobo 规划失败后回退 VLA 仍经常成功；
- cuRobo 规划成功也不保证 episode 成功。

因此希望新增一个更轻量的 `simple` 模式：

> VLA 先接近放置区域，距离 memory ready pose 约 10cm 时取消 VLA，用 `PoseController` 做短距离位姿微调，再交回 VLA 完成 Place。

本方案与现有 cuRobo placement **并行**，默认仍使用 cuRobo，不影响已有代码。

## 2. 目标

- 默认 `COSMOS_PLACE_MODE=curobo`，现有行为完全不变；
- 新增 `COSMOS_PLACE_MODE=simple`，作为实验模式；
- simple 模式不依赖 cuRobo、不依赖手持物点云提取；
- simple 模式失败后直接交回 VLA，不 fallback 到 cuRobo。

## 3. 模式定义

| 模式 | 行为 |
| --- | --- |
| `curobo`（默认） | 保持现有 HeldObjectPlanner / cuRobo placement 流程 |
| `simple` | Pick→Place 后不启动 cuRobo，VLA 接近 ready pose 后触发 PoseController 微调 |

环境变量：

```bash
COSMOS_PLACE_MODE=curobo   # 默认
COSMOS_PLACE_MODE=simple   # 实验
```

## 4. simple 模式流程

```text
Pick 完成
  -> 不调用 HeldObjectPlanner
  -> VLA 继续执行 Place

在 Place 阶段，每 4 个 action step 检测一次：
  -> 当前 EEF 到 memory ready_pose 的欧氏距离 <= 0.10m
  -> 触发微调

触发后：
  -> 清空 VLA action queue
  -> 使用 PoseController（含姿态修正）移动到 ready_pose
  -> 保持夹爪闭合

微调结束（收敛或超时）：
  -> 恢复 VLA 继续完成 Place
```

## 5. 关键设计

### 5.1 触发条件

- 使用当前 EEF 位置与 `ready_pose` 位置的三维欧氏距离；
- 阈值：`0.10m`；
- 检测频率：每 4 个 action step 一次；
- 触发条件不包含姿态误差。

### 5.2 微调控制器

- 复用 `memory_system/execute/recovery/controller.py` 中的 `PoseController`；
- 控制器同时修正位置和姿态；
- 由于 `PoseController` 没有 `finished` 属性，simple 分支内增加一个轻量 adapter，模拟 cuRobo `WaypointPoseController` 的 `CONVERGED / GOAL_NOT_CONVERGED / finished` 逻辑。

### 5.3 失败处理

- 如果一直没有触发：继续让 VLA 跑；
- 如果 `PoseController` 超时未收敛：恢复 VLA；
- 不 fallback 到 cuRobo。

## 6. 文件修改计划

### 6.1 新增文件

```text
memory_system/execute/planner/place_fine_aligner.py
```

内容：

- `PlaceFineAligner`：负责保存 ready_pose、检测触发、生成 PoseController；
- `PoseControllerAdapter`：为 `PoseController` 增加 `finished` / `status` 兼容逻辑。

预估 100~150 行。

### 6.2 修改已有文件

#### `cosmos_policy/experiments/robot/libero/run_libero_eval.py`

- 增加模块级 `_PLACE_ALIGNER = None`；
- 增加 `_maybe_start_place_fine_alignment(obs, frame)`；
- 在主循环非 alignment 分支调用；
- 默认 `_PLACE_ALIGNER is None` 时完全不影响现有逻辑。

#### `scripts/run_libero_smoke_test.py`

- 读取 `COSMOS_PLACE_MODE`，默认 `curobo`；
- `curobo` 分支保持现有逻辑；
- `simple` 分支只保存 ready_pose 并设置 `_PLACE_ALIGNER`，不调用 `HeldObjectPlanner`。

#### `scripts/run_libero10_k6_held_object_simple.sh`

- 新增脚本，复制现有 K6 脚本参数；
- 固定设置 `COSMOS_PLACE_MODE=simple`。

### 6.3 可选

- 更新 `K6_PICK_PLACE_INTEGRATION_PLAN.md` 或本文档，记录对比结果。

## 7. 兼容性

- 默认 `curobo`，现有 cuRobo placement 和 pick 阶段 planner 完全不变；
- `run_libero_eval.py` 的改动是纯新增，默认不生效；
- `PoseController` 本身不改动，通过 adapter 兼容；
- simple 模式不调用手持物点云提取，不依赖 cuRobo。

## 8. 验证计划

1. 用现有 `curobo` 模式跑一组 K6 baseline；
2. 用 `simple` 模式跑相同 K6 case；
3. 对比：

   - 干预触发率；
   - PoseController 收敛率；
   - episode 成功率；
   - 失败原因。

4. 根据结果决定后续默认模式和是否清理 cuRobo placement 路径。

## 9. 风险与待观察

- 10cm 阈值可能过早或过晚，需要实验调整；
- 每 4 步检测可能错过触发窗口；
- 直线/短距离 PoseController 仍可能撞到障碍物；
- 微调后恢复 VLA 可能仍失败；
- 姿态误差虽然不作为触发条件，但控制器会修正，需观察是否足够。
