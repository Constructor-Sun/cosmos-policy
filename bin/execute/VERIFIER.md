# 顺序 Verifier 设计与实现状态

Verifier 分为三个职责独立的阶段：

1. Phase verifier：判断当前是否在执行正确的 skill，并确认是否朝目标靠近。
2. Feasible-region verifier：判断机械臂是否进入可以执行末端动作的可行域。
3. Skill-completion verifier：判断当前 skill 是否完成，并决定何时进入下一 phase。

当前实现不训练额外模型，也不调用 VLM。  
所有 verifier 均为 observation-only，不直接修改 policy action。

## 一、Phase Verifier

### 1.1 文件

- 默认实现：`bin/execute/libero_phase_verifier.py`
- 离线目标构建：`bin/memory/build_libero_phase_targets.py`

### 1.2 离线目标

使用 `skill_memory/libero_10/phase_targets.pt`。

每个模板保存：

- task、demo、planner step、skill、arguments；
- frame、目标 bbox、目标中心、夹爪二维位置；
- `crop_rgb` / `crop_mask`；
- keypoints / descriptors。

Simulator 信息只用于离线构建，test-time 不读取 simulator object state。

### 1.3 匹配流程

1. 根据 task / planner step / skill / arguments 选择模板；
2. 对 masked crop 做多尺度模板匹配；
3. 跨 demo 聚类，至少 2 个 demo 支持同一区域才接受；
4. 返回粗粒度 `target_xy` 和 `matched_bbox_xyxy`。

### 1.4 方向判定

设上一时刻夹爪位置为 `g_prev`，当前为 `g_now`，目标为 `target_now`：

```text
progress = distance(g_prev, target_now) - distance(g_now, target_now)
```

- `progress > 0`：靠近；
- `progress < 0`：远离；
- 第一次出现 progress 时直接决定：
  - 靠近 -> `PHASE_OK`；
  - 明显远离 -> `PHASE_ERROR`。

输出：

- `PHASE_OK`：目标有视觉支持且正在靠近；
- `PHASE_ERROR`：第一次 progress 就明显远离；
- `PHASE_UNKNOWN`：视觉共识不足或模板缺失。

## 二、Feasible-region Verifier

### 2.1 文件

`bin/execute/libero_feasible_region_verifier.py`

### 2.2 输入

- 当前 task、planner step、skill、arguments；
- 当前目标中心、matched bbox、夹爪二维位置；
- `segments_ready_fixed16.json` 中的 `ready_frame`。

### 2.3 距离定义

```text
normalized_distance = distance(gripper, target_center) / bbox_diagonal
```

`ready_distance` 来自成功 demo 的 ready frame。  
运行时如果当前 normalized distance 小于等于至少 2 个 demo 的 ready distance，则锁存 `FEASIBLE`。

### 2.4 输出

- `FEASIBLE`：进入可行域并锁存；
- `NOT_FEASIBLE`：进入前持续停滞或折返；
- `FEASIBLE_UNKNOWN`：证据不足或仍在接近。

当前阈值偏松，通常 phase 确认后第一次 feasible check 就会 `FEASIBLE`。  
因此它更适合作为“粗过滤”，不单独承担“下一个动作是否成功”的预测。

## 三、Skill-completion Verifier

### 3.1 文件

实际 eval 使用：

`bin/execute/libero_skill_completion_verifier_geo.py`

它基于 wrist camera 和 wrist memory 判断完成，不再使用 VAE/DTW completion。

### 3.2 Wrist Memory

`skill_memory/libero_10/wrist_completion_targets.pt` 仍由 `bin/memory/build_libero_phase_targets.py` 生成。

当前 Pick 判断不再使用 SIFT 每帧重检测，只使用模板里的 `target_center_xy` 作为初始 ROI 中心先验。

### 3.3 Pick 完成条件

```text
夹爪闭合
+ 3D EEF 从闭合位置开始的累积位移 >= 0.04
+ qpos gap > 0.003
+ wrist 固定 ROI 内 GFTT + LK 光流 + RANSAC 单应性
+ 光流内点比例 flow_inlier_ratio >= 0.5
+ 物体中心无单帧跳变（<= 10px）
+ 最近 5 帧中至少 4 帧满足
```

核心信号是：

- 夹爪移动时，wrist 固定 ROI 内的物体表面点表现为一致刚体运动；
- RANSAC 内点比例高说明物体仍被稳定抓取；
- 如果脱落、晃动或跟踪丢失，内点比例会下降或中心跳变。

### 3.4 Place 完成条件

```text
夹爪 closed -> open 的释放 transition
+ gripper open
+ 连续 2 个低层 step 确认
```

### 3.5 Open / Close / TurnOn

暂未使用 wrist 完成判断，回退到旧几何规则或 `COMPLETION_UNKNOWN`。

## 四、顺序执行 Monitor

### 4.1 文件

`bin/execute/libero_execution_monitor.py`

流程：

```text
PHASE_CHECK -> FEASIBLE_CHECK -> COMPLETION_CHECK -> 下一 phase
```

### 4.2 Phase 确认条件

```text
PHASE_OK
+ progress_px 存在
+ progress_px > 0
+ target_xy / bbox 存在
```

如果第一次 progress < 0，phase verifier 返回 `PHASE_ERROR`，不会进入 feasible。

### 4.3 干预信号

`ExecutionMonitorResult` 新增：

```python
should_intervene: bool
intervention_reason: str | None
```

触发条件：

- phase 第一次明显远离：`PHASE_ERROR` 可作为干预信号；
- 进入 feasible 后 2 个 action chunk（32 个低层 step）仍未 `SKILL_COMPLETE`：

```text
intervention_reason = "no_completion_within_two_chunks_after_feasible"
```

当前 `should_intervene` 信号会触发 pose recovery，具体见 [CORRECTION.md](./CORRECTION.md)。

### 4.4 日志

运行时按阶段输出：

- `[PHASE]`
- `[FEASIBLE]`
- `[COMPLETION]`
- `[VERIFIER SUMMARY]`

每条记录包含 timestep、stage_before/stage_after、planner step、reason。

## 五、Memory 构建

### 5.1 Phase Targets

```bash
python bin/memory/build_libero_phase_targets.py \
  --input-dir LIBERO-Cosmos-Policy/success_only/libero_10_regen \
  --segments-manifest skill_memory/libero_10/segments_ready_fixed16.json \
  --output skill_memory/libero_10/phase_targets.pt \
  --max-per-task 10
```

### 5.2 Wrist Completion Targets

在同一个 builder 中增加：

```bash
python bin/memory/build_libero_phase_targets.py \
  --input-dir LIBERO-Cosmos-Policy/success_only/libero_10_regen \
  --segments-manifest skill_memory/libero_10/segments_ready_fixed16.json \
  --output skill_memory/libero_10/phase_targets.pt \
  --wrist-completion-output skill_memory/libero_10/wrist_completion_targets.pt \
  --max-per-task 10
```

它会从每个 segment 的 success 区间附近取一帧，用 wrist segmentation 提取物体 crop。

## 六、Eval 接入

- 开关：`enable_phase_verifier`
- smoke runner：`COSMOS_PHASE_VERIFIER=1`
- 入口：`cosmos_policy/experiments/robot/libero/run_libero_eval.py`

所有结果只记录，不修改 policy action，也不执行恢复。

## 七、当前已知问题与下一步

1. Feasible verifier 阈值偏松，基本进入即 `FEASIBLE`，后续需要更严格的可行域定义。
2. Pick 的光流 ROI 目前使用模板 target_center_xy 或 wrist 图像中心；如果模板中心偏离实际抓取位置，ROI 可能不包含完整物体，需要更多真实 rollout 校准。
3. Open / Close / TurnOn 的完成判断尚未用 wrist 实现。
4. 恢复策略已接入 `phase_pose_recovery`，但 PlaceOn 的 2D 方向判定仍偏严格，可能造成反复 recovery，具体见 [CORRECTION.md](./CORRECTION.md)。
5. 下一步可以接入 policy value / future prediction，用于更早判断“下一个动作是否可能成功”。
