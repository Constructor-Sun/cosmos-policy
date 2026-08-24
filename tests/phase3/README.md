# Stage 3 / Phase 3：RGB-D Phase 回放测试

本目录是 **test-only** 的 Phase 3 测试，用于验证 training demo 中各个 skill phase
是否能在 3D 信息下与其他候选物体区分。

## 支持范围

当前支持所有实际执行的 skill phase：

- `Pick`
- `PlaceIn`
- `PlaceOn`
- `Close`
- `TurnOn`

`Open` 在 LIBERO-10 中均为 `already_satisfied`，不参与测试。

## 各 skill 的 Phase 目标定义

| skill | 跟踪目标 | 候选物体类型 |
|---|---|---|
| Pick | `arguments.item` | item-like 物体 |
| PlaceIn | `arguments.target` | target-like 物体/区域 |
| PlaceOn | `arguments.target` | target-like 物体/区域 |
| Close | `arguments.target` | target-like 物体/区域 |
| TurnOn | `arguments.target` | target-like 物体/区域 |

## 设计

- 每条 LIBERO-10 demo 从全局 `t=0` 开始完整回放。
- 对每个执行的 skill phase：
  - 正确目标走 **Memory 模板匹配** 路径：
    `template match -> mask -> RGB-D -> 3D representative point`
  - 其他候选物体直接从 **simulator segmentation** 获取真实 mask，作为测试参考。
- 默认 `frame_stride = 4`，表示离线回放 demo 时每 4 帧采样一次。
- 注意：`frame_stride` 只是本离线测试的采样间隔，不等同于在线 rollout 的 phase 检查间隔。

## 在线 rollout 的 phase 检查间隔

在线 phase 检查由 `cosmos_policy/experiments/robot/libero/run_libero_eval.py` 控制，与 `tests/phase3` 的 `frame_stride` 无关。

当前在线配置：

- action chunk 仍为 16 步；
- phase 检查每 8 个环境 step 一次；
- 实际检查点约为 `t=10, 18, 26, 34, 42, ...`。

## Phase 异常语义

- 远离（away）和停滞（stall）统一视为 Phase 异常；
- 连续 `evidence_updates` 次异常即触发 `PHASE_ERROR`；
- 只有距离明显减小的正常接近才会清零异常计数。

## 文件

| 文件 | 说明 |
|---|---|
| `harness.py` | 环境、demo、manifest、候选选择、模板生成等测试辅助 |
| `run_phase3.py` | 主测试脚本，输出排名、Feasible、tolerance sweep |
| `visualize_phase3.py` | 生成单个 phase 的 MP4 可视化 |

## 运行

```bash
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy

python tests/phase3/run_phase3.py \
  --max-demos 2 \
  --frame-stride 4 \
  --out tests/phase3/results.json
```

可视化单个 phase：

```bash
python tests/phase3/visualize_phase3.py \
  --task "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
  --demo demo_0 \
  --step 3 \
  --out tests/phase3/phase3_failure.mp4
```

## 当前全 skill 结果（70 phases）

```text
phases                         70
correct_rank_1                 66
rank1_rate                     0.942857
correct_rank_top2              69
rank_top2_rate                 0.985714
correct_rank_top3              69
rank_top3_rate                 0.985714
correct_approach_positive      64
correct_approach_positive_rate 0.914286
phase_normal_end               56
phase_normal_end_rate          0.8
phase_normal_end_strict        53
phase_normal_end_strict_rate   0.757143
premature_gripper_change       15
premature_rate                 0.214286
```

### 分 skill 概况

| skill | n | rank1 | top2 | approach positive | normal end | premature |
|---|---:|---:|---:|---:|---:|---:|
| Pick | 32 | 29 | 31 | 31 | 32 | 2 |
| PlaceIn | 18 | 18 | 18 | 17 | 6 | 12 |
| PlaceOn | 14 | 13 | 14 | 14 | 12 | 0 |
| Close | 4 | 4 | 4 | 0 | 4 | 0 |
| TurnOn | 2 | 2 | 2 | 2 | 2 | 1 |

### tolerance sweep 摘要（全 skill）

| tolerance_m | correct_no_error | any_distractor_error |
|---:|---:|---:|
| 0.001 | 42 / 70 | 23 / 70 |
| 0.002 | 42 / 70 | 24 / 70 |
| 0.005 | 41 / 70 | 23 / 70 |
| 0.01 | 38 / 70 | 25 / 70 |
| 0.02 | 32 / 70 | 33 / 70 |
| 0.05 | 16 / 70 | 53 / 70 |

`evidence_updates` 固定为 2。远离和停滞统一计入连续异常计数。

当前没有 tolerance 能同时满足“正确目标 100% 无错误”和“有干扰物错误”。

## 已知剩余问题

1. PlaceIn 的 normal end 较低，主要因为 gripper 状态改变被记为 premature，且 Feasible entry 常为 None；
2. Close 的 approach 全为负，可能因为 target 3D 代表点应取门/把手而不是 region 中心；
3. Pick 仍有少量同类型物体 rank2；
4. 一个 `alphabet_soup_1` rank4 且 approach 为负；
5. 全 skill tolerance sweep 下正确目标 no-error 只有 60% 左右，需要进一步调参或按 skill 分别处理。

## 待补充

- 非 Pick phase 的可视化案例；
- 按 skill 分别选择 tolerance 的分析；
- PlaceIn / Close 的 Phase 语义调整。
