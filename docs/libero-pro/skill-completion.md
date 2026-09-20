# Skill 完成判定

阶段机（`PhaseEventRecorder` / `VLASkillRuntime`）靠每个 skill 的语义规则判断
"这一阶段完成了吗"。规则只读**可观测信号**（点云 + 本体感受），不读仿真状态，
因为要迁移到真机。

## 规则与实测

| skill | 规则 | 判据 | swap（VLA rollout） | demo（成功示范） |
|---|---|---|---|---|
| Pick | `PickCompletionChecker` | 目标点云随夹爪刚性移动 **且** z 抬升 ≥2cm，连续 5 帧 | TP27 FP0 FN1 TN132 | TP159 FP0 FN1 TN0 |
| PlaceIn/On | `ReleaseSkillCompletion` | 夹爪张开超 baseline 连续 **5** 帧 | TP21 FP5 FN0 TN134 | 非末段 TP126 FN2 |
| TurnOn | `TurnOnCompletion` | **夹住后** EE 净转角 ≥ **35°** | TP10 FP0 FN10 TN0 | — |
| Open / Close | `TimeoutOnly` | 无数值规则（见下） | — | — |

Δ = 规则触发帧 − 真值成立帧，中位：Pick **+5**、Place **+4~+21**（深容器更大）、TurnOn **+3**。
都是确认期的代价，不是延迟缺陷。

**已知边界**（各自写在模块 docstring 里）：

- Place 测的是"释放"不是"放进去"——松手未落入容器会误报（swap 里 5 例）
- TurnOn 测的是"手腕转了"——目标已经开着则不触发（K8 全 10 例），由 timeout 兜底
- Open 是 `ensure_open` 合成插入的**前置条件、不是任务目标**，不计入 stuck 统计
- Close 非瓶颈（swap 里没有 episode 卡在 Close）

## 三种测试

| 测什么 | 命令 | 通过判据 |
|---|---|---|
| **回放保真度** | `replay_measure.py --mode fidelity --hdf5 <ep>` | `max|d| == 0`，否则下游全部作废 |
| **规则 vs 真值** | `run_measure.sh` / `run_measure_demo.sh` | 看 **FP 列**——误报是危险方向 |
| **真值自检** | `--mode demo --validate-truth` | 成功 demo 末帧 goal 必为真，须 10/10 |

## 跑

```bash
OUT=$PWD/experiments/liberopro/measure_swap_to bash scripts/libero_pro/run_measure.sh
N_DEMOS=10 OPEN_FRAMES=5 bash scripts/libero_pro/run_measure_demo.sh
python scripts/libero_pro/aggregate_measure.py --json-glob "<out>/json/*.json"
```

两条路径的数据源和前置条件不同：

- `run_measure.sh` 回放采集的动作，suite 是 `libero_10_swap`，需要 `LIBERO_CONFIG_PATH`
  指向 LIBERO-PRO
- `run_measure_demo.sh` 直接恢复 sim 状态，suite 是 `libero_10`，要求 `LIBERO_CONFIG_PATH`
  **不设**（走 LIBERO-plus 的 BDDL）

**两者不能在同一进程内混跑**：`memory_system/pointcloud_action/offline` 会把 LIBERO-plus
插到 `sys.path` 最前，导致 `benchmark.get_benchmark_dict()` 解析到错误的 suite。
