# Cosmos Policy Counterfactual Phases

本文档汇总 Cosmos Policy 在 LIBERO-plus 对齐任务上的 counterfactual 分析流程、数据位置和当前结果。

当前流程分为三个阶段：

| phase | status | purpose | main output |
| --- | --- | --- | --- |
| Phase 1 | done | 运行 clean + 所有 perturb，得到 paired success/failure labels | paired smoke-test summary / episodes |
| Phase 2 | done | 在 preserved vs flipped 上统计 hidden angular shift，并分析其与 action error 的关系 | `summary.json`, `points_last.csv`, `analysis_last.json` |
| Phase 3 | not implemented | 根据 mean shift 对 action 做 recovery/intervention，衡量 action recovery | pending |

## Phase 1: Clean + All Perturb Smoke Test

### 默认运行目标

脚本现在默认进入 paired smoke test 模式：

```sh
cd /data3/liu/exp/counterfactual/external/cosmos-policy
GPU_ID=0 ./run_libero_smoke_test.sh
```

等价于默认使用以下关键配置：

```sh
SMOKE_MODE=paired
SMOKE_PAIR_SUITE=libero_10
SMOKE_PAIR_BASE_TASK=KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
SMOKE_PAIR_CLEAN_LANGUAGE="put the black bowl in the bottom drawer of the cabinet and close it"
SMOKE_PAIR_PERT_NAME=camera_viewpoints,background_textures,light_conditions,objects_layout,robot_initial_states,sensor_noise
SMOKE_NUM_PAIRS=20
SMOKE_SEED=7
SMOKE_DETERMINISTIC_RESET=true
SMOKE_DETERMINISTIC_RESET_SEED=0
SMOKE_HF_HUB_OFFLINE=1
SMOKE_RUN_ID=vla_jepa_kitchen_scene4_seed7_20case
SMOKE_RESULTS_DIR=./experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_20case
```

目标任务与 VLA-JEPA phase1 的任务一致：

```text
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
```

VLA-JEPA 参考结果路径：

```text
/data3/liu/exp/counterfactual/external/VLA-JEPA/results/phase1/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/summary.json
```

### Perturbation 映射

默认运行 clean 加 6 类非 language perturbation：

| condition | LIBERO-plus category | task pattern |
| --- | --- | --- |
| clean | clean | base task |
| camera_viewpoints | Camera Viewpoints | `_view_50_0_100_0_0_initstate_0` |
| background_textures | Background Textures | `_table_5` |
| light_conditions | Light Conditions | `_light_5` |
| objects_layout | Objects Layout | `_add_*` variants |
| robot_initial_states | Robot Initial States | `_view_0_0_100_0_0_initstate_274` |
| sensor_noise | Sensor Noise | `_view_0_0_100_0_0_initstate_0_noise_5` |

`objects_layout` 按 VLA-JEPA 的方式处理：枚举匹配 base task 的 Objects Layout variants，按 task name 排序，取前 `SMOKE_NUM_PAIRS` 个，每个 variant 跑 1 个 episode。其他 perturbation 是同一个 perturbed task 跑 `SMOKE_NUM_PAIRS` 个 episode。

### Language Perturbation

默认不跑 `language_instructions`，因为严格 language perturbation 需要对应 paraphrase instruction 的 T5 embedding。

当前 VLA-JEPA 对齐的 language task 是：

```text
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_5_view_0_0_100_0_0_initstate_0
```

对应的 language instruction 是：

```text
insert the darkcolored mixing container into the lower storage compartment of the kitchen storage unit and secure it shut
```

推荐先用本地 T5-11B 生成 extra embedding pkl：

```sh
cd /data3/liu/exp/counterfactual/external/cosmos-policy

NUMBA_CACHE_DIR=/tmp/cosmospolicy-numba \
MPLCONFIGDIR=/tmp/cosmospolicy-matplotlib \
PYTHONPATH=../LIBERO-plus \
CUDA_VISIBLE_DEVICES=0 \
/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/make_libero_plus_t5_embeddings.py \
  --output ./experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl
```

`bin/make_libero_plus_t5_embeddings.py` 默认从本地模型目录读取：

```text
/data3/liu/exp/counterfactual/checkpoints/t5-11b
```

生成 pkl 后跑 language perturbation：

```sh
SMOKE_PAIR_PERT_NAME=language_instructions \
SMOKE_NUM_PAIRS=20 \
SMOKE_RUN_ID=vla_jepa_kitchen_scene4_seed7_language_20case \
SMOKE_RESULTS_DIR=./experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_20case \
SMOKE_T5_EXTRA_EMBEDDINGS=./experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl \
GPU_ID=0 ./run_libero_smoke_test.sh
```

如果要跳过预生成，也可以允许 smoke test 内部在线计算：

```sh
SMOKE_PAIR_PERT_NAME=language_instructions \
SMOKE_T5_ALLOW_STRICT_COMPUTE=true \
GPU_ID=0 ./run_libero_smoke_test.sh
```

但这种方式会在 eval 进程内加载 T5-11B，内存峰值和失败面更大；预生成 extra embedding pkl 更稳。

注意：非 language perturbation 使用 base instruction embedding 是有意设计，不代表环境干扰失效。环境变化来自 perturbed task 的 BDDL / init state / scene 设置；base embedding 只是在保持语言指令不变的条件下测试视觉或状态扰动。

对 `language_instructions`，脚本使用 strict instruction mode，不允许 fallback 到 base instruction。缺少 T5 embedding 时会报错，除非显式允许在线计算或提供 extra embedding。

### 当前 20-case 结果

当前已经完成 VLA-JEPA 对齐任务的 20-case smoke test。非 language perturbation 和 language perturbation 是分两次跑的，之后合并到同一个 summary 中。

合并结果路径：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json
```

Cosmos-Policy 当前结果：

| condition | Cosmos success | success rate | clean success, perturb success | clean success, perturb fail | flipped episodes |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 20/20 | 100% | - | - | - |
| camera_viewpoints | 15/20 | 75% | 15 | 5 | 3, 6, 13, 18, 19 |
| background_textures | 0/20 | 0% | 0 | 20 | 0-19 |
| light_conditions | 12/20 | 60% | 12 | 8 | 0, 2, 3, 5, 6, 11, 15, 18 |
| objects_layout | 20/20 | 100% | 20 | 0 | none |
| robot_initial_states | 1/20 | 5% | 1 | 19 | 0, 2-19 |
| sensor_noise | 20/20 | 100% | 20 | 0 | none |
| language_instructions | 19/20 | 95% | 19 | 1 | 13 |

这里的 `clean success, perturb success` 对应 preserved success；`clean success, perturb fail` 对应从 clean 成功翻转到 perturb 失败。当前 clean 在两个 run 中都是 20/20，因此没有 `clean fail, perturb success` recovery case。

与 VLA-JEPA phase1 的 20-case 结果对齐比较：

| condition | Cosmos | VLA-JEPA | alignment note |
| --- | ---: | ---: | --- |
| clean | 20/20 | 20/20 | task match |
| camera_viewpoints | 15/20 | 5/20 | task match |
| background_textures | 0/20 | 20/20 | task match |
| light_conditions | 12/20 | 8/20 | task match |
| objects_layout | 20/20 | 20/20 | per-episode variant names match |
| robot_initial_states | 1/20 | 3/20 | task match |
| sensor_noise | 20/20 | 2/20 | task match |
| language_instructions | 19/20 | 19/20 | task match |

`objects_layout` 的 VLA-JEPA summary 使用 aggregate pattern，所以 top-level task name 不是逐个 variant 名；但 Cosmos 的 per-episode `task_name` 与 VLA-JEPA 的 per-episode `variant_name` 一致。

### 配对与 seed/init-state 审计

已生成配对审计文件：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/seed_and_init_state_alignment_audit.json
```

审计结论：

| condition | seed | deterministic reset | instruction mode | init-state hash vs clean | interpretation |
| --- | --- | --- | --- | --- | --- |
| clean | 7 | true, seed 0 | task | match | base condition |
| camera_viewpoints | 7 | true, seed 0 | base | match | 只改变 camera/view 参数 |
| background_textures | 7 | true, seed 0 | base | match | 只改变 table/background texture |
| light_conditions | 7 | true, seed 0 | base | match | 只改变 light condition |
| objects_layout | 7 | true, seed 0 | base | different, expected | object layout 本身就是 perturb |
| robot_initial_states | 7 | true, seed 0 | base | match | object/env init array 与 clean 相同；robot initstate 由 LIBERO-plus env wrapper 解析 `initstate_274` 后施加 |
| sensor_noise | 7 | true, seed 0 | base | match | 只改变 sensor/image noise |
| language_instructions | 7 | true, seed 0 | strict | match | 只改变 language instruction/T5 embedding |

因此，当前 20-case 设置满足配对测试要求：在同一个 VLA-JEPA 任务、同一 suite、同一 seed、同一 deterministic reset、同一 episode index 下比较 clean 和 perturb。除了被测 perturb 维度以外，其余可由脚本控制和审计的因素保持一致。

两个例外需要按 perturb 定义理解：

- `objects_layout`：object layout 就是被测扰动，所以 init-state hash 不应与 clean 相同；脚本按 VLA-JEPA 的方式枚举并排序 object-layout variants，每个 variant 跑 1 个 episode。
- `robot_initial_states`：审计中的 object/env init array hash 与 clean 相同是正常的；真正的 robot 初始状态扰动来自 LIBERO-plus env wrapper 对 task name 中 `initstate_274` 的解析和施加。

### 输出文件

默认输出目录：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_20case/
```

主要文件：

```text
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__selected__20pair_summary.json
clean/episodes.json
camera_viewpoints/episodes.json
background_textures/episodes.json
light_conditions/episodes.json
objects_layout/episodes.json
robot_initial_states/episodes.json
sensor_noise/episodes.json
logs/*.txt
```

language perturbation 输出目录：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_20case/
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__language_instructions__20pair_summary.json
clean/episodes.json
language_instructions/episodes.json
logs/*.txt
```

当前合并 summary 和配对审计目录：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json
seed_and_init_state_alignment_audit.json
```

`summary.json` 记录每个 condition 的 success rate、successes、num_trials、log path 和 episodes path。

每个 `episodes.json` 记录 per-episode metadata，包括：

```text
run_id
suite
seed
deterministic_reset
deterministic_reset_seed
base_task
condition
category
task_id
task_name
language
episode
init_state_index
success
log_path
instruction_mode
```

这些字段用于后续统计 preserved vs flipped、action error、angular error 和 recovery 关系。

### 可复现性设置

脚本中固定了以下设置：

```text
SMOKE_SEED=7
SMOKE_DETERMINISTIC_RESET=true
SMOKE_DETERMINISTIC_RESET_SEED=0
PYTHONHASHSEED=$SMOKE_SEED
CUBLAS_WORKSPACE_CONFIG=:4096:8
randomize_seed=False
deterministic=True
```

因此，同一机器、同一 checkpoint、同一代码、同一结果目录配置下，重复运行应得到几乎完全一致的 episode 选择和结果。CUDA、GPU kernel、driver 或硬件层面的不可控非确定性不在脚本可保证范围内。

如果要比较两次运行，建议换一个 `SMOKE_RESULTS_DIR` 和 `SMOKE_RUN_ID`，避免新日志和旧日志混在同一目录中。

### 日志清理

脚本 monkeypatch 了 `run_libero_eval` 的 `print`，过滤如下 step action spam：

```text
t: ...    action:
```

保留 query time、selected seed、predicted value、success rate、final results 等关键信息。

### 验证

修改后已做过脚本语法和 embedded Python 编译检查：

```sh
sh -n run_libero_smoke_test.sh
```

```sh
PYTHONPYCACHEPREFIX=/tmp/cosmospolicy-pycache \
/data2/haoze/miniconda3/envs/cosmospolicy/bin/python -m py_compile /tmp/cosmos_smoke_embedded.py
```

两项均通过。

### 注意事项

不要在 smoke test 正在运行时编辑 `run_libero_smoke_test.sh`。之前出现过类似：

```text
./run_libero_smoke_test.sh: 650: terministic_reset,: not found
```

这类错误通常来自 shell 正在执行脚本时文件被改写，导致解释器读到不完整或错位的内容。

## Phase 2: Preserved vs Flipped Angular Analysis

Phase 2 使用 Phase 1 的 paired metadata 定义 clean/perturb episode pair，并在 Cosmos Policy forward pass 中捕获 hidden states 与 action outputs，统计 preserved/flipped 的 angular shift 以及 angular shift 与 action error 的相关性。

标签定义：

```text
preserved = clean success && perturb success
flipped   = clean success && perturb fail
recovery  = clean fail    && perturb success
both_fail = clean fail    && perturb fail
```

当前 20-case clean run 是 20/20 success，因此本轮 all-perturb preserved/flipped 分析没有 `recovery` 和 `both_fail`。

### Phase 2 代码位置

```text
bin/run_phase2_angular_cosmos.py
bin/analyze_phase2_angular_cosmos.py
```

`run_phase2_angular_cosmos.py` 负责：

- 读取 Phase 1 paired summary 和 per-condition `episodes.json`
- 重新构造 clean/perturb 第一帧 observation
- 对 Cosmos Policy 跑 clean/perturb forward
- 捕获 selected DiT block 的 `video` slot hidden state 和 `action` slot hidden state
- 保存 action outputs、hidden states 和 pair-level `metrics.json`

`analyze_phase2_angular_cosmos.py` 负责：

- 汇总每个 pair 的 angular metrics 和 action error
- 输出 preserved/flipped angular mean/std/ratio
- 输出 angular shift 与 action error 的 Pearson / Spearman correlation
- 写出 row-level CSV 和 aggregate JSON

Objects Layout variants 使用 variant BDDL 和 LIBERO-plus new-object init state：

```text
../LIBERO-plus/libero/libero/bddl_files/libero_10/*_add_*.bddl
../LIBERO-plus/libero/libero/init_files/libero_newobj/libero_10/*_add_*.pruned_init
```

### Phase 2 指标定义

Cosmos Policy 的 LIBERO latent layout 中：

```text
2 current wrist image
3 current primary image
4 action
```

本阶段使用：

```text
video_embedding[layer]  = x[:, [2, 3], :, :, :]
action_embedding[layer] = x[:, [4], :, :, :]
```

对每个 clean/perturb pair：

```text
video_angle_deg         = arccos(cos(flatten(video_clean), flatten(video_pert)))
action_hidden_angle_deg = arccos(cos(flatten(action_clean), flatten(action_pert)))
action_error            = ||action_pert - action_clean|| / ||action_clean||
```

### Phase 2 输入数据

Phase 2 使用 Phase 1 的合并 summary：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/
  KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json
```

严格 language perturbation 额外使用：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl
```

### Phase 2 Capture Command

运行 clean + 所有 perturb，并保留 preserved + flipped：

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="../LIBERO-plus:${PYTHONPATH}" \
python bin/run_phase2_angular_cosmos.py \
  --groups preserved flipped \
  --layers last \
  --output-dir experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_perturb_preserved_flipped_last
```

### Phase 2 Analysis Command

```bash
python bin/analyze_phase2_angular_cosmos.py \
  --results-dir experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_perturb_preserved_flipped_last \
  --layer last \
  --groups preserved flipped \
  --csv experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_perturb_preserved_flipped_last/points_last.csv \
  --out-json experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_perturb_preserved_flipped_last/analysis_last.json
```

### Phase 2 输出数据

```text
experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_perturb_preserved_flipped_last/
├── summary.json
├── analysis_last.json
├── points_last.csv
├── background_textures/ep*/metrics.json
├── camera_viewpoints/ep*/metrics.json
├── language_instructions/ep*/metrics.json
├── light_conditions/ep*/metrics.json
├── objects_layout/ep*/metrics.json
├── robot_initial_states/ep*/metrics.json
└── sensor_noise/ep*/metrics.json
```

`summary.json` 保存 capture metadata 和 pair-level metrics。`points_last.csv` 是 row-level table，适合后续画 scatter plot。`analysis_last.json` 保存下面的 aggregate 结果。

### Phase 2 Pair Counts

`--layer last` 对应 DiT block `27`。本轮总计 140 个 clean/perturb pair。

| perturb | preserved | flipped |
| --- | ---: | ---: |
| background_textures | 0 | 20 |
| camera_viewpoints | 15 | 5 |
| language_instructions | 19 | 1 |
| light_conditions | 12 | 8 |
| objects_layout | 20 | 0 |
| robot_initial_states | 1 | 19 |
| sensor_noise | 20 | 0 |
| total | 87 | 53 |

### Phase 2 Video Angular: Preserved vs Flipped

Values are mean angular shift in degrees. `-` means the perturb has only one outcome group, so the ratio is not defined.

| perturb | n preserved | n flipped | preserved angular | flipped angular | flipped / preserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| background_textures | 0 | 20 | - | 42.429 | - |
| camera_viewpoints | 15 | 5 | 34.030 | 34.421 | 1.012x |
| language_instructions | 19 | 1 | 7.808 | 5.987 | 0.767x |
| light_conditions | 12 | 8 | 26.465 | 26.933 | 1.018x |
| objects_layout | 20 | 0 | 29.090 | - | - |
| robot_initial_states | 1 | 19 | 33.180 | 33.477 | 1.009x |
| sensor_noise | 20 | 0 | 17.420 | - | - |

### Phase 2 Action Hidden Angular: Preserved vs Flipped

Values are mean angular shift in degrees for the action latent slot hidden state.

| perturb | n preserved | n flipped | preserved angular | flipped angular | flipped / preserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| background_textures | 0 | 20 | - | 10.044 | - |
| camera_viewpoints | 15 | 5 | 8.086 | 9.108 | 1.126x |
| language_instructions | 19 | 1 | 4.387 | 4.139 | 0.943x |
| light_conditions | 12 | 8 | 7.029 | 7.006 | 0.997x |
| objects_layout | 20 | 0 | 8.889 | - | - |
| robot_initial_states | 1 | 19 | 10.824 | 12.646 | 1.168x |
| sensor_noise | 20 | 0 | 1.784 | - | - |

### Phase 2 Video Angular vs Action Error

| perturb | n | Pearson r | Spearman rho |
| --- | ---: | ---: | ---: |
| background_textures | 20 | 0.524 | 0.483 |
| camera_viewpoints | 20 | 0.878 | 0.910 |
| language_instructions | 20 | 0.707 | 0.708 |
| light_conditions | 20 | 0.483 | 0.441 |
| objects_layout | 20 | 0.070 | 0.059 |
| robot_initial_states | 20 | -0.070 | 0.056 |
| sensor_noise | 20 | 0.393 | 0.083 |
| POOLED | 140 | 0.563 | 0.669 |

### Phase 2 Action Hidden Angular vs Action Error

| perturb | n | Pearson r | Spearman rho |
| --- | ---: | ---: | ---: |
| background_textures | 20 | 0.687 | 0.624 |
| camera_viewpoints | 20 | 0.971 | 0.875 |
| language_instructions | 20 | 0.859 | 0.707 |
| light_conditions | 20 | 0.855 | 0.805 |
| objects_layout | 20 | 0.951 | 0.959 |
| robot_initial_states | 20 | 0.828 | 0.741 |
| sensor_noise | 20 | 0.484 | 0.310 |
| POOLED | 140 | 0.882 | 0.925 |

### Phase 2 Interpretation

Preserved-vs-flipped mean angular gap is weak and condition-dependent. Several perturbations are one-sided (`background_textures` all flipped; `objects_layout` and `sensor_noise` all preserved), so flipped/preserved ratios should not be treated as a stable separation metric in this run.

The angular-vs-action-error relationship is much stronger for `action_hidden` than for `video`. Pooled `action_hidden` angular shift has Pearson `0.882` and Spearman `0.925` with relative action error, while pooled `video` angular shift has Pearson `0.563` and Spearman `0.669`. This suggests that Cosmos Policy's action hidden slot is a better proxy for action-output error than the observed visual slots, even when preserved/flipped outcome separation is modest.

## Phase 3: Mean-Shift Action Recovery

Phase 3 is now implemented as a clean DiT block-entry action-slot
intervention. It follows the FastWAM/VLA-JEPA two-stage design, but adapts the
hook site to Cosmos Policy's DiT latent sequence.

Cosmos LIBERO latent layout:

```text
2 current wrist image
3 current primary image
4 action
```

Phase 3 intervenes on the action slot at a selected DiT block entrance:

```text
x[:, [4], :, :, :] = x[:, [4], :, :, :] + alpha * direction
```

The default direction is computed per perturbation over all flipped samples:

```text
direction[perturb, layer] =
    mean_flipped(clean_action_slot_hidden[layer] - pert_action_slot_hidden[layer])
```

This matches the requested two-pass procedure:

1. **Run A / capture**: for each perturbation, run all flipped samples once,
   capture clean/pert action-slot hidden states, and save the perturb-specific
   mean shift.
2. **Run B / recovery**: for each flipped sample, rerun the perturbed forward
   with the perturb-specific mean shift injected at each requested layer/alpha.

### Phase 3 Code Position

```text
bin/cosmos_layer_shift.py
bin/cosmos_phase3_utils.py
bin/run_phase3_mean_shift_recovery_cosmos.py
```

`cosmos_layer_shift.py` provides a temporary `register_forward_pre_hook`
context on `model.net.blocks[L]`. No Cosmos model source files are modified.

By default the hook only captures/intervenes on the conditioned CFG pass. This
matches Phase 2, whose hidden capture ignores the unconditioned pass.

### Phase 3 Recovery Metric

The primary recovery ratio follows the Phase 3 definition:

```text
error(a, b) = ||a - b|| / (||b|| + eps)

recovery_vs_pert =
    1 - error(action_intervened, action_clean)
        / error(action_pert, action_clean)
```

The script also records the FastWAM/VLA-JEPA-style MSE recovery:

```text
mse_recovery_vs_pert =
    1 - MSE(action_intervened, action_clean)
        / MSE(action_pert, action_clean)
```

Interpretation:

```text
recovery > 0  intervention moved action closer to clean
recovery ~= 0 no improvement
recovery < 0  intervention moved action farther from clean
```

### Phase 3 Run Command

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="../LIBERO-plus:${PYTHONPATH}" \
/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/run_phase3_mean_shift_recovery_cosmos.py \
  --layers 0 mid last \
  --alphas 0 0.25 0.5 0.75 1.0 1.25 1.5 \
  --direction-source all_flipped_mean \
  --output-dir experiments/phase3_mean_shift_cosmos/kitchen_scene4_seed7_all_perturb_flipped_action_slot
```

Useful smaller test:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="../LIBERO-plus:${PYTHONPATH}" \
/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/run_phase3_mean_shift_recovery_cosmos.py \
  --conditions camera_viewpoints \
  --layers last \
  --alphas 0 1.0 \
  --max-pairs 1 \
  --output-dir experiments/phase3_mean_shift_cosmos/debug_camera_one_pair
```

### Phase 3 Outputs

```text
experiments/phase3_mean_shift_cosmos/<run>/
├── summary.json
├── background_textures/
│   ├── mean_shift_directions.pt
│   ├── direction_consistency.json
│   ├── metrics.json
│   └── ep*/...
├── camera_viewpoints/...
├── language_instructions/...
├── light_conditions/...
└── robot_initial_states/...
```

Per episode:

```text
epXX/action_clean.npy
epXX/action_pert.npy
epXX/hidden_action_clean.pt
epXX/hidden_action_pert.pt
epXX/L00/action_alpha_*.npy
epXX/L14/action_alpha_*.npy
epXX/L27/action_alpha_*.npy
```

`summary.json` includes:

```text
groups[condition].results[episode].layers[layer].per_alpha[alpha]
by_perturbation[condition].layers[layer][alpha]
```

The aggregate table reports mean/median action recovery, MSE recovery, hidden
recovery, and `positive_recovery_rate` for each perturbation/layer/alpha.
