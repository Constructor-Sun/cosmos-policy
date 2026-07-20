# GetLatentSpecific

目标：为 Phase8 V1 生成 `camera_viewpoints` perturbation 的 first-chunk correction 数据。

当前只保留一条主线：
1. 使用已经完成的 full rollout summary 作为 clean / perturbed 成败标签来源。
2. 重新前向 first chunk，只保存 `latent_pert` 和 `latent_shift = clean - pert`。

## 当前状态

截至 2026-07-19：

- Full rollout 已经跑完，并按 suite 分布在三个目录：
  - `experiments/phase8_full_rollout/gpu0_libero_spatial_camera_seed7_20pair`
  - `experiments/phase8_full_rollout/gpu1_libero_object_camera_seed7_20pair`
  - `experiments/phase8_full_rollout/gpu2_libero_goal_camera_seed7_20pair`
- 每个 suite 有 10 个 base task summary，合计 30 个 summary。
- 已合并的 summary list 是：
  - `experiments/phase8_full_rollout/main_suites_camera_seed7_20pair/summaries.txt`
- First-chunk collection 已完成，输出目录是：
  - `experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair`
- First-chunk 最终索引文件已经生成：
  - `experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/manifest.jsonl`
  - `experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/summary.json`
- 当前 `summary.json` 记录：
  - `num_pairs = 590`
  - `num_errors = 0`
- `manifest.jsonl` 行数和 `.pt` 文件数也都是 `590`。

## 固定配置

- Suites：`libero_spatial`, `libero_object`, `libero_goal`
- Condition：`camera_viewpoints`
- Groups：`preserved`, `flipped`
- 每个 base task：20 个 clean/pert episode pair
- Policy seed：`7`
- Deterministic reset seed：`0`
- Expected full rollouts：`30 tasks * 20 pairs * 2 = 1200`
- Full rollout total pairs：`30 tasks * 20 pairs = 600`
- Selected first-chunk samples：`590`

Camera perturb 使用 base instruction，不需要额外 language T5 embedding。first-chunk collector 可以显式传 `--t5-extra-embeddings ""` 跳过旧实验的 extra embedding 文件。

First-chunk 只选择 `--groups preserved flipped`。这两个 group 都要求 clean rollout 成功：

- `preserved`: clean success, perturb success
- `flipped`: clean success, perturb failure

因此 600 个 total pairs 中有 10 个没有进入 first-chunk 数据集：

- `both_fail`: 5
- `recovery`: 5

当前选中的 590 个样本分布：

- `preserved`: 449
- `flipped`: 141
- `libero_spatial`: 197
- `libero_object`: 198
- `libero_goal`: 195

## Full Rollout Summary List

当前合并命令是成功的。它从三个 GPU 分片目录收集 summary，而不是从 `main_suites_camera_seed7_20pair` 目录递归查找：

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

find experiments/phase8_full_rollout/gpu0_libero_spatial_camera_seed7_20pair \
     experiments/phase8_full_rollout/gpu1_libero_object_camera_seed7_20pair \
     experiments/phase8_full_rollout/gpu2_libero_goal_camera_seed7_20pair \
  -name '*__camera_viewpoints__20pair_summary.json' \
  | sort > experiments/phase8_full_rollout/main_suites_camera_seed7_20pair/summaries.txt
```

检查：

```bash
wc -l experiments/phase8_full_rollout/main_suites_camera_seed7_20pair/summaries.txt

find experiments/phase8_full_rollout/gpu0_libero_spatial_camera_seed7_20pair \
     experiments/phase8_full_rollout/gpu1_libero_object_camera_seed7_20pair \
     experiments/phase8_full_rollout/gpu2_libero_goal_camera_seed7_20pair \
  -name '*__camera_viewpoints__20pair_summary.json' \
  | wc -l
```

两条命令都应该返回 `30`。

## Generate First Chunk

用 GPU7 生成 first-chunk latent：

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

CUDA_VISIBLE_DEVICES=7 python3 bin/phase8_collect_first_chunk_pairs.py \
  --summary $(cat experiments/phase8_full_rollout/main_suites_camera_seed7_20pair/summaries.txt) \
  --conditions camera_viewpoints \
  --groups preserved flipped \
  --t5-extra-embeddings "" \
  --output-dir experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair
```

如果之前的 first-chunk 进程中断在写 `.pt` 样本阶段，但还没有生成 `manifest.jsonl`，可以继续用同一个 `--output-dir` 重跑；已有 `.pt` 会被同名覆盖。

如果 `manifest.jsonl` 已经存在，再重跑同一个目录会有追加重复行的风险。此时应换一个新的 `--output-dir`，或者先人工确认旧 manifest 是否需要保留。

## Completion Check

first-chunk 完成后应出现：

```text
experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/manifest.jsonl
experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/summary.json
experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/samples/<suite>/<base_task>/camera_viewpoints/epXXXX.pt
```

检查数量：

```bash
wc -l experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/manifest.jsonl

find experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair \
  -name '*.pt' \
  | wc -l
```

两条命令都应该返回 `590`。

检查 summary：

```bash
sed -n '1,220p' experiments/phase8_first_chunk_pairs/main_suites_camera_seed7_20pair/summary.json
```

重点看：

- `num_pairs` 应该是 `590`
- `num_errors` 应该是 `0`
- `conditions` 应该只有 `camera_viewpoints`
- `groups` 应该包含 `preserved` 和 `flipped`

## Saved Payload

每个 `.pt` 样本保存：

- `latent_pert["vae_video"]`
- `latent_pert["layers"][layer]["video"]`
- `latent_pert["layers"][layer]["action"]`
- `latent_shift`，同结构，公式固定为 `clean - pert`
- `meta.perturb_caused_failure = clean_success and not pert_success`

默认层是 `0`, `mid`, `last`，会解析成 VAE + DiT L0/L14/L27。

Full rollout 的 success/failure 标签来自已经完成的完整 rollout。first-chunk collection 会重新前向一次模型来拿 latent，但标签不依赖这次 first-chunk 重跑。

## Generate 7500 Grid First Chunk

截至 2026-07-20，新增一个不重新跑 full rollout 的 first-chunk grid 数据生成方案。该方案仍使用已有 30 个 `camera_viewpoints` summary 来确定：

- suite
- base task
- clean task / perturbed camera task
- clean instruction / pert instruction

但不再只使用旧 summary 中 20 个 episode 的 `preserved/flipped` pair。新模式按 `task x init_state_index x policy_seed` 枚举生成 first-chunk latent：

```text
3 suites * 10 tasks/suite * 50 init states/task * 5 policy seeds = 7500 samples
```

当前选择固定 `env_seed = 0`，只枚举 5 个 policy seed：

```text
policy seeds = 1009, 2003, 3001, 4001, 5003
env seed = 0
```

选择固定 `env_seed = 0` 的理由：

- Cosmos Policy 的 LIBERO eval 工具在创建环境后调用 `env.seed(0)`，并注释说明 seed 会影响 object positions，即使使用 fixed initial state。
- 官方 LIBERO eval 示例使用 `--deterministic True`，常见 deterministic reset seed 是 `0`。
- Cosmos Policy 的 LIBERO policy training 本身读取离线 HDF5 数据，不在线启动环境；训练随机性默认来自 `trainer.seed = 0` 和 `DistributedSampler(seed=0)`。
- 仓库中的 LIBERO dataset regeneration 脚本默认 `--deterministic=True`，并在重渲染/replay 过程中使用 `set_seed_everywhere(seed=0)`。

因此为了和官方训练/测试分布更一致，当前 7500 first-chunk 生成暂时不扩展 env seed，只扩展 policy seed。注意：`GetLatentSpecific.md` 中 20case 和 50case 的对比也显示 deterministic reset seed 会显著影响 group/result，所以这里固定 env/reset seed 是一个有意控制变量的选择，不表示 env seed 不重要。

新增采集模式在：

```text
bin/phase8_collect_first_chunk_pairs.py
```

新增参数：

- `--sample-mode task_init_policy_grid`
- `--policy-seeds`
- `--env-seeds`
- `--num-shards`
- `--shard-index`

输出路径会带上 init/policy/env seed，避免同一 task 的不同 seed 覆盖：

```text
samples/<suite>/<base_task>/camera_viewpoints/initXXXX_pseedYYYY_eseed0.pt
```

样本 meta / manifest 会记录：

- `group = task_init_policy_grid`
- `source_group`: 旧 20-pair summary 中同 init index 的 source group；只对 `0..19` 有意义
- `init_state_index`
- `policy_seed`
- `env_seed`
- `seed_config_index`

### Launch 4-GPU 7500 Collection

封装脚本：

```text
bin/run_phase8_collect_grid_7500_env0.sh
```

直接运行：

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

bin/run_phase8_collect_grid_7500_env0.sh
```

该脚本默认：

```text
RUN = main_suites_camera_grid_50init_5pseed_env0_train7500
GPU_IDS = 0 1 2 3
POLICY_SEEDS = 1009 2003 3001 4001 5003
ENV_SEED = 0
```

4 个 GPU 会同时启动。每个 shard 预期生成：

```text
7500 / 4 = 1875 samples
```

每个 shard 的 suite 分布预期均衡：

```text
libero_spatial: 625
libero_object: 625
libero_goal: 625
```

日志目录：

```text
experiments/phase8_first_chunk_pairs/main_suites_camera_grid_50init_5pseed_env0_train7500/logs/
```

输出目录：

```text
experiments/phase8_first_chunk_pairs/main_suites_camera_grid_50init_5pseed_env0_train7500/gpu0
experiments/phase8_first_chunk_pairs/main_suites_camera_grid_50init_5pseed_env0_train7500/gpu1
experiments/phase8_first_chunk_pairs/main_suites_camera_grid_50init_5pseed_env0_train7500/gpu2
experiments/phase8_first_chunk_pairs/main_suites_camera_grid_50init_5pseed_env0_train7500/gpu3
```

完成检查：

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

RUN=main_suites_camera_grid_50init_5pseed_env0_train7500

for i in 0 1 2 3; do
  echo gpu${i}
  wc -l experiments/phase8_first_chunk_pairs/${RUN}/gpu${i}/manifest.jsonl
  find experiments/phase8_first_chunk_pairs/${RUN}/gpu${i} -name '*.pt' | wc -l
done
```

每个 GPU 预期都是：

```text
1875 manifest rows
1875 .pt files
```

总计：

```text
7500 samples
```

如果要终止该生成任务：

```bash
pkill -f run_phase8_collect_grid_7500_env0.sh
pkill -f phase8_collect_first_chunk_pairs.py
```

如果是在运行脚本的同一个终端里，也可以直接按 `Ctrl+C`。脚本会尝试停止已启动的后台子进程。

## LIBERO-10 First-Chunk Action Recovery Eval

当前已经用 `last_action_10e/final.pt` 在 `libero_10` 的同一个 kitchen scene camera perturb 任务上做了 first-chunk online correction 测量。

这不是 full rollout success eval。当前 eval 只比较 first chunk 的 action 是否更接近 clean action：

```text
action_recovery_vs_pert = 1 - corrected_action_rel_error / baseline_action_rel_error
```

其中：

- `baseline_action_rel_error`: camera perturb action 到 clean action 的相对误差
- `corrected_action_rel_error`: 插入 correction 后 action 到 clean action 的相对误差
- `action_recovery_vs_pert > 0`: correction 后 action 更接近 clean
- `action_recovery_vs_pert < 0`: correction 后 action 更远离 clean

使用的 checkpoint：

```text
experiments/phase8_latent_shift/main_suites_camera_seed7_20pair/last_action_10e/final.pt
```

该 checkpoint 对应：

- epoch: `20`
- target layer: `last` -> DiT layer `27`
- target: `action`
- target rms: `9.434132316358403`
- model: token-wise MLP, `2048 -> 512 -> 2048`

Eval 脚本里有 `alpha` 参数，但训练时没有单独的 alpha。当前使用 `alpha=1.0`，它等价于原始训练尺度：

```text
pred_delta = MLP(pert_hidden) * target_rms
corrected_hidden = pert_hidden + pred_delta
```

因此当前结果没有额外缩放 correction。

### Seed / Reset Source Comparison

| source | data | policy seed | deterministic reset | deterministic reset seed |
| --- | --- | ---: | --- | ---: |
| training | `libero_spatial`, `libero_object`, `libero_goal` camera 20-pair full rollouts | `7` | `true` | `0` |
| case 20 | `libero_10` kitchen scene4 selected 20-pair camera eval | `7` | `true` | `0` |
| case 50 | `libero_10` kitchen scene4 selected 50-pair core perturb camera eval | `7` | `true` | `7` |

Training seed/reset comes from the three training rollout shards:

```text
experiments/phase8_full_rollout/gpu0_libero_spatial_camera_seed7_20pair/batch_summary.json
experiments/phase8_full_rollout/gpu1_libero_object_camera_seed7_20pair/batch_summary.json
experiments/phase8_full_rollout/gpu2_libero_goal_camera_seed7_20pair/batch_summary.json
```

All three use `seed = 7`, `deterministic_reset = true`, and `deterministic_reset_seed = 0`.

### LIBERO-10 20-Pair Camera All Clean-Success

Summary:

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__selected__20pair_summary.json
```

Eval output:

```text
experiments/phase8_eval_latent_correction/libero10_kitchen_scene4_seed7_20case/last_action_epoch20_all20_alpha1
```

Rollout source:

- suite: `libero_10`
- task: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
- policy seed: `7`
- deterministic reset seed: `0`
- clean success: `20/20`
- camera success: `15/20`
- evaluated clean-success pairs: `20/20`
- preserved episodes: `0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17`
- flipped episodes: `3, 6, 13, 18, 19`
- both_fail episodes: none
- recovery episodes: none

Aggregate eval:

```text
num_pairs = 20
num_errors = 0
mean_action_recovery_vs_pert = 0.07519534547555547
mean_hidden_recovery_vs_pert = 0.10247906096347441
mean_corrected_action_rel_error = 0.18825437095020828
```

Action recovery by group:

| group | n | mean action recovery | positive |
| --- | ---: | ---: | ---: |
| preserved | 15 | +5.18% | 8/15 |
| flipped | 5 | +14.53% | 4/5 |
| all | 20 | +7.52% | 12/20 |

Interpretation: across all 20 clean-success pairs, action relative error is reduced by about `7.52%` on average. The flipped subset is stronger than the preserved subset, but the preserved subset still has a small positive mean.

Per episode:

| episode | group | baseline rel err | corrected rel err | action recovery | action MSE recovery | hidden recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | preserved | 0.120431 | 0.136552 | -13.39% | -28.56% | +1.72% |
| 1 | preserved | 0.129647 | 0.144774 | -11.67% | -24.70% | +6.06% |
| 2 | preserved | 0.185556 | 0.170800 | +7.95% | +15.27% | +15.93% |
| 3 | flipped | 0.145983 | 0.099644 | +31.74% | +53.41% | +19.50% |
| 4 | preserved | 0.315186 | 0.349378 | -10.85% | -22.87% | -2.60% |
| 5 | preserved | 0.103208 | 0.090323 | +12.48% | +23.41% | +0.55% |
| 6 | flipped | 0.256336 | 0.242690 | +5.32% | +10.36% | +13.65% |
| 7 | preserved | 0.157641 | 0.178124 | -12.99% | -27.67% | +2.32% |
| 8 | preserved | 0.150611 | 0.168018 | -11.56% | -24.45% | +4.43% |
| 9 | preserved | 0.130388 | 0.099459 | +23.72% | +41.82% | +17.15% |
| 10 | preserved | 0.320314 | 0.349020 | -8.96% | -18.73% | -1.41% |
| 11 | preserved | 0.125036 | 0.055698 | +55.45% | +80.16% | +25.23% |
| 12 | preserved | 0.168407 | 0.132201 | +21.50% | +38.38% | +19.60% |
| 13 | flipped | 0.466622 | 0.483862 | -3.69% | -7.53% | +6.23% |
| 14 | preserved | 0.156395 | 0.146851 | +6.10% | +11.83% | +11.99% |
| 15 | preserved | 0.233664 | 0.265783 | -13.75% | -29.38% | -0.70% |
| 16 | preserved | 0.332199 | 0.320236 | +3.60% | +7.07% | +13.30% |
| 17 | preserved | 0.146979 | 0.102736 | +30.10% | +51.14% | +16.36% |
| 18 | flipped | 0.112995 | 0.085887 | +23.99% | +42.23% | +14.66% |
| 19 | flipped | 0.168837 | 0.143051 | +15.27% | +28.21% | +20.97% |

### LIBERO-10 50-Pair Core Perturb Camera All Clean-Success

Summary:

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_50case_corepert/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__selected__50pair_summary.json
```

Eval output:

```text
experiments/phase8_eval_latent_correction/libero10_kitchen_scene4_seed7_50case_corepert/last_action_epoch20_all50_alpha1
```

Rollout source:

- suite: `libero_10`
- task: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
- policy seed: `7`
- deterministic reset seed: `7`
- clean success: `46/50`
- camera success: `38/50`
- evaluated clean-success pairs: `46/50`
- preserved episodes: `0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48`
- flipped episodes: `4, 5, 7, 14, 25, 30, 41, 42, 49`
- both_fail episodes: `13, 24, 26`
- recovery episodes: `23`

Aggregate eval:

```text
num_pairs = 46
num_errors = 0
mean_action_recovery_vs_pert = -0.13134779337437244
mean_hidden_recovery_vs_pert = 0.008712492566940544
mean_corrected_action_rel_error = 0.15563782160262227
```

Action recovery by group:

| group | n | mean action recovery | positive |
| --- | ---: | ---: | ---: |
| preserved | 37 | -11.45% | 9/37 |
| flipped | 9 | -20.07% | 1/9 |
| all | 46 | -13.13% | 10/46 |

Interpretation: across the 46 clean-success pairs, action relative error gets worse by about `13.13%` on average. The hidden relative error has a small positive mean recovery, but this does not translate to action recovery.

Per episode:

| episode | group | baseline rel err | corrected rel err | action recovery | action MSE recovery | hidden recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | preserved | 0.146579 | 0.120802 | +17.59% | +32.08% | +17.59% |
| 1 | preserved | 0.082228 | 0.066569 | +19.04% | +34.46% | +17.42% |
| 2 | preserved | 0.111773 | 0.154877 | -38.56% | -92.00% | -0.53% |
| 3 | preserved | 0.154400 | 0.167187 | -8.28% | -17.25% | +0.59% |
| 4 | flipped | 0.232958 | 0.269908 | -15.86% | -34.24% | -7.25% |
| 5 | flipped | 0.169030 | 0.188292 | -11.40% | -24.09% | -5.30% |
| 6 | preserved | 0.163976 | 0.184380 | -12.44% | -26.43% | +0.52% |
| 7 | flipped | 0.104177 | 0.117499 | -12.79% | -27.21% | +1.63% |
| 8 | preserved | 0.180598 | 0.204312 | -13.13% | -27.98% | -4.62% |
| 9 | preserved | 0.097582 | 0.108883 | -11.58% | -24.50% | -2.31% |
| 10 | preserved | 0.218001 | 0.242716 | -11.34% | -23.96% | -2.05% |
| 11 | preserved | 0.128390 | 0.122553 | +4.55% | +8.89% | +12.21% |
| 12 | preserved | 0.206667 | 0.239529 | -15.90% | -34.33% | -1.46% |
| 14 | flipped | 0.182655 | 0.216703 | -18.64% | -40.76% | -1.92% |
| 15 | preserved | 0.112603 | 0.109037 | +3.17% | +6.23% | +9.17% |
| 16 | preserved | 0.168839 | 0.190890 | -13.06% | -27.83% | +6.25% |
| 17 | preserved | 0.111183 | 0.129702 | -16.66% | -36.09% | +4.74% |
| 18 | preserved | 0.118421 | 0.113770 | +3.93% | +7.70% | +11.92% |
| 19 | preserved | 0.109908 | 0.110642 | -0.67% | -1.34% | -0.76% |
| 20 | preserved | 0.109568 | 0.134270 | -22.55% | -50.17% | -6.75% |
| 21 | preserved | 0.105338 | 0.149742 | -42.15% | -102.08% | -6.74% |
| 22 | preserved | 0.204965 | 0.238439 | -16.33% | -35.33% | -9.35% |
| 25 | flipped | 0.070787 | 0.114525 | -61.79% | -161.75% | -4.42% |
| 27 | preserved | 0.178458 | 0.207160 | -16.08% | -34.75% | -3.26% |
| 28 | preserved | 0.122930 | 0.155199 | -26.25% | -59.39% | -5.30% |
| 29 | preserved | 0.174834 | 0.171566 | +1.87% | +3.70% | +9.82% |
| 30 | flipped | 0.104911 | 0.086921 | +17.15% | +31.36% | +13.30% |
| 31 | preserved | 0.112134 | 0.132219 | -17.91% | -39.03% | -7.21% |
| 32 | preserved | 0.102045 | 0.085643 | +16.07% | +29.56% | +17.83% |
| 33 | preserved | 0.087685 | 0.103906 | -18.50% | -40.42% | -2.16% |
| 34 | preserved | 0.116189 | 0.158075 | -36.05% | -85.10% | -3.63% |
| 35 | preserved | 0.149389 | 0.176022 | -17.83% | -38.83% | -2.01% |
| 36 | preserved | 0.182723 | 0.205388 | -12.40% | -26.35% | -0.66% |
| 37 | preserved | 0.171760 | 0.188658 | -9.84% | -20.64% | +4.18% |
| 38 | preserved | 0.130805 | 0.146873 | -12.28% | -26.08% | +1.70% |
| 39 | preserved | 0.098476 | 0.117285 | -19.10% | -41.85% | +1.90% |
| 40 | preserved | 0.111427 | 0.141861 | -27.31% | -62.09% | -4.04% |
| 41 | flipped | 0.106873 | 0.151634 | -41.88% | -101.30% | -25.23% |
| 42 | flipped | 0.199285 | 0.243734 | -22.30% | -49.58% | -12.52% |
| 43 | preserved | 0.129590 | 0.155654 | -20.11% | -44.27% | +5.47% |
| 44 | preserved | 0.143342 | 0.182139 | -27.07% | -61.46% | -7.39% |
| 45 | preserved | 0.112976 | 0.132794 | -17.54% | -38.16% | -0.41% |
| 46 | preserved | 0.117649 | 0.125622 | -6.78% | -14.01% | +8.63% |
| 47 | preserved | 0.125019 | 0.108226 | +13.43% | +25.06% | +9.34% |
| 48 | preserved | 0.108473 | 0.103545 | +4.54% | +8.88% | +14.38% |
| 49 | flipped | 0.162589 | 0.183990 | -13.16% | -28.06% | -1.22% |

### Current Interpretation

The 20-pair and 50-pair eval sets use the same policy seed but not the same deterministic reset seed:

```text
20case policy seed = 7, deterministic_reset_seed = 0
50case policy seed = 7, deterministic_reset_seed = 7
```

The 20case flipped episodes become different groups in the 50case run:

```text
ep3  -> preserved
ep6  -> preserved
ep13 -> both_fail
ep18 -> preserved
ep19 -> preserved
```

The training data for `last_action_10e/final.pt` contains only:

```text
libero_spatial, libero_object, libero_goal
```

It does not contain `libero_10`, so the `libero_10` eval is an out-of-distribution transfer test. Current result: the model shows positive transfer on the 20case all-clean-success set, but does not transfer robustly to the 50case core perturb all-clean-success set.
