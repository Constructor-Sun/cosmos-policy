# Latent Shift Learning

本文档汇总当前 Phase 8 latent-shift corrector 的目标、数据、模型、训练实现、
LIBERO-10 评估结果、delta-z 方向诊断，以及下一步实验优先级。

文档对应的主要实现为：

```text
bin/phase8_collect_first_chunk_pairs.py
bin/phase8_train_latent_shift.py
bin/phase8_correction_lib.py
bin/phase8_eval_latent_correction.py
```

## 1. 学习目标

corrector 学习从扰动后的 DiT hidden state 预测 clean-minus-perturbed shift：

```text
delta_z_true = z_clean - z_pert
delta_z_pred = corrector(z_pert)
z_corrected = z_pert + delta_z_pred
```

训练损失为归一化 delta-z 上的 MSE：

```text
target_rms = RMS(delta_z_true over the train set)
y = delta_z_true / target_rms
loss = mean((corrector(z_pert) - y) ** 2)
```

推理时恢复真实尺度：

```text
z_corrected = z_pert + alpha * target_rms * corrector(z_pert)
```

这里的残差连接位于校正注入阶段。corrector 本身输出 shift，不在 MLP 内部再次执行
`z_pert + MLP(z_pert)`，否则推理时会重复加入输入 hidden state。

## 2. 当前数据配置

### 2.1 Train

```text
experiments/phase8_first_chunk_pairs/
  all_suites_demo20_camera10_4pseed_env0_train8000
```

配置：

```text
8000 pairs
40 base tasks
libero_spatial + libero_object + libero_goal + libero_10
camera_viewpoints perturbation
train demonstration states
camera_train tuples
policy seeds = 1009, 2003, 3001, 4001
```

### 2.2 Explicit validation

```text
experiments/phase8_first_chunk_pairs/
  all_suites_demo20_camera10_4pseed_env0_isolated_val400
```

配置：

```text
400 pairs
40 base tasks
held-out demonstration states
camera_val tuples
policy seed = 5003
```

train/validation 显式隔离：

```text
state_hash overlap = empty
camera_tuple overlap = empty
policy_seed overlap = empty
```

LIBERO-10 在这里不是 base-policy 的 unseen-task 测试。Cosmos Policy base model 已使用
LIBERO-10 demonstrations。当前测试衡量的是 corrector 对 held-out initial state、camera tuple
和 policy sampling seed 的跨条件泛化。

## 3. 当前 corrector

checkpoint：

```text
experiments/phase8_latent_shift/rmsnorm_dropout01_cuda7/best.pt
```

模型配置：

```text
target layer = 27 (last)
target slots = action only
input/output dim = 2048
hidden dim = 512
dropout = 0.1
normalization = RMSNorm
parameters = 2,101,760
target_rms = 9.554084435395426
```

网络结构：

```text
RMSNorm(2048)
Linear(2048, 512)
GELU
Dropout(0.1)
Linear(512, 2048)
```

优化器：

```text
AdamW
learning rate = 1e-4
weight decay = 1e-4
gradient clipping = 1.0
```

## 4. 数据加载优化

原实现会在每个 epoch 中重复执行 `torch.load`：

```text
train: 8000 loads/epoch
train evaluation: 8000 loads/epoch
validation evaluation: 400 loads/epoch
```

40 epochs 总计约 656000 次 sample file loads，CPU、文件系统和反序列化开销远高于
这个小型 MLP 的计算开销。

当前实现默认启用：

```text
--preload
```

启动时只读取一次原始 `.pt` 文件，并且只缓存指定 layer/target 的：

```text
z_pert
delta_z_true
```

缓存使用 FP16；batch 移动到训练设备后再转换为 FP32。对于当前 `last + action`
配置，8000 train + 400 validation 的缓存约占 12-13 GiB CPU RAM。

后续 epoch：

```text
不再执行 torch.load
DataLoader 从内存缓存组合训练 batch
validation 使用批量 GPU forward
减少逐样本 GPU/CPU synchronization
```

内存不足时可以使用：

```text
--no-preload
```

这会恢复按需磁盘读取。preload 模式保持 `num_workers=0` 是有意的；数据已经在内存中，
增加 worker 可能复制大体积缓存，而不会解决文件读取问题。

## 5. 训练结果

训练共 40 epochs。最佳 validation checkpoint 位于 epoch 38：

| Split | Epoch | MSE delta | Hidden recovery |
|---|---:|---:|---:|
| Train | 38 | 54.5385 | 0.3468 |
| Validation | 38 | 53.5575 | 0.2544 |
| Train | 40 | 54.1058 | 0.3521 |
| Validation | 40 | 53.7980 | 0.2522 |

validation recovery 在后半程基本停留在 0.24-0.254。继续增加 epoch 更可能扩大
train/validation gap，而不是解决当前方向误差。

## 6. LIBERO-10 20-case 评估

任务：

```text
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
```

条件：

```text
camera_viewpoints
20 paired cases
15 preserved
5 flipped
```

结果目录：

```text
experiments/phase8_eval_latent_correction/
  rmsnorm_dropout01_cuda7_libero10_camera20_alignment
```

### 6.1 Action recovery

| Group | N | Mean recovery | Positive | Positive fraction |
|---|---:|---:|---:|---:|
| Preserved | 15 | 0.3659 | 14 | 0.933 |
| Flipped | 5 | 0.3228 | 5 | 1.000 |
| All | 20 | 0.3551 | 19 | 0.950 |

其他结果：

```text
mean hidden recovery = 0.3971
mean corrected action relative error = 0.1382
mean action MSE recovery = 0.5574
num errors = 0
```

`action_recovery_vs_pert` 使用 relative L2 error，而 `action_mse_recovery_vs_pert`
使用 squared error。因此 0.3551 不表示只消除了 35.51% 的 action squared error；同一批
case 的 action MSE recovery 为约 55.74%。

## 7. Delta-z 角度诊断

定义：

```text
t = delta_z_true = z_clean - z_pert
p = delta_z_pred
r = ||p|| / ||t||
c = cosine(t, p)
theta = arccos(c)
```

hidden recovery 可写为：

```text
R = 1 - ||p - t||^2 / ||t||^2
  = 2 * r * cos(theta) - r^2
```

固定预测方向、只优化标量 alpha 时：

```text
alpha_opt = dot(t, p) / ||p||^2
maximum scaled recovery = cos(theta)^2
```

评估同时报告两种 delta：

1. `direct predictor`：在未校正的 baseline `z_pert` 上直接计算 corrector 输出；
2. `online effective`：完整在线多步校正后计算 `z_corrected - z_pert`。

结果：

| Delta | Cosine | Angle | Norm ratio | Optimal alpha | Oracle scaled recovery |
|---|---:|---:|---:|---:|---:|
| Direct predictor | 0.6474 | 49.522 deg | 0.6366 | 1.0733 | 0.4224 |
| Online effective | 0.6477 | 49.494 deg | 0.6371 | 1.0724 | 0.4228 |

### 7.1 结论

当前主要瓶颈是方向误差，不是预测长度。

```text
actual hidden recovery ~= 0.397
per-sample optimal scalar recovery ~= 0.423
```

即使允许每个 case 独立选择最佳 scalar，理论提升也只有约 2.6 个百分点。单一全局
`alpha` 的实际提升通常还会更小。

`cosine^2 ~= 0.42` 表示预测方向只能覆盖真实 delta-z 约 42% 的能量，约 58% 位于预测
方向的正交分量中。调整 `alpha` 不能恢复这些正交分量。

direct 和 online effective 指标几乎相同，说明在当前 20 cases 上，多步在线注入没有
显著改变最终 delta 的方向或尺度。因此“只保存最后一次 denoising capture、在线对所有
conditioned passes 注入”的不一致不是这批结果的首要瓶颈，但仍应在后续训练设计中修正。

## 8. 当前限制

### 8.1 Token-wise mapping

MLP 独立处理每个 `14 x 14` spatial token，共享同一组权重，没有：

```text
spatial attention
convolution
token mixing
global context aggregation
```

camera viewpoint perturbation 产生空间几何变化。仅使用单 token 的 2048 维向量，很难恢复
涉及跨位置对应关系的 clean representation。

### 8.2 条件不足

corrector 只接收 perturbed action hidden，不接收：

```text
camera parameters
reference/clean camera definition
task identity
policy seed
denoising timestep or sigma
video hidden tokens
```

在 MSE 训练下，模型趋向学习：

```text
E[delta_z | z_pert]
```

如果相似的 `z_pert` 对应多个不同的 clean counterfactual shift，模型只能输出条件均值，
形成不可通过延长训练解决的误差。

### 8.3 Last-layer action-only correction

当前只在 layer 27 block entry 校正 action slot。未校正的 video slots 仍会进入最后一个
DiT block，并可能再次影响 action representation。因此即使 action slot shift 完全正确，
也不保证最终 action 完全等于 clean action。

### 8.4 Bottleneck

`2048 -> 512 -> 2048` 将每个 token 压缩到 512 维。该瓶颈可能不足以表达复杂的
camera-induced shift family。

### 8.5 Denoising-step mismatch

collector 在每个 conditioned pass 都覆盖 capture 字典，最终 sample 只保存最后一次
denoising pass。online corrector 则在所有 conditioned passes 上运行，且模型没有 timestep
输入。当前测试中 direct/effective 差异很小，但更完整的模型仍应使用所有 denoising steps
训练并加入 timestep conditioning。

### 8.6 Regularization

当前使用 Dropout 0.1、Weight Decay 1e-4 和 RMSNorm。这些配置可能影响精确回归，但
`optimal_alpha` 接近 1，说明幅度并未严重失真。现有证据不支持将 RMSNorm 或 alpha 视为
当前约 50 度方向误差的主因。

## 9. 下一步实验优先级

### 9.1 Oracle true-delta intervention

优先在 layer 27 action slot 注入真实：

```text
delta_z_true = z_clean - z_pert
```

目的：测量 `last + action-only` 的 action recovery 上限。

解释：

```text
oracle action recovery close to 1
  -> 主要问题是 corrector 预测方向

oracle action recovery clearly below 1
  -> last/action-only 本身存在硬上限，需要更早层或联合校正 video slots
```

### 9.2 引入空间建模

候选结构：

```text
lightweight 2D convolution over the 14 x 14 grid
token-mixing MLP
small Transformer block
```

目标是让每个 token 的 shift 预测能够使用邻域和全局空间信息。

### 9.3 Video-action joint correction

尝试：

```text
--target video_action
```

或者让 action corrector 读取 video hidden 作为 context，但只输出 action delta。

### 9.4 Denoising timestep conditioning

采集所有 denoising steps 的 paired hidden states，并将 timestep/sigma embedding 输入
corrector。训练和在线注入应使用一致的 pass distribution。

### 9.5 Direction-aware objective

可以在 MSE 外增加 per-sample flattened delta cosine loss：

```text
loss = mse_loss + lambda_cos * (1 - cosine(delta_pred, delta_true))
```

也可以单独约束 norm：

```text
loss_norm = abs(||delta_pred|| / (||delta_true|| + eps) - 1)
```

cosine loss 应在完整 sample delta 上计算，而不是只对单个 scalar 或全 batch 合并计算。

### 9.6 Capacity and regularization ablation

在保持数据和 split 不变的情况下比较：

```text
hidden_dim = 512 / 1024 / 2048
dropout = 0.0 / 0.1
RMSNorm / no input norm
token-wise MLP / spatial corrector
```

单纯调 `alpha` 的优先级较低。可以测试 `alpha=1.05` 或 `1.07`，但现有 oracle scalar
结果表明其 hidden-recovery 上限提升有限。

### 9.7 Conditional VAE corrector

训练脚本提供 `--architecture cvae`，直接建模：

```text
p(delta_z | z_pert)
```

模型包含两个不同的 latent distributions：

```text
conditional prior: p(u | z_pert)
training posterior: q(u | z_pert, delta_z_true)
decoder:            delta_z_pred = g(z_pert, u)
```

训练时从 posterior 重参数化采样并优化：

```text
loss = reconstruction_mse + beta * KL(q || p)
```

KL 按 batch 和 latent dimensions 取平均，支持 linear warmup 和 per-dimension free bits。
模型记录 raw KL、参与 KL 的 latent fraction、prior/posterior standard deviation，用于诊断
posterior collapse。

推理时没有真实 shift，因此在线 hook 使用 conditional prior mean 解码，保持单次 correction
确定性。`LatentShiftCVAE.sample_shifts(z_pert, K)` 可以另外生成 K 个 prior candidates，供
best-of-K oracle、分布覆盖率和 uncertainty gating 实验使用；它不会在默认 rollout 中随机
选择 shift。

推荐首个配置：

```bash
python bin/phase8_train_latent_shift.py \
    --input-dir experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_train8000 \
    --val-input-dir experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_isolated_val400 \
    --output-dir experiments/phase8_latent_shift/cvae_h512_z32_beta001 \
    --target-layer last \
    --target action \
    --architecture cvae \
    --hidden-dim 512 \
    --cvae-latent-dim 32 \
    --cvae-beta 0.01 \
    --cvae-kl-warmup-epochs 10 \
    --cvae-free-bits 0.01 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --epochs 40 \
    --preload \
    --device cuda
```

validation hidden recovery 使用 prior mean，代表可部署的单点 correction 效果。Posterior
reconstruction 只说明 encoder/decoder 的拟合能力，不能代替 prior inference 指标。

## 10. 训练命令

```bash
set -eu

REPO=/data1/liu/exp/counterfactual/external/cosmos-policy
ROOT=$REPO/experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_train8000
VAL=$REPO/experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_isolated_val400
OUT=$REPO/experiments/phase8_latent_shift/rmsnorm_dropout01_cuda7

mkdir -p "$OUT"
cd "$REPO"

CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
python bin/phase8_train_latent_shift.py \
    --input-dir "$ROOT" \
    --val-input-dir "$VAL" \
    --output-dir "$OUT" \
    --target-layer last \
    --epochs 40 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --dropout 0.1 \
    --weight-decay 1e-4 \
    --preload \
    --device cuda \
    2>&1 | tee "$OUT/train.log"
```

## 11. Angle-aware evaluation command

```bash
set -eu

REPO=/data1/liu/exp/counterfactual/external/cosmos-policy
PYTHON=/data1/liu/miniconda3/envs/cosmospolicy/bin/python
CKPT=$REPO/experiments/phase8_latent_shift/rmsnorm_dropout01_cuda7/best.pt
SUMMARY=$REPO/experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json
POLICY=$REPO/../../checkpoints/Cosmos-Policy-LIBERO-Predict2-2B
OUT=$REPO/experiments/phase8_eval_latent_correction/rmsnorm_dropout01_cuda7_libero10_camera20_alignment

mkdir -p "$OUT"
cd "$REPO"

CUDA_VISIBLE_DEVICES=7 \
PYTHONNOUSERSITE=1 \
LIBERO_PLUS_PATH="$REPO/../LIBERO-plus" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
HF_HUB_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
NUMBA_CACHE_DIR=/tmp/cosmospolicy-numba-alignment-gpu7 \
MPLCONFIGDIR=/tmp/cosmospolicy-matplotlib-alignment-gpu7 \
PYTHONUNBUFFERED=1 \
"$PYTHON" bin/phase8_eval_latent_correction.py \
    --checkpoint "$CKPT" \
    --summary "$SUMMARY" \
    --output-dir "$OUT" \
    --policy-dir "$POLICY" \
    --t5-extra-embeddings "" \
    --conditions camera_viewpoints \
    --groups preserved flipped \
    --max-pairs 20 \
    --alpha 1.0 \
    --seed 7 \
    --reset-seed 0 \
    --num-denoising-steps 5 \
    --device cuda \
    --fail-fast \
    2>&1 | tee "$OUT/eval.log"
```

评估输出：

```text
results.jsonl       per-case metrics
summary.json        aggregated metrics and delta alignment
eval.log            console output
```

## 12. 当前结论

当前 RMSNorm + Dropout 0.1 corrector 对 LIBERO-10 camera 20-case 有稳定正向效果：

```text
19/20 cases improve
mean action recovery = 0.355
mean action MSE recovery = 0.557
mean hidden recovery = 0.397
```

但 delta-z angle 约 49.5 度，将 scalar-rescaling recovery 上限限制在约 0.423。当前主要问题
是 shift direction prediction，而不是 alpha 或 norm calibration。下一步应先完成 oracle
true-delta intervention，随后优先测试 spatial/video-conditioned corrector，而不是仅增加 epoch
或继续微调全局 alpha。
