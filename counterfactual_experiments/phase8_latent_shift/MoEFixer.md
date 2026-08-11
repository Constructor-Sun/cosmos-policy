# MoE Fixer：Phase 8 Latent Shift Mixture of Experts

本文档说明 Phase 8 latent-shift corrector 中 Mixture of Experts（MoE）的设计、
训练方式、指标解释、first-chunk 评估与完整 rollout 结果。

对应的主要实现为：

~~~text
bin/phase8_correction_lib.py
bin/phase8_train_latent_shift.py
bin/phase8_eval_latent_correction.py
bin/phase8_fast_full_rollout.py
run_phase8_fast_full_rollout.sh
~~~

当前推荐 checkpoint：

~~~text
experiments/phase8_latent_shift/moe_full_e16_top2_cuda6/best.pt
~~~

它是 16 专家、top-2 路由的 full MoE。它在 first-chunk 离线评估上有效，但当前
trajectory-wide 持续校正会破坏 rollout，不能直接作为完整轨迹修复器使用。

## 1. 学习目标

corrector 从扰动后的 DiT hidden state 预测 clean-minus-perturbed shift：

~~~text
delta_z_true = z_clean - z_pert
delta_z_pred = corrector(z_pert)
z_corrected = z_pert + alpha * target_rms * delta_z_pred
~~~

训练目标经过全局 RMS 归一化：

~~~text
y = delta_z_true / target_rms
MSE = mean((delta_z_pred - y)²)
~~~

当前数据的 target RMS 为：

~~~text
target_rms = 9.554084435395426
~~~

残差连接只在注入时执行。专家网络输出的是 shift，而不是完整 hidden state；否则会把
z_pert 重复加入两次。

## 2. 为什么引入 MoE

单个 MLP 必须用同一组参数拟合所有扰动类型。MoE 增加多个专家，并让 router 根据每个
样本的 hidden 特征动态选择 top-k 专家：

~~~text
z_pert
  ├─ sample-level router ──> top-2 experts + gates
  ├─ selected expert A ────> delta_A
  └─ selected expert B ────> delta_B

delta_z_pred = gate_A * delta_A + gate_B * delta_B
~~~

这里的“专家分工”由训练数据和路由共同学习，例如不同专家可能分别擅长某类 camera
扰动、某类任务状态或某种 latent-shift 方向。实现没有硬编码“专家 1 负责维度
1–100”这样的语义。

## 3. Router

输入 hidden 的最后一维是 2048。router 不对每个 token 单独路由，而是对一个 batch
sample 路由一次：

1. 对所有 token/spatial 轴做 mean pooling。
2. 对相同位置做 RMS pooling。
3. 拼接为 4096 维路由特征。
4. 经过 Linear(4096, num_experts) 得到 logits。
5. 选择 top_k_experts，并对被选中的 logits 做 softmax。

mean 保留有符号的整体结构，RMS 保留可能在 mean 中相互抵消的局部能量。一个样本选出的
专家组合会用于该样本的全部 token。

训练时默认给 router logits 加 0.1 的 Gaussian noise，推理时 model.eval() 自动关闭
noise。只执行被选中的专家，避免同时保留全部专家的 activation。

当前默认设置：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| num_experts | 16 | 专家总数 |
| top_k_experts | 2 | 每个样本激活的专家数 |
| router_temperature | 1.0 | router softmax 温度 |
| router_noise | 0.1 | 仅训练阶段使用的 logit noise |

8–16 个专家、top-2 或 top-3 是合理起点。专家更多并不自动带来更好的“全局覆盖”，因为
router 可能集中使用少量专家，所以必须同时观察 expert load。

## 4. 三种专家模式

### 4.1 full：当前默认和推荐模式

每个专家都预测完整的 2048 维 shift：

~~~text
RMSNorm(2048)
router: pooled 4096 -> 16

each expert:
  Linear(2048, 512)
  GELU
  Dropout(0.1)
  Linear(512, 2048)
~~~

对一个样本，top-2 专家的两个完整 shift 按 gate 加权求和。专家的参数彼此独立，因此仍可
学习不同校正模式，但不会预先限制它们只能修改固定通道。

优点：

- 每个被选专家都可以校正全部 2048 维。
- 不依赖“shift 在原始 channel 坐标中分段稀疏”的假设。
- 当前验证集和 first-chunk LIBERO-10 结果最好。

缺点：

- 参数量和计算量高于 low-rank。
- 更容易在 8000 个训练 pair 上过拟合。

### 4.2 low_rank：学习稠密低秩方向

每个专家学习自己的 dense basis：

~~~text
B_e: 2048 x rank
c_e(z): rank-dimensional coefficients
delta_e = B_e * c_e(z)
~~~

rank=32 时，每个专家先预测 32 个系数，再通过学习到的 2048×32 basis 展开成完整
2048 维 shift。basis 的每个列向量都可以在全部 2048 个原始维度上非零。

因此该实现属于“学习到的低秩方向”，但不是固定 PCA：

- 没有把离线 PCA basis 直接载入模型。
- basis 与系数网络端到端联合训练。
- basis 使用正交初始化。
- 训练中加入 basis orthogonality loss，抑制重复方向。

低秩并不等于只修改少数 raw channels。一个 rank-32 子空间完全可以在 2048 个坐标上表现
为稠密 shift。

当前 rank-32 结果明显弱于 full MoE，说明单专家 32 维 basis 的容量或优化目前不够。可以
后续比较 rank=64/128，或者用离线 PCA 初始化 basis，但这两项当前尚未实现。

### 4.3 partitioned：固定原始维度分片

16 个专家把 2048 维连续切成 16 段，每段约 128 维。专家 e 只输出自己的固定 slice：

~~~text
expert 0 -> dimensions    0:128
expert 1 -> dimensions  128:256
...
expert 15 -> dimensions 1920:2048
~~~

严格 partitioned + top-2 对每个样本只产生两个 slice，其余约 1792 个维度直接输出零。
“16 个专家总体覆盖 2048 维”只表示跨专家集合的覆盖；它不表示单个样本得到了全维校正。

对 32 个现有 action 样本的诊断中，把 2048 维分成 16 个连续段后，能量最高的两段平均
只覆盖约 21% 的真实 shift energy。因此 top-2 partitioned 会在单样本上漏掉约 79% 的
待校正能量。这个数字是当前样本上的经验诊断，不是对所有数据的数学定理，但足以说明它
不应作为默认模式。

## 5. “shift 很低维”和“shift 分布很散”并不矛盾

需要区分两个坐标系中的概念：

- 低秩：许多 shift 可以由少量共同方向的线性组合表示。
- raw-channel 稀疏：单个 shift 只在少数原始 2048 维坐标上非零或集中。

例如一个 PCA direction 本身可以在 2048 个坐标上全部非零。所有 shift 即使都位于少量
PCA directions 张成的低维子空间中，在 raw channel 坐标下仍可能非常分散。

当前已有 PCA-128 诊断可以解释大约 93% 的 shift energy，这支持“数据近似低秩”；而
top-2 连续 raw-channel 分段只能覆盖约 21%，说明它并不具有对应的分段稀疏性。这正是
low_rank 模式使用 dense learned basis、full 模式允许全维输出的原因。

## 6. 训练损失与路由正则

full 和 partitioned 模式的总损失为：

~~~text
loss =
    mse_loss
  + 1e-2 * load_balance_loss
  + 1e-3 * router_z_loss
~~~

low_rank 额外加入：

~~~text
+ 1e-3 * basis_orthogonality_loss
~~~

各项作用：

- mse_loss：拟合归一化后的真实 delta-z。
- load_balance_loss：减少专家塌缩到少数 route 的风险。
- router_z_loss：限制 router logits 尺度。
- basis_orthogonality_loss：减少同一 low-rank expert 内重复的 basis directions。

## 7. 日志指标怎么读

训练日志示例：

~~~text
epoch=023 loss=0.589643 mse=0.574891
train_rec=0.3944 val_rec=0.2575
route_min=0.033 route_max=0.130
~~~

含义：

- loss：MSE 加 MoE 辅助损失后的总 loss。
- mse：归一化 delta-z 上的 reconstruction MSE。
- train_rec / val_rec：hidden recovery，相对“不做校正”的 MSE 改善率。
- route_min / route_max：该 epoch 中最少和最多被选中的 expert load。

hidden recovery 定义为：

~~~text
recovery = 1 - MSE(z_corrected, z_clean) / MSE(z_pert, z_clean)
~~~

解释：

- recovery > 0：校正后更接近 clean。
- recovery = 0：与不校正相同。
- recovery < 0：校正使 hidden 更差。
- recovery = 0.25：相对 perturbed baseline，hidden MSE 降低约 25%。

16 个专家完全均匀时，每个专家的 route load 约为 1/16=0.0625。route_min/max 只反映
负载均衡，不代表校正质量。low-rank 在 epoch 23 的 0.041–0.077 比 full 的
0.033–0.130 更均匀，但其 val recovery 反而更低。

## 8. 当前训练结果

训练数据：

~~~text
train:
experiments/phase8_first_chunk_pairs/
  all_suites_demo20_camera10_4pseed_env0_train8000

explicit validation:
experiments/phase8_first_chunk_pairs/
  all_suites_demo20_camera10_4pseed_env0_isolated_val400
~~~

两者隔离了 state hash、camera tuple 和 policy seed。目标为 layer 27 的 action slots。

### 8.1 Full MoE

~~~text
checkpoint:
experiments/phase8_latent_shift/moe_full_e16_top2_cuda6/best.pt

experts = 16
top-k = 2
mode = full
best epoch = 52
validation hidden recovery = 0.2692
~~~

训练继续到 epoch 80 后，train recovery 上升到 0.5500，但 validation recovery 回落到
0.2414，说明后期出现明显 train/validation gap。评估应使用 best.pt，而不是 final.pt。

### 8.2 Low-rank MoE

~~~text
checkpoint:
experiments/phase8_latent_shift/moe_low_rank_e16_top2_r32_cuda7/best.pt

experts = 16
top-k = 2
mode = low_rank
rank = 32
observed best validation hidden recovery ≈ 0.1949
~~~

rank-32 的 validation recovery 大致在 0.18–0.195 平台，明显低于 full MoE。

## 9. 两种 MoE 训练命令

以下命令从 cosmos-policy 仓库根目录运行。没有设置 Python 解释器绝对路径，也没有重复
设置 num_experts=16、top_k_experts=2、dropout=0.1、epochs=80 等现有默认值。控制台输出
同时由 tee 写入 train.log。

### 9.1 GPU 6：full MoE

~~~bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy

OUT=experiments/phase8_latent_shift/moe_full_e16_top2_cuda6
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 \
python bin/phase8_train_latent_shift.py \
    --input-dir experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_train8000 \
    --val-input-dir experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_isolated_val400 \
    --output-dir "$OUT" \
    --target-layer last \
    --batch-size 32 \
    --eval-batch-size 64 \
    --architecture moe \
    2>&1 | tee "$OUT/train.log"
~~~

### 9.2 GPU 7：low-rank MoE

~~~bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy

OUT=experiments/phase8_latent_shift/moe_low_rank_e16_top2_r32_cuda7
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
python bin/phase8_train_latent_shift.py \
    --input-dir experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_train8000 \
    --val-input-dir experiments/phase8_first_chunk_pairs/all_suites_demo20_camera10_4pseed_env0_isolated_val400 \
    --output-dir "$OUT" \
    --target-layer last \
    --batch-size 32 \
    --eval-batch-size 64 \
    --architecture moe \
    --expert-mode low_rank \
    2>&1 | tee "$OUT/train.log"
~~~

## 10. 三个不同层级的评估

### 10.1 Validation hidden recovery

这是训练脚本在 held-out latent pairs 上计算的指标，只回答：

> 预测的 delta-z 是否让保存的 first-chunk hidden 更接近 clean hidden？

它不运行环境，也不直接衡量 action 或任务成功率。

### 10.2 First-chunk action recovery

对同一个 LIBERO-10 任务的 20 个 paired cases，full MoE 结果为：

~~~text
mean action recovery vs perturbed = 0.3849
mean hidden recovery vs perturbed = 0.4653
positive action recovery = 19/20
~~~

结果目录：

~~~text
experiments/phase8_eval_latent_correction/
  moe_full_e16_top2_libero10_kitchen_scene4_20case
~~~

这说明在与训练目标一致的 first policy chunk 上，checkpoint 大多数时候能让 hidden 和
action 更接近 clean。

但 action recovery 是连续值误差指标，不等于 rollout success recovery。一个 action
更接近 clean，不保证随后几百步的闭环轨迹一定成功。

复现该 20-case first-chunk 评估：

~~~bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy

OUT=experiments/phase8_eval_latent_correction/moe_full_e16_top2_libero10_kitchen_scene4_20case
mkdir -p "$OUT" /tmp/cosmospolicy-numba-moe-eval /tmp/cosmospolicy-matplotlib-moe-eval

CUDA_VISIBLE_DEVICES=6 \
PYTHONNOUSERSITE=1 \
LIBERO_PLUS_PATH="$PWD/../LIBERO-plus" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
HF_HUB_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
NUMBA_CACHE_DIR=/tmp/cosmospolicy-numba-moe-eval \
MPLCONFIGDIR=/tmp/cosmospolicy-matplotlib-moe-eval \
PYTHONUNBUFFERED=1 \
python bin/phase8_eval_latent_correction.py \
    --checkpoint experiments/phase8_latent_shift/moe_full_e16_top2_cuda6/best.pt \
    --summary experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__selected__20pair_summary.json \
    --output-dir "$OUT" \
    --policy-dir ../../checkpoints/Cosmos-Policy-LIBERO-Predict2-2B \
    --t5-extra-embeddings "" \
    --max-pairs 20 \
    2>&1 | tee "$OUT/eval.log"
~~~

### 10.3 完整 rollout recovery

完整 rollout 使用相同的 initial states 分别运行：

~~~text
20 perturbed episodes
20 perturbed + corrected episodes
total = 40 rollouts
~~~

不再额外运行 clean episodes。perturbed 是对照组，corrected 是实验组。

指标定义：

~~~text
failure recovery =
  perturbed 失败、corrected 成功的 episode 数
  / perturbed 失败 episode 数

preservation =
  perturbed 成功、corrected 仍成功的 episode 数
  / perturbed 成功 episode 数

harm rate =
  perturbed 成功、corrected 变失败的 episode 数
  / perturbed 成功 episode 数
~~~

## 11. 当前完整 rollout 结果

任务：

~~~text
libero_10:
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
~~~

结果：

| 条件 | 成功数 | 成功率 |
|---|---:|---:|
| perturbed | 16/20 | 0.800 |
| corrected | 0/20 | 0.000 |

配对恢复指标：

~~~text
perturbed failures = 4
recovered failures = 0
failure recovery = 0/4 = 0%

perturbed successes = 16
preserved successes = 0
preservation = 0/16 = 0%

harmed successes = 16
harm rate = 16/16 = 100%
~~~

summary：

~~~text
experiments/phase8_full_rollout/
  moe_full_e16_top2_libero10_kitchen_scene4_20pair/
  libero_10/
  KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/
  KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
  __camera_viewpoints__20pair_summary.json
~~~

终端中的：

~~~text
Total episodes: 20
Total successes: 0
Overall success rate: 0.0%
~~~

指的是刚刚完成的 corrected 条件自身的 20 个 episodes，不是把 perturbed 和 corrected
合成 20 个，也不是 40 个 rollout 的平均成功率。真正的配对解释是：perturbed 原本成功
16 个，加入 corrector 后 20 个全部失败。

所有 20 个 corrected episode log 都是 Success: False，因此这不是 summary 解析错误。

## 12. 为什么离线有效、完整 rollout 却归零

最可能的原因是训练分布与注入调度不一致：

- 训练数据只有每个 episode 的 first chunk。
- first-chunk 评估也只校正第一个 policy chunk。
- 当前 phase8_fast_full_rollout.py 在 corrected episode 的每一次 get_action 调用中都注册
  correction hook。
- 轨迹进入后续状态后，hidden 已不再来自训练时的 first-chunk 分布。
- 持续注入可能在每个 action chunk 累积偏差，并改变后续 observation，使闭环分布进一步
  漂移。

因此当前实验能够得出的结论是：

> 这个 first-chunk checkpoint 不适合 trajectory-wide persistent correction。

它还不能单独证明：

> first-query-only correction 一定无效。

也不能用 first-chunk action recovery 0.3849 推断 rollout failure recovery 应为
38.49%；两者衡量的是不同对象。

## 13. 完整 rollout 运行命令

以下命令运行一个 LIBERO-10 任务的 20 perturbed + 20 corrected rollouts。wrapper 默认
使用 EGL，避免 headless 环境中的 glGetError / OSMesa 初始化错误。

~~~bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy

OUT=experiments/phase8_full_rollout/moe_full_e16_top2_libero10_kitchen_scene4_20pair
mkdir -p "$OUT"

GPU_ID=6 \
PHASE8_SUITES=libero_10 \
PHASE8_RESULTS_DIR="$OUT" \
PHASE8_CORRECTOR_CHECKPOINT=experiments/phase8_latent_shift/moe_full_e16_top2_cuda6/best.pt \
PHASE8_EXTRA_ARGS="--only-task libero_10:KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
./run_phase8_fast_full_rollout.sh \
    2>&1 | tee "$OUT/rollout.log"
~~~

脚本默认 resume。若目标目录已经包含完整 episode 文件，它可能复用结果。需要全新实验时应
使用新的 OUT 目录名，而不是删除或覆盖已有结果。

## 14. Headless OpenGL 错误

如果出现：

~~~text
AttributeError: 'NoneType' object has no attribute 'glGetError'
~~~

通常是 PyOpenGL 在无显示器节点上没有正确初始化 GL backend。当前 wrapper 默认导出：

~~~text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
~~~

因此优先通过 run_phase8_fast_full_rollout.sh 启动。不要在同一进程中先 import OpenGL/
MuJoCo 后再修改这两个变量，因为 backend 往往在 import 阶段就已经确定。

## 15. 下一步实验

最小且信息量最高的下一步不是继续调 full MoE 的 expert 数，而是比较 correction schedule：

1. 每个 corrected episode 只在第一次 policy query 注入 corrector。
2. 后续 action chunks 使用原始 policy。
3. 仍运行相同的 20 perturbed + 20 corrected 配对 rollout。
4. 比较 failure recovery、preservation 和 harm rate。

当前 phase8_fast_full_rollout.py 尚未提供 first-query-only 的 CLI；它会在每次 policy query
应用校正。后续可以加入 correction-max-queries=1 或 correction-schedule=first，并确保
query counter 在每个 episode 开始时重置。

如果 first-query-only 仍降低成功率，则说明 first-chunk latent/action recovery 没有转化
为任务级恢复。若要支持持续轨迹校正，需要采集中间轨迹状态上的 clean/perturbed paired
chunks，并用这些 trajectory-distributed pairs 重新训练，而不能只依赖初始 chunk。

## 16. 当前建议

- 训练架构：优先 full e16 top-2。
- low_rank：作为容量/泛化消融，不作为当前最佳模型。
- partitioned：除非先验证 raw-channel 分段稀疏，否则不要使用 top-2 作为默认设置。
- checkpoint：评估 best.pt，不使用后期过拟合的 final.pt。
- 离线判断：同时看 validation hidden recovery 和 first-chunk action recovery。
- 最终判断：必须看 paired rollout 的 failure recovery、preservation 和 harm rate。
- 部署状态：当前 full MoE checkpoint 不应做 trajectory-wide persistent correction。
