# Perturb-Specific Subspace 实验路线

本文档记录当前 `Cosmos-Policy + LIBERO-plus` 反事实实验的下一步路线。近期目标不是扩展 layer/sigma，而是先把数据规模、held-out 验证和 causal intervention 做扎实。

## 核心问题

当前临时结果显示：

```text
video latent:        perturb-specific subspace 很强，跨 perturb 投影很低
action-slot hidden:  仍然主要是 perturb-specific，但跨 perturb overlap 变大
final action output: 不同 perturb 的 error directions 明显更重叠
```

因此更合理的研究命题是：

```text
H0: perturb shift 是无结构噪声。
H1: 所有 perturb 共享一个 global latent damage subspace。
H2: 每类 perturb 有自己的低维 latent damage subspace。
H3: 这些 perturb-specific subspace 的有害性来自它们和 action-critical directions 的重叠。
H4: 一个 perturb-specific correction dictionary 比单一 global correction 更能恢复 action / rollout。
```

近期重点验证 `H2 -> H3 -> H4`。如果后续发现跨 perturb transfer 很强，再回到 `H1`。

## 当前 N=50 证据摘要

当前较完整的一轮结果来自：

```text
paired smoke:
  experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_50case_corepert/

Phase 2 artifacts:
  experiments/phase2_angular_cosmos/vla_jepa_kitchen_scene4_seed7_50case_corepert_preserved_flipped_last/

Phase 3 subspace probe:
  experiments/phase3_perturb_subspace_probe/vla_jepa_kitchen_scene4_seed7_50case_corepert_last/
```

这轮 `corepert` 只包含 5 类扰动：

```text
camera_viewpoints
light_conditions
robot_initial_states
background_textures
sensor_noise
```

另外两类 `language_instructions` 和 `objects_layout` 没有包含在这轮 N=50 corepert 中。它们机制不同：

```text
language_instructions: 改的是 language / T5 conditioning，不是纯 video perturbation。
objects_layout: 改的是场景布局 / 任务语义，不宜直接并入普通视觉扰动 claim。
```

### 关键表述边界

当前最稳的表述是：

```text
At the final transformer layer's video-latent hidden representation,
perturbation-induced shifts are compact, low-rank, and strongly
perturb-specific.
```

中文可写为：

```text
在最后一层 transformer block 的 video-latent hidden representation 中，
不同扰动引起的表示位移不是无结构高维噪声；
每类扰动主要集中在一个相对低维、扰动特异的全局子空间中。
```

不要写成：

```text
所有扰动都在同一个 video latent subspace 中。
```

因为 N=50 结果显示，video latent 层面不同扰动的子空间重叠很小；到 `hidden_action` 和最终 `action` 时才逐渐汇聚。

### hidden_video 的 shape 和 PCA 含义

当前讨论的 `hidden_video` 是最后一层 block 27 的 video latent：

```text
shape: (1, 2, 14, 14, 2048)
dim after flatten: 1 * 2 * 14 * 14 * 2048 = 802816
```

各维度含义：

```text
1: batch size
2: 两个 video latent slots, 即 latent indices [2, 3]
14 x 14: spatial grid
2048: transformer hidden channel / feature dimension
```

PCA 前会把整个 tensor flatten 成一个 802816 维向量。`r90 = 10` 的含义是：

```text
在整个 flattened video-latent representation 中，
前 10 个全局 PCA disturbance patterns 能解释 90% shift energy。
```

这不是每个 spatial token 各有 10 个方向，也不是 2048 channel 维度里的 10 个局部方向。每个 PCA direction 都可以 reshape 回：

```text
(2, 14, 14, 2048)
```

因此它表示一个覆盖两个 video slots、所有空间位置和所有 hidden channels 的全局扰动模式。

### r90 和 top-k explained energy

对每个扰动单独收集：

```text
delta_i = h_pert_i - h_clean_i
X_p = [delta_1; ...; delta_n]
```

然后对 `X_p` 做 PCA。若 singular values 为 `s1, s2, ...`：

```text
top-k explained = (s1^2 + ... + sk^2) / sum_j sj^2
r90 = 最小的 k, 使 top-k explained >= 0.90
```

当前表格是 in-sample PCA summary，即每个 perturb 的所有可用样本一起 fit PCA 后得到的统计；强结论还需要 held-out split / bootstrap 验证。

### 低维性: hidden_video N=50

`hidden_video` 层的 N=50 corepert 结果：

| Perturb | Samples | r90 | Top16 explained | Within-class cosine |
|---|---:|---:|---:|---:|
| background_textures | 46 | 10 | 93.3% | 0.783 |
| camera_viewpoints | 46 | 8 | 94.3% | 0.805 |
| light_conditions | 46 | 20 | 87.7% | 0.650 |
| robot_initial_states | 46 | 20 | 87.5% | 0.520 |
| sensor_noise | 46 | 6 | 95.5% | 0.825 |

解释：

```text
background / camera / sensor_noise 最集中。
light / robot_initial_states 更分散，但仍然明显可压缩。
即使最难的 light / robot，也只用约 20 个全局方向解释约 88% energy。
```

面向老师的傻瓜式说法：

```text
video latent 有 802816 维，但每类扰动只需要约 6-20 个全局方向，
就能解释接近 90% 的扰动变化。
```

上表是 in-sample PCA（所有可用样本一起 fit）的结果。为进一步排除「低维只是样本不足的假象」，对每类扰动做了 N-scaling 验证（subsample n=8/12/16/24/32/40/46，每个 n 重复 100 次取平均 r90）：

| Perturb | n=12 r90 | n=24 r90 | n=40 r90 | full n=46 r90 | 饱和？ |
|---|---|---|---:|---:|---:|---:|---|
| background_textures | 4.8 | 7.1 | 9.3 | 10 | ⚠️ 缓慢增长 |
| camera_viewpoints | 4.1 | 6.0 | 7.8 | 8 | ⚠️ 缓慢增长 |
| light_conditions | 7.1 | 12.4 | 18.1 | 20 | ❌ 近似线性增长 |
| robot_initial_states | 7.6 | 12.7 | 18.0 | 20 | ❌ 近似线性增长 |
| sensor_noise | 3.3 | 5.0 | 6.0 | 6 | ✅ 已饱和 |

```text
sensor_noise 在 video latent 层的低维性最可靠（r90 在 n=12 后基本不再增长）。
background / camera 增长在放缓，但 N=46 时 r90 可能仍被低估 1-2 个方向。
light_conditions 和 robot_initial_states 的 r90 随 N 近似线性增长，
  其真实低维度可能高于当前 N=46 的估计值；
  需要更大样本量才能下结论。
```

此外，held-out projection（34 train / 12 test, 100 repeats, k=8）显示：

| Perturb | In-sample top8 ev | Held-out test proj | Gap |
|---|---:|---:|---:|
| background_textures | 0.889 | 0.832 | -0.057 |
| camera_viewpoints | 0.903 | 0.851 | -0.052 |
| light_conditions | 0.804 | 0.708 | -0.096 |
| robot_initial_states | 0.780 | 0.661 | -0.119 |
| sensor_noise | 0.917 | 0.870 | -0.047 |

```text
robot 和 light 的 held-out gap 最大（>0.09），
与 N-scaling 中这两类扰动 r90 仍在增长的结论一致——
当前 in-sample r90 可能低估了真实的子空间维度。
sensor_noise 和 background 泛化最好（gap < 0.06）。
```

### 同一 perturb 内部是否所有样本都重合

同一种扰动下，不同 episode 的 `delta_i` 不是完全相同方向，但高度集中。直接看同类样本之间 pairwise cosine：

| Perturb | Pairwise cosine p10 | Mean | p90 |
|---|---:|---:|---:|
| background_textures | 0.735 | 0.783 | 0.839 |
| camera_viewpoints | 0.756 | 0.805 | 0.856 |
| light_conditions | 0.582 | 0.650 | 0.731 |
| robot_initial_states | 0.379 | 0.520 | 0.674 |
| sensor_noise | 0.765 | 0.825 | 0.882 |

再看每个样本投到自己 perturb 的 PCA 子空间上能解释多少 energy：

| Perturb | Top8 mean | Top8 min | Top16 mean | Top16 min |
|---|---:|---:|---:|---:|
| background_textures | 0.889 | 0.825 | 0.933 | 0.904 |
| camera_viewpoints | 0.903 | 0.854 | 0.943 | 0.920 |
| light_conditions | 0.804 | 0.717 | 0.877 | 0.824 |
| robot_initial_states | 0.780 | 0.591 | 0.875 | 0.830 |
| sensor_noise | 0.917 | 0.854 | 0.955 | 0.936 |

因此更准确的结论是：

```text
同一 perturb 下样本不是完全同一方向，
而是落在一个紧凑的低维扇面 / 子空间中。
```

### 不同扰动是否在同一个 subspace

`hidden_video`, k=8, all group 的 cross-reconstruction matrix：

| train -> test | background | camera | light | robot | sensor |
|---|---:|---:|---:|---:|---:|
| background | 0.889 | 0.035 | 0.048 | 0.076 | 0.006 |
| camera | 0.037 | 0.903 | 0.034 | 0.027 | 0.060 |
| light | 0.048 | 0.033 | 0.804 | 0.038 | 0.006 |
| robot | 0.046 | 0.013 | 0.026 | 0.780 | 0.004 |
| sensor | 0.007 | 0.062 | 0.008 | 0.012 | 0.917 |

汇总：

```text
hidden_video k=8 diagonal mean    = 0.859
hidden_video k=8 off-diagonal mean = 0.031
ratio                              = 27.4x
```

这说明：

```text
同一扰动自己的子空间能解释自己的 shift；
几乎解释不了其他扰动的 shift。
```

所以 video latent 层面支持的是 perturb-specific subspaces，而不是一个 shared global video subspace。

Bootstrap stability 验证（100 resamples, k=8）：所有扰动的 top-1 PC 在 bootstrap 下的主角度 <5°（sensor 2.1° / camera 2.3° / background 2.6° / light 3.8° / robot 4.4°），子空间方向可重复。light_conditions 和 robot_initial_states 的 top PC 角度略大，与其 N-scaling 不饱和一致——采样波动导致子空间估计仍有不确定性。

### 层间变化: video -> hidden_action -> action

上表是 in-sample cross-reconstruction。held-out cross-reconstruction（34 train / 12 test, 100 repeats）完全复现了结论：hidden_video off-diag 均值仅 0.02-0.04（ratio 20-40x），action off-diag 高达 0.71-0.90（ratio 仅 1.1-1.3x）。

N=50 corepert 的 k=8 cross-reconstruction 汇总（in-sample）：

| Space | Diagonal mean | Off-diagonal mean | Ratio |
|---|---:|---:|---:|
| hidden_video | 0.859 | 0.031 | 27.4x |
| hidden_action | 0.823 | 0.267 | 3.1x |
| action | 0.952 | 0.756 | 1.3x |

解释：

```text
hidden_video:
  不同扰动路径高度 source-specific。

hidden_action:
  仍有 perturb-specific structure，但跨扰动 overlap 已明显变大。

action:
  已经是最终动作输出，不同扰动自然会汇聚到较共享的 action directions。
```

这支持如下核心结论：

```text
不同扰动在 video latent 上表现为方向各异的低维子空间；
这些扰动信号经过 transformer 逐层处理后，在 action 输出端
逐渐汇聚到共享的低维 error subspace 中。
```

对 action 空间单独做 held-out / bootstrap / N-scaling 验证：

| Perturb | action r90 | N-scaling 饱和？ | held-out test k=8 | bootstrap top PC |
|---|---:|---:|---:|---:|---:|
| background_textures | 5 | ✅ | 0.946 | 1.2° |
| camera_viewpoints | 5 | ✅ | 0.942 | 1.0° |
| light_conditions | 6 | ✅ | 0.912 | 1.6° |
| robot_initial_states | 2 | ✅ | 0.989 | 0.4° |
| sensor_noise | 10 | ⚠️ 仍在增长 | 0.793 | 2.9° |

```text
action 层的低维性总体比 video latent 更经得起验证：
  - light / robot 在 video 层 r90 线性增长，在 action 层已饱和
  - held-out projection 几乎追平 in-sample（robot 0.989 vs 0.994）
  - bootstrap top PC angle < 3° 对所有扰动

例外仍是 sensor_noise：在 action 层 r90=10 且 N-scaling 仍缓慢增长，
  held-out projection 也最低（0.79）。
  这与 sensor_noise 的物理性质一致——它是 per-pixel Gaussian noise，
  在 action 输出端没有强烈的结构化 bias。
```

更精确的渐进式表述：

```text
每种扰动在 video latent 上各自低维且 source-specific（off-diag 均值 0.03）；
这些多源扰动在 hidden_action 层开始融合（off-diag 升至 0.27）；
到 action 输出端，不同扰动已高度重叠在共享的 error directions 中（off-diag 0.76）。
```

### 因果验证：video slot vs action slot 干预

以上是表征层面的分析（PCA/cross-reconstruction）。因果层面，Phase 3 的 mean-shift recovery 实验分别在 video slot 和 action slot 上做了干预：

```text
干预方式:  x[:, slot] += α * mean_flipped(clean_h - pert_h)
测量指标:  recovery = 1 - ||a_corrected - a_clean|| / ||a_pert - a_clean||
位点:      DiT block 27 (last) 入口
方向:      每种 perturb 的 flipped 样本的 mean delta（1 个方向）
```

**Video slot 干预 — 完全无效：**

| Perturb | Best α | Best recovery | Positive rate | n |
|---|---:|---:|---:|---:|
| background_textures | 0 | **0.0%** | 0% | 20 |
| camera_viewpoints | 0 | **0.0%** | 0% | 5 |
| light_conditions | 1.25 | **+0.3%** | 75% | 8 |
| robot_initial_states | 1.0 | **+0.1%** | 68% | 19 |

```text
所有 video slot 干预的 recovery 都在 0% 附近，即使 alpha 扫了 0.25-1.5。
video latent 上的扰动偏移不能通过线性加回 mean direction 来纠正 action 输出。
这与 video 层的 source-specific 低维性一致：
  扰动在 video 上各自独立，经过 27 层 transformer 的非线性变换后，
  简单的线性干预在源头层不起作用。
```

**Action slot 干预 — 部分有效：**

| Perturb | Best α | Best recovery | Positive rate | n |
|---|---:|---:|---:|---:|
| background_textures | 1.0 | **11.6%** | 80% | 20 |
| camera_viewpoints | 0.75 | **28.3%** | 100% | 5 |
| light_conditions | 1.0 | **24.1%** | 87.5% | 8 |
| robot_initial_states | 1.0 | **38.3%** | 89.5% | 19 |

```text
在 action slot 上干预效果显著优于 video slot（11-38% vs ~0%）。
这从因果层面验证了收敛叙事：
  扰动信号在 action slot 附近已经汇聚到可被线性纠正的 error directions。

注意 robot_initial_states 在 video N-scaling 上表现最差（r90 线性增长），
但在 action slot 上 recovery 最好（38%）。
这说明 video 层的「分散」恰恰反映了复杂的非线性变换过程，
而这种复杂性在 action 端反而被压缩为了更可纠正的低维 error。
```

### Phase C: Ceiling 分析 — PCA subspace correction

Phase C 回答了 Subspace.md 规划的核心问题：**top-k PCA 子空间纠正比单方向 mean-shift 能多恢复多少？**

实验设计：
```text
方向类型:
  mean_shift     – 全局 mean delta（所有样本共享一个方向）
  oracle_k{N}    – 每个样本的 delta 投影到自己的 top-N PCA 上（per-sample adaptive）
  oracle_full    – 完整 delta（绝对天花板）
  cross_{X}_k{N} – 用 perturb X 的 PCA 投影当前样本的 delta（跨扰动功能等价性）
  merged_k{N}    – 用所有 perturb 合并的 PCA 投影（共享 error subspace 的可行性）
  mean_component – delta 沿 mean 方向的投影分量
  within_k{N}    – top-N PCA 减去 mean 的分量
  orthogonal_k{N}– delta 在 top-N PCA 外的残差

位点: action slot, block 27 (last)
α: 0.5, 1.0（oracle 理论上 α=1.0 最优）
```

**Self-recovery: oracle 远超 mean-shift：**

| Perturb | mean_shift | oracle_k1 | oracle_k4 | oracle_k8 | oracle_full |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 11.6% | 41.6% | 57.5% | **70.5%** | 79.2% |
| camera_viewpoints | 27.7% | 55.8% | 79.9% | **79.9%** | 79.9% |
| light_conditions | 24.1% | 40.3% | 58.8% | **79.2%** | 79.2% |
| robot_initial_states | 38.3% | 58.0% | 70.8% | **75.8%** | 79.5% |
| language_instructions | 80.8% | 80.8% | 80.8% | 80.8% | 80.8% |

```text
oracle_k1 已经将 mean_shift 翻倍（41-58% vs 12-38%）：
  PC1 是方差最大的方向，per-sample 投影系数保留了「每个样本偏离了多远」的信息，
  而 mean 把所有样本平均后丢失了这些差异。

oracle_k8 已接近天花板（71-80% vs full 79-80%）：
  top-8 PCA 几乎捕获了所有 action-relevant 扰动信息。
```

**Decomposition: orthogonal 完全无效（关键证据）：**

| Perturb | mean_component | within_k8 | orthogonal_k8 |
|---|---:|---:|---:|---:|
| background_textures | 11.5% | **43.8%** | **0.2%** |
| camera_viewpoints | 35.3% | 19.4% | **0.0%** |
| light_conditions | 26.2% | 33.0% | **0.0%** |
| robot_initial_states | 40.9% | 11.3% | **1.3%** |

```text
orthogonal_k8 recovery ≈ 0%：top-8 PCA 之外的残差对 action 完全没有影响。
这从因果层面证明了 PCA subspace 的特殊性——
它不是表征层面的 artifact，而是真正包含了所有 action-critical 扰动信息。
```

**Cross-perturb: 跨扰动纠正可行，但不完全等价：**

| Perturb | best cross recovery | source | vs self mean_shift | vs self oracle_k8 |
|---|---:|---:|---:|---:|---:|
| background_textures | 39.1% | robot PCA | **3.4x better** | 55% of oracle |
| camera_viewpoints | 30.4% | robot PCA | 1.1x | 38% |
| light_conditions | 45.1% | camera PCA | **1.9x better** | 57% |
| robot_initial_states | 39.1% | bg PCA | 1.0x | 52% |

```text
关键发现：cross-recovery 经常优于 mean_shift self-recovery。
  camera→light (45.1%) > light mean_shift (24.1%)
  robot→background (39.1%) > background mean_shift (11.6%)

这意味着：别人的 PCA 方向（因为捕获了高方差结构）比自己的 mean 更好地
代表了 action-relevant 信息。共享 error subspace 确实存在且可被利用。
```

**Merged PCA: 统一 dictionary 可行：**

| Perturb | mean_shift | merged_k8 | oracle_k8 (self) | gap (merged vs self) |
|---|---:|---:|---:|---:|---:|
| background_textures | 11.6% | **64.8%** | 70.5% | -5.7% |
| camera_viewpoints | 27.7% | **67.6%** | 79.9% | -12.3% |
| light_conditions | 24.1% | **57.6%** | 79.2% | -21.6% |
| robot_initial_states | 38.3% | **71.4%** | 75.8% | -4.4% |

```text
merged_k8 在所有 perturbation 上都大幅超过 mean_shift，
且与 self oracle_k8 的差距不大（4-22%）。
一个统一的 merged PCA dictionary 就能有效纠正所有扰动。
```

**Phase C 总结：**

```text
1. PCA subspace 是因果的：orthogonal_k8 recovery ≈ 0%，
   action-critical 信息全部在 top-k 子空间中。

2. mean-shift 远不是天花板：oracle_k1 翻倍，oracle_k8 接近 80% recovery。
   证实了「subspace 内方向分散 → mean 信号衰减」的直觉。

3. 跨扰动共享存在但不等价：
   cross-recovery 经常 > mean_shift self-recovery，证明共享结构可用；
   但 cross/self 只有 38-57%，不同 perturb 的 action-critical 方向仍部分特异。

4. Merged PCA 是实用路径：统一 dictionary 接近 per-perturb oracle，
   不需要知道扰动来源就能纠正。
```

### 面向老师的三张傻瓜图

第一张：低维性。

```text
Video latent dim = 802816.
每类扰动只需 6-20 个全局方向解释接近 90% energy。
（light/robot 的实际低维度可能更高，N-scaling 尚未饱和。）
```

第二张：扰动特异性。

```text
hidden_video k=8:
  diagonal mean    = 0.859
  off-diagonal mean = 0.031
  ratio             = 27.4x

自己的子空间解释自己，几乎解释不了别人。
```

第三张：小 dictionary 可覆盖多源扰动，且逐步收敛。

```text
5 classes * top16 directions = 80 directions
80 / 802816 = about 0.01%
top16 explains 87.5% - 95.5% perturb energy

但更重要的是：这些 perturb-specific 的 video 子空间
经过 transformer 处理后逐渐汇聚：
  hidden_video off-diag 0.031 → hidden_action 0.267 → action 0.756
即不同视觉扰动最终在 action 输出端高度重叠。
```

slide 话术：

```text
The data suggest that visual perturbations do not create arbitrary
high-dimensional noise in the video-latent representation. Each perturbation
source induces a compact, source-specific low-rank disturbance pattern.
Across five sources, a tiny dictionary of about 80 latent directions explains
most of the perturbation-induced variation.
```

中文：

```text
当前数据说明，扰动不是在 video embedding 里制造随机高维噪声。
每类扰动都主要落在一个很小的、扰动特异的子空间里。
五类扰动合起来，用大约 80 个全局 latent directions 就能覆盖大部分扰动变化。
因此，学习一个 multi-source latent disturbance space 是有希望的。
```

### VAE / 刚进入模型的层面

当前 N=50 subspace probe 讨论的是最后一层 `hidden_video`，不是 VAE 层。

真正 VAE / DiT input 层面的 video latent 大约是：

```text
full VAE latent shape: (1, 16, 9, 28, 28)
video conditioning slots: [2, 3]
VAE video latent shape: (16, 2, 28, 28)
flatten dim: 16 * 2 * 28 * 28 = 25088
```

已有 `phase6_vae_containment` 不是 VAE 层 PCA/subspace 低维性实验；它测的是 VAE delta 在 action-sensitive Jacobian 子空间上的 containment，不适合支持“VAE 层扰动低维”这个 claim。

如果要回答“扰动刚进入模型时是否已经低维”，需要额外捕获 VAE video latent，并对：

```text
delta_vae = vae_video_pert - vae_video_clean
```

重复本文件的 PCA / cross-reconstruction / held-out 验证。

## 为什么先不扩展 layer / sigma

layer/sigma 能回答机制形成过程，但当前最薄弱处不是机制定位，而是统计可信度：

```text
此前每类 flipped 样本只有 5-20 个
目前已有 seed7 N=50 corepert，但仍只有单 task / 单 seed
当前主要结论仍基于 in-sample PCA，held-out 稳定性还需要系统测
```

所以近期先固定：

```text
layer: last, 即 block 27
spaces: hidden_video, hidden_action, action
sigma: 继续使用 Phase 2 已保存 endpoint artifact
```

等低维性和 action-critical 因果性成立后，再扩展 layer/sigma。

## 实验递进

### Phase A: 扩展样本并验证低维性

目标：

```text
证明每类 perturb 的 latent shift direction 在 held-out 数据上仍然低维。
```

测量：

```text
delta_i = h_pert_i - h_clean_i
S_p(k) = PCA(delta_i for perturb p)
```

核心指标：

```text
r80 / r90: 解释 80% / 90% shift energy 所需维度
top-k explained energy
mean pairwise cosine
mean resultant length
held-out projection energy
bootstrap principal angle stability
```

判据：

```text
随着 N 从 20 增加到 50/100，r80/r90 不线性增长，而是趋于饱和。
held-out projection energy 接近 train projection energy。
bootstrap 得到的 top-k subspace 夹角稳定。
```

### Phase B: 区分 global vs perturb-specific

目标：

```text
判断是一个 global subspace，还是每类 perturb 各自一套 subspace。
```

做 cross-reconstruction matrix：

```text
R[p -> q] = ||Proj_{S_p(k)} delta_q||^2 / ||delta_q||^2
```

解释：

```text
diagonal 高、off-diagonal 低:
  perturb-specific subspace

diagonal 和 off-diagonal 都高:
  shared/global subspace

video latent diagonal 高但 action output off-diagonal 高:
  不同视觉扰动路径不同，但最终压到相近 action error directions
```

### Phase C: 验证 action-critical，而不是普通视觉痕迹

目标：

```text
证明 S_p 不只是描述视觉变化，而是真的控制 action error。
```

做 subspace correction：

```text
h_corrected = h_pert - alpha * Proj_{S_p(k)}(h_pert - h_clean)
```

比较：

```text
action recovery = 1 - ||a_corrected - a_clean|| / ||a_pert - a_clean||
```

必须有 controls：

```text
same-rank random subspace
orthogonal complement
preserved-only subspace
random episode pairing subspace
same-norm random direction
```

判据：

```text
只有 flipped perturb-specific S_p 能显著恢复 action。
random / orthogonal / preserved-only 不能达到同等恢复。
```

### Phase D: Rollout-level recovery

目标：

```text
证明 action-level recovery 能转化成 environment success recovery。
```

做法：

```text
在 rollout 中对 perturbed observation 的 selected hidden site 施加 subspace correction。
统计 perturb fail -> corrected success 的比例。
```

判据：

```text
corrected success rate 明显高于 perturbed baseline 和 random-subspace baseline。
```

## 分组原则

主分析必须分开：

```text
flipped   = clean success && perturb fail
preserved = clean success && perturb success
```

`all = flipped + preserved` 只作为总体 perturb shift 的诊断，不应用来定义 failure subspace。

原因：

```text
preserved shift 可能是 harmless adaptation
flipped shift 才是 failure-associated damage
合并会把 benign 和 harmful directions 混在一起
```

如果某个 perturb 只有 flipped 或只有 preserved：

```text
仍可用于低维性分析
但不能用于 preserved-vs-flipped failure discriminator
适合作为强扰动 / 弱扰动 control
```

## 数据扩展建议

近期优先扩展三个维度：

```text
1. N: 每个 condition 从 20 pair 扩到 50 或 100 pair
2. seed: 至少 3 个 seed
3. task: 至少 3 个 LIBERO kitchen/task variants
```

推荐第一轮：

```text
conditions:
  camera_viewpoints
  light_conditions
  robot_initial_states
  background_textures   # flipped-only control 可能较多
  sensor_noise          # preserved-only control 可能较多

skip for now:
  language_instructions # 需要额外 T5 embedding，先不作为主线
  objects_layout        # variant 机制不同，可后续单独分析
```

注意：

```text
SMOKE_NUM_PAIRS 增大能覆盖更多 init_state_index。
SMOKE_SEED 改变 policy diffusion seed。
SMOKE_DETERMINISTIC_RESET_SEED 建议和 SMOKE_SEED 一起改，避免所有 run 完全共享 reset state。
```

如果后续发现不同 seed 仍大量复用同一批 init states，可以再给 smoke script 增加 `SMOKE_INIT_OFFSET` 或显式 init index list。先不提前改。

## 扩展数据命令模板

以下命令以 `N=50, seed=11` 为例。每个 seed/task 应使用独立输出目录，避免覆盖旧结果。

### 1. Phase 1: 运行 paired smoke test

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy

SEED=11
N=50
TASK=KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
CLEAN_LANGUAGE="put the black bowl in the bottom drawer of the cabinet and close it"
PERTS=camera_viewpoints,light_conditions,robot_initial_states,background_textures,sensor_noise
RUN=vla_jepa_kitchen_scene4_seed${SEED}_${N}case_corepert

SMOKE_NUM_PAIRS=${N} \
SMOKE_SEED=${SEED} \
SMOKE_DETERMINISTIC_RESET_SEED=${SEED} \
SMOKE_PAIR_BASE_TASK=${TASK} \
SMOKE_PAIR_CLEAN_LANGUAGE="${CLEAN_LANGUAGE}" \
SMOKE_PAIR_PERT_NAME=${PERTS} \
SMOKE_RUN_ID=${RUN} \
SMOKE_RESULTS_DIR=./experiments/paired_smoke_${RUN} \
GPU_ID=0 \
./run_libero_smoke_test.sh
```

输出 summary 通常是：

```text
experiments/paired_smoke_${RUN}/${TASK}__selected__${N}pair_summary.json
```

### 2. Phase 2: 捕获 action / hidden artifacts

```bash
SUMMARY=experiments/paired_smoke_${RUN}/${TASK}__selected__${N}pair_summary.json
PHASE2=experiments/phase2_angular_cosmos/${RUN}_preserved_flipped_last

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="../LIBERO-plus:${PYTHONPATH}" \
/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/run_phase2_angular_cosmos.py \
  --summary ${SUMMARY} \
  --groups preserved flipped \
  --layers last \
  --seed ${SEED} \
  --reset-seed ${SEED} \
  --conditions camera_viewpoints light_conditions robot_initial_states background_textures sensor_noise \
  --output-dir ${PHASE2}
```

### 3. Phase 3a: 离线 subspace probe

```bash
SUBSPACE=experiments/phase3_perturb_subspace_probe/${RUN}_last

/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/perturb_subspace_probe_cosmos.py \
  --results-dir ${PHASE2} \
  --output-dir ${SUBSPACE} \
  --spaces hidden_video hidden_action action \
  --groups flipped preserved all \
  --fit-groups flipped preserved all \
  --ks 1 2 4 8 16
```

主要看：

```text
${SUBSPACE}/subspace_summary_hidden_video.csv
${SUBSPACE}/subspace_summary_hidden_action.csv
${SUBSPACE}/cross_reconstruction_hidden_video.csv
${SUBSPACE}/cross_reconstruction_hidden_action.csv
```

如果需要保存 top-k basis 给后续 intervention：

```bash
/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/perturb_subspace_probe_cosmos.py \
  --results-dir ${PHASE2} \
  --output-dir ${SUBSPACE}_with_bases \
  --spaces hidden_action hidden_video \
  --ks 1 2 4 8 \
  --save-bases \
  --basis-k 8
```

注意：basis 文件可能较大，默认不保存。

### 4. Phase 3b: 离线 held-out / bootstrap / N-scaling 验证

这一步不需要重新 rollout，也不跑 policy model。它只读取 Phase 2 保存的 paired artifacts，在 episode-level deltas 上做：

```text
held-out projection:
  train episodes fit PCA, test episodes measure projection energy

bootstrap stability:
  bootstrap resample fit top-k subspace, compare principal angles

N-scaling:
  subsample n = 8/12/16/24/32/40/46, 看 r80/r90 是否趋于饱和
```

命令：

```bash
VALID=experiments/phase3_perturb_subspace_validation/${RUN}_hidden_video

/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/validate_perturb_subspace_cosmos.py \
  --results-dir ${PHASE2} \
  --output-dir ${VALID} \
  --spaces hidden_video \
  --groups all \
  --ks 4 8 16 \
  --split-repeats 100 \
  --bootstrap-repeats 100 \
  --subsample-repeats 100 \
  --subsample-ns "8 12 16 24 32 40 46" \
  --seed 0
```

主要看：

```text
${VALID}/heldout_projection_hidden_video.csv
${VALID}/bootstrap_stability_hidden_video.csv
${VALID}/n_scaling_hidden_video.csv
```

如果要同时比较最后一层 `hidden_video -> hidden_action -> action`：

```bash
VALID=experiments/phase3_perturb_subspace_validation/${RUN}_all_spaces

/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/validate_perturb_subspace_cosmos.py \
  --results-dir ${PHASE2} \
  --output-dir ${VALID} \
  --spaces hidden_video hidden_action action \
  --groups all \
  --ks 4 8 16 \
  --split-repeats 100 \
  --bootstrap-repeats 100 \
  --subsample-repeats 100 \
  --subsample-ns "8 12 16 24 32 40 46" \
  --seed 0
```

如果要拆 `flipped / preserved`：

```bash
VALID=experiments/phase3_perturb_subspace_validation/${RUN}_hidden_video_groups

/data2/haoze/miniconda3/envs/cosmospolicy/bin/python \
  bin/validate_perturb_subspace_cosmos.py \
  --results-dir ${PHASE2} \
  --output-dir ${VALID} \
  --spaces hidden_video \
  --groups flipped preserved \
  --ks 4 8 \
  --split-repeats 100 \
  --bootstrap-repeats 100 \
  --subsample-repeats 100 \
  --subsample-ns "6 8 12 16 24 32 40" \
  --min-train 6 \
  --min-test 2 \
  --seed 1
```

注意：如果某些 group 样本太少，例如 `sensor_noise/flipped`，脚本会自然跳过或给出很少重复结果。主张“video perturbation 低维”时，优先看 `--groups all`。

## 多 seed 扩展

建议先跑：

```text
seed = 7, 11, 13
N = 50
same base task
same perturb list
```

命令上只需要改：

```bash
SEED=7
SEED=11
SEED=13
```

每个 seed 分别得到：

```text
paired_smoke_<RUN>/
phase2_angular_cosmos/<RUN>_preserved_flipped_last/
phase3_perturb_subspace_probe/<RUN>_last/
```

第一步先不要强行 pool。先逐 seed 比较：

```text
r80/r90 是否稳定
cross-reconstruction diagonal/off-diagonal pattern 是否稳定
flipped/preserved 差异是否稳定
```

如果逐 seed 结果一致，再做 pooled subspace。当前 `perturb_subspace_probe_cosmos.py` 是单 results-dir 输入；需要 pooled 分析时再扩展成 `--results-dirs`。

## 多 task 扩展

选择 task 时优先满足：

```text
clean success rate 高
camera/light/robot perturb 下同时有 preserved 和 flipped
任务不应太简单到所有 perturb 都 preserved
任务也不应太难到 clean 不稳定
```

每个 task 独立设置：

```bash
TASK=<new_libero_task_name>
CLEAN_LANGUAGE="<natural language instruction for this task>"
RUN=<task_short_name>_seed${SEED}_${N}case_corepert
```

其余命令不变。

建议至少：

```text
3 tasks × 3 seeds × 50 pairs
```

如果算力允许，再扩展到：

```text
3 tasks × 5 seeds × 100 pairs
```

## 结果判读

### 支持 perturb-specific subspace（video latent 层）

```text
hidden_video:
  train p -> test p projection high (in-sample mean diag 0.859)
  train p -> test q projection low (in-sample mean off-diag 0.031, ratio 27.4x)

held-out cross-reconstruction 复现: ratio 20-40x
bootstrap top PC angle < 5°: 子空间方向可重复
```

⚠️ 限定：light_conditions 和 robot_initial_states 在 video latent 层的 r90 随 N 近似线性增长（n=12→46: 7→20），held-out gap 也最大（0.10-0.12）。这两类在 video 层的低维性结论需更大 N 确认。

### 支持逐步收敛到共享 action error subspace

```text
cross-reconstruction off-diag:
  hidden_video 0.031 → hidden_action 0.267 → action 0.756

不同视觉扰动路径在 video latent 上几乎正交；
经过 transformer 处理后逐渐汇聚到共享的 action error directions。
```

### 支持 action 层低维性（比 video 层更可靠）

```text
action output:
  r90 = 2-10（远小于 video latent 的 6-20）
  N-scaling: light/robot 的 r90 在 n=24 已饱和（vs video 层线性增长）
  held-out projection ≈ in-sample（robot 0.989 vs 0.994, gap < 0.01）
  bootstrap top PC angle < 3°
```

⚠️ 限定：sensor_noise 在 action 层仍较分散（r90=10, held-out 0.79, N-scaling 未完全饱和），与其 per-pixel noise 的物理性质一致。

### 支持 action-critical（Phase C 已验证）

```text
✅ orthogonal_k8 recovery ≈ 0%：
   top-k PCA 之外的残差对 action 无影响 → subspace 是因果的

✅ oracle_k8 recovery >> mean_shift（71-80% vs 12-38%）：
   top-k PCA 包含几乎所有 action-relevant 扰动信息

✅ merged_k8 ≈ oracle_k8（差距 4-22%）：
   统一 PCA dictionary 可有效纠正所有扰动，不需知道来源

⚠️ cross-recovery 38-57% of oracle self：
   共享 error subspace 存在且可用（cross > mean_shift self），
   但不是完全等价
```

### 支持 rollout-level recovery（待验证 — Phase D）

```text
subspace correction 在 action-level recovery 上已验证（70-80% ceiling）
↓ 需要验证这能转化为实际 rollout success rate 提升
```

### 反证情形

```text
r80/r90 随 N 线性增长:
  当前低维主要是样本少造成的
  → 当前 video latent 的 light/robot 部分符合此情形，需注意措辞

held-out projection energy 显著低于 in-sample:
  in-sample PCA 过拟合
  → video latent 的 robot/light gap > 0.09，其余可接受

random subspace intervention 也能恢复:
  不是 perturb-specific action-critical subspace

preserved-only subspace 同样恢复 flipped:
  failure subspace 与 benign perturb shift 没有区分
```

## 近期最小可发表实验包

建议最小闭环：

```text
1. 3 seeds × 50 pairs × 1 task                                          [seed7 done]
2. hidden_video / hidden_action subspace summary                          [✅ done]
3. train/test held-out projection + bootstrap + N-scaling                [✅ hidden_video + action]
4. cross-perturb matrix                                                   [✅ done]
5. top-k subspace intervention on action error (Phase C ceiling)         [✅ done]
6. cross-perturb & merged PCA recovery                                    [✅ done]
7. rollout-level recovery (Phase D)                                       [待做]
```

在这个闭环完成前，不建议优先扩展 layer/sigma。
