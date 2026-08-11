# Gated Latent Canonicalizer：实验目的

## 研究问题

Cosmos Policy 在视觉背景发生变化时，即使机器人状态、物体状态、任务指令和动作历史保持不变，VAE 编码得到的 video latent 仍会偏离 clean latent。这个偏移可能继续传播到冻结的 DiT，并改变最终策略输出。

本项目希望验证一个最小问题：

> 能否只在冻结 VAE 与冻结 DiT 之间增加一个小型模块，将 perturbed video latent 映射回对应的 clean video latent，同时尽量保持 clean 输入不变？

这个模块暂称为 **Gated Latent Canonicalizer（GLC）**。

## 当前实验范围

当前阶段只训练和比较 video-latent canonicalizer：

```text
perturbed video latent
        ↓
trainable canonicalizer
        ↓
predicted clean video latent
```

输入和目标均来自 Cosmos Policy VAE 已缓存的标准化 latent：

```text
[B, 16, 2, 28, 28]
```

两个 latent slot 分别对应 wrist camera 和 primary camera。

当前实验不会：

- 更新 Cosmos Policy VAE；
- 更新 Cosmos Policy DiT；
- 修改 proprio、action、future-state 或 value latent；
- 使用 action label 训练 canonicalizer；
- 使用 action latent 或 rollout success 作为训练目标；
- 改变 Cosmos Policy 的 diffusion sampling 设置。

因此，这一阶段可以直接使用缓存 latent 训练，不需要加载完整 Cosmos Policy。

## 数据目录与划分

默认读取：

```text
dataset/paired_libero_plus_background_libero10_500
```

该路径相对于 Cosmos Policy 仓库根目录。训练程序优先读取合并后的
`manifest.jsonl`；如果根目录尚未合并，则自动汇总所有可用的
`shard-*/manifest.jsonl`。两种目录形式使用相同的训练命令。

训练程序不使用 manifest 原有的 `train/val` 标签，而是读取所有 episode，
按照 `policy_seed` 分组后重新进行 9:1 划分。划分前对不同 policy seed 进行
确定性随机打乱，默认 `split_seed=0`。

同一个 `policy_seed` 下的全部 episode 只能进入同一个 split，因此：

```text
train policy seeds ∩ validation policy seeds = ∅
```

实际使用的 train/validation seed 列表会写入每次实验的 `config.json`，用于
检查划分和复现实验。

原始 episode 文件包含数百帧以及训练不需要的其他字段，不适合在随机
frame sampling 时反复执行 `torch.load`。训练前先运行 `prepare_data.py`，将
clean/perturbed latent 打包为固定大小的连续 tensor shards。训练进程通过
memory mapping 读取这些 shards；多个实验可以共享操作系统 page cache，
而不需要各自复制整套 latent 到私有内存。

## 监督信号

数据中的 clean 与 perturbed 图像来自相同 simulator state，并共享相同任务、机器人状态、物体状态和动作轨迹。每个 perturbed latent 都有严格配对的 clean latent。

主要目标为：

```text
perturbed latent → clean latent
```

同时加入 clean identity 目标：

```text
clean latent → clean latent
```

训练损失为：

```text
L = MSE(model(perturbed), clean)
  + identity_weight * MSE(model(clean), clean)
```

第一项学习消除背景扰动产生的 latent 偏移，第二项限制模块不要破坏原始 clean latent。

## 模型与对照关系

当前实现比较四种具有统一输入、输出和 residual 写回方式的模型：

| 模型 | 空间/相机 token mixing | Sample-level gate |
|---|---:|---:|
| `mlp` | 否 | 否 |
| `transformer` | 是 | 否 |
| `gated_mlp` | 否 | 是 |
| `glc` | 是 | 是 |

这形成一个两因素对照：

```text
                         无 Gate          有 Gate
无 Attention            MLP              Gated MLP
有 Attention            Transformer      GLC
```

通过该设计可以分别判断：

1. 逐 token 的 channel mapping 是否已经足够；
2. 空间及跨相机 token mixing 是否带来额外收益；
3. sample-level gate 是否改善 perturbed 恢复或 clean 保持；
4. attention 与 gate 的组合是否优于单独使用其中一个组件。

所有 correction 分支的输出层均为零初始化，使模型训练开始时接近严格恒等映射。

## 核心评估指标

当前阶段只评估 video latent：

### Baseline MSE

```text
MSE(perturbed, clean)
```

表示未修正的 latent 偏差。

### Corrected MSE

```text
MSE(model(perturbed), clean)
```

表示 canonicalizer 修正后的剩余误差。

### Relative recovery

```text
1 - corrected_mse / baseline_mse
```

大于零表示修正有效，越接近 1 表示恢复越完整。

### Clean drift

```text
MSE(model(clean), clean)
```

表示模块对原始 clean latent 的破坏程度。

对于 gated 模型，还记录平均 gate 值，但当前不把 gate 当作具有明确监督含义的扰动分类器。

## 当前阶段能够支持的结论

如果模型在未参与训练的 background variant 上降低 corrected MSE，并保持较低 clean drift，可以说明：

> 使用严格配对的 clean/perturbed 数据，可以训练一个小型 residual adapter，在不更新 Cosmos Policy 主体的情况下恢复 background-perturbed VAE video latent。

四模型对照还可以判断收益主要来自 token-wise 非线性映射、attention、gate，还是 attention 与 gate 的组合。

## 当前阶段不能支持的结论

仅凭 latent reconstruction 结果，暂时不能证明：

- action prediction 已经恢复；
- LIBERO rollout success rate 已经恢复；
- 方法能够泛化到 camera、lighting、noise 或组合扰动；
- gate 学会了可解释的扰动检测；
- GLC 一定优于 DiT fine-tuning、LoRA 或其他使用 action label 的方法。

这些问题需要在 latent 实验成立后，将训练好的 canonicalizer 插入冻结 Cosmos Policy，并在保持其他输入与采样设置不变的条件下单独验证。

## 下一阶段判定条件

只有当 GLC 或其 baseline 在 validation paired latent 上同时满足以下条件，才进入冻结 DiT 和 rollout 实验：

1. corrected MSE 稳定低于未修正 baseline MSE；
2. relative recovery 在不同 validation episode 上为正；
3. clean drift 明显小于原始 clean/perturbed gap；
4. 结果不是仅由个别 task 或少数 frame 主导；
5. GLC 相对无 gate Transformer 的增益足以支持 gate 的必要性。

当前代码的唯一目的，就是以尽可能少的训练和工程变量回答上述问题。
