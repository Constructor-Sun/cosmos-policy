# Denoise Hessian With Clean-Action Loss

本文档记录当前 Cosmos Policy preserved/flipped Hessian 实验的设计、loss 定义、实现状态和已知计算问题。

## Goal

目标是比较 Phase 1 已有 paired samples 中 preserved 与 flipped 样本在真实 denoising trajectory 附近的局部曲率：

```text
lambda_max(H_x L) = max eigenvalue of Hessian of loss L around point x
```

轨迹必须是实际生成路径：

```text
noise -> clean
noise -> perturbed
```

不使用 clean -> perturb interpolation，也不使用 mean-shift intervention。

当前样本来源是：

```text
experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/
```

覆盖 7 类 perturb：

```text
camera_viewpoints
background_textures
light_conditions
objects_layout
robot_initial_states
sensor_noise
language_instructions
```

## Loss Choice

当前使用的 loss 是 clean-success reference action loss：

```text
a_clean = extract_action(clean_final_sample).detach()
L(x) = MSE(extract_action(pred_x0(x)), a_clean)
```

其中：

- `clean_final_sample` 来自同一个 paired sample 的 clean branch 最终 denoised latent。
- clean branch 在 Phase 1 中全部 success，因此 `a_clean` 可作为成功策略参考。
- clean 和 perturbed branch 共享同一个 `a_clean` target。
- action target 使用 normalized action space，因为 `extract_action` 直接从 model latent action slot 读出训练尺度 action。

这个 loss 不是 expert supervised training loss。它的含义是：

```text
perturbed denoising trajectory 附近，朝向 clean-success action 的局部 loss landscape 曲率。
```

### Validation: `a_clean` as Expert Action Proxy

使用 `a_clean` 作为 target 的一个潜在顾虑是循环论证：Hessian loss `L = MSE(a_pred, a_clean)` 和 action deviation `MSE(a_pert, a_clean)` 共享同一 reference，二者在定义上不完全独立。

为验证 `a_clean` 是否可以作为真正的 expert action 代理，我们在 LIBERO-Cosmos-Policy 训练数据上做了直接对比：对 task "put the black bowl in the bottom drawer of the cabinet and close it" 的 30 个成功 demo，用训练数据第一帧 observation 跑一次 clean branch，提取 action chunk，与训练数据自身的 expert action 对比。

结果（`experiments/validate_clean_vs_training/clean_vs_training.json`）：

| 指标 | 值 |
| --- | ---: |
| MSE | 0.00077 ± 0.00016 |
| RMSE | 0.0276 |
| Cosine Similarity | 0.9981 (min 0.9972) |

模型几乎完美复现训练数据中的专家策略（CosSim > 0.99）。`a_clean` 与 `a_expert` 在数值上高度一致，无论选哪个做 Hessian target 结果都不会有实质性区别。此前担忧的"Hessian loss 和 action deviation 共享同一 reference 导致循环论证"在实践中不构成问题。

## Denoising Steps And Sites

默认 denoising steps 是 50。默认取四分位位置：

```text
0, 12, 25, 37, 49
```

默认 sites：

```text
vae
layer0
mid
last
```

每个 site 上估计 Hessian 最大特征值。hidden site 的 loss 是通过替换该层输出并继续 forward 到 `pred_x0` 后计算 action loss。

## HVP Method

精确 autograd HVP 在 2B DiT 上容易 OOM，因为需要二阶 autograd graph。

当前脚本支持有限差分 HVP：

```text
H(x)v ~= [grad L(x + eps v) - grad L(x - eps v)] / (2 eps)
```

然后用 Lanczos 估计最大特征值。

推荐完整运行使用：

```text
--hvp-mode finite_diff
```

默认 `--fd-eps 1e-2`。

## Current Command

完整 clean + 7 perturb preserved/flipped 运行命令：

```sh
cd /data3/liu/exp/counterfactual/external/cosmos-policy

.venv/bin/python bin/run_denoise_hessian_cosmos.py \
  --target-source clean_final_action \
  --hvp-mode finite_diff \
  --output-dir experiments/phase3_denoise_hessian/kitchen_scene4_seed7_all_perturb_clean_final_action_fd
```

默认 `--conditions` 不设置时会使用 combined summary 中所有 perturb。默认 `--groups preserved flipped`。

## Current Results And Motivation Evidence

当前完整结果位于：

```text
experiments/phase3_denoise_hessian/kitchen_scene4_seed7_all_perturb_clean_final_action_fd/
```

输出完整性：

```text
denoise_hessian_detail.csv: 5600 data rows
denoise_hessian_delta.csv: 2800 data rows
summary.json: 140 pairs
```

行数与默认配置一致：

```text
140 pairs x 2 branches x 5 denoising calls x 4 sites = 5600 detail rows
140 pairs x 5 denoising calls x 4 sites = 2800 delta rows
```

### Simple Curvature Observation

最适合作为 motivation 的直接证据是：

```text
clean trajectory 的曲率在 preserved/flipped 之间接近；
perturbed trajectory 的曲率明显升高；
flipped cases 的 perturbed 曲率和曲率增量更高。
```

`call=49` 的 median `lambda_max`：

| site | group | lambda_clean | lambda_perturbed | delta_lambda | perturbed / clean |
| --- | --- | ---: | ---: | ---: | ---: |
| layer0 | preserved | 1.93 | 65.13 | +63.60 | 33.67x |
| layer0 | flipped | 1.89 | 127.81 | +125.96 | 67.78x |
| mid | preserved | 0.0023 | 0.3518 | +0.3498 | 150.6x |
| mid | flipped | 0.0024 | 0.6980 | +0.6950 | 294.0x |
| vae | preserved | 43.03 | 44.05 | +1.11 | 1.02x |
| vae | flipped | 43.06 | 49.93 | +7.01 | 1.16x |

这个表可以支持的简单结论是：

```text
Effective perturbations move the denoising trajectory from relatively smooth clean regions
into sharper high-curvature regions of the clean-action loss landscape.
```

中文表述：

```text
有效扰动会把 denoising trajectory 从相对平滑的 clean 区域推入更尖锐的 clean-action loss 区域。
```

### Relation To Action Deviation

如果需要补充说明“高曲率是否和错误动作有关”，可以使用 action deviation 与曲率的相关性。

这里的 action deviation 是 `summary.json` 中的：

```text
pert_action_mse_to_clean_final
```

简化后的 condition-centered Spearman correlation：

| location | comparison | correlation |
| --- | --- | ---: |
| call 37, vae | action deviation vs perturbed curvature | 0.9676 |
| call 37, vae | action deviation vs curvature increase | 0.9661 |
| call 49, layer0 | action deviation vs perturbed curvature | 0.8110 |
| call 49, layer0 | action deviation vs curvature increase | 0.8094 |

这说明：

```text
在不同 samples 之间，action deviation 更大的样本通常也有更高的 perturbed-trajectory 曲率。
```

这个相关性适合作为 supporting evidence。上文 Validation 已确认 `a_clean` 与 `a_expert` 数值上高度一致（CosSim > 0.99），共享 reference 在实践中不构成循环论证。

### Recommended Claim

推荐使用的稳妥表述：

```text
Our Hessian analysis shows that effective perturbations are associated with a large
clean-to-perturbed curvature increase along the denoising trajectory. Clean trajectories
have similar local curvature across preserved and flipped cases, while flipped perturbations
enter substantially sharper regions of the clean-action loss landscape. This motivates
constraining large Hessian eigenvalues to smooth the local loss landscape.
```

不要使用的过强表述：

```text
High clean curvature causes perturbation susceptibility.
Perturbations exploit the top Hessian eigenvector.
Curvature strictly determines action error.
```

当前实验没有直接测量 perturbation direction 与 top Hessian eigenvector 的 alignment，也没有证明 `lambda_clean` 可以稳定预测后续 perturbation susceptibility。

## Current Implementation Notes

脚本：

```text
bin/run_denoise_hessian_cosmos.py
```

已实现：

- `--target-source clean_final_action`
- `--target-source expert_action`，保留但不作为当前主实验解释
- `--hvp-mode finite_diff`
- model parameters frozen before HVP
- denoise trajectory capture under `torch.no_grad()`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- output CSV/JSON 中记录 `target_source`, `hvp_mode`, `fd_eps`

## Clean Branch Repetition

当前脚本按 `(condition, episode)` pair 逐个运行：

```text
clean branch
perturbed branch
```

因此同一个 `episode` 会在不同 perturb condition 下重复运行 clean branch。

这在语义上是正确的，但计算上有重复。当前默认设置下，同一个 episode 跨 perturb 的 clean branch 输入相同：

```text
same clean task
same clean instruction
same clean observation/init_state
same noise_seed = seed * 100000 + episode
same solver/guidance/steps
```

因此可以缓存 clean branch。

建议后续实现 clean cache，key 至少包含：

```text
episode
init_state_index
task_name
language
noise_seed
num_denoising_steps
selected_calls
layer_map
solver_option
guidance
```

缓存内容：

```text
BranchResult for clean
clean_final_action target
```

这样 7 perturb x 20 episodes 中 clean branch 可从最多约 140 次降到约 20 次。

## Interpretation Caveats

`last` site 的数值不建议用于主结论。当前结果中 `last` 有大量 `lambda_max = 0` 或接近 0 的情况，可能与 hidden replacement dtype 和 finite-difference step size 在高维 hidden state 上的数值精度有关。主结论建议只使用：

```text
vae
layer0
mid
```

对以下 perturb，shared clean action target 解释较自然：

```text
camera_viewpoints
background_textures
light_conditions
sensor_noise
language_instructions
```

对以下 perturb 需要更谨慎：

```text
objects_layout
robot_initial_states
```

因为物体布局或机器人初始状态变化后，真实最优 low-level action 可能改变。此时 `clean_final_action` 更适合作为“偏离 clean 成功策略”的 reference，而不是 ground-truth expert action。
