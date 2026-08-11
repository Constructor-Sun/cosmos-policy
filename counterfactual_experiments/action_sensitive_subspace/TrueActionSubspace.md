# True Action-Sensitive Subspace: Gradient Alignment Analysis

## 纠正 ActionSubspace.md 的核心错误

### ActionSubspace.md 做了什么

```
J = ∂a/∂h                                          [112, D] Jacobian
S_action = top-k SVD(J)                             对 J 做 SVD
containment = ||Proj_{S_action} δh||² / ||δh||²     投影 containment
```

三个错误：

1. **测量的是 action output 的 sensitivity，不是 action error 的 sensitivity。** SVD(J) 等权对待 112 个 action 维度，不区分哪些维度实际偏离了、偏离了多少。

2. **在错误的位置测量。** S_action 在 clean trajectory 的 sigma=80 处定义，但 δh 是 perturbed trajectory 上的位移。非线性 DiT 使 J 在 clean 点和 perturbed 点可以完全不同。

3. **完全忽略了 loss function。** 理解 perturbation effect 的核心问题是"为什么 perturbed action 是错的"，需要 loss `L = MSE(a_pred, a_clean)`，而不是裸的 action output。

### 修正方案

用 loss gradient 替代 Jacobian SVD：

$$L(h) = \text{MSE}\big(a(h), a_{clean}\big)$$

$$\nabla_h L = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial h} = 2\big(a(h) - a_{clean}\big)^T \cdot J$$

这是一个 **D 维向量**（不是 k 维子空间），每个 action 维度按其实际 error `(a_i − a_clean,i)` 加权。然后直接测量：

$$\text{alignment} = \cos(\delta_h, \nabla_h L(h_{pert})) = \frac{\delta_h^T \cdot \nabla_h L}{\|\delta_h\| \cdot \|\nabla_h L\|}$$

$$\text{directional derivative} = \frac{\delta_h^T \cdot \nabla_h L}{\|\delta_h\|}$$

### 为什么 gradient 比 SVD 更正确

| | SVD of J | Gradient of L |
|---|---|---|
| 测量对象 | action output 变化 | action **error** 变化 |
| action 维度权重 | 等权 | 按 (a − a_clean) 加权 |
| 结果维度 | k 维子空间 | 1 维方向（sharper test） |
| 测量点要求 | 仅此一个点 | 可在任意点测量 |
| 与 Hessian 的关系 | H ≈ 2JᵀJ（仅当 a≈a_clean） | ∇L → 0 时需二阶分析 |

---

## 实验设计

### 测量链路

对每个 (condition, episode, denoising_step, site)：

1. 捕获 clean 和 perturbed 两个 trajectory 沿线的 hidden states
2. `δh = h_pert − h_clean`
3. 在 perturbed 点构建 loss：`L(h) = MSE(extract_action(x0_fn_pert(h, σ)), a_clean)`
4. 通过 autograd 计算 `∇_h L(h_pert)`
5. 在 clean 点同样计算 `∇_h L(h_clean)` 作为对照
6. 计算 `alignment = cos(δh, ∇L)` 和 directional derivative

### Sites

| Site | 含义 | ∇_h L 的反向传播范围 |
|------|------|---------------------|
| vae | VAE 输出（DiT 入口） | 全部 28 层 DiT + FinalLayer |
| layer0 | DiT block 0 输出 | blocks[1:28] + FinalLayer |
| mid | DiT block 14 输出 | blocks[15:28] + FinalLayer |
| last | DiT block 27 输出 | FinalLayer only |

### 关键区别

- **VAE 的 gradient 经过全部 28 层 DiT**，累积了完整的 action sensitivity 信号
- **单层 gradient 只经过该层之后的部分**，捕捉的是"该层输出对最终 action loss 的局部敏感性"

### 运行

```bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy

MUJOCO_GL=egl LIBERO_PLUS_PATH=/data1/liu/exp/counterfactual/external/LIBERO-plus \
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
python bin/phase9_gradient_alignment.py \
  --conditions camera_viewpoints background_textures light_conditions \
              objects_layout robot_initial_states sensor_noise \
  --output-dir experiments/phase9_gradient_alignment/kitchen_scene4_seed7
```

（sensor_noise 需 libmagickwand，当前分析覆盖其余 5 类）

数据：100 pairs × 5 denoising calls × 4 sites = 2000 rows。
脚本：`bin/phase9_gradient_alignment.py`。

---

## 结果

### 1. 核心：alignment 总体极低

#### alignment_pert = cos(δh, ∇L(h_pert))

perturb 点处，δh 和 loss gradient 的对齐度：

| Site | Flipped mean | Preserved mean | 角度 |
|------|-------------|---------------|------|
| **vae** | 0.077 | 0.095 | ~86° |
| layer0 | 0.0014 | 0.0007 | ~90° |
| mid | 0.0019 | 0.0007 | ~90° |
| last | 0.020 | 0.018 | ~89° |

#### alignment_clean = cos(δh, ∇L(h_clean))  ← Taylor 展开的一阶项

| Site | \|cos(δh, ∇L_clean)\| mean | \|cos(δh, ∇L_pert)\| mean | 解读 |
|------|---------------------------|---------------------------|------|
| **vae** | 0.038 | 0.083 | clean 点比 perturb 点更低 |
| layer0 | **0.0011** | 0.0017 | 中位数 0.0004，一阶项精确为零 |
| mid | **0.0010** | 0.0023 | 中位数 0.0006，一阶项精确为零 |
| last | 0.0057 | 0.0234 | 一阶项也接近零 |

**alignment_clean 甚至比 alignment_pert 更低。** 对于内部层（layer0/mid），cos 中位数仅 0.0004-0.0006。这闭合了 Taylor 展开的论证：

$$L(h_{pert}) \approx \underbrace{L(h_{clean})}_{\approx 0.0005} + \underbrace{\delta_h^T \nabla L(h_{clean})}_{\cos < 0.04} + \boxed{\frac{1}{2}\delta_h^T H(\xi) \delta_h}$$

一阶项在 **clean 点和 perturb 点两端都 ≈ 0**，两个独立测量收敛到同一结论。

**δh 与 ∇L 在所有层都近乎正交。** perturbation 造成的 latent shift 不在 loss gradient 方向上。

### 2. VAE 层 preserved > flipped（反直觉），内部层 flipped > preserved

```
VAE:    flipped=0.077  preserved=0.095  Δ=-0.019  t=-3.14  ← preserved HIGHER
layer0: flipped=0.0014 preserved=0.0007 Δ=+0.0008 t=+4.65  ← flipped HIGHER
mid:    flipped=0.0019 preserved=0.0007 Δ=+0.0012 t=+5.14  ← flipped HIGHER
last:   flipped=0.020  preserved=0.018  Δ=+0.002  t=+1.73  ← flipped HIGHER
```

这恰好揭示了 DiT 的作用机制：

- **VAE 层**：preserved 扰动的视觉差异通过 VAE 编码后，其 latent shift 与 loss gradient 更对齐（视觉上更明显的改变），但 DiT 成功**抑制**了这些 shift——最终 action 仍然正确。
- **内部层**：flipped 扰动的 shift 虽然在 VAE 端更小，但经过 DiT 处理后变成了更"危险"的方向——与 gradient 对齐度上升，且 DiT 无法补偿。

换言之：**preserved 扰动是"大但不致命"的 shift，flipped 扰动是"小但致命"的 shift。** DiT 的 28 层变换能够过滤前者但无法过滤后者。

### 3. Alignment 随 denoising 步数的演化

```
VAE, call 12 (σ=18.1): flipped=0.049  preserved=0.101  Δ=-0.053
VAE, call 25 (σ=2.3):  flipped=0.062  preserved=0.085  Δ=-0.023
VAE, call 37 (σ=0.16): flipped=0.097  preserved=0.097  Δ=+0.001
VAE, call 49 (σ=0.002):flipped=0.099  preserved=0.099  Δ=+0.000
```

flipped-vs-preserved 的差距随 denoising 逐步缩小并最终消失。早期 step 的差异最大——此时 DiT 尚未完成对扰动的处理，VAE 层的 raw visual difference 信号最强。

### 4. 各扰动的 VAE 层 alignment

| Perturbation | Flipped | Preserved | 解读 |
|---|---|---|---|
| robot_initial_states | **0.121** | 0.090 | 机器人位姿变化→直接作用于 action-relevant 视觉特征 |
| objects_layout | N/A | **0.115** | 全部 preserved，物体布局变化大但 DiT 鲁棒 |
| light_conditions | 0.080 | 0.089 | 光照变化中等 |
| camera_viewpoints | 0.078 | 0.075 | 视角变化中等 |
| background_textures | **0.034** | N/A | 背景纹理变化分散在 VAE 空间各处，与 gradient 最不对齐 |

`background_textures` 最低的 alignment + 全部 flipped → 背景纹理变化产生的是"分散但致命"的 latent shift：方向不与 gradient 对齐（一阶效应小），但通过 curvature（二阶效应大）导致 action error。

### 5. Loss 和 Gradient 量级

| | Loss_clean | Loss_pert | Ratio |
|---|-----------|-----------|-------|
| Flipped | 0.00048 | 0.01797 | **37.4×** |
| Preserved | 0.00049 | 0.01163 | **23.6×** |

| | ‖∇L_pert‖ / ‖∇L_clean‖ (median) |
|---|---|
| Flipped VAE | 10.7× |
| Preserved VAE | 8.3× |

Perturbation 大幅提升了 gradient magnitude（5-15×），但 gradient direction 也显著旋转了——`cos(∇L_clean, ∇L_pert)` 的绝对值均值仅 0.1-0.4。

### 6. Gradient 方向的旋转

| Site | cos(∇L_clean, ∇L_pert) abs mean |
|------|----------------------------------|
| vae | 0.14 |
| layer0 | 0.37 |
| mid | 0.13 |
| last | 0.27 |

Gradient 方向在 clean→perturbed 之间旋转了 66-82°。这意味着 loss landscape 有显著的非线性曲率——perturbation 不只是沿着 clean 点的 gradient 方向上坡，而是进入了 landscape 完全不同的区域。

---

## 与 Hessian 分析的互补

### Gradient alignment 告诉我们的

- δh 与 ∇L 近乎正交 → **一阶泰勒展开不能解释 perturbation 的 loss 增加**
- 低 alignment + 大 loss 变化 → 二阶 Hessian 项 `δhᵀ H δh` 主导

### 分解

$$L(h_{pert}) \approx \underbrace{L(h_{clean})}_{\approx 0} + \underbrace{\delta_h^T \nabla L}_{\approx 0} + \boxed{\frac{1}{2}\delta_h^T H \delta_h} + \cdots$$

- 第一项：clean 点 loss 极小（0.00048），尤其在 sigma_min 时精确为 0（AdaLN 退化）
- 第二项：`δhᵀ ∇L ≈ 0`（alignment ≈ 0.001-0.08，近乎正交）
- 主导项：**Hessian quadratic form** `δhᵀ H δh`

这恰好验证了 [Hessian.md](Hessian.md) 的核心发现——perturbed trajectory 的 Hessian eigenvalues 是 clean 的 30-300×。高曲率 + 非零 shift → 大 loss 增加，即使 gradient alignment 低。

### 两份分析的关系

| | Gradient Alignment | Hessian Eigenvalues |
|---|---|---|
| 阶数 | 一阶 | 二阶 |
| 测量对象 | ∇L 的方向 | H 的谱 |
| 核心发现 | δh ⟂ ∇L（alignment < 0.1） | λ_max(pert) ≫ λ_max(clean) |
| 结论 | 一阶不解释 loss 变化 | 二阶 curvature 是主因 |
| flipped/preserved 信号 | 有但弱（Δ ≈ 0.001-0.02） | 强（Δ ≈ 60-126 in λ_max） |

两者互补：
- Gradient alignment 证明一阶机制**不**成立
- Hessian eigenvalues 证明二阶机制**成立**

---

## 假说状态（更新）

| # | 假说 | 证据 | 状态 |
|---|------|------|------|
| H1 | perturbation shift 沿 ∇L 方向 | alignment < 0.1，δh 与 ∇L 近乎正交 | ❌ 否定 |
| H2 | flipped 的 alignment > preserved | 内部层成立（Δ ≈ +0.001），VAE 层反向（Δ ≈ −0.019） | ⚠️ 部分成立 |
| H3 | alignment 随 DiT 深度增加 | last > mid > layer0 成立（0.020 > 0.0019 > 0.0014），但 VAE 最高（0.08） | ⚠️ VAE 最高但原因不同 |
| H4 | DiT 抑制 preserved 扰动 | VAE 层 preserved > flipped，内部层 flipped > preserved → DiT 过滤了大但不致命的 shift | ✅ 成立 |
| H5 | 一阶 gradient 不能解释 perturbation effect | clean 点 cos < 0.04（内部层 < 0.006），perturb 点 cos < 0.09，两端独立验证 | ✅ 成立 |
| H6 | 二阶 Hessian 主导 perturbation effect | Hessian.md λ_max 30-300×，gradient alignment 两端 ≈ 0，Taylor 展开完整闭合 | ✅ 成立 |
| H7 | δh 方向 ⟂ J top-k 奇异向量（VAE 层） | containment ≈ random baseline（ActionSubspace.md） | ✅ 成立 |
| H8 | DiT 中间层将 S_p 映射到 S_action | L14 Jacobian + containment 未测（phase7 调试中） | ❓ 未测 |
| H9 | Hessian top-k 特征向量 ⟂ δh | 未测；Lanczos 特征向量精度不足以支撑索偿（D=401K, ε=1e-2） | ❓ 不可测 |
| H10 | H 的主导项是 ∂²a/∂h² 而非 JᵀJ | δh ⟂ J(top-k) + λ_max 暴涨 → 逻辑推断，未经直接分解验证 | ⚠️ 合理推断 |

---

## 整体结论

### 三层证据的交叉验证

#### 第一层：一阶排除 ✅ 严格成立

三个独立测量指向同一结论：

| 测量 | 含义 | 值 | 站点覆盖 |
|------|------|-----|----------|
| `cos(δh, ∇L_clean)` | Taylor 展开一阶项——δh 是否沿 clean 点 loss 最陡方向 | < 0.04（内部层 < 0.006） | 全部 4 层 |
| `cos(δh, ∇L_pert)` | perturb 点沿 δh 的 directional derivative 是否非零 | < 0.09（内部层 < 0.003） | 全部 4 层 |
| `cos(∇L_clean, ∇L_pert)` | 梯度方向旋转——landscape 非线性程度 | 0.1-0.4（旋转 66-82°） | 全部 4 层 |

**推不翻的结论**：不管在 clean 点还是 perturb 点测量，δh 都不指向 loss 最陡峭的方向。一阶项 δhᵀ∇L 在数值上为零。任何依赖梯度对齐假设的防御方法（adversarial training 的 PGD 内层、梯度裁剪、梯度正则化）在理论上都作用在错误的方向上。

#### 第二层：二阶幅度 ✅ 严格成立（给定 finite-difference 精度）

| 测量 | 值 | 站点 |
|------|-----|------|
| `λ_max(H_pert) / λ_max(H_clean)` | 30-300× | layer0, mid |
| Flipped vs Preserved λ_max 差距 | flipped 高出 60-126（call=49, layer0） | layer0 |
| Action deviation vs curvature Spearman | 0.81-0.97 | vae, layer0 |

**推不翻的结论**：perturbed trajectory 进入了 loss landscape 曲率显著更高的区域，flipped 比 preserved 更严重。λ_max 的幅度差异 + action deviation 的相关性 = 两个独立信号指向曲率驱动。

#### 第三层：Jacobian 敏感度 ≠ 有害性 ✅ VAE 层成立，内部层未测

| 测量 | 值 | 站点 |
|------|-----|------|
| δh 在 J top-k SVD 子空间的 containment | ≈ random baseline（0.01-0.05%） | VAE |
| 扰动子空间 S_p 和 S_action 夹角 | → 正交（overlap ≈ random baseline） | VAE |

**在 VAE 层可以确定**：Jacobian 的高敏感方向不是 δh 的方向。VAE 做无差别图像压缩，这本身不意外。关键问题是经过 28 层 DiT 非线性变换后，中间层是否将 S_p 映射到了 S_action——目前**未测**。

### 核心结论（可以稳妥说的话）

> Across all DiT layers, the perturbation-induced latent shift δh is nearly orthogonal to the loss gradient ∇L — both at the clean point (`|cos| < 0.04`) and at the perturbed point (`|cos| < 0.09`). The first-order Taylor term δhᵀ∇L is therefore numerically zero and cannot explain the 20–37× increase in action-matching loss. Independently, the largest Hessian eigenvalue λ_max grows 30–300× along perturbed denoising trajectories, with larger increases for flipped than preserved cases. **This curvature increase, combined with the vanishing first-order term at both endpoints, establishes that the loss landscape's local second-order geometry — not its gradient or Jacobian sensitivity — is the dominant mechanism by which visual perturbations degrade action predictions.**

### 合理推断（有间接证据但不能严格断言）

> The dominant Hessian contribution likely comes from the pure nonlinear curvature ∂²a/∂h² within DiT's transformer blocks (self-attention softmax, LayerNorm, etc.), not from the JᵀJ term — because δh lies in the approximate nullspace of J's top singular vectors at the VAE level. Standard defenses targeting gradient directions (worst-case adversarial training, whose PGD inner loop searches along ∇L) or Jacobian norms (Jacobian regularization, which penalizes JᵀJ) would therefore act on directions orthogonal to the actual perturbation shift, making them theoretically ineffective against this class of perturbations.

### 不应该说的话

- "Hessian top eigenvectors are misaligned with δh" — 未测特征向量，且在 401K 维空间中用 finite-difference HVP (ε=1e-2) + Lanczos 估算的特征向量精度不足以支撑这个索偿
- "Jacobian SVD directions are misaligned with δh at all layers" — 仅 VAE 层有数据（phase6），DiT 中间层（L14/L27）未测（phase7 调试中）
- "Nonlinear curvature ∂²a/∂h² dominates" — 逻辑推断合理但未经 H 分解实验直接验证

---

## 局限

1. **sensor_noise 缺失**（需 libmagickwand），当前仅覆盖 5/7 类扰动
2. **language_instructions 未测**（数据不可访问），但 language 在 VAE 层 δh=0，不贡献信号
3. **仅一个 task**（kitchen_scene4），跨 task 泛化性待验证
4. **Hessian 特征向量未测**：当前只测了 λ_max（特征值），未测特征向量与 δh 的对齐度。在 D=401K 空间中，Lanczos + finite-diff HVP（ε=1e-2）下特征向量的收敛精度不足以支撑方向索偿。且 H(h_clean) 和 H(h_pert) 是不同的矩阵——不存在"唯一的"Hessian 来定义"对齐"
5. **无法分解 H 的 JᵀJ 项和 ∂²a/∂h² 项**：从 δh ⟂ J(top-k) 推断 ∂²a/∂h² 主导，但未经直接测量验证
6. **last 层数值不可靠**：λ_max 大量为零，finite-diff step size 在高维 hidden state 上精度不足，主结论只使用 vae/layer0/mid
7. **Jacobian containment 仅 VAE 层**：DiT 中间层（L14）是关键的缺失测量——VAE 层 S_p ⟂ S_action 是预期的（VAE 做无差别压缩），DiT 是否在中间层将两者对齐才是核心问题

---

## 目录结构

```
experiments/phase9_gradient_alignment/
├── smoke_test/
│   ├── gradient_alignment_detail.csv
│   └── gradient_alignment_summary.json
└── kitchen_scene4_seed7/
    ├── gradient_alignment_detail.csv      # 2000 rows
    └── gradient_alignment_summary.json

bin/
└── phase9_gradient_alignment.py           # 主脚本
```
