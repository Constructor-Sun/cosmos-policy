# Action-Sensitive Subspace Coverage Analysis

核心问题：DiT 的哪些 latent 方向对 action 输出最敏感？7 类视觉扰动的 latent shift 是否落在这些敏感方向上？

## 方法论

### S_action 的定义（模型属性，与扰动无关）

S_action = Jacobian ∂a/∂h 的 top-k right singular vectors。

- σ_i：沿 v_i 方向扰动 latent 时对 action 输出的放大系数
- top-k 构成 action-sensitive subspace

在 **sigma=80**（第一个 denoising step，AdaLN 激活，attention 工作）测量。sigma_min 处 FinalLayer 退化（rank=0），无法定义有意义的 S_action。

### 测量点

| 测量点 | 含义 | Jacobian 方式 | 状态 |
|--------|------|---------------|------|
| **VAE output** | 视觉 latent（DiT 入口），扰动进入处 | Autograd 通过 28 层 DiT + FinalLayer | ✅ |
| **Layer 14** | DiT 中间层 | 需要 autograd 通过 blocks[15:] | ⚠️ 未做 |
| **Layer 27** | DiT 最后，FinalLayer 之前 | FinalLayer 权重 W_linear SVD | ✅ |

### 测量链路

```
Input (256×256) → [VAE] → Video Latent (25K dim)
                              ↓ [DiT ×28 blocks]
                           L14 (401K dim)
                              ↓
                           L27 (401K dim)
                              ↓ [FinalLayer: AdaLN + Linear]
                           Action (112 dim)
```

- **VAE**：无差别图像压缩，不区分 action-relevant vs irrelevant
- **DiT blocks**：通过 self/cross-attention 提取 action-relevant 信号
- **FinalLayer**：线性读出，AdaLN 在 sigma_min 时压缩导致退化

### Containment

```
containment = ||Proj_{S_action} delta_h||² / ||delta_h||²
random baseline = k / D
```

---

## 关键发现

### 1. VAE 层 S_action：高度集中

| k | 累积 sensitivity | 占 VAE 空间 |
|---|-----------------|-------------|
| 1 | 83.0% | 0.004% (1/25088) |
| 3 | 91.9% | 0.012% |
| 5 | 93.1% | 0.020% |
| 10 | 95.0% | 0.040% |

σ₁/σ₂ = 3.6x。单一方向占 83%。有效秩 = 112（满秩），但主要 sensitivity 在 top-3。

### 2. VAE 层 containment：扰动与 S_action 几乎正交

127 pairs（sigma=80）：

| Perturbation | Flipped N | k=3 containment | Preserved N | k=3 containment |
|---|---|---|---|---|
| light_conditions | 8 | **0.000584** | 12 | 0.000542 |
| background_textures | 20 | 0.000485 | — | — |
| robot_initial_states | 19 | 0.000427 | 1 | — |
| camera_viewpoints | 5 | 0.000133 | 15 | 0.000104 |
| sensor_noise | — | — | 20 | 0.000027 |
| language_instructions | — | 0 | 0 | 0 |
| *Random baseline* | — | *0.000120* | — | — |

- Flipped mean / Preserved mean ≈ 2x（方向性存在但绝对值极小）
- 所有值都在 random baseline 的量级（0.01-0.05%）
- language_instructions = 0（只改 T5 text，不改 VAE）
- sensor_noise 最低（噪声均匀分散在所有方向上）

### 3. 各扰动 S_p 在 VAE 处是低维的

| Perturbation | Flipped N | r80 | r90 |
|---|---|---|---|
| camera_viewpoints | 5 | 3 | 4 |
| light_conditions | 8 | 5 | 6 |
| robot_initial_states | 19 | 10 | 14 |
| background_textures | 20 | 11 | 15 |
| sensor_noise | 20 | 13 | 16 |

### 4. S_p 和 S_action 在 VAE 空间中正交

以 light_conditions flipped（S_p ≈ 5 维）和 S_action k=3 为例：
- 随机期望重叠：~0.0006
- 实际重叠：~0.00058

**在 VAE 空间中，perturb-specific subspace 和 action-sensitive subspace 基本是正交的低维子空间。** DiT 的作用是将不相关的 S_p 通过 28 层 attention 逐步映射到 S_action 上。

### 5. L27 层：FinalLayer Jacobian 极稀疏

FinalLayer 只有 28 个 grid position × 4 个 output channel 对 action 有贡献。
- S_action 维度：112 / 401,408 = 0.028%
- 4 个 action-relevant 权重向量的 singular values 几乎相等 [0.20, 0.16, 0.16, 0.15] → 近似各向同性
- 在 sigma_min 时 AdaLN 压缩 (1+scale)≈0，Jacobian 完全退化

### 6. L27 containment（phase7，待完成调试）

phase7 脚本已覆盖 VAE + L27 双层 containment，但当前存在运行时 bug：
- `DiTForward` 已修复为用真实图像构造 data_batch
- `project_L27` 已修复 temporal 维度处理
- `errors` 已改为 list 类型

---

## 假说状态

| 假说 | 证据 |
|------|------|
| **H1**: 所有扰动共享 global latent damage subspace | ❌ 跨扰动 flipped centroid cos ≈ 0，不共享 |
| **H2**: 每类扰动有独立低维 S_p | ✅ r80 在 3-15 维内，远小于 D=25088 |
| **H3**: S_p 的有害性来自与 S_action 的重叠 | ⚠️ VAE 层两者正交；需要在 L14/L27 验证 |
| **H4**: perturb-specific correction > global correction | ⚠️ 未测 |

**核心未解决问题**：DiT 的中间层（L14）是否将 S_p 和 S_action 对齐了？目前只有 VAE（入口）的测量——S_p 和 S_action 在此处正交。需要在 L14 测 Jacobian + containment。

---

## 脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `bin/phase5_jacobian_autograd.py` | ∂a/∂h at VAE output, sigma=80 | ✅ |
| `bin/phase6_vae_containment.py` | VAE delta_h → S_action 投影 (127/140) | ✅ |
| `bin/phase6b_delta_h_dimensionality.py` | 各扰动 delta_h_VAE 有效维度 | ✅ |
| `bin/phase4_jacobian.py` | FinalLayer 权重 SVD（L27, sigma_min） | ✅ |
| `bin/phase7_multilayer.py` | VAE + L27 双层 containment | 🔄 调试中 |

---

## 运行

```bash
cd /data3/liu/exp/counterfactual/external/cosmos-policy
export PATH=/data2/haoze/miniconda3/bin:$PATH
eval "$(/data2/haoze/miniconda3/bin/conda shell.bash hook)"
conda activate cosmospolicy

# Phase 5: VAE Jacobian (约 6 min，112 backward passes)
MUJOCO_GL=egl PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
python bin/phase5_jacobian_autograd.py --sigma 80

# Phase 6: VAE containment (约 12 min，140 pairs × 2 VAE encodes)
MUJOCO_GL=egl LIBERO_PLUS_PATH=/data3/liu/exp/counterfactual/external/LIBERO-plus \
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
python bin/phase6_vae_containment.py \
  --summary experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/...combined_summary.json

# Phase 7: 多层 containment (调试中)
MUJOCO_GL=egl LIBERO_PLUS_PATH=... PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
python bin/phase7_multilayer.py --sigma 80
```

## 目录结构

```
experiments/
├── phase5_jacobian_autograd/          # VAE ∂a/∂h + SVD
│   ├── jacobian.npy                   # [112, 25088]
│   ├── singular_values.npy            # [112]
│   ├── right_singular_vectors.npy     # [112, 25088]
│   └── summary.json
├── phase6_vae_containment/            # VAE δh → S_action 投影
│   ├── containment_results.csv
│   └── summary.json
└── phase7_multilayer/                 # L27 + VAE 双层 (待完成)
    ├── containment.csv
    └── summary.json
```
