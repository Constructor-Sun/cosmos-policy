# Causal Hessian: Taylor Validation & Intervention

本文档记录对 Hessian 曲率因果性的两个直接检验：

1. **Taylor 展开验证**：$\Delta L \stackrel{?}{\approx} \nabla L^T\delta_h + \frac{1}{2}\delta_h^T H \delta_h$，分别检查一阶项和二阶项的实际贡献
2. **Inference-time 干预**：消除 δh 在 top-k Hessian 特征向量上的投影，观察 action error 是否下降

---

## 1. 实验设置

### 数据

- **Taylor 验证**：当前 CSV 实际包含 camera_viewpoints，11 episodes（9 preserved + 2 flipped）× 2 calls (37, 49) × 2 sites (vae, layer0) = 44 rows
- **干预实验**：robot_initial_states，1 flipped episode × 1 call (37) × 3 sites (vae, layer0, mid) × k=3

### 方法

- Lanczos：finite difference HVP，ε=1e-2，4 iterations（smoke），8 iterations（Taylor）
- Loss：`MSE(extract_action(pred_x0), a_clean)`，EDM weighted
- Sites：vae（完整 28 层 gradient），layer0（block 0 输出），mid（block 14 输出）

### 核心指标

| 指标 | 含义 |
|---|---|
| `cos(δh, ∇L)` | 一阶方向对齐度；不能单独代表一阶项的数值大小 |
| `gradient_term` | 一阶项 $\nabla L^T\delta_h$ |
| `predicted_quadratic_term` | 二阶项 $\frac{1}{2}\delta_h^T H\delta_h$ |
| `prediction_ratio` | `(gradient_term + predicted_quadratic_term) / actual ΔL`，应该 ≈ 1.0 |
| `recovery` | (mse_before − mse_after) / (mse_before − mse_clean)，> 0 表示干预有效 |
| `proj_fraction` | δh 能量在 top-k 特征空间中的占比 |
| `cos(δh, v_max)` | δh 与 top 特征向量的对齐度 |

---

## 2. Taylor 展开验证

### 2.1 小 cosine 不等于一阶项可忽略

camera_viewpoints 的样本上：

```
call 37, vae:     cos(δh, ∇L) ∈ [+0.040, +0.120]   中位数 ≈ +0.061
call 37, layer0:  cos(δh, ∇L) ∈ [+0.001, +0.003]   中位数 ≈ +0.002
call 49, vae:     cos(δh, ∇L) ∈ [+0.040, +0.121]   中位数 ≈ +0.063
call 49, layer0:  cos(δh, ∇L) ∈ [+0.0003, +0.002]  中位数 ≈ +0.0007
```

这些 cosine 确实很小，但不能据此推出 $\delta_h^T \nabla L \approx 0$，因为

$$
\delta_h^T \nabla L = \|\delta_h\|\,\|\nabla L\|\cos(\delta_h,\nabla L).
$$

当前位移范数很大（VAE 层约 137--153，layer0 约 222--267），因此小 cosine 仍然可以产生很大的点积。原始 `taylor_validation.csv` 直接记录的贡献比例证明，一阶项不能忽略。

按每行 `term / delta_loss_actual` 后取中位数：

| Site | Call | 一阶项 / actual $\Delta L$ | 二阶项 / actual $\Delta L$ | 两项合计 / actual $\Delta L$ |
|---|---:|---:|---:|---:|
| vae | 37 | **188.05%** | 1.33% | 190.14% |
| vae | 49 | **200.24%** | 2.02% | 202.17% |
| layer0 | 37 | **13.05%** | 0.0056% | 12.91% |
| layer0 | 49 | **0.300%** | 0.0017% | 0.298% |

例如 episode 0、VAE、call 37：actual $\Delta L=0.14942$，一阶项为 0.31830（213.0%），二阶项为 0.00645（4.31%）。call 49 上 actual $\Delta L=712.24$，一阶项为 1425.54（200.15%），二阶项为 14.39（2.02%）。

因此，这组 Taylor 数据不能支持“一阶导数没有用”或“一阶项为零”。它支持的是更细的结论：**方向对齐度低，但 VAE 层一阶点积在数值上占主导；layer0 的局部一阶、二阶项都不足以解释实际 loss 变化。**

### 2.2 VAE 层总 Taylor 预测约为 2× actual ΔL

```
call 37, vae:   prediction_ratio ∈ [1.83, 2.17]    中位数 = 1.93
call 49, vae:   prediction_ratio ∈ [2.00, 2.13]    中位数 = 2.02
```

**关键发现**：这里的 `prediction_ratio` 是一阶项与二阶项之和除以 actual $\Delta L$，不是 Hessian 二阶项单独的比例。约 2× 的预测几乎全部来自一阶项；Hessian 二阶项的中位贡献仅为 1.33%（call 37）和 2.02%（call 49）。

总 Ratio 在这些 episode × 2 个 call 之间较稳定，且 **preserved 和 flipped 无明显系统差异**：

```
call 37, vae:  preserved median ratio = 1.93   (n=9)
               flipped   median ratio = 1.90   (n=2)
call 49, vae:  preserved median ratio = 2.03   (n=9)
               flipped   median ratio = 2.00   (n=2)
```

不能把这个 ~2× 归因于 Hessian。当前结果反而表明，VAE 层在 clean anchor 上的一阶局部线性预测约为 actual $\Delta L$ 的 1.88--2.00 倍，而二阶修正很小。总预测高估约 2×，说明沿完整 $\delta_h$ 的有限位移已经超出可靠的局部 Taylor 区间，或 loss/trajectory 的缩放口径仍需核查；现有数据不足以区分这两种原因。

### 2.3 Layer0 的一阶和二阶局部项都不能解释 loss 变化

```
call 37, layer0:  prediction_ratio ∈ [0.07, 0.36]
call 49, layer0:  prediction_ratio ∈ [0.001, 0.004]
```

这里此前展示的 `prediction_ratio` 同样是“一阶 + 二阶”的总和，而非 Hessian 单项。拆开后，call 37 的一阶项中位贡献为 13.05%，二阶项仅 0.0056%；call 49 分别为 0.300% 和 0.0017%。因此 layer0 处两阶局部展开都系统性低估 actual $\Delta L$，不能据此单独判定“二阶主导”或“一阶无用”。

---

## 3. Inference-time 干预 V1：δh 投影法（❌ 方法有误）

### 3.1 方法

```
h_corrected = h_pert − proj_{top-3 eigenvectors}(δh)
recovery = (mse_before − mse_after) / (mse_before − mse_clean)
```

### 3.2 结果

**robot_initial_states, episode 0, flipped** (overall action_mse = 0.0347):

| Site | λ_max | proj_fraction | cos(δh, v_max) | recovery |
|---|---|---|---|---|
| vae | 0.10 | 1.23% | 0.0123 | **+0.0069** |
| layer0 | 0.62 | 0.10% | 0.0010 | **+0.0005** |
| mid | 0.0075 | 0.17% | 0.0017 | **+0.0027** |

所有三个 site 的 recovery 都接近 0。

### 3.3 批判：这个方法在几何上是错误的

上述解读（"δh 不指向高曲率方向 → 曲率非因果"）存在范畴错误：

- **Hessian 特征向量**描述的是 h_pert 处的**局部曲率**：从 h_pert 出发，往哪个方向走 loss 变化最快
- **δh = h_pert − h_clean** 是连接两点的**全局位移向量**，不是一个局部方向

把全局路径投影到局部曲率方向上，等于在问"回 clean 的路是否恰好沿着当前站的位置最陡的方向"——它凭什么是呢？

**正确的 Hessian 用途**：不是分解 δh，而是分解**梯度** g = ∇_h L(h_pert)。Hessian 告诉你哪些梯度方向可信（低曲率 = 平坦 = 可以走大步），哪些不可信（高曲率 = 陡峭 = 容易 overshoot）。

---

## 4. Inference-time 干预 V2：梯度方向分割法（✅ 方法修正）

### 4.1 方法

将梯度 g = ∇_h L(h_pert) 按 Hessian 特征向量分为低曲率和高曲率分量，做三种单步 correction 比较：

```
h_full = h_pert − α · g                （全方向梯度下降）
h_low  = h_pert − α · (I − V_k V_k^T) · g    （仅平坦方向）
h_high = h_pert − α · V_k V_k^T · g          （仅陡峭方向）
```

**预测**：如果 Hessian 曲率是因果的 → 在平坦方向走应该比在陡峭方向走恢复得更多：

```
recovery_low  >  recovery_full  >  recovery_high
```

直觉：高曲率 = 悬崖，梯度不可靠；低曲率 = 缓坡，梯度可信。

### 4.2 实验设置

- **数据**：camera_viewpoints，1 preserved episode × 1 call (37) × 3 sites (vae, layer0, mid)
- **Lanczos**：autograd HVP，24 iterations，top-k = {10, 20}
- **Loss**：纯 action MSE（`MSE(action(pred_x0), a_clean)`），无 EDM weights
- **步长 sweep**：α ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}，不归一化梯度方向
- **Recovery**：`(mse_before − mse_after) / (mse_before − mse_clean)`

### 4.3 结果

**camera_viewpoints, episode 0, preserved** (action_mse_before = 0.00293, action_mse_clean = 0.00013):

| Site | λ_max | ||g|| | best α | recovery_low | recovery_full | recovery_high |
|---|---|---|---|---|---|---|---|
| vae | 0.0007 | 0.0010 | 0.1 | **+0.0007** | +0.0003 | +0.0005 |
| layer0 | 0.0044 | ~0 | — | 0.0000 | 0.0000 | 0.0000 |
| mid | 0.0000 | ~0 | — | 0.0000 | 0.0000 | 0.0000 |

**所有 recovery 都接近 0。** 即使最大的 α=0.1，步长仍然太小，因为梯度范数 ||g|| 本身就极小（~0.001 at vae, ~0 at layer0/mid）。

### 4.4 解读

这个结果只说明：**在当前单个 preserved sample 的 h_pert 处，使用纯 action MSE 得到的梯度范数很小，所测试的单步梯度下降无法降低 action error。**

它不能推广成“一阶导数没有用”。Section 2 的 Taylor 验证使用 clean anchor 和 EDM-weighted loss，VAE 层的一阶项反而是主要数值贡献。两组实验的 anchor、loss scaling 和问题不同：Taylor 实验衡量一阶项对 clean-to-perturbed 有限位移的解释力，V2 衡量从 perturbed 点出发做单步局部校正是否有效。

这意味着：

1. **这个 sample 的 h_pert 局部梯度信号很弱**。它可能位于局部极小值、鞍点或平坦区域，但当前实验不足以区分三者
2. **当前单步曲率干预无法发挥作用**，因为这个点和这个 loss 下的梯度步长太小
3. **单步梯度干预不足以恢复**——可能需要：
   - 多步迭代梯度下降
   - 更大的步长（但现在 ||g|| ≈ 0，增大 α 也没用）
   - 完全不同的干预策略（如直接沿 δh 的某个子空间分量走，而非沿梯度）

### 4.5 V1 vs V2 对比

| | V1（δh 投影） | V2（梯度分割） |
|---|---|---|
| 投影对象 | δh（全局位移） | g（局部梯度） |
| 几何意义 | 错：把路径当方向 | 对：用曲率信息选梯度方向 |
| Recovery | ~0 | ~0 |
| 失败原因 | 方法错误 | 当前 sample/loss 下 ||g|| ≈ 0，单步移动不足 |

V1 碰巧也得到了 recovery ≈ 0，但原因是错的。V2 在正确的方法下得到了同样的 recovery ≈ 0，但原因完全不同：不是曲率不因果，而是当前 sample/loss 下的局部梯度不足以支持有效的单步移动。

---

## 5. 综合结论

### 5.1 修正后的因果判断

之前 V1 的结论"曲率不是因果的"**缺乏方法论支撑**——它的实验设计有几何错误。V2 修正了方法，但结果仍为 recovery ≈ 0：在所测单个 h_pert 和纯 action MSE 下，局部梯度太小，单步干预无法有效移动。

**目前的状态是"未证明"而非"已否证"**：我们不知道曲率是否因果。与此同时，Taylor CSV 已明确否定“一阶项可以由低 cosine 直接忽略”的说法。

### 5.2 现有结果能说什么

- **VAE 层一阶项是 Taylor 数值的主要贡献**：中位数为 actual $\Delta L$ 的 188%（call 37）和 200%（call 49）；二阶项仅为 1.33% 和 2.02%
- **低 cosine 不能推出一阶项无用**：点积还取决于 $\|\delta_h\|$ 和 $\|\nabla L\|$
- **Layer0 的局部一阶、二阶项都不足**：call 37 合计约 12.9%，call 49 合计约 0.30%，说明该层的局部二阶展开不能解释完整有限位移
- **单步梯度干预无效**（V2）只在当前单个 sample、perturbed anchor 和纯 action MSE 设置下成立，不能外推为梯度方法普遍无效

### 5.3 局限

| 局限 | 影响 |
|---|---|
| 仅 1 个 preserved sample | 需要在更多样本上确认 ||g|| ≈ 0 是普遍现象 |
| 仅测了 2 个 top-k (10, 20) | k 更大的影响未知 |
| ||g|| ≈ 0 | 梯度降无法检验曲率的因果性——检验工具本身失效 |
| 仅单步 | 多步迭代可能产生不同的 recovery 模式 |
| camera_viewpoints 的 preserved | flipped 样本可能有非零梯度 |
| Taylor 位移范数很大 | 总 Taylor 预测在 VAE 高估约 2×，高阶项或缩放口径可能不可忽略 |
| Taylor 与 V2 的 anchor/loss 不同 | 不能直接比较二者的梯度大小或推广为统一的一阶机制结论 |

### 5.4 下一步

1. **验证 ||g|| ≈ 0 的普遍性**：在 flipped samples 上测梯度范数，看是否为非零。如果 flipped 有显著梯度，在那里做 V2 才是有意义的因果检验
2. **多步迭代**：单步梯度不行就做多步（每步重算 Hessian），看在低曲率方向上迭代是否比全方向收敛得更好
3. **替代 loss**：当前 loss = `MSE(pred_action, a_clean)`。可以尝试在 latent space 上定义 loss（`MSE(pred_x0, x0_clean)`），可能梯度更丰富
4. **δh 归因**：用 Phase 4 的 action-sensitive subspace 分解 δh，直接沿 δh 的子空间分量移动，而不是沿梯度
