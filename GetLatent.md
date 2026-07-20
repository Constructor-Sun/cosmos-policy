# GetLatent — Multi-Layer Shift Embedding Collection

核心目标：对 LIBERO-Plus 的 4 个 suite（libero_10 / libero_spatial / libero_object / libero_goal），在所有 7 类视觉/语言扰动下，收集每个 policy rollout step 的 **video/action shift embedding**——即扰动相对于 clean 在 VAE / L0 / L14 / L27 四层的 hidden state 偏移量。

## 方法论

### 测量点

| 测量点 | 含义 | Shape | 来源 |
|--------|------|-------|------|
| **VAE** | 视觉 latent（DiT 入口），扰动进入处 | `[16, 2, 28, 28]` | `model.get_data_and_condition()` → `latent_state` video slots 2,3 |
| **L0** | 第一个 transformer block 输出 | `[14, 14, 2048]` | `blocks[0]` forward hook, action slot (temporal 4) |
| **L14** | DiT 中间层 | `[14, 14, 2048]` | `blocks[14]` forward hook, action slot (temporal 4) |
| **L27** | 最后一个 block，FinalLayer 之前 | `[14, 14, 2048]` | `blocks[27]` forward hook, action slot (temporal 4) |
| **Action** | 最终输出 | `[112]` | `x0_pred[:, :, [4], :, :].reshape(-1)[:112]` |

所有层使用 **sigma=80**（第一个 denoising step），与 Phase 5/6/7 保持一致。

### 测量链路

```
Image (256²×3)
  → [VAE] → Video Latent [16, 9, 28, 28]
       ↓ x_embedder (patchify)
     [9, 14, 14, 2048]
       ↓ Block 0  ──→  L0  (hook)
       ↓ Block 1..13
       ↓ Block 14 ──→  L14 (hook)
       ↓ Block 15..26
       ↓ Block 27 ──→  L27 (hook)
       ↓ FinalLayer (AdaLN + Linear)
     Action [112]
```

### Shift embedding 定义

对于 perturbation $P$、layer $\ell$、episode step $t$：

$$\delta h_{\ell, P}(t) = h_\ell^{\text{pert}}(t) - h_\ell^{\text{clean}}(t)$$

$$\delta a_P(t) = a^{\text{pert}}(t) - a^{\text{clean}}(t)$$

其中 clean 和 pert 共享相同的 deterministic reset seed，确保 observation 差异纯粹来自扰动本身。

### 保存策略

- **保存间隔 N=5**：每 5 个 policy step 保存一次全部 4 层 + action
- **精度**：float16
- **Temporal slot**：DiT 层只保存 action slot（temporal index 4），VAE 保存 video slots（temporal indices 2, 3）

---

## 扰动类型

共 7 类扰动，每类对 40 个 task 都生成 1 个 evaluation episode。

| 扰动 | 机制 | 一致性保证 |
|------|------|-----------|
| `clean` | 原始 BDDL | 基准，共享 seed |
| `camera_viewpoints` | 代码层修改相机内外参 | 相同 init state |
| `light_conditions` | BDDL `_light_*` 变体 | 相同 init state |
| `background_textures` | BDDL `_table_*` 变体 | 相同 init state |
| `robot_initial_states` | BDDL `_add_*` 变体（distractor 变化） | 相同 base init |
| `sensor_noise` | observation 层注入噪声 | 相同 init state |
| `language_instructions` | T5 text 同义词替换 | 相同 observation |
| `objects_layout` | BDDL 变体 或 代码层修改 | 相同 init state |

### Seed 一致性

同一 task 的 clean 和所有 perturbed episode 使用：
- 相同的 `deterministic_reset_seed`
- 相同的 `init_state_index`
- 相同的 `seed`

```python
# 所有 condition 共享的参数
seed = 7
init_state_index = 0
deterministic_reset_seed = 0
```

---

## Task 选取

4 个 suite，每 suite 取全部 10 个具有完整扰动 BDDL 覆盖的 unique base task：

| Suite | Base tasks | Light 变体 | Table 变体 | Add 变体 |
|-------|-----------|-----------|-----------|---------|
| libero_10 | 10 | 50/task | 28/task | 30/task |
| libero_spatial | 10 | 50/task | ~18/task | 30/task |
| libero_object | 10 | 50/task | 28/task | 30/task |
| libero_goal | 10 | 50/task | 28/task | 30/task |

**总计**：40 tasks × 8 conditions (clean + 7 perturb) × 1 seed = **320 episodes**

---

## 存储估算

### 单次保存大小（per snapshot, float16）

| 数据 | Shape | 大小 |
|------|-------|------|
| VAE (video slots) | `[16, 2, 28, 28]` | 50 KB |
| L0 (action slot) | `[14, 14, 2048]` | 784 KB |
| L14 (action slot) | `[14, 14, 2048]` | 784 KB |
| L27 (action slot) | `[14, 14, 2048]` | 784 KB |
| Action | `[112]` | 0.4 KB |
| **单 snapshot 合计** | | **~2.35 MB** |

### 总量估算（N=5, ~250 steps/episode）

| 项目 | 数量 |
|------|------|
| 每 episode 保存次数 (~50 snapshots) | ~117 MB |
| 320 episodes × ~117 MB | ~37 GB |
| manifest.parquet (320 行元数据) | ~100 KB |
| **总计** | **~37 GB** ✅ |

磁盘预算 200 GB，使用率 ~19%，余量充足。

---

## 输出目录结构

采用 **HDF5 per episode** 格式（robomimic / D4RL 风格）。一个 episode 一个 `.h5` 文件，所有 step 的 tensor 组织在一个 HDF5 group 内。

```
experiments/phase8_shift_embeddings/
├── manifest.parquet                     # 全局索引（pyarrow Parquet）
├── config.json                          # 运行参数记录
├── h5/
│   ├── libero_10/
│   │   ├── {task_name}/
│   │   │   ├── {task_name}__clean.h5
│   │   │   ├── {task_name}__camera_viewpoints.h5
│   │   │   ├── {task_name}__light_conditions.h5
│   │   │   ├── {task_name}__background_textures.h5
│   │   │   ├── {task_name}__robot_initial_states.h5
│   │   │   ├── {task_name}__sensor_noise.h5
│   │   │   ├── {task_name}__language_instructions.h5
│   │   │   └── {task_name}__objects_layout.h5
│   │   └── ...
│   ├── libero_spatial/...
│   ├── libero_object/...
│   └── libero_goal/...
└── summary.json                         # 全局统计
```

### HDF5 内部结构

```
{task}__{condition}.h5
  /data/
    /step_{t:04d}/             # episode step（如 step_0000, step_0005, ...）
      VAE       [16, 2, 28, 28]  float16
      L0        [14, 14, 2048]   float16
      L14       [14, 14, 2048]   float16
      L27       [14, 14, 2048]   float16
      action    [112]            float16
  /metadata (HDF5 attributes)
    suite, task_name, condition, seed, deterministic_reset_seed,
    init_state_index, success, total_steps, num_snapshots, save_interval
```

### 读取示例

```python
import h5py

with h5py.File("h5/libero_10/.../task__camera_viewpoints.h5", "r") as f:
    # 读元数据
    success = f.attrs["success"]
    
    # 读单个 layer 的单个 step（不加载其他 step）
    L0_step5 = f["data/step_0005/L0"][:]    # [14, 14, 2048] float16
    L0_step5 = f["data/step_0005/L0"][()]   # 同上
    
    # 遍历所有 step
    for step_name in f["data"]:
        vae = f[f"data/{step_name}/VAE"][()]
        ...
```

### manifest.parquet 格式

| 列 | 类型 | 示例 |
|----|------|------|
| suite | str | libero_10 |
| task_name | str | KITCHEN_SCENE4_put_bowl... |
| condition | str | camera_viewpoints |
| h5_path | str | h5/libero_10/.../task__camera_viewpoints.h5 |
| seed | int | 7 |
| success | bool | True |
| total_steps | int | 245 |
| num_snapshots | int | 49 |
| save_interval | int | 5 |

```python
import pandas as pd
df = pd.read_parquet("manifest.parquet")
# 筛选所有 flipped 的 camera_viewpoints episode
flipped = df[(df["condition"] == "camera_viewpoints") & (~df["success"])]
```

---

## 脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `bin/phase8_collect_latent.py` | 主收集脚本：启动 env → policy rollout → hook 4 层 → 每 N 步写入 HDF5 | 📝 待实现 |
| `bin/phase8_compute_delta.py` | 读取 clean + perturb HDF5，计算 delta 并追加写入 perturb HDF5 的 `/delta/` group | 📝 待实现 |
| `bin/phase8_generate_manifest.py` | 扫描 h5/ 目录，生成 manifest.parquet | 📝 待实现 |

### Delta 存储

`phase8_compute_delta.py` 对每个 perturbed HDF5，找到对应 clean HDF5，计算：

```
/delta/
  /step_{t:04d}/
    VAE       = VAE_pert - VAE_clean   # 同 step
    L0        = L0_pert  - L0_clean
    L14       = L14_pert - L14_clean
    L27       = L27_pert - L27_clean
    action    = action_pert - action_clean
```

Delta 直接追加写入 perturbed HDF5 的 `/delta/` group 中，不创建额外文件。

---

## 运行计划

### Pilot（验证 pipeline）

1 task（KITCHEN_SCENE4）× 8 conditions = 8 episodes

```bash
MUJOCO_GL=egl LIBERO_PLUS_PATH=/data3/liu/exp/counterfactual/external/LIBERO-plus \
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
python bin/phase8_collect_latent.py \
  --suite libero_10 \
  --task "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
  --save-interval 5 \
  --seed 7 \
  --output-dir experiments/phase8_shift_embeddings
```

预期：~10 min，~1 GB 存储

### 全量运行

```bash
MUJOCO_GL=egl LIBERO_PLUS_PATH=/data3/liu/exp/counterfactual/external/LIBERO-plus \
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
python bin/phase8_collect_latent.py \
  --all-suites \
  --save-interval 5 \
  --seed 7 \
  --output-dir experiments/phase8_shift_embeddings
```

预期：~7 hours，~38 GB 存储

---

## 实际运行记录（2026-07-18）

实际全量命令使用 `bin/collect_shift_embeddings.py --all-suites`，输出目录：

```
experiments/phase8_shift_embeddings/all_suites_full/
```

本次运行结果：

| 项目 | 数量 |
|------|------|
| attempted episodes | 328 |
| manifest rows / H5 files | 235 |
| success rows | 228 |
| perturbation failure rows | 7 |
| errors | 93 |
| disk usage | ~40 GB |

235 个 H5 文件均可正常打开，主要 dataset shape 一致：

| Key | Shape | dtype |
|-----|-------|-------|
| `VAE` | `[2, 9, 28, 28]` | float16 |
| `L0` / `L14` / `L27` | `[3, 14, 14, 2048]` | float16 |
| `action` | `[112]` | float16 |

当前最有价值的是 7 个 `clean success + perturbation failure` 样本：

| Suite | Task | Failed condition | Variant |
|-------|------|------------------|---------|
| libero_goal | `open_the_top_drawer_and_put_the_bowl_inside` | `background_textures` | `tb_21` |
| libero_goal | `put_the_bowl_on_top_of_the_cabinet` | `language_instructions` | `language_35` |
| libero_goal | `put_the_cream_cheese_in_the_bowl` | `language_instructions` | `language_33` |
| libero_goal | `turn_on_the_stove` | `language_instructions` | `language_36` |
| libero_object | `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | `objects_layout` | `add_13` |
| libero_object | `pick_up_the_butter_and_place_it_in_the_basket` | `objects_layout` | `level5_sample2` |
| libero_object | `pick_up_the_salad_dressing_and_place_it_in_the_basket` | `objects_layout` | `level3_sample3` |

93 个 errors 主要来自 `--all-suites` 的任务枚举问题，而不是 rollout 失败：

- `libero_goal` 的 10 个 `*_moved` tasks 没有完整 language perturbation 覆盖，合计约 80 个 errors。
- `libero_spatial` 中误解析出 `pick_up_the_black_bowl_from`，并且 `table_center` 的部分 perturbation 未写入 manifest。
- 因此分析时应优先使用 manifest 中已写出的 canonical rows，重点比较上表 7 个 failure rows 与对应 clean rows。

T5 embedding 状态：

- 原始 clean/base instruction embedding 已由 `Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl` 覆盖。
- LIBERO-plus `language_instructions` 额外 embedding 已生成到 `experiments/phase8_shift_embeddings/libero_main_suites_language_t5.pkl`。
- 对已写入 manifest 的 language variants，未发现 T5 embedding 缺失。

---

## 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 文件格式 | HDF5（无压缩） | 单文件 per episode（320 个），支持按 step/layer 随机读取，robomimic/D4RL 标准 |
| Manifest | Parquet | HF 生态原生，列式存储，pandas 直接读 |
| 保存间隔 | N=5 | 平衡轨迹分辨率和存储，5 步分辨率对行为分析足够 |
| DiT temporal slot | 仅 action slot (idx 4) | 全 9 slot 太大（20+ MB/step），action slot 最直接反映"路由到 action"的信号 |
| VAE temporal slot | video slots (idx 2, 3) | 这是 DiT 的图像输入区域，是扰动的直接入口 |
| 精度 | float16 | 存储减半，hidden state 本身是 bfloat16，精度损失可忽略 |
| sigma | 80 | 与 Phase 5/6/7 一致，第一个 denoising step，AdaLN 激活 |
| seed | 7 | 与 Phase 6/7 一致 |
| 每 task 1 init state | 是 | 最大化 task 覆盖度而非 seed 多样性 |
