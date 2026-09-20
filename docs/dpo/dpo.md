# TTA DPO

> 本文合并并取代原 `TTA_TRAINING_NOTES.md`（历史规划）、`TTA_TRAINING_IMPLEMENTATION.md`（实现说明）、`TTA_DPO_LAUNCH.md`（启动手册）三份文档。

## 1. 已有的 DPO 文章

- [Diffusion-DPO 原论文](https://arxiv.org/abs/2311.12908)（Wallace et al.）：扩散生成的偏好优化，经典目标。
- [作者官方实现](https://github.com/SalesforceAIResearch/DiffusionDPO)：参考噪声构造与正负相对去噪误差的计算。
- [Diffusers LoRA Diffusion-DPO 示例](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/diffusion_dpo)：adapter / reference 管理参考。

注意：Cosmos 用 EDM 和不同 latent 布局，不能直接换数据路径运行；本项目是 action-frame EDM 适配，不宣称算出精确轨迹 log probability。

## 2. 现在使用 DPO 的目的

背景：SFT 裸部署修复 22/68 = 32%，memory 修复机制上限 61/68 = 90%，差 58 个点；132 个 base 会做的任务回退 5.3%。DPO 的三个结构性增量：

1. **不要求逐帧克隆闭环动作**，只要求去噪方向偏好 chosen、排斥 rejected——直接攻击 SFT 与 memory 修复之间的 58 点差距（开环 BC 模仿闭环控制器的结构性损失）。
2. **rejected 信号显式惩罚辨别型失败**（"别接近错的那一侧"）——对准 LRSCENE5/LRSCENE2 这类对比型失败，SFT 的纯正样本信号对此很弱。
3. **reference = base 本身**，为无偏好信号的区域提供锚定——132 回退有望显著缩小。

不解决什么：阶段覆盖缺口（训练对来自同一批 68 case、同一批失败相位），该缺口要靠按阶段造数据或部署时 memory 兜底。

**判决矩阵（跑前注册，不得事后改）**：

| 68 case 成功率 | 结论 | 下一步 |
|---|---|---|
| ≥ 45%（≥31/68，超 SFT 达 13 case ≈ 2σ） | 偏好通道胜出 | 加大 DPO，SFT 降级 |
| 30–40% | 与 SFT 平手 | 主攻按阶段修复数据 |
| < 30% | 偏好信号不稳 | 混合部署（SFT + memory 兜底） |

独立专项指标：132 回退 ≤7；LRSCENE5+LRSCENE2 合计 ≥8/32（SFT 基线 2/32）。

## 3. DPO 算法的构成

**代码**（全部在 `memory_system/tta/`）：

| 文件 | 职责 |
|---|---|
| `dataset.py` | 每个 item = 一个 pair（chosen/rejected 各 1 个 chunk，K=1），复用 `libero_dataset.build_action_chunk_sample` 组装 |
| `model.py` | `TTADPOModel`：加载策略、LoRA 注入、共享噪声 DPO 前向、adapter 存取 |
| `dpo_train.py` | 单卡自定义循环（无端口、无分布式）、AdamW 仅 LoRA 参数、adapter 保存 |
| `test_tta.py` | CPU 测试 + `--gpu` smoke |
| `tools/` | 一次性数据准备：`replay_rejected` / `collect_chosen` / `build_manifest` |

**数据**：manifest 为 JSON `{"pairs": [{pair_id, task, init, chosen_path, rejected_path, chosen_success, rejected_success}]}`。episode 即 eval 采集格式；约定 obs[k] 在 action[k] 执行前；chunk 只取完整段（start+16 ≤ T）；归一化复用 SFT 统计。关键性质：chosen 与 rejected 在 t* 之前动作逐位相同，对比只在失败点之后分叉。

**目标公式**（两侧共享同一份 sigma/epsilon，bf16 前向）：

```
E(side)   = edm_loss_per_frame[side, action_latent_idx]     # 动作帧去噪误差
delta     = E_policy(side) - E_reference(side)              # reference = 关 LoRA、no_grad
margin    = beta * (delta_rejected.sum() - delta_chosen.sum())
loss      = -logsigmoid(margin)
```

K=1 时是单 chunk 偏好目标，不是全轨迹似然。整 pair 单前向 + backward 峰值 ~10.9GB。

**LoRA**：rank 8 / alpha 16 / dropout 0，q/k/v/o/mlp，共 280 层 11.5M 参数；base、VAE、T5 全冻结。默认超参 beta 0.1、lr 1e-5、batch = 1 pair。

**训练命令**（单卡，自动选显存最空的卡，预算 20GB；训练前必须 `export IMAGINAIRE_OUTPUT_ROOT=/data1/liu/exp/counterfactual/checkpoints`）：

```bash
python memory_system/tta/dpo_train.py \
  --manifest <pairs.json> \
  --checkpoint /data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt \
  --dataset-stats-path training/tta_sft_metadata/dataset_statistics.json \
  --t5-text-embeddings-path training/tta_sft_metadata/t5_embeddings.pkl \
  --output-dir memory_system/tta/results/runs --max-steps 500
```

**评测接入**：`run_libero_eval.py --adapter_path <adapter.pt>`；或经 `tests/tta/merge_dpo_adapter.py` 合并成完整 ckpt 后走 `scripts/eval_sft_68.py` / `eval_sft_success_132.py`。

**数据构建要点**：rejected 补图用 `tools/replay_rejected.py`（确定性重放，必须显式传 `--diagnosis-root` 指向 `tta_failure_screening_68`，工具默认指向只剩 1 个 case 的 v3 目录）；manifest 用 `tools/build_manifest.py`（按 summary 成功配对会产 incomplete pair，过滤 `chosen_path=null` 后校验）。

**已验证**：CPU 22/22 + 真模型 GPU 验证 PASS（LoRA 注入 280 层/11.5M、adapter 开关等价、梯度仅 LoRA、峰值 7.51GB），记录见 `memory_system/tta/results/VALIDATION_RECORD_20260909.md`。环境坑：eval 链前 `export PATH=/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH`；本机无外网（`HF_HUB_OFFLINE=1`）；路径全用绝对路径；eval 对子进程失败静默，跑完必须数 summary。

## 4. DPO 流程涉及的文档

| 文档 | 关系 |
|---|---|
| `docs/repair/libero-plus-repair.md` | 修复流水线——chosen 轨迹的来源；数据构建命令见其 §2 |
| `docs/sft/sft.md` | SFT 流水线、LoRA 死初始化 bug 与 `grad_flow_probe.py` 守则、SFT 基线结果（22/68、125/132）——DPO 的对照基线与数据来源 |
| `memory_system/tta/results/VALIDATION_RECORD_20260909.md` | 实现验证记录与未验证假设清单 |
| `scripts/eval_sft_68.py` / `scripts/eval_sft_success_132.py` | 评测脚本（68 失败 case / 132 通过 case） |

## 5. DPO 的实验结果、可能的视频

**数据与训练资产（实测，2026-09-19）**：

- manifest 共 5 版：`memory_system/tta/results/manifests/` 下 v1=52 对 → v2/v2_notstar=59 → v2_preview56=56 → **v3=57 对**（最新）；
- rejected 补图：`experiments/tta/dpo_rejected_images/`（73 个 hdf5）；
- 训练 run：`memory_system/tta/results/runs/` 下 11 个（2026-09-13/14，含 `anchor_only`、`beta1_pure` 消融）；日志在 `experiments/tta/redo68/dpo_train_*.log`；
- merged ckpt：`training/` 下 7 个（`tta_dpo_s100_merged`、`tta_dpo_s500_merged`、`tta_dpo_s500_anchor1`、`tta_dpo_s500_anchor_win`、`tta_dpo_s500_pure_win`、`tta_dpo_500_anchor_only`、`tta_dpo_500_beta1_pure`）。

**验证阶梯**（L0→L3，逐层通过才进下一层）：L0 训练动力学（margin 稳定转正）→ L1 机制读数（`tests/tta/compare_base_vs_merged_train_loss.py`，teacher-forcing 下 chosen chunk 误差相对 reference 下降、rejected 上升或少降，否则只是整体漂移）→ L2 小规模 rollout（KSCENE4 20 case，对照 SFT 的 11/20）→ L3 全量（68 case 对照 22/68、132 case 对照 125/132）。

**评测结果（L2 层，KSCENE4 20 case，`scripts/experiments/tta_dpo_eval_ks4/`）**：

| 配置 | 成功率 |
|---|---|
| base | 0/20 |
| s100 / s500 | 0/20 |
| 500_anchor_only / 500_beta1_pure | 0/20 |
| s500_anchor1 / s500win_anchor / s500win_pure | 0/20 |

全部 0/20：**DPO 各 checkpoint 未超过 base，远低于 SFT 的 11/20**。L0 训练动力学也未通过——s500 的 margin 在 −0.21 ~ +0.72 间震荡、loss 在 log2 附近，未稳定转正；`anchor_only` 变体的 loss 由 anchor（SFT）项主导（dpo 项 ≈ log2、margin ≈ 0）。L3（68/132 全量）从未运行。

按 §2 判决矩阵：当前 DPO 通道**无正证据**（< 30% 档，偏好信号不稳），pragmatic 方向是混合部署（SFT + memory 兜底）或先修偏好信号本身。

**视频**：无。ks4 评测未开数据采集；需要轨迹/视频时给 `eval_sft_68.py` 加 `SMOKE_DATA_COLLECTION=1` 重跑。
