# TTA SFT

> 本文合并并取代原 `TTA_SFT_RESULTS.md`（结果分析）、`TTA_SFT_V2_PIPELINE.md`（v2 流程）、`TTA_LoRAFix.md`（LoRA 死初始化 bug）三份文档。

## 1. 已有的 SFT 文章

原三份文档未收集 SFT 专栏，以下是本训练实际依赖的方法出处：

- [LoRA](https://arxiv.org/abs/2106.09685)（Hu et al., 2021）：低秩适配，本项目所有微调的载体（PEFT inject_adapter_in_model）。
- [EDM](https://arxiv.org/abs/2206.00364)（Karras et al., 2022）：Cosmos Policy 的去噪训练目标，SFT loss 即动作 latent 上的 EDM 去噪误差。
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)（Chi et al., 2023）：动作扩散 + 行为克隆的代表工作；本项目 SFT 本质是其 LoRA 化变体。

## 2. 现在使用 SFT 的目的

让 policy 学到「扰动初始位置 → 完整成功轨迹」，从而**独立裸部署**——不依赖 prefix 回放、memory 机器和对齐控制器。v2 关键设定：修复 rollout 的录制从扰动初始位置开始（含回放前缀，剪掉 settle/wait 静止段），这样 SFT 后的模型从扰动起点就能直接跑通。

已知局限（结论保留，勿重蹈）：

1. 数据量与成功率**零相关**：LRSCENE5 拿第二多数据（19 条）只有 5%，KSCENE6 用 3 条就 100%。
2. 分界线是**容差**：成功 ≥50% 的全是"放进区域/推拉/上台面"类；失败的全是"空间关系精放/辨别"类——开环 BC 误差累积在容差小的任务上必然越界。
3. **闭环→开环克隆差距**：裸 SFT 32% vs memory 修复 90%，58 个点是用开环 BC 模仿闭环控制器动作的结构性损失。
4. **阶段覆盖缺口**：每条轨迹只含一个相位的一次修复（绝大多数是 Pick），Place/关门等相位的修复从未进监督信号。
5. **部署代价**：132 个 base 会做的任务回退 5.3%（分布漂移）。

## 3. SFT 算法的构成

- **训练形式**：LoRA SFT（rank 8 / alpha 16），标准 AdamW，2030 iters ≈ 100% 数据覆盖，6–8 卡；loss 是 base 管线的动作 EDM 去噪损失，无偏好项。
- **数据**：61 条修复成功轨迹（7 个 LIBERO-10 任务、robotinit 变体；每条 = 扰动初始态前缀回放 + 闭环控制器修复 + base 策略续行至成功），在 `training/tta_sft_success_v2/`；归一化统计与 T5 嵌入在 `training/tta_sft_metadata/`。
- **LoRA 死初始化修复（前提，勿删）**：`build_net()` 在 meta device 上建网，PEFT 的标准初始化作用在 meta 张量上是空操作，`to_empty` 后 LoRA 变成未初始化内存（恰好全零），`init_weights()` 不认识动态注入的 lora_A/lora_B → A=B=0 进入训练，梯度双双恒为零死锁（修复前表面 loss 2.2→0.014 全正常但模型没学）。根因修复是在 `build_net()` 的 `net.init_weights()` 后按 PEFT 标准重初始化（A=kaiming_uniform(a=√5)、B=zeros）；已验证：重训后 280/280 层非零、ks4 2/17 → 8/17。
- **训练命令**（启动 2 分钟内核验：`Resuming ckpt` 指向 base `.pt` 且 keys=`['model']`；`Remapped 687 checkpoint keys`；首步 loss ~0.2 量级，窗口均值单调下降）：

```bash
TTA_SFT_NPROC=8 TTA_SFT_JOB_NAME=<新名字> TTA_SFT_MAX_ITER=2030 TTA_SFT_SAVE_ITER=500 \
scripts/smoke_tta_repair_sft.sh 0,1,2,3,4,5,6,7 full
```

- **合并**：`scripts/merge_lora_ckpt.py` 把 LoRA checkpoint 合并成完整 `.pt` 供裸评测。
- **已知坑**：`IMAGINAIRE_OUTPUT_ROOT` 必须指向 /data1（否则写满根分区）；同 `TTA_SFT_JOB_NAME` 会续训旧 checkpoint，新实验必须换名；评测默认不写轨迹，需要轨迹/视频加 `SMOKE_DATA_COLLECTION=1`；剪裁只允许剪 settle/wait 静止帧，运动段一帧不能丢。
- **流程守则**：任何新训练配置，全量跑之前先 `python tests/tta/grad_flow_probe.py`，期望 LoRA 梯度非零张量 560/560。

## 4. SFT 流程涉及的代码

| 环节 | 代码 / 路径 |
|---|---|
| 数据采集（chosen 轨迹） | `scripts/tta_repair_batch.py`（repair 模式 + `SMOKE_DATA_COLLECTION=1`，命令见 `../repair/libero-plus-repair.md` §2） |
| 训练数据 | `training/tta_sft_success_v2/`（hdf5）+ `training/tta_sft_metadata/{dataset_statistics.json, t5_embeddings.pkl}` |
| 训练启动器（含 LoRA 重初始化） | `tests/tta/train_tta_sft_fixed.py` → `scripts/smoke_tta_repair_sft.sh` → cosmos_policy 原 trainer |
| LoRA 初始化根因位置 | `cosmos_policy/_src/predict2/models/text2world_model.py` 的 `build_net()` |
| LoRA 合并 | `scripts/merge_lora_ckpt.py` |
| 评测 | `scripts/eval_sft_68.py`（68 失败 case）、`scripts/eval_sft_success_132.py`（132 通过 case） |
| 探针 | `tests/tta/grad_flow_probe.py`（梯度连通性）、`tests/tta/init_state_probe.py`、`tests/tta/compare_base_vs_merged_train_loss.py` |
| merged checkpoint | `training/tta_sft_merged_v2_2030.pt`（主结果）、另有 `tta_sft_ks4_fix_merged` / `tta_sft_v2fix_merged` / `tta_sft_slices_merged_686` / `tta_sft_v2slice_merged_2664` 等变体 |

## 5. SFT 的实验结果、可能的视频

评测目录聚合实测（2026-09-19 核对 summary）：

| 评测目录 | 配置 | 结果 |
|---|---|---|
| `scripts/experiments/tta_sft_v2fix_eval` | v2 修复版，68 个 base 失败 case（修复能力） | **22/68 = 32%** |
| `scripts/experiments/tta_sft_v2fix_eval_132` | v2 修复版，132 个 base 通过 case（部署回归：检查 SFT 有没有把原本会的做废） | **125/132 = 94.7%**（回退 7 个） |

对照：base 0/68（按构造）、memory 修复机制上限 61/68 = 90%；200 case 净效果 +22 恢复 −7 回退 = **+15**。

按任务分解（成功率 vs 训练数据量）：

| 任务 | 训练数据 | 68 case 成功率 | 任务性质 |
|---|---|---|---|
| KSCENE6 微波炉关门 | 3 | 100% (4/4) | 容差大 |
| LRSCENE6 盘+布丁 | 4 | 75% (3/4) | 容差中 |
| KSCENE8 双锅上灶 | 1 | 50% (1/2) | 容差大 |
| KSCENE4 抽屉 | 17 | 55% (11/20) | 容差中 |
| KSCENE3 开灶 | 6 | 17% (1/6) | 容差偏小 |
| LRSCENE2 进篮 | 11 | 8% (1/12) | 空间辨别 |
| LRSCENE5 左右盘 | 19 | 5% (1/20) | 精确辨位 |

结论与教训见 §2 的五条局限；SFT 的后继通道（DPO）见 `../dpo/dpo.md`。

**视频**：无。SFT 评测未开数据采集；需要轨迹/视频时给 `eval_sft_68.py` 加 `SMOKE_DATA_COLLECTION=1` 重跑。修复批次（数据来源侧）的视频路径在 `../repair/libero-plus-repair.md` §3。
