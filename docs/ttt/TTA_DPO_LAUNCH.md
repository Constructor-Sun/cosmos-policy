# TTA DPO 启动文档（临时，v2 修订版）

目的：在 SFT（`TTA_SFT_RESULTS.md`）之后启动 DPO 通道，并按预先注册的验证阶梯判定其有效性。本文档自包含。

必读前置（同目录）：`TTA_LoRAFix.md`、`TTA_SFT_RESULTS.md`（DPO 的对照基线在此）。

**v2 修订说明**：v1 文档的 rejected 数据位置、配对账目有错误（写了工具默认路径而非实际数据位置、61 对实为 56 对），本版全部以实测核实为准。教训：**工具的默认路径 ≠ 数据现实，每个资产必须数一遍**。

---

## 0. 环境（全部踩过坑，逐条遵守）

- 仓库：`/data1/liu/exp/counterfactual/external/cosmos-policy`（下文 `$REPO`）
- Python：**必须用绝对路径** `/data1/liu/miniconda3/envs/cosmospolicy/bin/python`（下文 `$PY`）
- **坑 1**：调用 `scripts/run_libero_smoke_test.sh`（eval 链）前必须 `export PATH=/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH`，否则其内部 `exec python` 报 `python: not found`，且评测脚本静默失败（`check=False`）。
- **坑 2**：任何训练启动前 `export IMAGINAIRE_OUTPUT_ROOT=/data1/liu/exp/counterfactual/checkpoints`，否则 checkpoint 写到 `/tmp` 打爆根分区（历史事故）。
- **坑 3**：本机**无外网**。模型构建时 HuggingFace tokenizer 解析会重试 ~1 分钟后回退本地缓存（正常现象）。加速可设 `HF_HUB_OFFLINE=1` 与 `HF_HUB_CACHE=/tmp/tta_hf_cache/huggingface-hub`。
- **坑 4**：所有 `--work-dir` / `--out` 用**绝对路径**，否则产物落到 `scripts/scripts/` 嵌套目录。
- **坑 5**：eval 脚本对子进程失败静默，跑完必须统计 summary 文件数量核对。
- GPU：`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits` 选空闲卡；DPO 训练单卡（自动挑显存最空的卡，预算 20GB）；评测每卡 ~10GB。GPU 使用需用户授权。

## 1. 资产清单（全部实测核实，2026-09-13）

| 资产 | 路径 | 数量/说明 |
|---|---|---|
| chosen（成功轨迹） | `$REPO/training/tta_sft_success_v2/` | 61 个 hdf5（success=True），tag 格式 `KSCENE4-init004` |
| **rejected 正源** | `$REPO/memory_system/pointcloud_action/results/tta_failure_screening_68/<task>/<initNNN>/episode.h5` | **68 个**，只有 actions+proprio（无图像，Step 1 补） |
| ~~勿用~~ | `$REPO/experiments/tta/tta_phase_check_repro_68/` | 也是 68 个 episode.h5，但那是 09-09 第一次修复批次的诊断源，**与 chosen 前缀不同源**（实测同 case 前缀动作差 0.017）；`tta_phase_check_v3` 只剩 **1** 个 case |
| case 元数据 | `$REPO/memory_system/pointcloud_action/results/tta_screening_meta.json` | 68 tags |
| 修复汇总 | `$REPO/memory_system/pointcloud_action/results/tta_repair_68_summary.json` | 68 case，task_success=True **62** 个 |
| base checkpoint | `/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt` | DPO 的 policy 与 reference |
| 统计 + T5 | `$REPO/training/tta_sft_metadata/{dataset_statistics.json,t5_embeddings.pkl}` | 与 SFT 同源 |
| SFT 对照基线 | 68 case **22/68=32%**；132 case **125/132=94.7%**；KSCENE4 子集 **11/20** | 判定 DPO 的对照 |

**rejected 正源判定依据（关键，勿再改动）**：chosen 数据内嵌的前缀动作与 screening_68 的 episode.h5 逐位一致（抽测 KSCENE4-init004 前 50 步最大差 = 0.000000；repro_68 同 case 差 0.017）。即 chosen 的前缀回放正是从 screening_68 读取的（`tta_repair_batch.py` 的默认 `--result-dir`）。这带来一个理想性质：**chosen 与 rejected 在 t\* 之前动作逐位相同，对比对只在失败点之后分叉**。

**配对账目（实测）**：chosen 61 ∩ rejected 68 = **61 对**（上限）。但 `build_manifest.py` 以"原修复汇总成功（62 个）"为配对依据：会产出 62 个 pair，其中 6 个无 chosen 文件（KSCENE4-init005/009/023、KSCENE6-init020、LRSCENE2-init016、LRSCENE5-init011——v2 重采时漂移为失败），validate 会**报错**；同时 5 个有效 chosen 被丢弃（KSCENE4-init013、KSCENE6-init013、LRSCENE5-init005/017、LRSCENE6-init021——原修复失败但 v2 重采成功，h5 的 success 属性是真相）。按 Step 2 的后处理实际得到 **56 对**；61 对需小改 builder（见 Step 2 备注）。

## 2. 执行步骤

### Step 0 环境自检
```bash
cd $REPO
$PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # 期望 True N
```

### Step 1 rejected 重放补图（确定性重放，无需 policy 推理；单卡 2–3 小时）
```bash
$PY memory_system/tta/tools/replay_rejected.py \
  --diagnosis-root $REPO/memory_system/pointcloud_action/results/tta_failure_screening_68 \
  --out-dir $REPO/experiments/tta/dpo_rejected_images
```
- **必须显式传 `--diagnosis-root`**：工具默认指向 `tta_phase_check_v3`（只剩 1 个 case），不传则只产出 1 个。
- 产物为**扁平文件**：`<out-dir>/tta_rejected--<tag>--success=False.hdf5`（与 build_manifest 的 glob 匹配）。
- **数量关口**：完成后 `ls $REPO/experiments/tta/dpo_rejected_images | wc -l` 应为 **68**（重放意外成功的 case 会被工具标记丢弃，每少 1 个记录 1 个）。

### Step 2 构建 manifest（预期 56 对，不是 61）
```bash
$PY memory_system/tta/tools/build_manifest.py \
  --chosen-dir $REPO/training/tta_sft_success_v2 \
  --rejected-dir $REPO/experiments/tta/dpo_rejected_images \
  --out $REPO/memory_system/tta/results/manifests/dpo_pairs_v1.json
# 丢弃 6 个 incomplete pair（chosen_path=null，validate 对其报错）：
$PY -c "
import json; p='$REPO/memory_system/tta/results/manifests/dpo_pairs_v1.json'
d=json.load(open(p)); d['pairs']=[x for x in d['pairs'] if x.get('chosen_path')]
json.dump(d,open(p,'w'))"
# 校验（--validate-only 接 manifest 路径，只能 build 之后用）：
$PY memory_system/tta/tools/build_manifest.py --validate-only $REPO/memory_system/tta/results/manifests/dpo_pairs_v1.json
```
- 账目：62 pair（summary 成功）→ 6 个无 chosen 丢弃 → **56 对**，校验应零错误。
- 已知限制：5 个有效 chosen（漂移 case，h5 success=True 为真相）被 builder 的 summary 判据丢弃。要拿回它们需小改 `build_pairs()`（按 chosen∩rejected 直接配对，56→61 对）——**需用户批准，非必需**，首跑用 56 对即可。

### Step 3 DPO 训练（单卡，自动选卡）
```bash
cd $REPO && IMAGINAIRE_OUTPUT_ROOT=/data1/liu/exp/counterfactual/checkpoints \
$PY memory_system/tta/dpo_train.py \
  --manifest $REPO/memory_system/tta/results/manifests/dpo_pairs_v1.json \
  --checkpoint /data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt \
  --dataset-stats-path $REPO/training/tta_sft_metadata/dataset_statistics.json \
  --t5-text-embeddings-path $REPO/training/tta_sft_metadata/t5_embeddings.pkl \
  --output-dir $REPO/memory_system/tta/results/runs \
  --max-steps 100
```
- 默认超参：beta 0.1、lr 1e-5、LoRA 8/16、batch=1 pair、chunk 16、pair 内共享 sigma/epsilon（已验证实现）。
- **步数消融**：依次 `--max-steps 100 / 500 / 1000`（各 1–2 小时），adapter 落在 `memory_system/tta/results/runs/run_<时间戳>/adapter_step<NNN>.pt`（格式 `{"meta","lora_state"}`，拒绝覆盖）。

### Step 4 adapter → merged ckpt 转换（新代码，**两处均需用户批准**）
1. `tests/tta/merge_dpo_adapter.py`（~30 行新文件）：加载 base → `add_lora(meta)` → `load_state_dict(lora_state, strict=False)` → 逐层 `merge()` → 剥 lora key 存 .pt。参照 `scripts/merge_lora_ckpt.py` + `memory_system/tta/model.py:attach_adapter` 拼装。输出命名：`training/tta_dpo_s{100,500,1000}_merged.pt`。
2. L1 探针改造：`tests/tta/compare_base_vs_merged_train_loss.py` 目前面向 SFT 的 base-vs-merged 且无分侧；需改为"reference(base) vs base+adapter"并**分 chosen/rejected 两侧**报告误差。

### Step 5 验证阶梯（逐层通过才进下一层）

- **L0 训练动力学**（读训练日志）：margin 从 0/负**稳定转正**；loss/margin 无塌缩。（注：`dpo_train.py` 只打聚合 loss/margin/grad_norm，分侧误差在 L1 查。）
- **L1 机制读数**（teacher-forcing，用 Step 4.2 改造的探针）：**chosen chunk 误差相对 reference 下降，rejected chunk 上升或少降**——rejected 侧是 DPO 独有信号；只有 chosen 降而 rejected 不动 = 整体漂移而非对比学习，停下。
- **L2 小规模 rollout**（KSCENE4 20 case，对照 SFT 的 11/20）：
```bash
export PATH=/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH
TAGS=$($PY -c "import json;m=json.load(open('memory_system/pointcloud_action/results/tta_screening_meta.json'));print(','.join(sorted(k for k in m if k.startswith('KSCENE4'))))")
$PY scripts/eval_sft_68.py --gpus <空闲卡> \
  --ckpt training/tta_dpo_s500_merged.pt \
  --work-dir /data1/liu/exp/counterfactual/external/cosmos-policy/scripts/experiments/tta_dpo_eval_ks4 \
  --tags "$TAGS"
```
- **L3 全阶梯**（对照 22/68 与 125/132）：
```bash
# 68 case（默认全部 tags）
$PY scripts/eval_sft_68.py --gpus <空闲卡> \
  --ckpt training/tta_dpo_s500_merged.pt \
  --work-dir /data1/liu/exp/counterfactual/external/cosmos-policy/scripts/experiments/tta_dpo_eval_68
# 132 case（脚本为 scripts/eval_sft_success_132.py）
$PY scripts/eval_sft_success_132.py --gpus <空闲卡> \
  --ckpt training/tta_dpo_s500_merged.pt \
  --work-dir /data1/liu/exp/counterfactual/external/cosmos-policy/scripts/experiments/tta_dpo_eval_132
```

### Step 6 判决矩阵（跑前注册，不得事后改；本节取代 TTA_SFT_RESULTS.md 第 4 节中 ≥45% 的表述，两文档统一为此）

| 结果 | 结论 | 下一步 |
|---|---|---|
| 68 成功率 ≥ **45%（≥31/68）**（超过 SFT 22/68 达 13 个 case ≈ 2σ 二项噪声） | 偏好通道胜出 | 加大 DPO；SFT 降级 |
| 30–40% | 与 SFT 平手 | 主攻按阶段修复数据（`TTA_SFT_RESULTS.md` 第 3 节第 4 条"阶段覆盖缺口"） |
| <30% | 偏好信号不稳 | 混合部署（SFT + memory 按需兜底） |

独立于主判据的两个专项指标：
- **132 回退 ≤7 且越少越好**（DPO reference 锚定的结构性承诺）；
- **辨别型任务专项**：LRSCENE5+LRSCENE2 合计 ≥8/32（SFT 基线 2/32）——对比信号的理论靶点；聚合涨而这两个任务不动 = 机制理解需修正。

## 3. 陷阱速查（全部真实踩过）

| 症状 | 原因 | 解法 |
|---|---|---|
| eval 后 0 个 summary | smoke test 内 `exec python` 找不到 | `export PATH=/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH` |
| 磁盘写满 / checkpoint 损坏 | 未设输出根，写 /tmp | `IMAGINAIRE_OUTPUT_ROOT=/data1/liu/exp/counterfactual/checkpoints` |
| 产物出现在 `scripts/scripts/` | work-dir 相对路径 | 全部绝对路径 |
| HuggingFace 重试刷屏 | 无外网 | 等回退；或 `HF_HUB_OFFLINE=1` + `HF_HUB_CACHE=/tmp/tta_hf_cache/huggingface-hub` |
| rejected 只有 1 个 case | 工具默认 diagnosis-root 指 v3（仅剩 1 case） | **显式传 screening_68 路径**（本档 Step 1） |
| 训练 loss 好看但模型没学 | 历史事故（LoRA 死初始化） | L0 检查是强制项 |
| manifest 校验报 incomplete pair | builder 按 summary 成功配对，6 个无 chosen | Step 2 的后处理过滤（已写入命令） |
