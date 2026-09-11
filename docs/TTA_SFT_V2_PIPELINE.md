# TTA SFT v2 流程（prefix 修复版）

原理：修复 rollout 的录制从**扰动初始位置**开始（含回放前缀，剪掉 settle/wait 静止段），
SFT 学到"扰动起点 → 完整成功轨迹"，可独立部署（裸评测，无 prefix/memory 机器）。

## 1. 采集（68 case，8 卡约 20 分钟）

```bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy
/data1/liu/miniconda3/envs/cosmospolicy/bin/python scripts/tta_repair_batch.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --work-dir /data1/liu/exp/counterfactual/external/cosmos-policy/scripts/experiments/tta_repair_work_v2
```

校验（每条都必须过）：`frame_indices[0]==0`；第0帧 proprio ≈ 筛选录制帧0（距离<0.02）；
前缀段连续 0..t*-1；接缝后连续 10,11,...。

## 2. 汇总数据集

```bash
mkdir -p training/tta_sft_success_v2
cp scripts/experiments/tta_repair_work_v2/results/*/logs/rollout_data/*success=True*.hdf5 \
   training/tta_sft_success_v2/
ls training/tta_sft_success_v2 | wc -l   # 61 (v2, 16228帧)
```

## 3. SFT 训练（2030 步 = 100% 覆盖，约 50 分钟）

```bash
TTA_SFT_NPROC=8 TTA_SFT_JOB_NAME=tta_repair_sft_v2 TTA_SFT_MAX_ITER=2030 TTA_SFT_SAVE_ITER=500 \
scripts/smoke_tta_repair_sft.sh 0,1,2,3,4,5,6,7 full 2>&1 | tee training/tta_repair_sft_v2_$(date +%m%d_%H%M).log
```

启动 2 分钟内核验：`Resuming ckpt` 指向 base `.pt` 且 keys=`['model']`；
`Remapped 687 checkpoint keys`；首步 loss ~0.2 量级。
窗口均值应单调下降；单点波动是 sigma 随机性，正常。

## 4. 合并 LoRA（约 5 分钟）

```bash
CUDA_VISIBLE_DEVICES=2 /data1/liu/miniconda3/envs/cosmospolicy/bin/python scripts/merge_lora_ckpt.py \
  --ckpt /data1/liu/exp/counterfactual/checkpoints/tta_repair_sft/cosmos_policy/cosmos_v2_finetune/tta_repair_sft_v2/checkpoints/iter_000002030/model \
  --out training/tta_sft_merged_v2_2030.pt
```

## 5. 裸评测 68 case（8 卡约 20 分钟）

```bash
/data1/liu/miniconda3/envs/cosmospolicy/bin/python scripts/eval_sft_68.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --ckpt training/tta_sft_merged_v2_2030.pt \
  --work-dir scripts/experiments/tta_sft_eval_v2
ls scripts/experiments/tta_sft_eval_v2/results/*/*_summary.json | wc -l   # 必须=68
```

## 历史教训（勿重蹈）

- **checkpoint 输出写满根分区**：必须 `IMAGINAIRE_OUTPUT_ROOT` 指向 /data1（脚本已内置）。
- **DCP 加载键名不匹配**：训练启动加载 base `.pt` 时需键名重映射（dcp.py 已修复，
  日志必须出现 `Remapped 687`，否则主干是随机初始化，训练必废）。
- **同 job name 会续训旧 checkpoint**：新实验必须换 `TTA_SFT_JOB_NAME`。
- **评测不写轨迹**：需要轨迹分析时加 `SMOKE_DATA_COLLECTION=1`。
- **剪裁**：录制起点=扰动初始位置；只允许剪 settle/wait 静止帧，运动段一帧不能丢。
