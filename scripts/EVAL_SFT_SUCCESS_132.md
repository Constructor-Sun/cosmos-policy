# SFT 原成功样本评测

评测原始 200 个 case 中除去 68 个基线失败后剩余的 132 个 case：

```bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy
/data1/liu/miniconda3/envs/cosmospolicy/bin/python scripts/eval_sft_success_132.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --ckpt training/tta_sft_merged_v2_2030.pt \
  --work-dir scripts/experiments/tta_sft_eval_success_132
```

汇总结果：`scripts/experiments/tta_sft_eval_success_132/baseline_success_132_eval_summary.json`。
