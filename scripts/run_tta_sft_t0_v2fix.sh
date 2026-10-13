#!/usr/bin/env bash
# v2fix 配方 × t0 数据 × N 卡：与 9/13 的 tta_repair_sft_v2_fix 逐项相同，
# 仅数据目录与 GPU 数不同。
#
# 入口是归档的 v2fix 启动器
#   REPO/tests/tta/archived_lorafix/train_tta_sft_fixed.py
# 它 = stock train.py + （1）物化后 LoRA reinit（A=kaiming, B=0）
#      （2）optimizer 换标准 AdamW（optim_type=adamw，删 master_weights、capturable=False）
#      （3）dataloader num_workers=0、persistent_workers=False
#      （4）save_iter=400、warm_up_steps=[25,0]
# 输出根沿用 v2fix 的默认：/data1/liu/exp/counterfactual/checkpoints
#   （如需改回 tta_repair_sft 树：启动前 export IMAGINAIRE_OUTPUT_ROOT=...）
#
# 用法: bash scripts/run_tta_sft_t0_v2fix.sh [GPU列表]   # 例: 1 或 0,1,2,3,4,5
#   nproc 自动 = GPU 列表长度（可用 TTA_SFT_NPROC 覆盖）
#   max_iter 默认 2742 = t0 数据在 6 卡下的 1 epoch（同 tta_sft_t0_v2）。
#   若改卡数，1 epoch 需按比例折算（单卡 ≈ 16452），用 TTA_SFT_MAX_ITER 覆盖。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GPU_ID="${1:-0,1,2,3,4,5}"
IFS=',' read -ra _GPU_LIST <<< "$GPU_ID"
NPROC="${TTA_SFT_NPROC:-${#_GPU_LIST[@]}}"
DATA_DIR="${TTA_SFT_SOURCE_DIR:-$REPO_ROOT/training/tta_sft_success_t0}"
MAX_ITER="${TTA_SFT_MAX_ITER:-2742}"
JOB_NAME="${TTA_SFT_JOB_NAME:-tta_sft_t0_v2fix}"
ENV_DIR="${TTA_SFT_ENV_DIR:-/data1/liu/miniconda3/envs/cosmospolicy}"

PYTHON_BIN="$ENV_DIR/bin/python"
TORCHRUN_BIN="$ENV_DIR/bin/torchrun"
METADATA_DIR="$REPO_ROOT/training/tta_sft_metadata"
FIXED_LAUNCHER="$REPO_ROOT/tests/tta/archived_lorafix/train_tta_sft_fixed.py"

for required in "$PYTHON_BIN" "$TORCHRUN_BIN" "$DATA_DIR" "$METADATA_DIR" "$FIXED_LAUNCHER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

cd "$REPO_ROOT"

# 离线 tokenizer 缓存（与 smoke_tta_repair_sft.sh 相同的前置）
"$PYTHON_BIN" -c 'from memory_system.tta.model import setup_offline_hf_cache; setup_offline_hf_cache()'

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/tmp/tta_hf_cache/huggingface-hub
export TTA_REPAIR_SFT_METADATA_DIR="$METADATA_DIR"

echo "v2fix recipe on t0: gpus=$GPU_ID nproc=$NPROC max_iter=$MAX_ITER job=$JOB_NAME data=$DATA_DIR"

"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" \
  "$FIXED_LAUNCHER" \
  --job-name "$JOB_NAME" --max-iter "$MAX_ITER" --data-dir "$DATA_DIR"

echo "v2fix-recipe run completed."
