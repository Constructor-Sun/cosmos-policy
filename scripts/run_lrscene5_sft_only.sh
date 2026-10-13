#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${TTA_SFT_ENV_DIR:-/data1/liu/miniconda3/envs/cosmospolicy}"
GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
JOB_NAME="${TTA_SFT_JOB_NAME:-lrscene5_sft_only_$(date +%Y%m%d_%H%M%S)}"
MAX_ITER="${TTA_SFT_MAX_ITER:-2742}"
PROBE_EVERY="${TTA_MASK_PROBE_EVERY:-100}"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/tmp/tta_hf_cache/huggingface-hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node="${#GPU_LIST[@]}" \
  scripts/train_lrscene5_sft_only.py --job-name "$JOB_NAME" --max-iter "$MAX_ITER" \
  --probe-every "$PROBE_EVERY"
