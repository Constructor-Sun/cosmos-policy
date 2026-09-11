#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GPU_ID="${1:-0}"
MODE="${2:-smoke}"   # smoke: 1 sample, 1 iter, 1 GPU | full: full dataset, configurable iters/GPUs

ENV_DIR="${TTA_SFT_ENV_DIR:-/data1/liu/miniconda3/envs/cosmospolicy}"
PYTHON_BIN="$ENV_DIR/bin/python"
TORCHRUN_BIN="$ENV_DIR/bin/torchrun"
SOURCE_DIR="${TTA_REPAIR_SFT_SOURCE_DIR:-$REPO_ROOT/training/tta_sft_success_v2}"
METADATA_DIR="${TTA_REPAIR_SFT_METADATA_DIR:-$REPO_ROOT/training/tta_sft_metadata}"
BASE_CHECKPOINT="${TTA_REPAIR_SFT_BASE_CHECKPOINT:-/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt}"

for required in "$PYTHON_BIN" "$TORCHRUN_BIN" "$SOURCE_DIR" "$METADATA_DIR" "$BASE_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mapfile -t SAMPLES < <(find -L "$SOURCE_DIR" -maxdepth 1 -type f -name '*.hdf5' | sort)
if (( ${#SAMPLES[@]} == 0 )); then
  echo "No HDF5 samples found in $SOURCE_DIR" >&2
  exit 1
fi

if [[ "$MODE" == "full" ]]; then
  DATA_DIR="$SOURCE_DIR"
  NPROC="${TTA_SFT_NPROC:-4}"
  MAX_ITER="${TTA_SFT_MAX_ITER:-500}"
  SAVE_ITER="${TTA_SFT_SAVE_ITER:-100}"
  WARM_UP="${TTA_SFT_WARM_UP:-25}"
  JOB_NAME="${TTA_SFT_JOB_NAME:-tta_repair_sft_full}"
  TRAIN_ARGS=(
    experiment=tta_repair_sft
    "trainer.max_iter=$MAX_ITER"
    "scheduler.cycle_lengths=[$MAX_ITER,100000000000000]"
    "scheduler.warm_up_steps=[$WARM_UP,0]"
    "checkpoint.save_iter=$SAVE_ITER"
    "job.name=$JOB_NAME"
  )
else
  SMOKE_DATA_DIR="$(mktemp -d /tmp/tta_sft_data.XXXXXX)"
  SAMPLE="${SAMPLES[0]}"
  SMOKE_LINK="$SMOKE_DATA_DIR/$(basename "$SAMPLE")"
  ln -s "$(realpath "$SAMPLE")" "$SMOKE_LINK"

  cleanup() {
    [[ ! -L "$SMOKE_LINK" ]] || unlink "$SMOKE_LINK"
    [[ ! -d "$SMOKE_DATA_DIR" ]] || rmdir "$SMOKE_DATA_DIR"
  }
  trap cleanup EXIT

  DATA_DIR="$SMOKE_DATA_DIR"
  NPROC=1
  TRAIN_ARGS=(
    experiment=tta_repair_sft
    trainer.max_iter=1
    checkpoint.save_iter=999999999
    dataloader_train.num_workers=0
    dataloader_train.persistent_workers=false
    job.name=tta_repair_sft_smoke
  )
fi

cd "$REPO_ROOT"

# Recreate the private offline tokenizer cache layout used by the existing TTA code.
"$PYTHON_BIN" -c 'from memory_system.tta.model import setup_offline_hf_cache; setup_offline_hf_cache()'

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/tmp/tta_hf_cache/huggingface-hub
export TTA_REPAIR_SFT_ROLLOUT_DIR="$DATA_DIR"
export TTA_REPAIR_SFT_METADATA_DIR="$METADATA_DIR"
export TTA_REPAIR_SFT_BASE_CHECKPOINT="$BASE_CHECKPOINT"
# Checkpoints go under /data1 (large disk), next to other model weights — never /tmp (small root partition).
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/data1/liu/exp/counterfactual/checkpoints/tta_repair_sft}"

echo "SFT sample source: $DATA_DIR (mode=$MODE, nproc=$NPROC)"

"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" \
  -m cosmos_policy.scripts.train \
  --config cosmos_policy/config/config.py \
  -- \
  "${TRAIN_ARGS[@]}"

echo "SFT run completed."
