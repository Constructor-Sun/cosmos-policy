#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/liu/exp/counterfactual/external/cosmos-policy
DATA_ROOT=${DATA_ROOT:-$ROOT/dataset/mixed_clean_camera_background_light_libero10_hdf5}
DATA=$DATA_ROOT/train
MODEL_DIR=${1:-/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs}
JOB_NAME=${JOB_NAME:-mixed_clean_camera_background_light_libero9_adapter_sft}
OUT=$OUTPUT_ROOT/cosmos_policy/counterfactual_adapter_sft/$JOB_NAME
CONFIG=counterfactual_experiments/direct_sft/adapter/config.py

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-20477}
MAX_ITER=${MAX_ITER:-1000}
SAVE_ITER=${SAVE_ITER:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-12}
LR=${LR:-1e-5}
export CUDA_VISIBLE_DEVICES

if [[ ! -d "$DATA" ]]; then
  echo "Missing mixed training dataset: $DATA" >&2
  exit 1
fi

for DOMAIN in clean camera background light; do
  COUNT=$(find "$DATA/$DOMAIN" -maxdepth 1 -type f -name '*.hdf5' | wc -l)
  if [[ "$COUNT" -ne 9 ]]; then
    echo "Expected 9 HDF5 task files in $DATA/$DOMAIN, found $COUNT" >&2
    exit 1
  fi
done

for REQUIRED_FILE in \
  Cosmos-Policy-LIBERO-Predict2-2B.pt \
  tokenizer/tokenizer.pth \
  libero_t5_embeddings.pkl \
  libero_dataset_statistics.json \
  dataset_statistics_post_norm.json; do
  if [[ ! -f "$MODEL_DIR/$REQUIRED_FILE" ]]; then
    echo "Missing required model file: $MODEL_DIR/$REQUIRED_FILE" >&2
    exit 1
  fi
done

ln -sfn "$MODEL_DIR/libero_dataset_statistics.json" "$DATA/dataset_statistics.json"
ln -sfn "$MODEL_DIR/dataset_statistics_post_norm.json" "$DATA/dataset_statistics_post_norm.json"
mkdir -p "$OUT" /tmp/cosmospolicy-numba /tmp/cosmospolicy-matplotlib
export HF_HUB_OFFLINE=1 IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT PYTHONPATH=$ROOT
export NUMBA_CACHE_DIR=/tmp/cosmospolicy-numba MPLCONFIGDIR=/tmp/cosmospolicy-matplotlib
cd "$ROOT"

python -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  -m cosmos_policy.scripts.train \
  --config="$CONFIG" \
  -- \
  experiment=cosmos_predict2_2b_480p_libero_adapter \
  "dataloader_train.dataset.data_dir=$DATA" \
  "dataloader_train.dataset.t5_text_embeddings_path=$MODEL_DIR/libero_t5_embeddings.pkl" \
  dataloader_train.dataset.rollout_data_dir= \
  dataloader_train.dataset.demonstration_sampling_prob=1.0 \
  "dataloader_train.batch_size=$BATCH_SIZE" \
  "dataloader_train.num_workers=$NUM_WORKERS" \
  trainer.grad_accum_iter=1 \
  "trainer.max_iter=$MAX_ITER" \
  trainer.run_validation=false \
  "optimizer.lr=$LR" \
  "model.config.tokenizer.vae_pth=$MODEL_DIR/tokenizer/tokenizer.pth" \
  "checkpoint.load_path=$MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" \
  checkpoint.load_training_state=false \
  checkpoint.load_ema_to_reg=false \
  "checkpoint.save_iter=$SAVE_ITER" \
  job.group=counterfactual_adapter_sft \
  "job.name=$JOB_NAME"
