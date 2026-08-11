#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3

ROOT=/data1/liu/exp/counterfactual/external/cosmos-policy
DATA=$ROOT/dataset/paired_libero_plus_light_libero10_500_hdf5/libero_10/train
MODEL_DIR=${1:-/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}
OUTPUT_ROOT=$ROOT/outputs
OUT=$OUTPUT_ROOT/cosmos_policy/counterfactual_direct_sft/light_libero10_500_fixed

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
export HF_HUB_OFFLINE=1 IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT
export NUMBA_CACHE_DIR=/tmp/cosmospolicy-numba MPLCONFIGDIR=/tmp/cosmospolicy-matplotlib
cd "$ROOT"

python -m torch.distributed.run --nproc_per_node=4 --master_port=20477 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  -- \
  experiment=cosmos_predict2_2b_480p_libero \
  "dataloader_train.dataset.data_dir=$DATA" \
  "dataloader_train.dataset.t5_text_embeddings_path=$MODEL_DIR/libero_t5_embeddings.pkl" \
  dataloader_train.dataset.rollout_data_dir= dataloader_train.dataset.demonstration_sampling_prob=1.0 \
  dataloader_train.batch_size=16 dataloader_train.num_workers=12 trainer.grad_accum_iter=1 \
  trainer.max_iter=1000 trainer.run_validation=false optimizer.lr=1e-5 \
  "model.config.tokenizer.vae_pth=$MODEL_DIR/tokenizer/tokenizer.pth" \
  "checkpoint.load_path=$MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" \
  checkpoint.load_training_state=false checkpoint.load_ema_to_reg=false checkpoint.save_iter=100 \
  job.group=counterfactual_direct_sft job.name=light_libero10_500_fixed
