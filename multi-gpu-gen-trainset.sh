#!/bin/sh
set -eu

CONDITION=${CONDITION:-camera}
OUTPUT_DIR=${OUTPUT_DIR:-dataset/paired_libero_plus_${CONDITION}_libero10_500}
LIBERO_PLUS_PATH=${LIBERO_PLUS_PATH:-"$PWD/../LIBERO-plus"}
PYTHON=${PYTHON:-python3}
export LIBERO_PLUS_PATH

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  bin/collect_libero_plus_variant_dataset.py \
  --condition "$CONDITION" \
  --variant-catalog configs/libero_plus_variant_catalog.json \
  --variant-splits configs/libero_plus_variant_splits.json \
  --task-limit 10 \
  --examples-per-task 50 \
  --train-per-task 45 \
  --max-attempts-per-task 200 \
  --camera-train-offset-deg 5 \
  --camera-val-fraction 0.1 \
  --camera-split-seed 0 \
  --policy-seeds 1009 2003 3001 4001 \
  --val-policy-seeds 5003 \
  --env-seed 0 \
  --output-dir "$OUTPUT_DIR" \
  --save-rgb-videos \
  --fail-fast
