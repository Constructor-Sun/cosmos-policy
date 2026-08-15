#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
GPU_ID=${GPU_ID:-7}
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-4}
COSMOS_TARGET_DEMOS=${COSMOS_TARGET_DEMOS:-3}
COSMOS_TARGET_DEMOS_DIR=${COSMOS_TARGET_DEMOS_DIR:-$REPO_ROOT/vector_db_demos_camera_aligned_v1}
COSMOS_EARLY_MANIFOLD_THRESHOLD=${COSMOS_EARLY_MANIFOLD_THRESHOLD:-0.1}
COSMOS_EARLY_MANIFOLD_MAX_OFFSET=${COSMOS_EARLY_MANIFOLD_MAX_OFFSET:-15}
COSMOS_EARLY_MANIFOLD_CORRECTION_CHUNKS=${COSMOS_EARLY_MANIFOLD_CORRECTION_CHUNKS:-1}
COSMOS_ROLLOUT_SUBDIR=${COSMOS_ROLLOUT_SUBDIR:-08-13-libero10-all-plus-init-chunk1}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_all_plus_init_chunk1}

if [ ! -d "$COSMOS_TARGET_DEMOS_DIR" ]; then
    echo "ERROR: memory directory not found: $COSMOS_TARGET_DEMOS_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"

# task_idx:robot_init_variant
for pair in \
    0:273 \
    1:274 \
    2:270 \
    3:269 \
    4:268 \
    5:271 \
    6:282 \
    7:265 \
    8:267 \
    9:276
do
    task_idx=${pair%%:*}
    variant=${pair#*:}

    echo "=== task_idx=$task_idx robot_init=$variant ==="

    NUM_CASES="$NUM_CASES" \
    SEED="$SEED" \
    GPU_ID="$GPU_ID" \
    COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
    COSMOS_ONLINE_DETECTION=0 \
    COSMOS_TARGET_DEMOS="$COSMOS_TARGET_DEMOS" \
    COSMOS_TARGET_DEMOS_DIR="$COSMOS_TARGET_DEMOS_DIR" \
    COSMOS_EARLY_MANIFOLD_SCORE=0 \
    COSMOS_EARLY_MANIFOLD_INJECT=0 \
    COSMOS_EARLY_MANIFOLD_THRESHOLD="$COSMOS_EARLY_MANIFOLD_THRESHOLD" \
    COSMOS_EARLY_MANIFOLD_MAX_OFFSET="$COSMOS_EARLY_MANIFOLD_MAX_OFFSET" \
    COSMOS_EARLY_MANIFOLD_CORRECTION_CHUNKS="$COSMOS_EARLY_MANIFOLD_CORRECTION_CHUNKS" \
    COSMOS_DATA_COLLECTION= \
    COSMOS_VECTOR_DB= \
    COSMOS_ROLLOUT_SUBDIR="$COSMOS_ROLLOUT_SUBDIR" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    sh ./run_libero10_experiment.sh \
        --task-idx "$task_idx" \
        --perturbation robot_initial_states \
        --variant "$variant" \
        --num-cases "$NUM_CASES" \
        --seed "$SEED" \
        --gpu "$GPU_ID" \
        --init-state-offset "$COSMOS_INIT_STATE_OFFSET"
done

echo "All LIBERO-10 tasks complete: $OUTPUT_ROOT"
