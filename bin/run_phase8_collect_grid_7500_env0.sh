#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

RUN=${RUN:-all_suites_demo20_camera10_4pseed_env0_train8000}
GPU_IDS=${GPU_IDS:-"0 1 4 5 6 7"}
POLICY_DIR=${POLICY_DIR:-"$REPO_ROOT/../../checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"}
DEMO_ROOT=${DEMO_ROOT:-LIBERO-Cosmos-Policy/success_only}
CAMERA_CONFIG=${CAMERA_CONFIG:-../LIBERO-plus/libero/libero/benchmark/task_classification.json}
POLICY_SEEDS=${POLICY_SEEDS:-"1009 2003 3001 4001"}
ENV_SEED=${ENV_SEED:-0}

cd "$REPO_ROOT"

NUM_SHARDS=$(set -- $GPU_IDS; echo "$#")
if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "GPU_IDS is empty" >&2
    exit 1
fi
if [ ! -d "$DEMO_ROOT" ] || [ ! -f "$CAMERA_CONFIG" ]; then
    echo "Missing demo root or camera classification file" >&2
    exit 1
fi
if [ ! -f "$POLICY_DIR/libero_dataset_statistics.json" ]; then
    echo "Missing dataset statistics: $POLICY_DIR/libero_dataset_statistics.json" >&2
    exit 1
fi
if [ ! -f "$POLICY_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" ]; then
    echo "Missing policy checkpoint: $POLICY_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" >&2
    exit 1
fi

OUT_ROOT="experiments/phase8_first_chunk_pairs/${RUN}"
mkdir -p "${OUT_ROOT}/logs"

PIDS=""

cleanup() {
    if [ -n "$PIDS" ]; then
        echo "Stopping child processes: $PIDS" >&2
        kill $PIDS 2>/dev/null || true
    fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

SHARD_INDEX=0
for GPU_ID in $GPU_IDS; do
    LOG_PATH="${OUT_ROOT}/logs/gpu${GPU_ID}.log"
    OUT_DIR="${OUT_ROOT}/gpu${GPU_ID}"
    EXPECTED_PAIRS=$((8000 / NUM_SHARDS))
    if [ "$SHARD_INDEX" -lt $((8000 % NUM_SHARDS)) ]; then
        EXPECTED_PAIRS=$((EXPECTED_PAIRS + 1))
    fi
    if [ $((SHARD_INDEX % 2)) -eq 0 ]; then
        SHARD_SEEDS="1009,3001"
    else
        SHARD_SEEDS="2003,4001"
    fi
    echo "Launching gpu=${GPU_ID} shard=${SHARD_INDEX}/${NUM_SHARDS} pairs=${EXPECTED_PAIRS} seeds=${SHARD_SEEDS} log=${LOG_PATH}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    NUMBA_CACHE_DIR="/tmp/cosmospolicy-numba-${RUN}-gpu${GPU_ID}" \
    MPLCONFIGDIR="/tmp/cosmospolicy-matplotlib-${RUN}-gpu${GPU_ID}" \
    python bin/phase8_collect_first_chunk_pairs.py \
        --sample-mode task_init_policy_grid \
        --train8000 \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --demo-root "$DEMO_ROOT" \
        --camera-classification "$CAMERA_CONFIG" \
        --conditions camera_viewpoints \
        --policy-dir "$POLICY_DIR" \
        --t5-extra-embeddings "" \
        --policy-seeds $POLICY_SEEDS \
        --env-seeds "$ENV_SEED" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$SHARD_INDEX" \
        --skip-existing \
        --output-dir "$OUT_DIR" \
        > "$LOG_PATH" 2>&1 &
    PIDS="$PIDS $!"
    SHARD_INDEX=$((SHARD_INDEX + 1))
done

echo "Started child processes:$PIDS"
echo "Logs: ${OUT_ROOT}/logs"

STATUS=0
for PID in $PIDS; do
    if ! wait "$PID"; then
        STATUS=1
    fi
done

trap - INT TERM

if [ "$STATUS" -eq 0 ]; then
    echo "All shards completed successfully."
else
    echo "At least one shard failed. Check logs in ${OUT_ROOT}/logs." >&2
fi

exit "$STATUS"
