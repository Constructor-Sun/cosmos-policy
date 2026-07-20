#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECKPOINT_ROOT=$(CDPATH= cd -- "$REPO_ROOT/../../checkpoints" && pwd)
BASE_MODEL_DIR="$CHECKPOINT_ROOT/Cosmos-Predict2-2B-Video2World"
POLICY_MODEL_DIR="$CHECKPOINT_ROOT/Cosmos-Policy-LIBERO-Predict2-2B"
HF_REPO_DIR="$CHECKPOINT_ROOT/huggingface-hub/models--nvidia--Cosmos-Predict2-2B-Video2World"
HF_REVISION="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"

GPU_ID=${GPU_ID:-0}
PHASE8_SUITES=${PHASE8_SUITES:-"libero_spatial libero_object libero_goal"}
PHASE8_NUM_PAIRS=${PHASE8_NUM_PAIRS:-20}
PHASE8_SEED=${PHASE8_SEED:-7}
PHASE8_DETERMINISTIC_RESET_SEED=${PHASE8_DETERMINISTIC_RESET_SEED:-0}
PHASE8_RESULTS_DIR=${PHASE8_RESULTS_DIR:-"$REPO_ROOT/experiments/phase8_full_rollout/main_suites_camera_seed7_20pair"}
PHASE8_SAVE_VIDEOS=${PHASE8_SAVE_VIDEOS:-false}
PHASE8_RESUME=${PHASE8_RESUME:-true}
PHASE8_TASK_LIMIT=${PHASE8_TASK_LIMIT:-0}
PHASE8_EXTRA_ARGS=${PHASE8_EXTRA_ARGS:-}

SAVE_VIDEOS_FLAG=--no-save-videos
case "$PHASE8_SAVE_VIDEOS" in
    1|true|TRUE|yes|YES)
        SAVE_VIDEOS_FLAG=--save-videos
        ;;
esac

RESUME_FLAG=--resume
case "$PHASE8_RESUME" in
    0|false|FALSE|no|NO)
        RESUME_FLAG=--no-resume
        ;;
esac

if [ ! -f "$BASE_MODEL_DIR/model-480p-16fps.pt" ]; then
    echo "Missing base checkpoint: $BASE_MODEL_DIR/model-480p-16fps.pt" >&2
    exit 1
fi

if [ ! -f "$BASE_MODEL_DIR/tokenizer/tokenizer.pth" ]; then
    echo "Missing tokenizer checkpoint: $BASE_MODEL_DIR/tokenizer/tokenizer.pth" >&2
    exit 1
fi

if [ ! -f "$POLICY_MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" ]; then
    echo "Missing policy checkpoint: $POLICY_MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" >&2
    exit 1
fi

mkdir -p "$HF_REPO_DIR/refs" "$HF_REPO_DIR/snapshots"
printf '%s' "$HF_REVISION" > "$HF_REPO_DIR/refs/main"
ln -sfnT "$BASE_MODEL_DIR" "$HF_REPO_DIR/snapshots/$HF_REVISION"

mkdir -p "${TMPDIR:-/tmp}/cosmospolicy-numba" "${TMPDIR:-/tmp}/cosmospolicy-matplotlib"

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/cosmospolicy-numba"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cosmospolicy-matplotlib"
export HF_HUB_CACHE="$CHECKPOINT_ROOT/huggingface-hub"
export HF_HUB_OFFLINE=1
export PYTHONHASHSEED="$PHASE8_SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export COSMOS_POLICY_MODEL_DIR="$POLICY_MODEL_DIR"

python bin/phase8_fast_full_rollout.py \
    --suites $PHASE8_SUITES \
    --num-pairs "$PHASE8_NUM_PAIRS" \
    --seed "$PHASE8_SEED" \
    --deterministic-reset-seed "$PHASE8_DETERMINISTIC_RESET_SEED" \
    --policy-dir "$POLICY_MODEL_DIR" \
    --output-root "$PHASE8_RESULTS_DIR" \
    --task-limit "$PHASE8_TASK_LIMIT" \
    "$SAVE_VIDEOS_FLAG" \
    "$RESUME_FLAG" \
    $PHASE8_EXTRA_ARGS
