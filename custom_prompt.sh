#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GPU_ID=${GPU_ID:-0}
PROMPT=${PROMPT:-Pick up the small bowl and move it to the knife block on the left}
T5_MODEL_PATH=${T5_MODEL_PATH:-$REPO_ROOT/../../checkpoints/t5-11b}
T5_EMBEDDING_PATH=${T5_EMBEDDING_PATH:-$REPO_ROOT/experiments/custom_prompt_t5.pkl}
RESULTS_DIR=${RESULTS_DIR:-$REPO_ROOT/experiments/cosmos_robotinit_custom_prompt_5}

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="../LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}" \
    python bin/make_libero_plus_t5_embeddings.py \
    --prompt "$PROMPT" \
    --model-path "$T5_MODEL_PATH" \
    --device cuda \
    --local-files-only \
    --output "$T5_EMBEDDING_PATH"

GPU_ID="$GPU_ID" \
SMOKE_ONLY_CONDITION=perturb \
SMOKE_PAIR_PERT_NAME=robot_initial_states \
SMOKE_NUM_PAIRS=5 \
SMOKE_PROMPT_OVERRIDE="$PROMPT" \
SMOKE_T5_EXTRA_EMBEDDINGS="$T5_EMBEDDING_PATH" \
SMOKE_RESULTS_DIR="$RESULTS_DIR" \
SMOKE_RUN_ID=cosmos_robotinit_custom_prompt_5 \
sh ./run_libero_smoke_test.sh
