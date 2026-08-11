#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK=${LIBERO10_TASK:-KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it}
LANGUAGE=${LIBERO10_LANGUAGE:-put the black bowl in the bottom drawer of the cabinet and close it}
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}
CLEAN_GPU=${CLEAN_GPU:-4}
BACKGROUND_GPU=${BACKGROUND_GPU:-5}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_clean_background_20cases}

mkdir -p "$OUTPUT_ROOT/clean" "$OUTPUT_ROOT/background" "$OUTPUT_ROOT/tmp-clean" "$OUTPUT_ROOT/tmp-background"

cd "$REPO_ROOT"

TMPDIR="$OUTPUT_ROOT/tmp-clean" \
SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
GPU_ID="$CLEAN_GPU" \
SMOKE_GL_BACKEND=egl \
SMOKE_ONLY_CONDITION=clean \
SMOKE_PAIR_SUITE=libero_10 \
SMOKE_PAIR_BASE_TASK="$TASK" \
SMOKE_PAIR_CLEAN_LANGUAGE="$LANGUAGE" \
SMOKE_NUM_PAIRS="$NUM_CASES" \
SMOKE_SEED="$SEED" \
SMOKE_RESULTS_DIR="$OUTPUT_ROOT/clean" \
SMOKE_RUN_ID="clean_gpu${CLEAN_GPU}_${NUM_CASES}cases" \
sh ./run_libero_smoke_test.sh >"$OUTPUT_ROOT/clean.log" 2>&1 &
CLEAN_PID=$!

TMPDIR="$OUTPUT_ROOT/tmp-background" \
SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
GPU_ID="$BACKGROUND_GPU" \
SMOKE_GL_BACKEND=egl \
SMOKE_ONLY_CONDITION=background \
SMOKE_PAIR_SUITE=libero_10 \
SMOKE_PAIR_BASE_TASK="$TASK" \
SMOKE_PAIR_CLEAN_LANGUAGE="$LANGUAGE" \
SMOKE_PAIR_PERT_NAME=background_textures \
SMOKE_PAIR_PERT_CATEGORY="Background Textures" \
SMOKE_PAIR_PERT_TASK="${TASK}_table_5" \
SMOKE_NUM_PAIRS="$NUM_CASES" \
SMOKE_SEED="$SEED" \
SMOKE_RESULTS_DIR="$OUTPUT_ROOT/background" \
SMOKE_RUN_ID="background_gpu${BACKGROUND_GPU}_${NUM_CASES}cases" \
sh ./run_libero_smoke_test.sh >"$OUTPUT_ROOT/background.log" 2>&1 &
BACKGROUND_PID=$!

printf 'clean: GPU %s, PID %s, log %s\n' "$CLEAN_GPU" "$CLEAN_PID" "$OUTPUT_ROOT/clean.log"
printf 'background: GPU %s, PID %s, log %s\n' "$BACKGROUND_GPU" "$BACKGROUND_PID" "$OUTPUT_ROOT/background.log"

set +e
wait "$CLEAN_PID"
CLEAN_STATUS=$?
wait "$BACKGROUND_PID"
BACKGROUND_STATUS=$?
set -e

printf 'clean exit status: %s\n' "$CLEAN_STATUS"
printf 'background exit status: %s\n' "$BACKGROUND_STATUS"

if [ "$CLEAN_STATUS" -ne 0 ] || [ "$BACKGROUND_STATUS" -ne 0 ]; then
    exit 1
fi
