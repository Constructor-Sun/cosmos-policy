#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK=${LIBERO10_TASK:-KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it}
LANGUAGE=${LIBERO10_LANGUAGE:-put the black bowl in the bottom drawer of the cabinet and close it}
NUM_CASES=${NUM_CASES:-1}
SEED=${SEED:-7}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}
ROBOTINIT_GPU=${ROBOTINIT_GPU:-7}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_robotinit_single}
COSMOS_ROLLOUT_SUBDIR=${COSMOS_ROLLOUT_SUBDIR:-08-06}

# --- Action offset (patch, disabled by default) ---
# AMOUNT = total 6-dim offset over the entire duration (not per-step):
#   dx,dy,dz,droll,dpitch,dyaw  (m, m, m, rad, rad, rad)
#   dx>0 = forward, dy>0 = left, dz>0 = up
# DURATION = number of steps to spread the total offset across (default 16 = 1 chunk)
# Set COSMOS_OFFSET_START_T to a non-negative timestep to enable.
# Example: 1cm forward at t=160 over 1 chunk → START_T=160, AMOUNT="0.01,0,0,0,0,0", DURATION=16
COSMOS_OFFSET_START_T=${COSMOS_OFFSET_START_T:-80}
COSMOS_OFFSET_AMOUNT=${COSMOS_OFFSET_AMOUNT:-"-5,-4,1,0,0,-15"}
COSMOS_OFFSET_DURATION=${COSMOS_OFFSET_DURATION:-16}
# --- Data collection (save trajectory HDF5 for offline analysis) ---
COSMOS_DATA_COLLECTION=${COSMOS_DATA_COLLECTION:-}
# --- Vector DB (save VAE latents + proprio at action chunk boundaries) ---
COSMOS_VECTOR_DB=${COSMOS_VECTOR_DB:-}
COSMOS_VECTOR_DB_DIR=${COSMOS_VECTOR_DB_DIR:-}
# --- Init state offset (default 0 = first init state) ---
# Episode N = init_state_index N-1. Set OFFSET=2 for episode 3.
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-4}

mkdir -p "$OUTPUT_ROOT/robotinit" "$OUTPUT_ROOT/tmp-robotinit"

cd "$REPO_ROOT"

COSMOS_ROLLOUT_SUBDIR="$COSMOS_ROLLOUT_SUBDIR" \
COSMOS_SKIP_PLAIN_ROLLOUT=1 \
COSMOS_OFFSET_START_T="$COSMOS_OFFSET_START_T" \
COSMOS_OFFSET_AMOUNT="$COSMOS_OFFSET_AMOUNT" \
COSMOS_OFFSET_DURATION="$COSMOS_OFFSET_DURATION" \
COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
SMOKE_DATA_COLLECTION="$COSMOS_DATA_COLLECTION" \
COSMOS_VECTOR_DB="$COSMOS_VECTOR_DB" \
COSMOS_VECTOR_DB_DIR="$COSMOS_VECTOR_DB_DIR" \
TMPDIR="$OUTPUT_ROOT/tmp-robotinit" \
SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
GPU_ID="$ROBOTINIT_GPU" \
SMOKE_GL_BACKEND=egl \
SMOKE_ONLY_CONDITION=perturb \
SMOKE_PAIR_SUITE=libero_10 \
SMOKE_PAIR_BASE_TASK="$TASK" \
SMOKE_PAIR_CLEAN_LANGUAGE="$LANGUAGE" \
SMOKE_PAIR_PERT_NAME=robot_initial_states \
SMOKE_PAIR_PERT_CATEGORY="Robot Initial States" \
SMOKE_PAIR_PERT_TASK="${TASK}_view_0_0_100_0_0_initstate_274" \
SMOKE_NUM_PAIRS="$NUM_CASES" \
SMOKE_SEED="$SEED" \
SMOKE_RESULTS_DIR="$OUTPUT_ROOT/robotinit" \
SMOKE_RUN_ID="robotinit_single_gpu${ROBOTINIT_GPU}" \
sh ./run_libero_smoke_test.sh
# echo "Robotinit single case evaluation complete: $OUTPUT_ROOT"
