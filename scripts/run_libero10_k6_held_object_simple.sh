#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-0}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}
ROBOTINIT_GPU=${ROBOTINIT_GPU:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_robotinit_single}
COSMOS_ROLLOUT_SUBDIR=${COSMOS_ROLLOUT_SUBDIR:-08-28}

# --- Data collection (save trajectory HDF5 for offline analysis) ---
COSMOS_DATA_COLLECTION=${COSMOS_DATA_COLLECTION:-}
# --- Vector DB (save VAE latents + proprio at action chunk boundaries) ---
COSMOS_VECTOR_DB=${COSMOS_VECTOR_DB:-}
COSMOS_VECTOR_DB_DIR=${COSMOS_VECTOR_DB_DIR:-}
# --- One-shot initial alignment ---
COSMOS_INITIAL_ALIGNMENT=${COSMOS_INITIAL_ALIGNMENT:-1}
COSMOS_CUROBO_JOINT_EXECUTION=${COSMOS_CUROBO_JOINT_EXECUTION:-0}
COSMOS_URDF_ROBOT_FILTER=${COSMOS_URDF_ROBOT_FILTER:-1}
# --- Per-step initial-alignment debug logging ---
COSMOS_DEBUG_INIT_ALIGN=${COSMOS_DEBUG_INIT_ALIGN:-0}
# --- Held-object Pick->Place pipeline ---
COSMOS_HELD_OBJECT=${COSMOS_HELD_OBJECT:-1}
# --- Held-object debug point cloud images ---
COSMOS_HELD_OBJECT_DEBUG=${COSMOS_HELD_OBJECT_DEBUG:-0}
# --- Active skill completion (needed to detect Pick->Place transition) ---
COSMOS_SKILL_COMPLETION_ACTIVE=${COSMOS_SKILL_COMPLETION_ACTIVE:-1}
COSMOS_SKILL_COMPLETION_ITEM=${COSMOS_SKILL_COMPLETION_ITEM:-white_yellow_mug_1}
# --- Place mode: use simple VLA + PoseController fine alignment ---
COSMOS_PLACE_MODE=${COSMOS_PLACE_MODE:-simple}
# --- Init state offset (default 0 = first init state) ---
# Episode N = init_state_index N-1. Set OFFSET=2 for episode 3.
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-4}

RESULT_NAME=robotinit-simple
MODE_NAME=normal

mkdir -p "$OUTPUT_ROOT/$RESULT_NAME" "$OUTPUT_ROOT/tmp-robotinit"

cd "$REPO_ROOT"

# Legacy manual-injection controls are deliberately unsupported.
unset COSMOS_OFFSET_START_T COSMOS_OFFSET_AMOUNT COSMOS_OFFSET_DURATION COSMOS_ONLINE_INJECT

run_task() {
    task=$1
    language=$2
    variant=$3
    task_output="$OUTPUT_ROOT/$RESULT_NAME"

    mkdir -p "$task_output"
    echo "[K6] COSMOS_HELD_OBJECT=$COSMOS_HELD_OBJECT COSMOS_PLACE_MODE=$COSMOS_PLACE_MODE"

    COSMOS_ROLLOUT_SUBDIR="$COSMOS_ROLLOUT_SUBDIR" \
    COSMOS_SKIP_PLAIN_ROLLOUT=1 \
    COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
    SMOKE_DATA_COLLECTION="$COSMOS_DATA_COLLECTION" \
    COSMOS_VECTOR_DB="$COSMOS_VECTOR_DB" \
    COSMOS_VECTOR_DB_DIR="$COSMOS_VECTOR_DB_DIR" \
    COSMOS_INITIAL_ALIGNMENT="$COSMOS_INITIAL_ALIGNMENT" \
    COSMOS_CUROBO_JOINT_EXECUTION="$COSMOS_CUROBO_JOINT_EXECUTION" \
    COSMOS_URDF_ROBOT_FILTER="$COSMOS_URDF_ROBOT_FILTER" \
    COSMOS_DEBUG_INIT_ALIGN="$COSMOS_DEBUG_INIT_ALIGN" \
    COSMOS_HELD_OBJECT="$COSMOS_HELD_OBJECT" \
    COSMOS_HELD_OBJECT_DEBUG="$COSMOS_HELD_OBJECT_DEBUG" \
    COSMOS_SKILL_COMPLETION_ACTIVE="$COSMOS_SKILL_COMPLETION_ACTIVE" \
    COSMOS_SKILL_COMPLETION_ITEM="$COSMOS_SKILL_COMPLETION_ITEM" \
    COSMOS_PLACE_MODE="$COSMOS_PLACE_MODE" \
    TMPDIR="$OUTPUT_ROOT/tmp-robotinit" \
    SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
    GPU_ID="$ROBOTINIT_GPU" \
    SMOKE_GL_BACKEND=egl \
    SMOKE_ONLY_CONDITION=perturb \
    SMOKE_PAIR_SUITE=libero_10 \
    SMOKE_PAIR_BASE_TASK="$task" \
    SMOKE_PAIR_CLEAN_LANGUAGE="$language" \
    SMOKE_PAIR_PERT_NAME=robot_initial_states \
    SMOKE_PAIR_PERT_CATEGORY="Robot Initial States" \
    SMOKE_PAIR_PERT_TASK="$variant" \
    SMOKE_NUM_PAIRS="$NUM_CASES" \
    SMOKE_SEED="$SEED" \
    SMOKE_RESULTS_DIR="$task_output" \
    SMOKE_RUN_ID="k6_held_object_simple_gpu${ROBOTINIT_GPU}" \
    sh ./run_libero_smoke_test.sh
}

run_task \
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it \
    "put the yellow and white mug in the microwave and close it" \
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_270

printf 'K6 held-object simple evaluation completed.\nResults: %s\n' "$OUTPUT_ROOT/$RESULT_NAME"
