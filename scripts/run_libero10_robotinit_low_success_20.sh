#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NUM_CASES=${NUM_CASES:-5}
SEED=${SEED:-7}
ROBOTINIT_GPU=${ROBOTINIT_GPU:-6}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_robotinit_low_success_20}
ROLLOUT_SUBDIR=${ROLLOUT_SUBDIR:-libero_10_robotinit_low_success}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-0}
# --- One-shot initial alignment (default on for this low-success rerun set) ---
COSMOS_INITIAL_ALIGNMENT=${COSMOS_INITIAL_ALIGNMENT:-1}
COSMOS_CUROBO_JOINT_EXECUTION=${COSMOS_CUROBO_JOINT_EXECUTION:-1}

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/tmp" "$REPO_ROOT/rollouts/$ROLLOUT_SUBDIR"
cd "$REPO_ROOT"

run_task() {
    task=$1
    language=$2
    variant=$3
    task_output="$OUTPUT_ROOT/$task"

    mkdir -p "$task_output"
    printf 'Running %s (%s cases)\n' "$task" "$NUM_CASES"

    COSMOS_ROLLOUT_SUBDIR="$ROLLOUT_SUBDIR" \
    COSMOS_SKIP_PLAIN_ROLLOUT=1 \
    COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
    COSMOS_INITIAL_ALIGNMENT="$COSMOS_INITIAL_ALIGNMENT" \
    COSMOS_CUROBO_JOINT_EXECUTION="$COSMOS_CUROBO_JOINT_EXECUTION" \
    TMPDIR="$OUTPUT_ROOT/tmp" \
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
    SMOKE_RUN_ID="low_success_robotinit_gpu${ROBOTINIT_GPU}" \
    sh ./run_libero_smoke_test.sh >"$task_output/run.log" 2>&1
}

run_task \
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it \
    "put the yellow and white mug in the microwave and close it" \
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_270

run_task \
    KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
    "put both moka pots on the stove" \
    KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_269

run_task \
    LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket \
    "put both the alphabet soup and the tomato sauce in the basket" \
    LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_271

run_task \
    LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate \
    "put the white mug on the left plate and put the yellow and white mug on the right plate" \
    LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_265

run_task \
    STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy \
    "pick up the book and place it in the back compartment of the caddy" \
    STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_276


printf 'All low-success robot-init evaluations completed.\nResults: %s\nVideos: %s\n' \
    "$OUTPUT_ROOT" "$REPO_ROOT/rollouts/$ROLLOUT_SUBDIR"
