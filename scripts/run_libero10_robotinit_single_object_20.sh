#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-0}
ROBOTINIT_GPU=${ROBOTINIT_GPU:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_robotinit_single_object}
ROLLOUT_SUBDIR=${ROLLOUT_SUBDIR:-08-31-curobo-single-object}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-0}

COSMOS_INITIAL_ALIGNMENT=${COSMOS_INITIAL_ALIGNMENT:-1}
COSMOS_CUROBO_JOINT_EXECUTION=${COSMOS_CUROBO_JOINT_EXECUTION:-0}
COSMOS_URDF_ROBOT_FILTER=${COSMOS_URDF_ROBOT_FILTER:-1}
COSMOS_DEBUG_INIT_ALIGN=${COSMOS_DEBUG_INIT_ALIGN:-1}
COSMOS_HELD_OBJECT=${COSMOS_HELD_OBJECT:-1}
COSMOS_HELD_OBJECT_DEBUG=${COSMOS_HELD_OBJECT_DEBUG:-1}
COSMOS_SKILL_COMPLETION_ACTIVE=${COSMOS_SKILL_COMPLETION_ACTIVE:-1}

mkdir -p "$OUTPUT_ROOT/tmp"
cd "$REPO_ROOT"

unset COSMOS_OFFSET_START_T COSMOS_OFFSET_AMOUNT COSMOS_OFFSET_DURATION COSMOS_ONLINE_INJECT

run_task() {
    task=$1
    language=$2
    variant=$3
    item=$4
    task_output="$OUTPUT_ROOT/$task"
    mkdir -p "$task_output"
    printf 'Running %s (%s cases), item=%s\n' "$task" "$NUM_CASES" "$item"

    COSMOS_ROLLOUT_SUBDIR="$ROLLOUT_SUBDIR" \
    COSMOS_SKIP_PLAIN_ROLLOUT=1 \
    COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
    COSMOS_INITIAL_ALIGNMENT="$COSMOS_INITIAL_ALIGNMENT" \
    COSMOS_CUROBO_JOINT_EXECUTION="$COSMOS_CUROBO_JOINT_EXECUTION" \
    COSMOS_URDF_ROBOT_FILTER="$COSMOS_URDF_ROBOT_FILTER" \
    COSMOS_DEBUG_INIT_ALIGN="$COSMOS_DEBUG_INIT_ALIGN" \
    COSMOS_HELD_OBJECT="$COSMOS_HELD_OBJECT" \
    COSMOS_HELD_OBJECT_DEBUG="$COSMOS_HELD_OBJECT_DEBUG" \
    COSMOS_SKILL_COMPLETION_ACTIVE="$COSMOS_SKILL_COMPLETION_ACTIVE" \
    COSMOS_SKILL_COMPLETION_ITEM="$item" \
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
    SMOKE_RUN_ID="single_object_held_robotinit_gpu${ROBOTINIT_GPU}" \
    sh ./run_libero_smoke_test.sh >"$task_output/run.log" 2>&1
}

run_task \
    KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    "turn on the stove and put the moka pot on it" \
    KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_273 \
    moka_pot_1

run_task \
    KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it \
    "put the black bowl in the bottom drawer of the cabinet and close it" \
    KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_274 \
    akita_black_bowl_1

run_task \
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it \
    "put the yellow and white mug in the microwave and close it" \
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_270 \
    white_yellow_mug_1

run_task \
    STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy \
    "pick up the book and place it in the back compartment of the caddy" \
    STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_276 \
    black_book_1

printf 'Single-object held-object evaluations completed.\nResults: %s\nVideos: %s\n' \
    "$OUTPUT_ROOT" "$REPO_ROOT/rollouts/$ROLLOUT_SUBDIR"
