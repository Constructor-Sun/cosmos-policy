#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-0}
ROBOTINIT_GPU=${ROBOTINIT_GPU:-2}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
MAX_PARALLEL=${MAX_PARALLEL:-4}
_gpu_index=0
_n_gpus=0
for _g in $GPU_IDS; do _n_gpus=$((_n_gpus+1)); done
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/../experiments/libero10_all10_held_object}
ROLLOUT_SUBDIR=${ROLLOUT_SUBDIR:-08-31-all10-held-object}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-0}

# --- Data collection / vector DB / rollouts ---
COSMOS_DATA_COLLECTION=${COSMOS_DATA_COLLECTION:-}
COSMOS_VECTOR_DB=${COSMOS_VECTOR_DB:-}
COSMOS_VECTOR_DB_DIR=${COSMOS_VECTOR_DB_DIR:-}
COSMOS_SKIP_PLAIN_ROLLOUT=${COSMOS_SKIP_PLAIN_ROLLOUT:-1}
# --- One-shot initial alignment ---
COSMOS_INITIAL_ALIGNMENT=${COSMOS_INITIAL_ALIGNMENT:-1}
COSMOS_CUROBO_JOINT_EXECUTION=${COSMOS_CUROBO_JOINT_EXECUTION:-0}
COSMOS_URDF_ROBOT_FILTER=${COSMOS_URDF_ROBOT_FILTER:-1}
COSMOS_DEBUG_INIT_ALIGN=${COSMOS_DEBUG_INIT_ALIGN:-0}
# --- Held-object / skill transition ---
COSMOS_HELD_OBJECT=${COSMOS_HELD_OBJECT:-1}
COSMOS_HELD_OBJECT_DEBUG=${COSMOS_HELD_OBJECT_DEBUG:-0}
COSMOS_PLACE_MODE=${COSMOS_PLACE_MODE:-curobo}
COSMOS_SKILL_COMPLETION_ACTIVE=${COSMOS_SKILL_COMPLETION_ACTIVE:-1}

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/tmp"
cd "$REPO_ROOT"

unset COSMOS_OFFSET_START_T COSMOS_OFFSET_AMOUNT COSMOS_OFFSET_DURATION COSMOS_ONLINE_INJECT

run_task() {
    task=$1
    language=$2
    variant=$3
    item=$4
    task_output="$OUTPUT_ROOT/$task"

    # Round-robin GPU assignment for multi-GPU parallel runs.
    _gpu=""
    _i=0
    for _g in $GPU_IDS; do
        if [ "$_i" -eq "$_gpu_index" ]; then
            _gpu=$_g
            break
        fi
        _i=$((_i + 1))
    done
    if [ -z "$_gpu" ]; then
        _gpu=$ROBOTINIT_GPU
    fi
    _gpu_index=$(( (_gpu_index + 1) % _n_gpus ))

    mkdir -p "$task_output"
    printf 'Running %s (%s cases), item=%s\n' "$task" "$NUM_CASES" "$item"

    COSMOS_ROLLOUT_SUBDIR="$ROLLOUT_SUBDIR" \
    COSMOS_SKIP_PLAIN_ROLLOUT="$COSMOS_SKIP_PLAIN_ROLLOUT" \
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
    COSMOS_PLACE_MODE="$COSMOS_PLACE_MODE" \
    COSMOS_SKILL_COMPLETION_ACTIVE="$COSMOS_SKILL_COMPLETION_ACTIVE" \
    COSMOS_SKILL_COMPLETION_ITEM="$item" \
    TMPDIR="$OUTPUT_ROOT/tmp" \
    SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
    GPU_ID="$_gpu" \
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
    SMOKE_RUN_ID="all10_held_object_gpu${_gpu}" \
    sh ./run_libero_smoke_test.sh >"$task_output/run.log" 2>&1 &
}

# Single-object tasks.
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

wait

# Multi-object tasks: the active completion reads each item from phase
# arguments.  COSMOS_SKILL_COMPLETION_ITEM is only used by read-only shadow
# telemetry, so we set it to the first Pick item here.
run_task \
    KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
    "put both moka pots on the stove" \
    KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_269 \
    moka_pot_2

run_task \
    LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket \
    "put both the alphabet soup and the cream cheese box in the basket" \
    LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_268 \
    alphabet_soup_1

run_task \
    LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket \
    "put both the alphabet soup and the tomato sauce in the basket" \
    LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_271 \
    alphabet_soup_1

run_task \
    LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket \
    "put both the cream cheese box and the butter in the basket" \
    LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_282 \
    cream_cheese_1

wait

run_task \
    LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate \
    "put the white mug on the left plate and put the yellow and white mug on the right plate" \
    LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_265 \
    porcelain_mug_1

run_task \
    LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate \
    "put the white mug on the plate and put the chocolate pudding to the right of the plate" \
    LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_267 \
    porcelain_mug_1

wait

printf 'All 10 LIBERO-10 held-object evaluations completed.\nResults: %s\nVideos: %s\n' \
    "$OUTPUT_ROOT" "$REPO_ROOT/rollouts/$ROLLOUT_SUBDIR"
