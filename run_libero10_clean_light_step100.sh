#!/usr/bin/env bash

set -euo pipefail

GPU_ID=${1:?Usage: sh run_libero10_clean_light_step100.sh GPU_ID NUM_CASES}
NUM_CASES=${2:?Usage: sh run_libero10_clean_light_step100.sh GPU_ID NUM_CASES}

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK=KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
CHECKPOINT="$REPO_ROOT/outputs/cosmos_policy/counterfactual_direct_sft/light_libero10_500/checkpoints/iter_000000500/model"
RUN_ROOT="$REPO_ROOT/experiments/step100_clean_light_gpu${GPU_ID}_${NUM_CASES}cases"

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

GPU_ID="$GPU_ID" \
SMOKE_POLICY_CKPT_PATH="$CHECKPOINT" \
SMOKE_GL_BACKEND=egl \
SMOKE_ONLY_CONDITION=both \
SMOKE_PAIR_SUITE=libero_10 \
SMOKE_PAIR_BASE_TASK="$TASK" \
SMOKE_PAIR_CLEAN_LANGUAGE="put the black bowl in the bottom drawer of the cabinet and close it" \
SMOKE_PAIR_PERT_NAME=light_conditions \
SMOKE_PAIR_PERT_CATEGORY="Light Conditions" \
SMOKE_PAIR_PERT_TASK="${TASK}_light_5" \
SMOKE_NUM_PAIRS="$NUM_CASES" \
SMOKE_SEED=7 \
SMOKE_RESULTS_DIR="$RUN_ROOT" \
SMOKE_RUN_ID="step100_clean_light_gpu${GPU_ID}" \
sh ./run_libero_smoke_test.sh 2>&1 \
    | tee "$RUN_ROOT/rollout.log" \
    | awk '
        /Query [0-9]+\/[0-9]+ \(seed [^)]+\): Predicted value =/ { next }
        /t=[0-9]+: (Current base seed|Selected seed)/ { next }
        /Query [0-9]+\/[0-9]+: Action query time =/ { next }
        /fori_loop: [0-9]+/ { next }
        { print; fflush() }
    '

echo "Evaluation complete: $RUN_ROOT"
