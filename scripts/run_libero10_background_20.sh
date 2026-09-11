#!/bin/sh
#
# run_libero10_background_20.sh
#
# Run CosmosPolicy LIBERO-10 background-texture perturbation evaluation with
# phase verifier / recovery / feasible recovery enabled.
#
# This script mirrors the run-plan style of run_libero10_all_7dims_baseline.sh:
# it reads configs/libero10_experiment_tasks.json and picks a valid background
# variant per task (prefer 5(table), otherwise the first table variant).
#
# Usage:
#   sh scripts/run_libero10_background_20.sh
#
# Useful overrides:
#   GPU_IDS="0 1 2 3" NUM_CASES=10 SEED=7 OUTPUT_ROOT=/path ROLLOUT_SUBDIR=background \
#     COSMOS_INIT_STATE_OFFSET=0 sh scripts/run_libero10_background_20.sh
#
# Scope:
#   Default runs all 10 LIBERO-10 tasks from libero10_experiment_tasks.json.
#   Set TASK_SCOPE=other9 to skip KITCHEN_SCENE4 (the same 9 tasks used by
#   run_libero10_other9_robotinit_20.sh).
#

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GPU_ID=${GPU_ID:-${ROBOTINIT_GPU:-7}}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
MAX_PARALLEL=${MAX_PARALLEL:-4}
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
TASK_SCOPE=${TASK_SCOPE:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/baselines/libero10_background_20}
ROLLOUT_SUBDIR=${ROLLOUT_SUBDIR:-background_initial}
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-0}
COSMOS_INITIAL_ALIGNMENT=${COSMOS_INITIAL_ALIGNMENT:-1}
COSMOS_HELD_OBJECT=${COSMOS_HELD_OBJECT:-1}
COSMOS_PLACE_MODE=${COSMOS_PLACE_MODE:-curobo}
COSMOS_SKILL_COMPLETION_ACTIVE=${COSMOS_SKILL_COMPLETION_ACTIVE:-1}
COSMOS_CUROBO_JOINT_EXECUTION=${COSMOS_CUROBO_JOINT_EXECUTION:-0}
COSMOS_URDF_ROBOT_FILTER=${COSMOS_URDF_ROBOT_FILTER:-1}
COSMOS_HELD_OBJECT_DEBUG=${COSMOS_HELD_OBJECT_DEBUG:-0}
SMOKE_GL_BACKEND=${SMOKE_GL_BACKEND:-egl}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/scripts/run_libero_smoke_test.py}

# Keep this background-perturbation evaluation free of data-collection machinery.
unset COSMOS_DATA_COLLECTION \
      COSMOS_VECTOR_DB \
      COSMOS_VECTOR_DB_DIR

# ---------------------------------------------------------------------------
# Generate a background run plan from configs/libero10_experiment_tasks.json.
#
# Output columns:
#   task_name|language|pert_name|pert_category|pert_task|variant_tag|variant_display
# ---------------------------------------------------------------------------
RUN_PLAN=$(python3 - "$REPO_ROOT/configs/libero10_experiment_tasks.json" "$TASK_SCOPE" <<'PY'
import json
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: run_plan.py <config> <scope>")
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

scope = sys.argv[2]
if scope not in ("all", "other9"):
    raise SystemExit(f"TASK_SCOPE must be 'all' or 'other9', got {scope!r}")

for idx, task in enumerate(cfg["tasks"]):
    if scope == "other9" and idx == 1:
        continue

    name = task["name"]
    lang = task["language"]
    perts = task["perturbations"]

    bg_variants = perts["background_textures"]["variants"]
    chosen = None
    for v in bg_variants:
        if v["id"] == 5 and "table" in v["kinds"]:
            chosen = (5, "table")
            break
    if chosen is None:
        v = bg_variants[0]
        chosen = (v["id"], v["kinds"][0])

    pert_task = f"{name}_{chosen[1]}_{chosen[0]}"
    variant_tag = f"{chosen[0]}_{chosen[1]}"
    variant_display = f"{chosen[0]} ({chosen[1]})"
    print(f"{name}|{lang}|background_textures|Background Textures|{pert_task}|{variant_tag}|{variant_display}")
PY
)

if [ -z "$RUN_PLAN" ]; then
    echo "ERROR: could not generate background run plan from $REPO_ROOT/configs/libero10_experiment_tasks.json" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

_n_gpus=0
for _g in $GPU_IDS; do _n_gpus=$((_n_gpus+1)); done
if [ "$_n_gpus" -eq 0 ]; then
    echo "ERROR: GPU_IDS is empty" >&2
    exit 1
fi
_gpu_index=0
_jobs=0

echo "======================================================"
echo "LIBERO-10 background_textures evaluation (no clean)"
echo "Output root : $OUTPUT_ROOT"
echo "GPU_IDS     : $GPU_IDS"
echo "NUM_CASES   : $NUM_CASES"
echo "SEED        : $SEED"
echo "TASK_SCOPE  : $TASK_SCOPE"
echo "InitAlign   : $COSMOS_INITIAL_ALIGNMENT"
echo "PlaceMode   : $COSMOS_PLACE_MODE"
echo "======================================================"

# ---------------------------------------------------------------------------
# Run background perturbations in parallel across GPU_IDS.
# ---------------------------------------------------------------------------
_run_plan_file=$(mktemp)
printf '%s\n' "$RUN_PLAN" > "$_run_plan_file"

while IFS='|' read -r task language pert_name pert_category pert_task variant_tag variant_display; do
    [ -z "$task" ] && continue

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
        _gpu=$GPU_ID
    fi
    _gpu_index=$(( (_gpu_index + 1) % _n_gpus ))

    pert_dir="$OUTPUT_ROOT/$task/$pert_name"
    mkdir -p "$pert_dir"

    echo ""
    echo "=== $pert_name: $task ==="
    echo "    variant: $variant_display"
    echo "    gpu: $_gpu"
    echo "    results: $pert_dir"

    (
        cd "$REPO_ROOT"
        COSMOS_INITIAL_ALIGNMENT="$COSMOS_INITIAL_ALIGNMENT" \
        COSMOS_HELD_OBJECT="$COSMOS_HELD_OBJECT" \
        COSMOS_PLACE_MODE="$COSMOS_PLACE_MODE" \
        COSMOS_SKILL_COMPLETION_ACTIVE="$COSMOS_SKILL_COMPLETION_ACTIVE" \
        COSMOS_CUROBO_JOINT_EXECUTION="$COSMOS_CUROBO_JOINT_EXECUTION" \
        COSMOS_URDF_ROBOT_FILTER="$COSMOS_URDF_ROBOT_FILTER" \
        COSMOS_HELD_OBJECT_DEBUG="$COSMOS_HELD_OBJECT_DEBUG" \
        COSMOS_ROLLOUT_SUBDIR="$ROLLOUT_SUBDIR" \
        COSMOS_SKIP_PLAIN_ROLLOUT=1 \
        COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
        SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
        GPU_ID="$_gpu" \
        SMOKE_GL_BACKEND="$SMOKE_GL_BACKEND" \
        SMOKE_ONLY_CONDITION=perturb \
        SMOKE_PAIR_SUITE=libero_10 \
        SMOKE_PAIR_BASE_TASK="$task" \
        SMOKE_PAIR_CLEAN_LANGUAGE="$language" \
        SMOKE_PAIR_PERT_NAME="$pert_name" \
        SMOKE_PAIR_PERT_CATEGORY="$pert_category" \
        SMOKE_PAIR_PERT_TASK="$pert_task" \
        SMOKE_NUM_PAIRS="$NUM_CASES" \
        SMOKE_SEED="$SEED" \
        SMOKE_RESULTS_DIR="$pert_dir" \
        SMOKE_RUN_ID="background_${variant_tag}_gpu${_gpu}_${task}" \
        sh "$REPO_ROOT/scripts/run_libero_smoke_test.sh" > "$pert_dir/run.log" 2>&1
    ) &
    _jobs=$((_jobs + 1))

    if [ "$_jobs" -ge "$MAX_PARALLEL" ]; then
        wait
        _jobs=0
    fi
done < "$_run_plan_file"

rm -f "$_run_plan_file"

wait

echo ""
echo "All LIBERO-10 background evaluations complete."
echo "Results under: $OUTPUT_ROOT"
