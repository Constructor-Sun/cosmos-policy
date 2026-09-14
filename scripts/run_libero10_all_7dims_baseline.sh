#!/bin/sh
#
# run_libero10_all_7dims_baseline.sh
#
# Run CosmosPolicy original/baseline evaluation over selected LIBERO-10
# tasks and LIBERO-plus perturbation dimensions (no clean, no robot_initial_states).
#
# For each task it runs:
#   - camera_viewpoints
#   - background_textures
#   - light_conditions
#   - objects_layout
#   - sensor_noise
#   - language_instructions
#
# Unlike a naive SMOKE_PAIR_PERT_NAME=all loop, this version reads
# configs/libero10_experiment_tasks.json and selects a valid per-task
# variant for each perturbation, because the valid variant IDs/specs are
# different across tasks.
#
# By default use four currently available GPUs; override with GPU_IDS=...
#
# Usage:
#   sh scripts/run_libero10_all_7dims_baseline.sh
#
# Useful overrides:
#   GPU_IDS="0 2 3 4" MAX_PARALLEL=4 NUM_CASES=20 SEED=7 OUTPUT_ROOT=/path \
#     sh scripts/run_libero10_all_7dims_baseline.sh
#

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GPU_ID=${GPU_ID:-7}
GPU_IDS=${GPU_IDS:-"0 2 3 4"}
MAX_PARALLEL=${MAX_PARALLEL:-4}
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_all_7dims_baseline}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-variants/libero10_all_7dims_20260914}
COSMOS_SKIP_PLAIN_ROLLOUT=${COSMOS_SKIP_PLAIN_ROLLOUT:-1}
SMOKE_GL_BACKEND=${SMOKE_GL_BACKEND:-egl}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/scripts/run_libero_smoke_test.py}
SMOKE_T5_EXTRA_EMBEDDINGS=${SMOKE_T5_EXTRA_EMBEDDINGS:-$REPO_ROOT/experiments/cache/libero_plus_language_t5_libero10.pkl}
SMOKE_T5_ALLOW_STRICT_COMPUTE=${SMOKE_T5_ALLOW_STRICT_COMPUTE:-false}

# Force original CosmosPolicy baseline behavior: disable data collection and
# vector-memory construction.
unset COSMOS_DATA_COLLECTION \
      COSMOS_VECTOR_DB \
      COSMOS_VECTOR_DB_DIR
export COSMOS_INIT_STATE_OFFSET=0

# ---------------------------------------------------------------------------
# Generate a run plan from configs/libero10_experiment_tasks.json.
#
# Output columns:
#   task_name|language|pert_name|pert_category|pert_task|variant_display
#
# For objects_layout, pert_task is left empty; the smoke-test script will
# use its built-in all_variants mode.
# ---------------------------------------------------------------------------
RUN_PLAN=$(python3 - "$REPO_ROOT/configs/libero10_experiment_tasks.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

def pick_first(d):
    return next(iter(d))

for idx, task in enumerate(cfg["tasks"]):
    name = task["name"]
    lang = task["language"]
    perts = task["perturbations"]

    # objects_layout is not in the JSON config; use smoke-test all_variants.
    print(f"{name}|{lang}|objects_layout|Objects Layout||objects_layout")

    # camera_viewpoints
    specs = perts["camera_viewpoints"]["specs"]
    spec = "50_0_100_0_0" if "50_0_100_0_0" in specs else pick_first(specs)
    p = specs[spec]
    pert_task = f"{name}_view_{p['h']}_{p['v']}_{p['s']}_{p['yaw']}_{p['pitch']}_initstate_0"
    print(f"{name}|{lang}|camera_viewpoints|Camera Viewpoints|{pert_task}|{spec}")
    # background_textures
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
    print(f"{name}|{lang}|background_textures|Background Textures|{pert_task}|{chosen[0]}:{chosen[1]}")

    # light_conditions
    light_ids = perts["light_conditions"]["variants"]
    light_id = 5 if 5 in light_ids else light_ids[0]
    pert_task = f"{name}_light_{light_id}"
    print(f"{name}|{lang}|light_conditions|Light Conditions|{pert_task}|{light_id}")

    # sensor_noise
    sensor_ids = [v["id"] for v in perts["sensor_noise"]["variants"]]
    sensor_id = 5 if 5 in sensor_ids else sensor_ids[0]
    pert_task = f"{name}_view_0_0_100_0_0_initstate_0_noise_{sensor_id}"
    print(f"{name}|{lang}|sensor_noise|Sensor Noise|{pert_task}|{sensor_id}")


    # language_instructions
    lang_ids = perts["language_instructions"]["variants"]
    lang_id = 5 if 5 in lang_ids else lang_ids[0]
    pert_task = f"{name}_language_{lang_id}_view_0_0_100_0_0_initstate_0"
    print(f"{name}|{lang}|language_instructions|Language Instructions|{pert_task}|{lang_id}")
PY
)

if [ -z "$RUN_PLAN" ]; then
    echo "ERROR: could not generate run plan from $REPO_ROOT/configs/libero10_experiment_tasks.json" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

_n_gpus=0
for _g in $GPU_IDS; do _n_gpus=$((_n_gpus + 1)); done
if [ "$_n_gpus" -eq 0 ]; then
    echo "ERROR: GPU_IDS is empty" >&2
    exit 1
fi
if [ "$MAX_PARALLEL" -lt 1 ]; then
    echo "ERROR: MAX_PARALLEL must be at least 1" >&2
    exit 1
fi
_gpu_index=0
_jobs=0

echo "======================================================"
echo "LIBERO-10 x selected LIBERO-plus dimensions baseline (no clean, no robot_init)"
echo "Output root : $OUTPUT_ROOT"
echo "GPU_IDS     : $GPU_IDS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "ROLLOUT_ROOT: $ROLLOUT_ROOT"
echo "NUM_CASES   : $NUM_CASES"
echo "SEED        : $SEED"
echo "======================================================"

# ---------------------------------------------------------------------------
# Run selected perturbations only.
# ---------------------------------------------------------------------------
_run_plan_file=$(mktemp)
printf '%s\n' "$RUN_PLAN" > "$_run_plan_file"

while IFS='|' read -r task language pert_name pert_category pert_task variant_display; do
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

    # For objects_layout, leave the default placeholder so the smoke-test
    # script enters all_variants mode.  For all other perturbations, use the
    # per-task resolved task name from config.
    if [ -n "$pert_task" ]; then
        resolved_pert_task="$pert_task"
    else
        resolved_pert_task="${task}_view_50_0_100_0_0_initstate_0"
    fi

    echo ""
    echo "=== $pert_name: $task ==="
    echo "    variant: $variant_display"
    echo "    gpu: $_gpu"
    echo "    results: $pert_dir"

    (
        cd "$REPO_ROOT"
        COSMOS_INIT_STATE_OFFSET=0 \
        COSMOS_ROLLOUT_SUBDIR="$ROLLOUT_ROOT/$pert_name/$task" \
        COSMOS_SKIP_PLAIN_ROLLOUT="$COSMOS_SKIP_PLAIN_ROLLOUT" \
        SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
        GPU_ID="$_gpu" \
        SMOKE_GL_BACKEND="$SMOKE_GL_BACKEND" \
        SMOKE_ONLY_CONDITION=perturb \
        SMOKE_PAIR_SUITE=libero_10 \
        SMOKE_PAIR_BASE_TASK="$task" \
        SMOKE_PAIR_CLEAN_LANGUAGE="$language" \
        SMOKE_PAIR_PERT_NAME="$pert_name" \
        SMOKE_PAIR_PERT_CATEGORY="$pert_category" \
        SMOKE_PAIR_PERT_TASK="$resolved_pert_task" \
        SMOKE_NUM_PAIRS="$NUM_CASES" \
        SMOKE_SEED="$SEED" \
        SMOKE_RESULTS_DIR="$pert_dir" \
        SMOKE_RUN_ID="all7dims_baseline_${pert_name}_gpu${_gpu}_${task}" \
        SMOKE_T5_EXTRA_EMBEDDINGS="$SMOKE_T5_EXTRA_EMBEDDINGS" \
        SMOKE_T5_ALLOW_STRICT_COMPUTE="$SMOKE_T5_ALLOW_STRICT_COMPUTE" \
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
echo "All selected LIBERO-10 evaluations complete (no clean, no robot_init)."
echo "Results under: $OUTPUT_ROOT"
