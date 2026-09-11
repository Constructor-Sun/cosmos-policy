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
# Default GPU is CUDA device 7; override with GPU_ID=...
#
# Usage:
#   sh scripts/run_libero10_all_7dims_baseline.sh
#
# Useful overrides:
#   GPU_ID=7 NUM_CASES=20 SEED=7 OUTPUT_ROOT=/path sh scripts/run_libero10_all_7dims_baseline.sh
#

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GPU_ID=${GPU_ID:-7}
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_all_7dims_baseline}
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

echo "======================================================"
echo "LIBERO-10 x selected LIBERO-plus dimensions baseline (no clean, no robot_init)"
echo "Output root : $OUTPUT_ROOT"
echo "GPU_ID      : $GPU_ID"
echo "NUM_CASES   : $NUM_CASES"
echo "SEED        : $SEED"
echo "======================================================"

# ---------------------------------------------------------------------------
# Run selected perturbations only.
# ---------------------------------------------------------------------------
printf '%s\n' "$RUN_PLAN" | while IFS='|' read -r task language pert_name pert_category pert_task variant_display; do
    [ -z "$task" ] && continue

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
    echo "    results: $pert_dir"

    (
        cd "$REPO_ROOT"
        COSMOS_INIT_STATE_OFFSET=0 \
        SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
        GPU_ID="$GPU_ID" \
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
        SMOKE_RUN_ID="all7dims_baseline_${pert_name}_gpu${GPU_ID}_${task}" \
        SMOKE_T5_EXTRA_EMBEDDINGS="$SMOKE_T5_EXTRA_EMBEDDINGS" \
        SMOKE_T5_ALLOW_STRICT_COMPUTE="$SMOKE_T5_ALLOW_STRICT_COMPUTE" \
        sh "$REPO_ROOT/scripts/run_libero_smoke_test.sh" > "$pert_dir/run.log" 2>&1
    )

    echo "    finished: $pert_dir"
done

echo ""
echo "All selected LIBERO-10 evaluations complete (no clean, no robot_init)."
echo "Results under: $OUTPUT_ROOT"
