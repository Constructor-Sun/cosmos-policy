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
#   GPU_ID=5 NUM_CASES=10 SEED=7 OUTPUT_ROOT=/path ROLLOUT_SUBDIR=background \
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
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
TASK_SCOPE=${TASK_SCOPE:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_background_20}
ROLLOUT_SUBDIR=${ROLLOUT_SUBDIR:-background}
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-0}
COSMOS_PHASE_VERIFIER=${COSMOS_PHASE_VERIFIER:-1}
COSMOS_PHASE_RECOVERY=${COSMOS_PHASE_RECOVERY:-1}
COSMOS_FEASIBLE_RECOVERY=${COSMOS_FEASIBLE_RECOVERY:-1}
SMOKE_GL_BACKEND=${SMOKE_GL_BACKEND:-egl}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/scripts/run_libero_smoke_test.py}

# Keep this script focused on phase/feasible detection; disable unrelated
# data-collection machinery unless explicitly needed.
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

echo "======================================================"
echo "LIBERO-10 background_textures evaluation (no clean)"
echo "Output root : $OUTPUT_ROOT"
echo "GPU_ID      : $GPU_ID"
echo "NUM_CASES   : $NUM_CASES"
echo "SEED        : $SEED"
echo "TASK_SCOPE  : $TASK_SCOPE"
echo "Phase       : verifier=$COSMOS_PHASE_VERIFIER recovery=$COSMOS_PHASE_RECOVERY feasible=$COSMOS_FEASIBLE_RECOVERY"
echo "======================================================"

# ---------------------------------------------------------------------------
# Run background perturbations.
# ---------------------------------------------------------------------------
printf '%s\n' "$RUN_PLAN" | while IFS='|' read -r task language pert_name pert_category pert_task variant_tag variant_display; do
    [ -z "$task" ] && continue

    pert_dir="$OUTPUT_ROOT/$task/$pert_name"
    mkdir -p "$pert_dir"

    echo ""
    echo "=== $pert_name: $task ==="
    echo "    variant: $variant_display"
    echo "    results: $pert_dir"

    (
        cd "$REPO_ROOT"
        COSMOS_PHASE_VERIFIER="$COSMOS_PHASE_VERIFIER" \
        COSMOS_PHASE_RECOVERY="$COSMOS_PHASE_RECOVERY" \
        COSMOS_FEASIBLE_RECOVERY="$COSMOS_FEASIBLE_RECOVERY" \
        COSMOS_ROLLOUT_SUBDIR="$ROLLOUT_SUBDIR" \
        COSMOS_SKIP_PLAIN_ROLLOUT=1 \
        COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
        SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
        GPU_ID="$GPU_ID" \
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
        SMOKE_RUN_ID="background_${variant_tag}_gpu${GPU_ID}_${task}" \
        sh "$REPO_ROOT/scripts/run_libero_smoke_test.sh" > "$pert_dir/run.log" 2>&1
    )

    echo "    finished: $pert_dir"
done

echo ""
echo "All LIBERO-10 background evaluations complete."
echo "Results under: $OUTPUT_ROOT"
