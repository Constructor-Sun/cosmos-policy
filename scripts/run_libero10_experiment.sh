#!/bin/sh
#
# run_libero10_experiment.sh — Generic launcher for LIBERO-10 perturbation experiments.
#
# Reads the task-perturbation mapping from configs/libero10_experiment_tasks.json
# and constructs the correct SMOKE_PAIR_PERT_TASK for any valid (task, perturbation, variant) combo.
#
# Usage:
#   # Robot initial states — variant 274 on task index 1 (KITCHEN_SCENE4)
#   ./run_libero10_experiment.sh --task-idx 1 --perturbation robot_initial_states --variant 274
#
#   # Same, but by task name
#   ./run_libero10_experiment.sh \
#       --task "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
#       --perturbation robot_initial_states --variant 274
#
#   # Background textures — table_5 on task 0
#   ./run_libero10_experiment.sh --task-idx 0 --perturbation background_textures --variant 5 --bg-kind table
#
#   # Light conditions
#   ./run_libero10_experiment.sh --task-idx 0 --perturbation light_conditions --variant 5
#
#   # Sensor noise — gaussian noise with ID 13
#   ./run_libero10_experiment.sh --task-idx 0 --perturbation sensor_noise --variant 13
#
#   # Language instructions
#   ./run_libero10_experiment.sh --task-idx 0 --perturbation language_instructions --variant 5
#
#   # Dry-run: just print what would be executed
#   ./run_libero10_experiment.sh --task-idx 1 --perturbation robot_initial_states --variant 274 --dry-run
#
# Overridable environment variables (same semantics as run_libero10_robotinit_20.sh):
#   NUM_CASES, SEED, GPU_ID, OUTPUT_ROOT, COSMOS_ROLLOUT_SUBDIR,
#   COSMOS_INIT_STATE_OFFSET, COSMOS_DATA_COLLECTION, COSMOS_VECTOR_DB,
#   COSMOS_VECTOR_DB_DIR, COSMOS_ONLINE_DETECTION, COSMOS_TARGET_DEMOS,
#   SMOKE_PYTHON_SCRIPT

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# -------------------------------------------------------------------
# Defaults (override via environment or CLI)
# -------------------------------------------------------------------
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-7}
GPU_ID=${GPU_ID:-7}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/experiments/libero10_experiment}
SMOKE_PYTHON_SCRIPT=${SMOKE_PYTHON_SCRIPT:-$REPO_ROOT/run_libero_smoke_test.py}

# Init state offset (default 0 = first init state)
COSMOS_INIT_STATE_OFFSET=${COSMOS_INIT_STATE_OFFSET:-0}

# -------------------------------------------------------------------
# Parse CLI arguments
# -------------------------------------------------------------------
TASK_IDX=""
TASK_NAME=""
PERTURBATION=""
VARIANT=""
BG_KIND=""
DRY_RUN=false

usage() {
    cat >&2 << 'EOF'
Usage: run_libero10_experiment.sh --perturbation <TYPE> --variant <VAL> [--task-idx N | --task NAME] [options]

Required:
  --perturbation TYPE    Perturbation type key (see list below)
  --variant VAL          Variant parameter value (integer for most types)

Task selection (one of):
  --task-idx N           Task index (0-9), ordered as in libero10_experiment_tasks.json
  --task NAME            Full base task name

Perturbation-specific:
  --bg-kind table|tb     (background_textures only) which background kind to use

Options:
  --num-cases N          Number of rollout trials (default: $NUM_CASES)
  --seed N               Random seed (default: $SEED)
  --gpu N                GPU device ID (default: $GPU_ID)
  --output-root DIR      Results output directory (default: $OUTPUT_ROOT)
  --init-state-offset N  Offset into init states (default: $COSMOS_INIT_STATE_OFFSET)
  --dry-run              Print the resolved config without running

Perturbation type keys:
  robot_initial_states   Variant = initstate number (e.g., 274)
  background_textures    Variant = background ID (e.g., 5), use --bg-kind table|tb
  light_conditions       Variant = light ID (e.g., 5)
  sensor_noise           Variant = noise ID (e.g., 13 for gaussian severity 3)
  language_instructions  Variant = language ID (e.g., 5)

Examples:
  # Robot init state 274 on KITCHEN_SCENE4 (task index 1)
  $0 --task-idx 1 --perturbation robot_initial_states --variant 274

  # Background table_5 on KITCHEN_SCENE3 (task index 0)
  $0 --task-idx 0 --perturbation background_textures --variant 5 --bg-kind table

  # Light condition 5 on the bowl-in-drawer task
  $0 --task "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      --perturbation light_conditions --variant 5

  # Dry-run to verify
  $0 --task-idx 1 --perturbation robot_initial_states --variant 274 --dry-run
EOF
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --task-idx)
            TASK_IDX="$2"; shift 2 ;;
        --task)
            TASK_NAME="$2"; shift 2 ;;
        --perturbation)
            PERTURBATION="$2"; shift 2 ;;
        --variant)
            VARIANT="$2"; shift 2 ;;
        --bg-kind)
            BG_KIND="$2"; shift 2 ;;
        --num-cases)
            NUM_CASES="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --output-root)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --init-state-offset)
            COSMOS_INIT_STATE_OFFSET="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --help|-h)
            usage ;;
        *)
            echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Validate required args
if [ -z "$PERTURBATION" ] || [ -z "$VARIANT" ]; then
    echo "ERROR: --perturbation and --variant are required" >&2
    usage
fi
if [ -z "$TASK_IDX" ] && [ -z "$TASK_NAME" ]; then
    echo "ERROR: --task-idx or --task is required" >&2
    usage
fi

# -------------------------------------------------------------------
# Resolve experiment config via Python helper
# -------------------------------------------------------------------
RESOLVE_SCRIPT="$REPO_ROOT/bin/resolve_experiment_config.py"

if [ ! -f "$RESOLVE_SCRIPT" ]; then
    echo "ERROR: resolve_experiment_config.py not found at $RESOLVE_SCRIPT" >&2
    exit 1
fi

# Build resolver args using set -- to handle spaces correctly
set -- --perturbation "$PERTURBATION" --variant "$VARIANT"
if [ -n "$TASK_IDX" ]; then
    set -- "$@" --task-idx "$TASK_IDX"
fi
if [ -n "$TASK_NAME" ]; then
    set -- "$@" --task "$TASK_NAME"
fi
if [ -n "$BG_KIND" ]; then
    set -- "$@" --bg-kind "$BG_KIND"
fi
if $DRY_RUN; then
    set -- "$@" --dry-run
fi

# Run resolver and capture output
RESOLVED=$(python3 "$RESOLVE_SCRIPT" "$@") || exit 1

# Evaluate the resolved variables
eval "$RESOLVED"

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Derive per-perturbation sub-directory name
# -------------------------------------------------------------------
PERT_SHORT=$(echo "$PERTURBATION" | tr '_' '-')
TASK_SHORT=$(echo "$SMOKE_PAIR_BASE_TASK" | sed 's/^\([A-Z_]*_[A-Z]*[0-9]*\)_.*/\1/')
RUN_DIR="$OUTPUT_ROOT/${TASK_SHORT}_${PERT_SHORT}_${VARIANT}"
RUN_MODE=normal
if [ "${COSMOS_ONLINE_DETECTION:-}" = "1" ]; then
    RUN_MODE=online-auto
    RUN_DIR="${RUN_DIR}_online-auto"
fi

# Override COSMOS_ROLLOUT_SUBDIR default to include perturbation type
COSMOS_ROLLOUT_SUBDIR=${COSMOS_ROLLOUT_SUBDIR:-$(date +%m-%d)}

# -------------------------------------------------------------------
# Dry-run: print what would run
# -------------------------------------------------------------------
if $DRY_RUN; then
    echo ""
    echo "=== Resolved Configuration ==="
    echo "SMOKE_PAIR_SUITE:           $SMOKE_PAIR_SUITE"
    echo "SMOKE_PAIR_BASE_TASK:       $SMOKE_PAIR_BASE_TASK"
    echo "SMOKE_PAIR_CLEAN_LANGUAGE:  $SMOKE_PAIR_CLEAN_LANGUAGE"
    echo "SMOKE_PAIR_PERT_NAME:       $SMOKE_PAIR_PERT_NAME"
    echo "SMOKE_PAIR_PERT_CATEGORY:   $SMOKE_PAIR_PERT_CATEGORY"
    echo "SMOKE_PAIR_PERT_TASK:       $SMOKE_PAIR_PERT_TASK"
    echo ""
    echo "NUM_CASES:                  $NUM_CASES"
    echo "SEED:                       $SEED"
    echo "GPU_ID:                     $GPU_ID"
    echo "INIT_STATE_OFFSET:          $COSMOS_INIT_STATE_OFFSET"
    echo "OUTPUT_ROOT:                $RUN_DIR"
    echo "ROLLOUT_SUBDIR:             ${COSMOS_ROLLOUT_SUBDIR:-(unset)}"
    if [ "$RUN_MODE" = "online-auto" ]; then
        echo "INJECTION_MODE:             automatic online detection + correction"
    else
        echo "INJECTION_MODE:             none (normal rollout)"
    fi
    echo ""
    echo "=== Shell command that would execute ==="
    echo "cd $REPO_ROOT && COSMOS_ROLLOUT_SUBDIR=... sh ./run_libero_smoke_test.sh"
    exit 0
fi

# -------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------
mkdir -p "$RUN_DIR" "$OUTPUT_ROOT/tmp-${PERT_SHORT}"

echo "=== Running experiment ==="
echo "Task:         $SMOKE_PAIR_BASE_TASK"
echo "Perturbation: $SMOKE_PAIR_PERT_NAME"
echo "Variant:      $VARIANT"
echo "Results:      $RUN_DIR"
echo ""

cd "$REPO_ROOT"

# Export optional vars only when non-empty (avoids int("") crashes)
maybe_export() {
    eval "val=\${$1:-}"
    if [ -n "$val" ]; then
        export "$1=$val"
    else
        # An exported-but-empty parent variable still appears in os.environ.
        # Remove it so Python defaults work as intended.
        unset "$1"
    fi
}
maybe_export COSMOS_ROLLOUT_SUBDIR
maybe_export COSMOS_DATA_COLLECTION
maybe_export COSMOS_VECTOR_DB
maybe_export COSMOS_VECTOR_DB_DIR
maybe_export COSMOS_ONLINE_DETECTION
maybe_export COSMOS_TARGET_DEMOS

# Legacy manual/dry-run injection controls are deliberately unsupported.
unset COSMOS_OFFSET_START_T COSMOS_OFFSET_AMOUNT COSMOS_OFFSET_DURATION COSMOS_ONLINE_INJECT

COSMOS_SKIP_PLAIN_ROLLOUT=1 \
COSMOS_INIT_STATE_OFFSET="$COSMOS_INIT_STATE_OFFSET" \
SMOKE_DATA_COLLECTION="${COSMOS_DATA_COLLECTION:-}" \
TMPDIR="$OUTPUT_ROOT/tmp-${PERT_SHORT}" \
SMOKE_PYTHON_SCRIPT="$SMOKE_PYTHON_SCRIPT" \
GPU_ID="$GPU_ID" \
SMOKE_GL_BACKEND=egl \
SMOKE_ONLY_CONDITION=perturb \
SMOKE_PAIR_SUITE="$SMOKE_PAIR_SUITE" \
SMOKE_PAIR_BASE_TASK="$SMOKE_PAIR_BASE_TASK" \
SMOKE_PAIR_CLEAN_LANGUAGE="$SMOKE_PAIR_CLEAN_LANGUAGE" \
SMOKE_PAIR_PERT_NAME="$SMOKE_PAIR_PERT_NAME" \
SMOKE_PAIR_PERT_CATEGORY="$SMOKE_PAIR_PERT_CATEGORY" \
SMOKE_PAIR_PERT_TASK="$SMOKE_PAIR_PERT_TASK" \
SMOKE_NUM_PAIRS="$NUM_CASES" \
SMOKE_SEED="$SEED" \
SMOKE_RESULTS_DIR="$RUN_DIR" \
SMOKE_RUN_ID="${PERT_SHORT}_${VARIANT}_${RUN_MODE}_gpu${GPU_ID}" \
sh ./run_libero_smoke_test.sh

echo ""
echo "Experiment complete: $RUN_DIR"
