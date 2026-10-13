#!/bin/sh
# Attention-location visualization on one of the 68 LIBERO-plus robotinit
# failure cases (memory_system/pointcloud_action/results/tta_screening_meta.json).
#
# Usage (run inside the cosmospolicy conda env, or put its bin on PATH):
#   sh tests/attention_viz/run_robotinit.sh KSCENE3-init006 [--gpu 3] [--every 1] [--ckpt PATH]
#
# Images land in outputs/attn_viz/<tag>/{base|<ckpt name>}/stepNNNN_{wrist,primary}.png
# --every N  save every Nth control step (default 1)
# --ckpt     policy checkpoint (.pt); default is the stock LIBERO base checkpoint
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../.." && pwd)
TAG=${1:?usage: run_robotinit.sh TAG [--gpu N] [--every N] [--ckpt PATH]}
shift

GPU=3; EVERY=1; CKPT=""; CLEAN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)   shift; GPU=$1 ;;
        --every) shift; EVERY=$1 ;;
        --ckpt)  shift; CKPT=$1 ;;
        --clean) CLEAN=1 ;;   # run the base task with default init (libero_10, no perturbation)
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

META="$REPO/memory_system/pointcloud_action/results/tta_screening_meta.json"
[ -f "$META" ] || { echo "missing $META" >&2; exit 1; }
command -v python >/dev/null || { echo "python not on PATH (activate the cosmospolicy env)" >&2; exit 1; }

# Same tag -> env mapping as scripts/eval_sft_68.py
eval "$(python - "$META" "$TAG" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))[sys.argv[2]]
LANGS = {
    "KSCENE3": "turn on the stove and put the moka pot on it",
    "KSCENE4": "put the black bowl in the bottom drawer of the cabinet and close it",
    "KSCENE6": "put the yellow and white mug in the microwave and close it",
    "KSCENE8": "put both moka pots on the stove",
    "LRSCENE2": "put both the alphabet soup and the tomato sauce in the basket",
    "LRSCENE5": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "LRSCENE6": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
}
base_task = meta.get("base_task", meta["task"])
short = sys.argv[2].split("-init")[0]
lang = meta.get("language") or LANGS[short]
pert_task = meta.get("pert_task") or meta.get("suite_task_name", meta["task"])
print("BASE_TASK=%s" % json.dumps(base_task))
print("LANG=%s" % json.dumps(lang))
print("PERT_TASK=%s" % json.dumps(pert_task))
print("INIT_OFF=%d" % int(meta.get("census_abs_init", 0)))
print("SEED=%d" % int(meta.get("seed", 7)))
PY
)"

if [ -n "$CKPT" ]; then
    CKPT=$(readlink -f "$CKPT")   # resolve before cd'ing to scripts/
    RUN=$(basename "$CKPT" .pt)
    export SMOKE_POLICY_CKPT_PATH="$CKPT"
else
    RUN=base
fi
if [ -n "$CLEAN" ]; then
    OUT="$REPO/outputs/attn_viz/${TAG%-init*}-clean/$RUN"
else
    OUT="$REPO/outputs/attn_viz/$TAG/$RUN"
fi
if [ -n "${ATTN_VIZ_INTERVENE:-}" ]; then
    OUT="$OUT-intv${ATTN_VIZ_INTERVENE}lam${ATTN_VIZ_LAMBDA:-0.5}blk${ATTN_VIZ_BLOCK_SEL:-last}"
fi
mkdir -p "$OUT"

cd "$REPO/scripts"
if [ -n "$CLEAN" ]; then
    GPU_ID=$GPU \
    MUJOCO_EGL_DEVICE_ID=$GPU \
    SMOKE_ONLY_CONDITION=clean \
    SMOKE_PAIR_SUITE=libero_10 \
    SMOKE_PAIR_BASE_TASK="$BASE_TASK" \
    SMOKE_PAIR_CLEAN_LANGUAGE="$LANG" \
    SMOKE_NUM_PAIRS=1 \
    SMOKE_SEED="$SEED" \
    SMOKE_RUN_ID="attn_viz_${TAG}_clean" \
    SMOKE_RESULTS_DIR="$OUT/smoke_results" \
    SMOKE_PYTHON_SCRIPT="$HERE/smoke_entry.py" \
    ATTN_VIZ_DIR="$OUT" \
    ATTN_VIZ_EVERY=$EVERY \
    sh run_libero_smoke_test.sh
    exit 0
fi
GPU_ID=$GPU \
MUJOCO_EGL_DEVICE_ID=$GPU \
SMOKE_ONLY_CONDITION=perturb \
SMOKE_PAIR_SUITE=libero_10 \
SMOKE_PAIR_BASE_TASK="$BASE_TASK" \
SMOKE_PAIR_CLEAN_LANGUAGE="$LANG" \
SMOKE_PAIR_PERT_NAME=robot_initial_states \
SMOKE_PAIR_PERT_CATEGORY="Robot Initial States" \
SMOKE_PAIR_PERT_TASK="$PERT_TASK" \
SMOKE_NUM_PAIRS=1 \
SMOKE_SEED="$SEED" \
COSMOS_INIT_STATE_OFFSET="$INIT_OFF" \
SMOKE_RUN_ID="attn_viz_${TAG}" \
SMOKE_RESULTS_DIR="$OUT/smoke_results" \
SMOKE_PYTHON_SCRIPT="$HERE/smoke_entry.py" \
ATTN_VIZ_DIR="$OUT" \
ATTN_VIZ_EVERY=$EVERY \
ATTN_VIZ_INTERVENE="${ATTN_VIZ_INTERVENE:-}" \
ATTN_VIZ_LAMBDA="${ATTN_VIZ_LAMBDA:-0.5}" \
ATTN_VIZ_BBOX="${ATTN_VIZ_BBOX:-}" \
ATTN_VIZ_BBOX2="${ATTN_VIZ_BBOX2:-}" \
ATTN_VIZ_PROPRIO_SRC="${ATTN_VIZ_PROPRIO_SRC:-}" \
ATTN_VIZ_BLOCK_SEL="${ATTN_VIZ_BLOCK_SEL:-last}" \
ATTN_VIZ_DUMP_OBJECTS="${ATTN_VIZ_DUMP_OBJECTS:-}" \
sh run_libero_smoke_test.sh
