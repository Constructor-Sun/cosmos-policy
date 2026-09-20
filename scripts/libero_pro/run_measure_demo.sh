#!/bin/bash
# Measure rule-vs-truth on LIBERO-10 demonstration trajectories.
#
# These demos carry full simulator states, so every frame is restored exactly
# (no action replay, no VLA).  Ten tasks x N demos, all successful by
# construction -> dense positive samples across every task.
#
# OPEN_FRAMES sets ReleaseSkillCompletion.required_open_frames so the Place
# confirmation window can be A/B tested without editing the runtime rule.
set -euo pipefail

R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"

DEMO_DIR=${DEMO_DIR:-$R/LIBERO-Cosmos-Policy/success_only/libero_10_regen}
OUT=${OUT:-$R/experiments/liberopro/measure_demo_libero10}
N_DEMOS=${N_DEMOS:-10}
OPEN_FRAMES=${OPEN_FRAMES:-16}
JOBS=${JOBS:-10}

# NOTE: libero_10 is a base suite whose BDDL files live in LIBERO-plus, so
# LIBERO_CONFIG_PATH must NOT point at the LIBERO-PRO config here.
unset LIBERO_CONFIG_PATH
export PYTHONPATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO:$R${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export COSMOS_SKILL_COMPLETION_SHADOW=1   # SegmentationRenderEnv for point clouds

export PY=/data1/liu/miniconda3/envs/cosmospolicy/bin/python
export R OUT N_DEMOS OPEN_FRAMES

mkdir -p "$OUT/json" "$OUT/logs"
echo "open_frames=$OPEN_FRAMES  n_demos=$N_DEMOS  -> $OUT"

ls "$DEMO_DIR"/*_demo.hdf5 | xargs -P "$JOBS" -I{} "$R/scripts/libero_pro/_demo_worker.sh" {}

echo "=== json produced: $(ls "$OUT/json"/*.json 2>/dev/null | wc -l) ==="
