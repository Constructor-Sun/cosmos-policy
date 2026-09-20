#!/bin/bash
# Replay every collected episode and measure rule-vs-truth agreement.
# Needs COSMOS_SKILL_COMPLETION_SHADOW=1 so get_libero_env builds a
# SegmentationRenderEnv (instance segmentation for the target point cloud).
set -euo pipefail

R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"

EP_DIR=${EP_DIR:-$R/experiments/liberopro/libero_10_swap_collect_seed7/local_logs/rollout_data}
OUT=${OUT:-$R/experiments/liberopro/measure_swap_seed7}
JOBS=${JOBS:-8}

export LIBERO_CONFIG_PATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO/configs/libero_pro
export PYTHONPATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO:$R${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export COSMOS_SKILL_COMPLETION_SHADOW=1

PY=/data1/liu/miniconda3/envs/cosmospolicy/bin/python
mkdir -p "$OUT/json" "$OUT/logs"

ls "$EP_DIR"/*.hdf5 | xargs -P "$JOBS" -I{} bash -c '
  f="{}"; R="'"$R"'"; OUT="'"$OUT"'"; PY="'"$PY"'"
  b=$(basename "$f" .hdf5)
  "$PY" "$R/scripts/libero_pro/replay_measure.py" \
      --mode measure --hdf5 "$f" --out "$OUT/json/$b.json" \
      > "$OUT/logs/$b.log" 2>&1
  echo "done: $b"
'

echo "=== json produced: $(ls "$OUT/json"/*.json 2>/dev/null | wc -l) ==="
