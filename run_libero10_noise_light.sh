SAME_GPU=${SAME_GPU:-4} \
RUN_ROOT=${RUN_ROOT:-"experiments/libero10_light_noise_$(date +%Y%m%d_%H%M%S)"} \
EVAL_CONDITION=${EVAL_CONDITION:-light} \
NUM_CASES=${NUM_CASES:-20} \
POLICY_CKPT_PATH=${POLICY_CKPT_PATH:-outputs/cosmos_policy/counterfactual_direct_sft/light_libero10_500/checkpoints/iter_000000100/model} \
LATENT_ADAPTER_CHECKPOINT=${LATENT_ADAPTER_CHECKPOINT:-} \
sh -c '
set -eu

TASK=KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it

mkdir -p "$RUN_ROOT"

case "$EVAL_CONDITION" in
light)
    echo "Running step-100 checkpoint on light_5, GPU $SAME_GPU..."
    GPU_ID="$SAME_GPU" \
    SMOKE_POLICY_CKPT_PATH="$POLICY_CKPT_PATH" \
    SMOKE_GL_BACKEND=egl \
    SMOKE_ONLY_CONDITION=background \
    SMOKE_PAIR_SUITE=libero_10 \
    SMOKE_PAIR_BASE_TASK="$TASK" \
    SMOKE_PAIR_PERT_NAME=light_conditions \
    SMOKE_PAIR_PERT_CATEGORY="Light Conditions" \
    SMOKE_PAIR_PERT_TASK="${TASK}_light_5" \
    SMOKE_NUM_PAIRS="${NUM_CASES:-20}" \
    SMOKE_SEED=7 \
    SMOKE_RESULTS_DIR="$RUN_ROOT/light" \
    SMOKE_RUN_ID="step100_light_5" \
    SMOKE_LATENT_ADAPTER="$LATENT_ADAPTER_CHECKPOINT" \
    sh ./run_libero_smoke_test.sh >"$RUN_ROOT/light.log" 2>&1
    ;;
noise)
    echo "Running step-100 checkpoint on noise_5, GPU $SAME_GPU..."
    GPU_ID="$SAME_GPU" \
    SMOKE_POLICY_CKPT_PATH="$POLICY_CKPT_PATH" \
    SMOKE_GL_BACKEND=egl \
    SMOKE_ONLY_CONDITION=background \
    SMOKE_PAIR_SUITE=libero_10 \
    SMOKE_PAIR_BASE_TASK="$TASK" \
    SMOKE_PAIR_PERT_NAME=sensor_noise \
    SMOKE_PAIR_PERT_CATEGORY="Sensor Noise" \
    SMOKE_PAIR_PERT_TASK="${TASK}_view_0_0_100_0_0_initstate_0_noise_5" \
    SMOKE_NUM_PAIRS="${NUM_CASES:-20}" \
    SMOKE_SEED=7 \
    SMOKE_RESULTS_DIR="$RUN_ROOT/noise" \
    SMOKE_RUN_ID="step100_noise_5" \
    SMOKE_LATENT_ADAPTER="$LATENT_ADAPTER_CHECKPOINT" \
    sh ./run_libero_smoke_test.sh >"$RUN_ROOT/noise.log" 2>&1
    ;;
*)
    echo "EVAL_CONDITION must be light or noise, got: $EVAL_CONDITION" >&2
    exit 2
    ;;
esac

python -c "
import json
from pathlib import Path

root = Path(\"$RUN_ROOT\")
condition = \"$EVAL_CONDITION\"
name = \"light_conditions\" if condition == \"light\" else \"sensor_noise\"
episodes = json.loads((root / condition / name / \"episodes.json\").read_text())
successes = sum(bool(x[\"success\"]) for x in episodes)

print()
print(\"========== Same-GPU Results ==========\")
print(f\"GPU:            $SAME_GPU\")
print(f\"Checkpoint:     $POLICY_CKPT_PATH\")
print(f\"{condition}:          {successes}/{len(episodes)} = {successes/len(episodes):.4f} ({100*successes/len(episodes):.1f}%)\")
print(f\"Results:        {root}\")
print(\"======================================\")
"
'
