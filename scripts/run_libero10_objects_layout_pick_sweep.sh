#!/bin/bash
set -u

REPO=/data1/liu/exp/counterfactual/external/cosmos-policy
cd "$REPO"

source /data1/liu/miniconda3/etc/profile.d/conda.sh
conda activate cosmospolicy

# Configurable via environment variables
NUM_GPUS=${NUM_GPUS:-8}
MAX_CASES_PER_TASK=${MAX_CASES_PER_TASK:-10}
MAX_STEPS=${MAX_STEPS:-200}
POINT_CLOUD_SOURCE=${POINT_CLOUD_SOURCE:-complete}
MEMORY=${MEMORY:-$REPO/memory_system/pointcloud_action/pointcloud_action_memory.pt}
OUTPUT=${OUTPUT:-$REPO/rollouts/libero10_objects_layout_pick_sweep_results.jsonl}
LOG_DIR=${LOG_DIR:-$REPO/rollouts/libero10_objects_layout_pick_sweep_logs}

mkdir -p "$LOG_DIR"

pids=""
cleanup() {
    echo "Stopping all parallel jobs..."
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i \
    python -m memory_system.pointcloud_action.eval.run_libero_plus_objects_layout_sweep \
        --memory "$MEMORY" \
        --max-cases-per-task "$MAX_CASES_PER_TASK" \
        --point-cloud-source "$POINT_CLOUD_SOURCE" \
        --max-steps "$MAX_STEPS" \
        --output "$OUTPUT" \
        --shard-id "$i" \
        --num-shards "$NUM_GPUS" \
        > "$LOG_DIR/shard_${i}.log" 2>&1 &
    pids="$pids $!"
done

wait

cat "$OUTPUT".shard*.jsonl > "$OUTPUT"
echo "Done. Merged results: $OUTPUT"
