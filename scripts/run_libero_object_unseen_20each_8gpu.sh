#!/bin/bash
set -u

REPO=/data1/liu/exp/counterfactual/external/cosmos-policy
cd "$REPO"

source /data1/liu/miniconda3/etc/profile.d/conda.sh
conda activate cosmospolicy

OUTPUT=${OUTPUT:-$REPO/rollouts/libero_object_unseen_20each_results.jsonl}
LOG_DIR=${LOG_DIR:-$REPO/rollouts/libero_object_unseen_20each_logs}
NUM_GPUS=${NUM_GPUS:-8}

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
    python -m memory_system.pointcloud_action.eval.run_libero_object_unseen_20each \
        --shard-id "$i" \
        --num-shards "$NUM_GPUS" \
        --output "$OUTPUT" \
        > "$LOG_DIR/shard_${i}.log" 2>&1 &
    pids="$pids $!"
done

wait

cat "${OUTPUT%.jsonl}".shard*.jsonl > "$OUTPUT"
echo "Done. Merged results: $OUTPUT"
