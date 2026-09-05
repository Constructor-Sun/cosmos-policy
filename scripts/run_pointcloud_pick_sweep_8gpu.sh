#!/bin/sh
set -eu

REPO=/data1/liu/exp/counterfactual/external/cosmos-policy
PYTHON=/data1/liu/miniconda3/envs/cosmospolicy/bin/python
OUTPUT=$REPO/rollouts/libero10_pick_sweep_waypoint_20_results.jsonl
VIDEO_DIR=$REPO/rollouts/libero10_pick_sweep_waypoint_20_videos
LOG_DIR=$REPO/rollouts/libero10_pick_sweep_waypoint_20_logs

mkdir -p "$VIDEO_DIR" "$LOG_DIR"

pids=""

cleanup() {
    echo ""
    echo "cancelling parallel shards..."
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

for i in $(seq 0 7); do
    gpu=$i
    log="$LOG_DIR/shard_${i}.log"
    (
        cd "$REPO"
        exec env CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON" -m memory_system.pointcloud_action.eval.run_libero10_pick_sweep \
            --max-per-task 20 \
            --max-steps 200 \
            --output "$OUTPUT" \
            --save-video-dir "$VIDEO_DIR" \
            --shard-id "$i" \
            --num-shards 8 \
            > "$log" 2>&1
    ) &
    pids="$pids $!"
    echo "started shard $i on GPU $gpu pid $!"
done

echo "waiting for all shards..."
for pid in $pids; do
    wait "$pid"
done

trap - INT TERM

echo "all shards done"
cat "$REPO"/rollouts/libero10_pick_sweep_waypoint_20_results.shard*.jsonl > "$OUTPUT"
echo "merged results: $OUTPUT"
echo "videos: $VIDEO_DIR/shard_*"
