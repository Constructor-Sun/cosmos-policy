#!/bin/bash
# Swap baseline WITH episodic data collection, for offline replay.
#
# 与 run_swap.sh 的唯一区别：
#   1) 传 --data_collection True，把每帧 actions/proprio/frame_indices 存成 hdf5
#   2) 默认 10 trials/task（100 episodes），而不是 20
#   3) 输出到 *_collect_seed7/，不覆盖已有基线
#
# 产物：$O/local_logs/rollout_data/*.hdf5 —— 之后所有离线回放的输入。
# 参考：docs/libero-pro/LIBERO_PRO_REPAIR.md
set -euo pipefail

R=$(cd "$(dirname "$0")/../.." && pwd)
S=${SUITE:-libero_10_swap}
G=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
P=${MAX_PARALLEL:-}
N=${NUM_TRIALS_PER_TASK:-10}
D=${SEED:-7}
M=${MODEL:-/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}
O=${OUTPUT_ROOT:-$R/experiments/liberopro/${S}_collect_seed${D}}

export LIBERO_CONFIG_PATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO/configs/libero_pro
export PYTHONPATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO:$R${PYTHONPATH:+:$PYTHONPATH}
export COSMOS_ROLLOUT_SUBDIR="libero-pro/${S}_collect_seed${D}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$R"
mkdir -p "$O/logs" "$O/local_logs"

GA=($G); NG=${#GA[@]}
[ -z "$P" ] && P=$NG

T=$(python -c "from libero.libero import benchmark; import sys; print(chr(10).join(benchmark.get_benchmark_dict()[sys.argv[1]]().get_task_names()))" "$S" | grep -E '^(KITCHEN|LIVING_ROOM|STUDY)_SCENE')

i=0
for t in $T; do
    g=${GA[$((i % NG))]}
    i=$((i + 1))

    CUDA_VISIBLE_DEVICES=$g python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path "$M/Cosmos-Policy-LIBERO-Predict2-2B.pt" \
        --dataset_stats_path "$M/libero_dataset_statistics.json" \
        --t5_text_embeddings_path "$M/libero_t5_embeddings.pkl" \
        --task_suite_name "$S" \
        --unnorm_key libero_10 \
        --task_filter "$t" \
        --num_trials_per_task "$N" \
        --seed "$D" \
        --data_collection True \
        --local_log_dir "$O/local_logs" \
        --run_id_note "$t" >"$O/logs/$t.log" 2>&1 &

    [ $((i % P)) -eq 0 ] && wait || true
done
wait || true

e=$(awk '/^Total episodes:/{s+=$3}END{print s+0}' "$O"/logs/*.log)
t=$(awk '/^Total successes:/{s+=$3}END{print s+0}' "$O"/logs/*.log)
h=$(find "$O" -name '*.hdf5' | wc -l)

echo "Suite: $S  trials/task=$N  seed=$D" | tee "$O/summary.txt"
echo "Total: $t/$e" | tee -a "$O/summary.txt"
echo "hdf5 files: $h (expect $((N * 10)))" | tee -a "$O/summary.txt"

# 没有 hdf5 说明 --data_collection 没生效，整轮作废，早点报出来
if [ "$h" -eq 0 ]; then
    echo "ERROR: no hdf5 produced -- data_collection did not take effect" >&2
    exit 1
fi
