#!/bin/bash
set -euo pipefail

R=$(cd "$(dirname "$0")/../.." && pwd)
S=libero_10_swap
G=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
P=${MAX_PARALLEL:-}
N=${NUM_TRIALS_PER_TASK:-20}
D=${SEED:-7}
M=${MODEL:-/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}
O=${OUTPUT_ROOT:-$R/experiments/liberopro/${S}_seed${D}}

export LIBERO_CONFIG_PATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO/configs/libero_pro
export PYTHONPATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO:$R${PYTHONPATH:+:$PYTHONPATH}
export COSMOS_ROLLOUT_SUBDIR="libero-pro/${S}_seed${D}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$R"
mkdir -p "$O/logs" "$O/local_logs"

GA=($G)
NG=${#GA[@]}
if [ -z "$P" ]; then P=$NG; fi

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
        --local_log_dir "$O/local_logs" \
        --run_id_note "$t" >"$O/logs/$t.log" 2>&1 &

    [ $((i % P)) -eq 0 ] && wait || true
done
wait || true

e=$(awk '/^Total episodes:/{s+=$3}END{print s+0}' "$O"/logs/*.log)
t=$(awk '/^Total successes:/{s+=$3}END{print s+0}' "$O"/logs/*.log)
r=$(awk -v a="$t" -v b="$e" 'BEGIN{print (b ? a/b : 0)}')

echo "Suite: $S" | tee "$O/summary.txt"
echo "Total: $t/$e = $r" | tee -a "$O/summary.txt"