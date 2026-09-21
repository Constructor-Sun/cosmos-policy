#!/bin/bash
# Oracle ready-pose alignment sweep on libero_10_swap (all 10 tasks).
#
# Smoke test (single task, single trial):
#   cd /data1/liu/exp/counterfactual/external/cosmos-policy
#   export LIBERO_CONFIG_PATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO/configs/libero_pro
#   export PYTHONPATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO:$PWD
#   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
#   export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONNOUSERSITE=1
#   CUDA_VISIBLE_DEVICES=0 python scripts/libero_pro/oracle_ready_eval.py \
#     --mode controller --suite libero_10_swap --task STUDY_SCENE1 \
#     --trials 1 --out experiments/liberopro/oracle_smoke
#
# 对照基线：experiments/liberopro/libero_10_swap_seed7/summary.txt = 0/200（10 任务全 0/20）
# 见 docs/libero-pro/LIBERO_PRO_REPAIR.md
set -euo pipefail

R=$(cd "$(dirname "$0")/../.." && pwd)
S=libero_10_swap
G=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
P=${MAX_PARALLEL:-}
N=${NUM_TRIALS_PER_TASK:-20}
D=${SEED:-7}
MODE=${MODE:-controller}   # ik | controller
M=${MODEL:-/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}
O=${OUTPUT_ROOT:-$R/experiments/liberopro/${S}_oracle_seed${D}}

export LIBERO_CONFIG_PATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO/configs/libero_pro
export PYTHONPATH=/data1/liu/exp/counterfactual/external/LIBERO-PRO:$R${PYTHONPATH:+:$PYTHONPATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export COSMOS_ROLLOUT_SUBDIR="libero-pro/oracle_seed${D}"

cd "$R"
mkdir -p "$O/logs" "$O/local_logs"

GA=($G); NG=${#GA[@]}
[ -z "$P" ] && P=$NG

# 任务列表 —— 对齐目标不再手工指定：参照体按技能取自各任务的 BDDL 阶段计划
# （Pick→item，Place/TurnOn→target），开局对齐第一阶段。
#
# KITCHEN_SCENE3 / KITCHEN_SCENE8  第一阶段都是 TurnOn（swap 变体的 BDDL 含
#                 turnon 谓词）：对齐到灶台 ready pose。moka_pot_2 的同型回退
#                 （-> moka_pot_1 记录）只在逐 Pick 对齐（stage 4）时才会用到。
TASKS=(
  "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
  "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
  "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
  "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)

i=0
for t in "${TASKS[@]}"; do
    g=${GA[$((i % NG))]}; i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g python scripts/libero_pro/oracle_ready_eval.py \
        --suite "$S" --task "$t" \
        --trials "$N" --seed "$D" --mode "$MODE" --model "$M" --out "$O" \
        >"$O/logs/$t.log" 2>&1 &
    [ $((i % P)) -eq 0 ] && wait || true
done
wait || true

e=$(awk '/^Total episodes:/{s+=$3}END{print s+0}' "$O"/logs/*.log)
t=$(awk '/^Total successes:/{s+=$3}END{print s+0}' "$O"/logs/*.log)
echo "Oracle ready-pose on $S (seed $D, mode $MODE, ${#PAIRS[@]} tasks)" | tee "$O/summary.txt"
echo "Total: $t/$e" | tee -a "$O/summary.txt"
echo "Baseline (no oracle): 0/200" | tee -a "$O/summary.txt"
