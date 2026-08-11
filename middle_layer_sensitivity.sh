mkdir -p experiments/phase8_directional_action_sensitivity/all4_suites_7pert/logs

for SHARD in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES=$SHARD \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  python bin/phase8_directional_action_sensitivity.py \
    --all-suites-grid \
    --layers 14 27 \
    --target action \
    --init-state-indices 0 \
    --jvp-epsilon 0.02 \
    --random-baselines 4 \
    --hook-mode pre \
    --t5-extra-embeddings \
      experiments/libero_plus_language_t5_main_suites.pkl \
      experiments/libero_plus_language_t5_libero10.pkl \
    --num-shards 7 \
    --shard-index $SHARD \
    --output-dir experiments/phase8_directional_action_sensitivity/all4_suites_7pert/shard${SHARD} \
    > experiments/phase8_directional_action_sensitivity/all4_suites_7pert/logs/shard${SHARD}.log 2>&1 &
done

wait