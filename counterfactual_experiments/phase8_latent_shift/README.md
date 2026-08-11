# Phase 8 Latent Shift

## Notes

- `GetLatent.md`: latent-shift dataset design.
- `GetLatentSpecific.md`: concrete first-chunk data generation and evaluation notes.
- `ShiftLearning.md`: latent-shift corrector data, training, evaluation, and limitations.
- `MoEFixer.md`: Mixture-of-Experts corrector design and rollout interpretation.

## Scripts

- `../../bin/phase8_collect_first_chunk_pairs.py`: collect clean/perturb first-chunk latent pairs.
- `../../bin/run_phase8_collect_grid_7500_env0.sh`: multi-GPU first-chunk collection launcher.
- `../../bin/phase8_correction_lib.py`: shared corrector models and runtime correction utilities.
- `../../bin/phase8_train_latent_shift.py`: train latent-shift correctors.
- `../../bin/phase8_eval_latent_correction.py`: evaluate first-chunk latent/action correction.
- `../../bin/phase8_fast_full_rollout.py`: full rollout evaluation with correction.
- `../../run_phase8_fast_full_rollout.sh`: root-level full rollout launcher.
- `../../bin/run_phase8_libero_goal_action_recovery.sh`: LIBERO goal action-recovery launcher.
- `../../bin/phase8_oracle_intervention.py`: oracle true-delta intervention.
- `../../bin/phase8_first_chunk_subspace.py`: first-chunk subspace analysis.
- `../../bin/phase8_eval_pca128_action_recovery.py`: PCA128 action-recovery evaluation.
- `../../bin/phase8_analyze_per_layer_shift.py`: per-layer shift analysis.
- `../../bin/phase8_plot_per_layer_shift.py`: per-layer shift plotting.
- `../../bin/phase8_trajectory_shift_dynamics.py`: trajectory shift dynamics analysis.
- `../../bin/phase8_umap_trajectory.py`: UMAP trajectory visualization.
- `../../bin/collect_shift_embeddings.py`: collect shift embeddings.
- `../../bin/compare_clean_vs_expert.py`: compare clean policy output with expert action.
- `../../bin/compare_model_vs_expert_direct.py`: direct model-vs-expert comparison.
- `../../bin/validate_clean_vs_training.py`: validate clean data against training states.
