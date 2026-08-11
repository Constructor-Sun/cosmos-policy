# Counterfactual Experiments

This directory groups the counterfactual experiment notes by research question.
The runnable scripts remain in `../bin/` to avoid breaking existing commands.

## Groups

- `phase1_smoke/`: clean vs perturb rollout setup and preserved/flipped labels.
- `phase2_angular_shift/`: hidden/action angular shift capture and analysis.
- `phase3_subspace_recovery/`: perturb-specific subspace and mean-shift recovery.
- `hessian_curvature/`: denoise Hessian, Taylor validation, and intervention analysis.
- `action_sensitive_subspace/`: action-sensitive subspace and gradient alignment.
- `phase8_latent_shift/`: first-chunk latent-shift data, corrector training, MoE, and rollout evaluation.

## Files Kept In Place

The original Cosmos Policy project files are intentionally left at the repository root
and under `cosmos_policy/`. The `bin/` scripts are also left in place for now.
