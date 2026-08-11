# Paired Camera Data Collection
Goal: collect clean/perturbed camera pairs for the adaptive AdaLN inverse adapter
without running perturbed closed-loop rollouts.

## Principle
Run only the clean trajectory. At each selected timestep, render two views from
the same simulator state:
```text
clean_obs_t = render(state_t, clean_camera)
pert_obs_t  = render(state_t, camera_viewpoint_variant)
```
Robot state, object state, action history, task, and timestep stay fixed.

## Why
Closed-loop clean and perturbed rollouts diverge after the first action. Later
frames mix camera effects with policy errors and state drift. This collection
isolates camera perturbation.

## Sample Fields
Each saved sample should contain:
```text
clean.front, clean.wrist              [T, H, W, 3]
perturbed.front, perturbed.wrist      [T, H, W, 3]
state                                 [T, state_dim]
action_context                        [T, action_dim] or chunked action input
instruction                           string
suite, base_task, init_state_index
policy_seed, env_seed, clean_success
camera_tuple                          azimuth/elevation/radius/x_offset/y_offset
```
Raw images are enough if training dynamically runs Cosmos preprocessing and VAE
encoding. Cached processed tensors or latents are optional speedups.

## Filtering
Use only clean-success trajectories for the first training set:
```text
clean_success == true
```
There is no perturbed success label because the perturbed branch is render-only.

## Camera Coverage
`_view_50_0_100_0_0_initstate_0` is only a smoke-test/default view. Formal
experiments should sample `Camera Viewpoints` from:
```text
../LIBERO-plus/libero/libero/benchmark/task_classification.json
```
Existing Phase 8 code expects 450 unique `initstate_0` camera tuples. Use a
held-out split, for example:
```text
train cameras: 360
val cameras:    45
test cameras:   45
```

## Code Changes Needed
1. Add a collector such as `bin/collect_paired_camera_render_dataset.py`.
2. Reuse Phase 8 task selection, camera parsing/splitting, metadata writing,
   and Cosmos preprocessing conventions.
3. Replace perturbed rollout execution with perturbed rendering from the clean
   simulator state.
4. Render strategy:
```text
preferred: switch camera parameters in the same env, render, restore
fallback: copy clean simulator state into a camera-variant env, render there
```
5. Save:
```text
manifest.jsonl
summary.json
samples/<suite>/<base_task>/camera_viewpoints/<sample_id>.pt
```

## Minimal Smoke Test
```text
1 task
5 clean-success trajectories
1 camera tuple
first 8-16 frames per trajectory
```
Then scale to multiple camera tuples and held-out camera validation.

## First Training Check
Train only `latent_adaln_adapter`:
```text
teacher = DiT_original(clean_input)
student = DiT_adaptive(perturbed_input)
loss = MSE(student, teacher)
```
Report alignment MSE against the frozen original perturbed baseline before full
rollout evaluation.
