# Video-Latent Canonicalization

This directory trains small residual adapters from paired perturbed and clean
Cosmos Policy VAE latents. It does not load or update the VAE, DiT, action head,
proprio latents, or value latents.

## Models

All models map `[B, 16, 2, 28, 28]` to the same shape and start as an identity
mapping through a zero-initialized correction output.

| Name | Token mixing | Sample gate |
|---|---|---|
| `mlp` | token-wise MLP | no |
| `transformer` | one self-attention block | no |
| `gated_mlp` | token-wise MLP | yes |
| `glc` | one self-attention block | yes |

The two latent slots are wrist and primary camera observations. Transformer
models use fixed 2D position encodings and learned camera-slot embeddings.

## Data

The default data directory is
`dataset/paired_libero_plus_background_libero10_500`. The loader first looks
for a merged `manifest.jsonl`; if it is absent, it automatically reads all
available `shard-*/manifest.jsonl` files. Each manifest row must have `path`,
`policy_seed`, and preferably `num_frames`. Each sample must contain:

```python
sample["latent"]["clean"]["vae_video"]       # [T, 16, 2, 28, 28]
sample["latent"]["perturbed"]["vae_video"]   # [T, 16, 2, 28, 28]
```

Manifest `train` and `val` labels are ignored. Episodes are grouped by
`policy_seed`, shuffled deterministically with `--split-seed`, and divided 9:1
by default. A policy seed is never shared by training and validation.

## Training

Pack the episode files once into contiguous, memory-mappable shards:

```bash
python counterfactual_experiments/inverse_project/prepare_data.py
```

Multiple perturbation datasets can be packed together by passing multiple
paths after `--data-dir`. If an in-progress dataset has no manifest yet, the
packer indexes its existing `shard-*/samples/**/*.pt` files. Policy seeds are
grouped across all sources before the train/validation split.

The default packed directory is `dataset/paired_libero_plus_background_libero10_500/packed_latents`.
Training processes map the same files with `torch.load(..., mmap=True)`, so four
experiments share the operating-system page cache without repeatedly loading
complete episode files or copying the full dataset into each process.

```bash
python counterfactual_experiments/inverse_project/train.py \
  --output-dir experiments/inverse_project/glc \
  --model glc
```

The loss is:

```text
MSE(model(perturbed), clean)
+ identity_weight * MSE(model(clean), clean)
```

Outputs are `config.json`, `metrics.jsonl`, `latest.pt`, and `best.pt`.
Validation reports baseline MSE, corrected MSE, relative recovery, clean drift,
and mean gate value for gated models.

Resume with:

```bash
python counterfactual_experiments/inverse_project/train.py \
  --output-dir OUTPUT \
  --model glc \
  --resume OUTPUT/latest.pt
```

## Loading

```python
from models import load_model

adapter = load_model("experiments/inverse_project/glc/best.pt", device="cuda")
adapter.eval()
corrected_video_latent = adapter(video_latent)
```
