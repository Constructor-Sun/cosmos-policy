#!/usr/bin/env python
"""Independent cross-check of the L1 probe: recompute per-side errors for one
pair chunk via the EVAL-SIDE injection path (attach_adapter on a fresh model)
plus a manual forward, and compare against the probe path (TTADPOModel
toggle). Two different code paths, same chunk, same seeded noise: if they
agree, the probe's numbers are not an artifact of its own machinery.

No training, no writes except this script's stdout.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from memory_system.tta.dataset import PreferenceDataset  # noqa: E402
from memory_system.tta.model import (  # noqa: E402
    TTADPOModel, attach_adapter, load_policy_model, share_noise_across_pairs,
)

BASE = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
ADAPTER = sys.argv[1]
META = str(REPO / "training/tta_sft_metadata")
GPU = sys.argv[2] if len(sys.argv) > 2 else "1"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

full = json.loads(Path(REPO / "memory_system/tta/results/manifests/dpo_pairs_v2.json").read_text())
tmp = REPO / "experiments/tta/redo68/crosscheck_manifest_1pair.json"
tmp.write_text(json.dumps({**full, "pairs": full["pairs"][:1]}))

ds = PreferenceDataset(
    manifest_path=str(tmp),
    t5_text_embeddings_path=str(META + "/t5_embeddings.pkl"),
    dataset_stats_path=str(META + "/dataset_statistics.json"),
    strict_labels=True,
)
batch = TTADPOModel.batch_to_device(default_collate([ds[0]]))
flat = {k: (v.flatten(0, 1) if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == 2 else v)
        for k, v in batch.items()}

SCALARS = {"world_model_sample_mask", "value_function_sample_mask", "rollout_data_mask",
           "action_latent_idx", "value_latent_idx", "current_proprio_latent_idx",
           "current_wrist_image_latent_idx", "current_image_latent_idx",
           "future_proprio_latent_idx", "future_wrist_image_latent_idx",
           "future_image_latent_idx", "fps"}
fixed = {}
for k, v in flat.items():
    if k in SCALARS and isinstance(v, torch.Tensor):
        v = v.reshape(v.shape[0], -1)
        if v.shape[1] == 1:
            v = v.squeeze(1)
    fixed[k] = v
flat = fixed

KW = dict(
    action_chunk=flat["actions"], action_indices=flat["action_latent_idx"],
    proprio=flat["proprio"], current_proprio_indices=flat["current_proprio_latent_idx"],
    future_proprio=flat["future_proprio"], future_proprio_indices=flat["future_proprio_latent_idx"],
    future_wrist_image_indices=flat["future_wrist_image_latent_idx"], future_wrist_image2_indices=None,
    future_image_indices=flat["future_image_latent_idx"], future_image2_indices=None,
    rollout_data_mask=flat["rollout_data_mask"], world_model_sample_mask=flat["world_model_sample_mask"],
    value_function_sample_mask=flat["value_function_sample_mask"],
    value_function_return=flat["value_function_return"], value_indices=flat["value_latent_idx"],
)


def errors_of(model, x0, cond, sigma, eps):
    out, _, _, _ = model.compute_loss_with_epsilon_and_sigma(x0, cond, eps, sigma, **KW)
    bidx = torch.arange(out["edm_loss_per_frame"].shape[0], device=x0.device)
    return out["edm_loss_per_frame"][bidx, flat["action_latent_idx"]].float()


# ---- path A: probe machinery (TTADPOModel toggle) ----
model_a = load_policy_model(BASE)
ttm = TTADPOModel(model_a, beta=0.1, lora_rank=8, lora_alpha=16, base_checkpoint=BASE)
ttm.load_adapter(ADAPTER)
torch.manual_seed(42)
prepared = ttm.prepare_inputs(batch)
with torch.no_grad():
    e_pol_a, e_ref_a = ttm.errors(batch, prepared=prepared)
print(f"[pathA probe ] e_policy={e_pol_a.tolist()} e_reference={e_ref_a.tolist()} "
      f"d_chosen={float(e_pol_a[0]-e_ref_a[0]):+.5f} d_rejected={float(e_pol_a[1]-e_ref_a[1]):+.5f}")
del model_a, ttm, prepared
torch.cuda.empty_cache()

# ---- path B: eval-side attach_adapter + separate reference model ----
model_pol = load_policy_model(BASE)
attach_adapter(model_pol, ADAPTER)
torch.manual_seed(42)
_, x0, cond = model_pol.get_data_and_condition(flat)
sigma, eps = model_pol.draw_training_sigma_and_epsilon(x0.size(), cond)
sigma, eps = share_noise_across_pairs(sigma.to(**model_pol.tensor_kwargs),
                                      eps.to(**model_pol.tensor_kwargs))
x0 = x0.to(**model_pol.tensor_kwargs)
with torch.no_grad():
    e_pol_b = errors_of(model_pol, x0, cond, sigma, eps)
del model_pol
torch.cuda.empty_cache()

model_ref = load_policy_model(BASE)
with torch.no_grad():
    e_ref_b = errors_of(model_ref, x0, cond, sigma, eps)
del model_ref
torch.cuda.empty_cache()

print(f"[pathB attach ] e_policy={e_pol_b.tolist()} e_reference={e_ref_b.tolist()} "
      f"d_chosen={float(e_pol_b[0]-e_ref_b[0]):+.5f} d_rejected={float(e_pol_b[1]-e_ref_b[1]):+.5f}")
dc_a, dr_a = float(e_pol_a[0] - e_ref_a[0]), float(e_pol_a[1] - e_ref_a[1])
dc_b, dr_b = float(e_pol_b[0] - e_ref_b[0]), float(e_pol_b[1] - e_ref_b[1])
ok = abs(dc_a - dc_b) < 1e-2 and abs(dr_a - dr_b) < 1e-2
print(f"[crosscheck] {'AGREE' if ok else 'DISAGREE'}: "
      f"d_chosen {dc_a:+.5f} vs {dc_b:+.5f}; d_rejected {dr_a:+.5f} vs {dr_b:+.5f}")
