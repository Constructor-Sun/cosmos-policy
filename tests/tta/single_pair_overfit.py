#!/usr/bin/env python
"""Implementation-faithfulness test: overfit ONE (pair, chunk) for 100 steps
under two objectives and track that chunk's per-side errors:

  dpo: the real training loss (reference-corrected Diffusion-DPO)
  bc : pure chosen-side denoising loss (what SFT/behavior-cloning optimizes)

Faithful implementation must show:
  bc : E_c drops fast (single-chunk overfit), E_r ~unchanged
  dpo: E_c drops, E_r rises (both direct pressures visible)

If E_c does NOT drop under bc, the forward/gradient path itself is broken.
If E_c rises under dpo on its OWN trained chunk, the DPO composition is
broken. Eval noise is fixed-seed so curves are comparable across modes;
training noise is fresh per step (as in the real loop).
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from memory_system.tta.dataset import PreferenceDataset  # noqa: E402
from memory_system.tta.model import TTADPOModel, load_policy_model  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "dpo"
GPU = sys.argv[2] if len(sys.argv) > 2 else "7"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

BASE = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
full = json.loads(Path(REPO / "memory_system/tta/results/manifests/dpo_pairs_v2.json").read_text())
tmp = REPO / "experiments/tta/redo68/overfit_manifest_1pair.json"
tmp.write_text(json.dumps({**full, "pairs": full["pairs"][:1]}))

ds = PreferenceDataset(
    manifest_path=str(tmp),
    t5_text_embeddings_path=str(REPO / "training/tta_sft_metadata/t5_embeddings.pkl"),
    dataset_stats_path=str(REPO / "training/tta_sft_metadata/dataset_statistics.json"),
    strict_labels=True,
)
model = load_policy_model(BASE)
ttm = TTADPOModel(model, beta=0.1, lora_rank=8, lora_alpha=16, base_checkpoint=BASE)
batch = TTADPOModel.batch_to_device(default_collate([ds[0]]))
opt = torch.optim.AdamW(ttm.lora_parameters(), lr=1e-5)


def eval_fixed_noise():
    """E_c / E_r on the trained chunk with FIXED eval noise."""
    with torch.no_grad():
        torch.manual_seed(777)
        prepared = ttm.prepare_inputs(batch)
        e_pol, e_ref = ttm.errors(batch, prepared=prepared)
    e_pol = e_pol.tolist(); e_ref = e_ref.tolist()
    return e_pol[0], e_pol[1], e_ref[0], e_ref[1]


ec0, er0, refc0, refr0 = eval_fixed_noise()
print(f"[{MODE}] step=0 E_c={ec0:.5f} E_r={er0:.5f} (ref {refc0:.5f}/{refr0:.5f})", flush=True)
for step in range(1, STEPS + 1):
    if MODE == "dpo":
        loss, margin, _ = ttm.dpo_forward(batch)
    else:
        e_pol, _e_ref = ttm.errors(batch)
        loss = e_pol.view(-1, 2)[:, 0].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ttm.lora_parameters(), max_norm=1e9)
    opt.step()
    if step % 10 == 0:
        ec, er, refc, refr = eval_fixed_noise()
        print(f"[{MODE}] step={step} E_c={ec:.5f} E_r={er:.5f} loss={float(loss):.5f}", flush=True)
