#!/usr/bin/env python
"""Pre-training hard gate for the 100-step DPO run (user-requested, 2026-09-13).

Four checks on the REAL manifest data + REAL base checkpoint, single GPU:

1. forward checksum — the fresh adapter has B=0, so policy == reference and
   the DPO loss has the analytic value log(2) ~= 0.6931 with margin 0.
   Any deviation means the forward plumbing is broken.
2. backward — LoRA gradients must be NONZERO and finite. Exactly-zero grads
   are the docs/sft/sft.md dead-init signature (A=B=0 deadlock); the correct
   init has A=kaiming (nonzero), so d(loss)/d(B) != 0. Frozen base must have
   no grads at all.
3. 3 optimizer steps — loss/margin stay finite, peak memory recorded
   (budget 20 GB per the training contract).
4. saves a dummy (slightly-stepped) adapter for the merge-chain smoke.

Exits nonzero on any violation. Uses a 2-pair temp manifest to keep the
episode-decode cost down; the loading code path is identical to training.
"""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from memory_system.tta.dataset import PreferenceDataset  # noqa: E402
from memory_system.tta.dpo_train import pick_gpu  # noqa: E402
from memory_system.tta.model import TTADPOModel, load_policy_model  # noqa: E402

DEFAULT_BASE = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_MANIFEST = str(REPO / "memory_system/tta/results/manifests/dpo_pairs_v3.json")
DEFAULT_META = str(REPO / "training/tta_sft_metadata")
OUT_DIR = REPO / "experiments/tta/redo68"


def fail(msg):
    print(f"[preflight] FAIL: {msg}")
    sys.exit(1)


def main():
    gpu = pick_gpu(20.0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[preflight] GPU {gpu}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = json.loads(Path(DEFAULT_MANIFEST).read_text())
    tmp_manifest = OUT_DIR / "preflight_manifest_2pairs.json"
    tmp = dict(full)
    tmp["pairs"] = full["pairs"][:2]
    tmp_manifest.write_text(json.dumps(tmp))

    ds = PreferenceDataset(
        manifest_path=str(tmp_manifest),
        t5_text_embeddings_path=str(Path(DEFAULT_META) / "t5_embeddings.pkl"),
        dataset_stats_path=str(Path(DEFAULT_META) / "dataset_statistics.json"),
        strict_labels=True,
    )
    model = load_policy_model(DEFAULT_BASE)
    ttm = TTADPOModel(model, beta=0.1, lora_rank=8, lora_alpha=16,
                      base_checkpoint=DEFAULT_BASE)
    batch = TTADPOModel.batch_to_device(default_collate([ds[0]]))

    # ---- check 1: forward checksum (untrained adapter -> loss == log 2) ----
    with torch.no_grad():
        prepared = ttm.prepare_inputs(batch)
        loss0, margin0, _ = ttm.dpo_forward(batch, prepared=prepared)
    print(f"[preflight] checksum: loss={float(loss0):.6f} (expect ~{math.log(2):.6f}) "
          f"margin={float(margin0):+.3e}")
    if abs(float(loss0) - math.log(2)) > 1e-2:
        fail(f"forward checksum: loss {float(loss0):.6f} != log2 within 1e-2")

    # ---- check 2: backward — nonzero finite LoRA grads, frozen base ----
    loss1, margin1, _ = ttm.dpo_forward(batch)
    loss1.backward()
    lora_grads = [p.grad for p in ttm.lora_parameters() if p.grad is not None]
    if not lora_grads:
        fail("no LoRA gradients at all")
    max_grad = max(float(g.abs().max()) for g in lora_grads)
    zero_grads = sum(1 for g in lora_grads if float(g.abs().max()) == 0)
    if max_grad <= 0 or not all(torch.isfinite(g).all() for g in lora_grads):
        fail(f"LoRA grads dead or non-finite: max={max_grad}")
    if max_grad == 0 or zero_grads == len(lora_grads):
        fail("ALL LoRA grads exactly zero — docs/sft/sft.md dead-init signature")
    base_grads = [p.grad for n, p in ttm.model.net.named_parameters()
                  if "lora_" not in n and p.grad is not None]
    if base_grads:
        fail(f"{len(base_grads)} frozen base params received gradients")
    print(f"[preflight] backward: {len(lora_grads)} LoRA grad tensors, "
          f"max|grad|={max_grad:.3e}, zero-tensors={zero_grads}, base grads=0 OK")

    # ---- check 3: 3 optimizer steps, finite, memory ----
    opt = torch.optim.AdamW(ttm.lora_parameters(), lr=1e-5)
    for step in range(1, 4):
        opt.zero_grad(set_to_none=True)
        loss, margin, _ = ttm.dpo_forward(batch)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ttm.lora_parameters(), max_norm=1e9)
        opt.step()
        if not (torch.isfinite(loss) and torch.isfinite(margin) and torch.isfinite(gn)):
            fail(f"step {step}: non-finite loss/margin/grad_norm")
        print(f"[preflight] step {step}: loss={float(loss):.6f} margin={float(margin):+.6f} "
              f"grad_norm={float(gn):.3e}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[preflight] peak memory: {peak:.2f} GB (budget 20)")
    if peak > 20:
        fail(f"peak memory {peak:.2f} GB exceeds the 20 GB budget")

    # ---- check 4: dummy adapter for the merge smoke ----
    dummy = OUT_DIR / f"dummy_adapter_{datetime.now().strftime('%H%M%S')}.pt"
    ttm.save_adapter(str(dummy), extra_meta={"step": 3, "note": "preflight dummy"})
    print(f"[preflight] dummy adapter -> {dummy}")

    print(f"[preflight] PASS ({datetime.now().isoformat()})")


if __name__ == "__main__":
    main()
