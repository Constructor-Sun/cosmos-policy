#!/usr/bin/env python
"""Merge a saved DPO LoRA adapter into the base policy and export a plain .pt
the eval chain (run_libero_eval via SMOKE_POLICY_CKPT_PATH) can load.

Same export convention as scripts/merge_lora_ckpt.py (post-merge state dict,
"base_layer." stripped, lora keys dropped) that the SFT round already proved
end-to-end on the eval side; the injection/load half mirrors
memory_system.tta.model.attach_adapter. Runs on CPU only.

Built-in sanity gate: the merged key set must EQUAL the base state-dict key
set (checked before saving).

Usage:
  python tests/tta/merge_dpo_adapter.py \
    --adapter memory_system/tta/results/runs/run_<ts>/adapter_step100.pt \
    --out training/tta_dpo_s100_merged.pt
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

DEFAULT_BASE = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="adapter .pt ({meta, lora_state})")
    ap.add_argument("--out", required=True, help="output merged .pt (must not exist)")
    ap.add_argument("--base", default="", help="base policy .pt (default: meta.base_checkpoint)")
    args = ap.parse_args()

    blob = torch.load(args.adapter, map_location="cpu")
    meta = blob["meta"]
    base = args.base or meta.get("base_checkpoint") or DEFAULT_BASE
    print(f"[merge] adapter step={meta.get('step')} rank={meta['lora_rank']} "
          f"alpha={meta['lora_alpha']} base={base}")

    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _config = load_model_from_checkpoint(
        experiment_name="cosmos_predict2_2b_480p_libero__inference_only",
        s3_checkpoint_dir=base,
        config_file="cosmos_policy/config/config.py",
        load_ema_to_reg=False,
        to_device="cpu",
    )
    pre_keys = set(model.state_dict().keys())

    model.add_lora(model.net, lora_rank=meta["lora_rank"], lora_alpha=meta["lora_alpha"],
                   lora_target_modules=meta["target_modules"], init_lora_weights=True)
    # adapters may carry training-time gradient-checkpointing wrapper prefixes;
    # normalize to the canonical (freshly-injected) module paths before loading
    state = {k.replace("_checkpoint_wrapped_module.", ""): v
             for k, v in blob["lora_state"].items()}
    result = model.net.load_state_dict(state, strict=False)
    lora_missing = [k for k in result.missing_keys if "lora_" in k]
    if result.unexpected_keys or lora_missing:
        raise RuntimeError(
            f"adapter key mismatch after normalization: unexpected={len(result.unexpected_keys)} "
            f"lora_missing={len(lora_missing)} e.g. {(result.unexpected_keys + lora_missing)[:3]}")
    print(f"[merge] adapter loaded: {len(state)} tensors")

    merged = 0
    for _name, mod in model.net.named_modules():
        if hasattr(mod, "lora_A") and len(getattr(mod, "lora_A", {})) > 0:
            mod.merge()
            merged += 1
    print(f"[merge] merged {merged} LoRA layers")

    sd = {k.replace("base_layer.", ""): v for k, v in model.state_dict().items()
          if "lora" not in k.lower()}
    if set(sd.keys()) != pre_keys:
        missing = sorted(pre_keys - set(sd.keys()))[:5]
        extra = sorted(set(sd.keys()) - pre_keys)[:5]
        raise RuntimeError(f"merged key set != base key set: missing={missing} extra={extra}")

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print(f"[merge] saved {out} ({len(sd)} keys)")


if __name__ == "__main__":
    main()
