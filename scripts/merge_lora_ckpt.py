#!/usr/bin/env python
"""Merge a LoRA-finetuned DCP checkpoint into base weights and export a plain
.pt checkpoint that run_libero_eval (base architecture, no LoRA) can load.

Usage:
  python scripts/merge_lora_ckpt.py \
    --ckpt <.../checkpoints/iter_000000500/model> \
    --out training/tta_sft_merged_iter500.pt
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="DCP model dir (containing .metadata)")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--lora-rank", type=int, help="LoRA rank used during training")
    ap.add_argument("--lora-alpha", type=int, help="LoRA alpha used during training")
    args = ap.parse_args()
    if (args.lora_rank is not None and args.lora_rank < 1) or (args.lora_alpha is not None and args.lora_alpha < 1):
        ap.error("LoRA rank and alpha must be positive")
    overrides = []
    if args.lora_rank is not None:
        overrides.append(f"model.config.lora_rank={args.lora_rank}")
    if args.lora_alpha is not None:
        overrides.append(f"model.config.lora_alpha={args.lora_alpha}")

    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    print("Loading base+LoRA model from DCP checkpoint...")
    model, _ = load_model_from_checkpoint(
        experiment_name="tta_repair_sft",
        s3_checkpoint_dir=args.ckpt,
        config_file="cosmos_policy/config/config.py",
        load_ema_to_reg=False,
        to_device="cpu",
        experiment_opts=overrides,
    )

    # Merge every LoRA layer (peft semantics: W += B @ A * alpha/rank).
    merged = 0
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and len(getattr(mod, "lora_A", {})) > 0:
            if (args.lora_rank is not None and mod.r['default'] != args.lora_rank
                    or args.lora_alpha is not None and mod.lora_alpha['default'] != args.lora_alpha):
                raise ValueError(f"Unexpected LoRA rank/alpha in {name}")
            mod.merge()
            merged += 1
    print(f"Merged {merged} LoRA layers.")

    sd = {k.replace("base_layer.", ""): v for k, v in model.state_dict().items()
          if "lora" not in k.lower()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    print(f"Saved merged checkpoint: {args.out} ({len(sd)} keys)")


if __name__ == "__main__":
    main()
