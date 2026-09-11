"""Single-GPU custom LoRA-DPO training loop for memory-guided TTA.

v1 (2026-09-09): one pair per step (K=1 per side), one forward over both sides
with shared sigma/epsilon, plain AdamW over LoRA parameters only. No ports, no
distributed init, no hydra/megatron/EMA/callback machinery — deliberately a
~100-line loop so a human can review all of it.

Safety constraints honoured: GPU = the one with the LOWEST memory usage, budget
20 GB; degrade/postpone when insufficient; adapter saved to a NEW file only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# MUST run before the imports below: dataset/model transitively import
# huggingface_hub, whose cache/offline config freezes at import time
# (review finding 3, 2026-09-09).
from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

from memory_system.tta.dataset import PreferenceDataset  # noqa: E402
from memory_system.tta.model import TTADPOModel, load_policy_model  # noqa: E402


def pick_gpu(budget_gb: float) -> int:
    """GPU with the LOWEST current memory usage; raise (postpone) if free < budget."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = sorted(
        ({"index": int(i), "used": int(u), "free": int(f)} for i, u, f in
         (line.split(",") for line in out.strip().splitlines())),
        key=lambda g: g["used"],
    )
    best = gpus[0]
    if best["free"] < int(budget_gb * 1024):
        raise RuntimeError(
            f"GPU {best['index']} is least-used but has {best['free']} MiB free (< {int(budget_gb * 1024)}). "
            "Postpone — do not retry."
        )
    return best["index"]


def parse_args():
    p = argparse.ArgumentParser(description="Memory-guided TTA LoRA-DPO training (single GPU)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True, help="policy .pt to adapt")
    p.add_argument("--dataset-stats-path", required=True)
    p.add_argument("--t5-text-embeddings-path", required=True)
    p.add_argument("--output-dir", default="memory_system/tta/results/runs")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0, help="0 = save only at the end")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-budget-gb", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device_index = pick_gpu(args.gpu_memory_budget_gb)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    print(f"[train] GPU {device_index} (least used), budget {args.gpu_memory_budget_gb} GB")

    torch.manual_seed(args.seed)
    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset = PreferenceDataset(
        manifest_path=args.manifest,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        seed=args.seed,
    )
    print(f"[train] pairs={len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    inner = load_policy_model(args.checkpoint)
    model = TTADPOModel(inner, beta=args.beta, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                        base_checkpoint=args.checkpoint)
    model.model.train()
    optimizer = torch.optim.AdamW(model.lora_parameters(), lr=args.lr)

    run_dir.mkdir(parents=True, exist_ok=True)
    step, epoch = 0, 0
    while step < args.max_steps:
        dataset.set_epoch(epoch)
        for batch in loader:
            batch = model.batch_to_device(batch)
            loss, margin, _e = model.dpo_forward(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.lora_parameters(), max_norm=1e9)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[train] step={step} loss={float(loss):.4f} margin={float(margin):.4f} "
                      f"grad_norm={float(grad_norm):.3e}")
            if args.save_every and step % args.save_every == 0:
                model.save_adapter(str(run_dir / f"adapter_step{step}.pt"),
                                   extra_meta={"step": step, "manifest": args.manifest})
            if step >= args.max_steps:
                break
        epoch += 1

    final_path = str(run_dir / f"adapter_step{step}.pt")
    model.save_adapter(final_path, extra_meta={"step": step, "manifest": args.manifest,
                                               "max_steps": args.max_steps})
    print(f"[train] done: {step} steps, adapter at {final_path}")


if __name__ == "__main__":
    main()
