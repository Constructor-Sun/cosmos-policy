"""Export a direct-SFT latent adapter from a Cosmos DCP checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader

from counterfactual_experiments.direct_sft.adapter import LatentCanonicalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Iteration checkpoint directory")
    parser.add_argument("output", type=Path, help="Output adapter .pt file")
    args = parser.parse_args()

    model_checkpoint = args.checkpoint / "model"
    if not (model_checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"missing DCP model checkpoint: {model_checkpoint}")

    adapter = LatentCanonicalizer(
        model_name="transformer",
        channels=16,
        slots=2,
        height=28,
        width=28,
        hidden_dim=64,
        mlp_dim=128,
        num_heads=4,
        dropout=0.0,
        gate_hidden_dim=32,
        initial_gate=0.12,
    )
    checkpoint_state = {
        f"latent_adapter.{key}": torch.empty_like(value, dtype=torch.bfloat16)
        for key, value in adapter.state_dict().items()
    }
    dcp.load(
        checkpoint_state,
        storage_reader=FileSystemReader(str(model_checkpoint)),
        planner=DefaultLoadPlanner(allow_partial_load=False),
    )
    adapter.load_state_dict(
        {
            key.removeprefix("latent_adapter."): value.float()
            for key, value in checkpoint_state.items()
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": adapter.config,
            "model_state_dict": adapter.state_dict(),
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
