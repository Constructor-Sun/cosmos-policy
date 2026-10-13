#!/usr/bin/env python3
"""Export a full-parameter DCP model checkpoint to the original policy .pt format."""
import argparse
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, help='DCP model directory containing .metadata')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name='tta_repair_sft', s3_checkpoint_dir=args.ckpt,
        config_file='cosmos_policy/config/config.py', load_ema_to_reg=False,
        to_device='cpu', experiment_opts=['model.config.use_lora=False'])
    if model.config.use_lora:
        raise ValueError('Full-parameter export must disable LoRA')
    state = model.state_dict()
    if not state or any('lora_' in key or 'base_layer.' in key for key in state):
        raise ValueError('Expected plain policy weights without adapter keys')
    state = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
             for key, value in state.items()}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    torch.save(state, temporary)
    temporary.replace(output)
    print(f'Exported full policy checkpoint: {output} ({len(state)} keys)', flush=True)


if __name__ == '__main__':
    main()
