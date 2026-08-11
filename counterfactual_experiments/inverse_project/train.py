#!/usr/bin/env python3
"""Train a residual canonicalizer on paired clean/perturbed video latents."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PackedLatentDataset
from models import MODEL_NAMES, LatentCanonicalizer, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="dataset/paired_libero_plus_background_libero10_500"
    )
    parser.add_argument("--packed-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--resume")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_loader(dataset: PairedLatentDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


def train_epoch(
    model: LatentCanonicalizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss, total_frames = 0.0, 0
    for batch in loader:
        perturbed = batch["perturbed"].to(device, dtype=torch.float32, non_blocking=True)
        clean = batch["clean"].to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.precision):
            pred_perturbed = model(perturbed)
            pred_clean = model(clean)
            loss_perturbed = F.mse_loss(pred_perturbed.float(), clean)
            loss_clean = F.mse_loss(pred_clean.float(), clean)
            loss = loss_perturbed + args.identity_weight * loss_clean
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * clean.shape[0]
        total_frames += clean.shape[0]
    return total_loss / max(total_frames, 1)


@torch.no_grad()
def validate(
    model: LatentCanonicalizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    sums = dict(baseline_mse=0.0, corrected_mse=0.0, clean_drift_mse=0.0, gate=0.0)
    frames = 0
    gated = model.gate is not None
    for batch in loader:
        perturbed = batch["perturbed"].to(device, dtype=torch.float32, non_blocking=True)
        clean = batch["clean"].to(device, dtype=torch.float32, non_blocking=True)
        with amp_context(device, args.precision):
            corrected, aux = model(perturbed, return_aux=True)
            clean_out = model(clean)
        count = clean.shape[0]
        sums["baseline_mse"] += F.mse_loss(perturbed, clean).item() * count
        sums["corrected_mse"] += F.mse_loss(corrected.float(), clean).item() * count
        sums["clean_drift_mse"] += F.mse_loss(clean_out.float(), clean).item() * count
        if gated:
            sums["gate"] += aux["gate"].float().mean().item() * count
        frames += count
    metrics = {key: value / max(frames, 1) for key, value in sums.items() if key != "gate"}
    metrics["recovery"] = 1.0 - metrics["corrected_mse"] / max(metrics["baseline_mse"], 1e-12)
    if gated:
        metrics["mean_gate"] = sums["gate"] / max(frames, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: LatentCanonicalizer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_name": model.config["model_name"],
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        args.model,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    start_epoch, global_step, best_mse = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != model.config:
            raise ValueError("resume checkpoint model config does not match command line")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_mse = float(checkpoint.get("metrics", {}).get("corrected_mse", best_mse))
    packed_dir = args.packed_dir or str(Path(args.data_dir) / "packed_latents")
    train_data = PackedLatentDataset(packed_dir, "train")
    val_data = PackedLatentDataset(packed_dir, "val")
    metadata = train_data.metadata
    if metadata["split_seed"] != args.split_seed or metadata["val_fraction"] != args.val_fraction:
        raise ValueError("packed dataset split settings do not match training arguments")
    train_loader = make_loader(train_data, args, shuffle=True)
    val_loader = make_loader(val_data, args, shuffle=False)
    config = vars(args).copy()
    config.update(
        model_config=model.config,
        train_frames=len(train_data),
        val_frames=len(val_data),
        packed_dir=str(Path(packed_dir).expanduser().resolve()),
        train_policy_seeds=metadata["train_policy_seeds"],
        val_policy_seeds=metadata["val_policy_seeds"],
    )
    write_json(output_dir / "config.json", config)
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, args)
        global_step += len(train_loader)
        metrics = validate(model, val_loader, device, args)
        metrics.update(epoch=epoch, global_step=global_step, train_loss=train_loss)
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, epoch, global_step, metrics)
        if metrics["corrected_mse"] < best_mse:
            best_mse = metrics["corrected_mse"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, epoch, global_step, metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
