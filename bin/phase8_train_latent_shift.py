#!/usr/bin/env python3
"""Train a small Phase 8 first-chunk latent-shift corrector."""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from phase8_correction_lib import (
    LatentShiftMLP,
    hidden_recovery,
    parse_layer_spec,
    resolve_manifest,
    select_target_from_capture,
    write_json,
)


def load_sample(root: pathlib.Path, row: dict[str, Any]) -> dict[str, Any]:
    return torch.load(root / row["path"], map_location="cpu", weights_only=False)


def available_layers(sample: dict[str, Any]) -> list[int]:
    if "latent_pert" in sample:
        return sorted(int(k) for k in sample["latent_pert"]["layers"])
    return sorted(int(k) for k in sample["hidden_pert"])


def select_saved_target(payload: dict[str, torch.Tensor], target: str) -> torch.Tensor:
    if target == "action":
        return payload["action"].float()
    if target == "video":
        return payload["video"].float()
    if target == "wrist":
        return payload["video"][:, :1].float()
    if target == "primary":
        return payload["video"][:, 1:2].float()
    if target == "video_action":
        return torch.cat([payload["video"].float(), payload["action"].float()], dim=1)
    raise ValueError(f"unknown target: {target}")


def sample_tensors(
    sample: dict[str, Any],
    *,
    layer: int,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = str(layer)
    if "latent_pert" in sample:
        pert = select_saved_target(sample["latent_pert"]["layers"][key], target)
        delta = select_saved_target(sample["latent_shift"]["layers"][key], target)
        return pert, delta, pert + delta
    pert = select_target_from_capture(sample["hidden_pert"][key].float(), target)
    clean = select_target_from_capture(sample["hidden_clean"][key].float(), target)
    return pert, clean - pert, clean


class ShiftDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        rows: list[dict[str, Any]],
        *,
        layer: int,
        target: str,
        target_rms: float,
    ) -> None:
        self.root = root
        self.rows = rows
        self.layer = int(layer)
        self.target = target
        self.target_rms = float(target_rms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pert, delta, _clean = sample_tensors(
            load_sample(self.root, self.rows[idx]),
            layer=self.layer,
            target=self.target,
        )
        return pert, delta / self.target_rms


def split_rows(rows: list[dict[str, Any]], val_frac: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) <= 1 or val_frac <= 0:
        return rows, []
    n_val = max(1, min(len(rows) - 1, int(round(len(rows) * val_frac))))
    return rows[n_val:], rows[:n_val]


def filter_rows(rows: list[dict[str, Any]], groups: list[str] | None, conditions: list[str] | None) -> list[dict[str, Any]]:
    out = rows
    if groups:
        keep = set(groups)
        out = [row for row in out if row.get("group") in keep]
    if conditions:
        keep = set(conditions)
        out = [row for row in out if row.get("condition") in keep]
    return out


def estimate_target_rms(
    root: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    layer: int,
    target: str,
) -> float:
    total_sq = 0.0
    total_n = 0
    for row in rows:
        _pert, delta, _clean = sample_tensors(load_sample(root, row), layer=layer, target=target)
        total_sq += float(torch.sum(delta.float() ** 2).item())
        total_n += int(delta.numel())
    if total_n == 0:
        return 1.0
    return max(math.sqrt(total_sq / total_n), 1e-8)


def evaluate(
    model: torch.nn.Module,
    root: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    layer: int,
    target: str,
    target_rms: float,
    device: torch.device,
) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    losses, recoveries = [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            pert, delta, clean = sample_tensors(load_sample(root, row), layer=layer, target=target)
            pred = model(pert.to(device).float()).cpu() * target_rms
            loss = torch.mean((pred - delta) ** 2).item()
            rec = hidden_recovery(clean, pert, pert + pred)
            losses.append(float(loss))
            recoveries.append(float(rec["hidden_recovery_vs_pert"]))
    return {
        "n": len(rows),
        "mse_delta": float(sum(losses) / len(losses)),
        "hidden_recovery_vs_pert": float(sum(recoveries) / len(recoveries)),
    }


def save_checkpoint(
    path: pathlib.Path,
    *,
    model: LatentShiftMLP,
    args,
    layer: int,
    dim: int,
    target_rms: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "dim": dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "target": {
                "layer": layer,
                "layer_spec": args.target_layer,
                "target": args.target,
                "target_rms": target_rms,
            },
            "train_args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-layer", default="0", help="integer layer or 'last'")
    p.add_argument("--target", choices=["action", "video", "video_action", "wrist", "primary"], default="action")
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--normalize-target", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root, rows = resolve_manifest(args.input_dir)
    rows = filter_rows(rows, args.groups, args.conditions)
    if not rows:
        raise RuntimeError("no rows selected from manifest")
    first = load_sample(root, rows[0])
    layer = parse_layer_spec(args.target_layer, available_layers(first))
    pert0, _delta0, _clean0 = sample_tensors(first, layer=layer, target=args.target)
    dim = int(pert0.shape[-1])
    train_rows, val_rows = split_rows(rows, args.val_frac, args.split_seed)
    target_rms = (
        estimate_target_rms(root, train_rows, layer=layer, target=args.target)
        if args.normalize_target
        else 1.0
    )

    out_dir = pathlib.Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = LatentShiftMLP(dim=dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = ShiftDataset(root, train_rows, layer=layer, target=args.target, target_rms=target_rms)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"rows train={len(train_rows)} val={len(val_rows)} layer={layer} target={args.target} rms={target_rms:.6g}", flush=True)
    best_score = -float("inf")
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_eval = evaluate(model, root, train_rows, layer=layer, target=args.target, target_rms=target_rms, device=device)
        val_eval = evaluate(model, root, val_rows, layer=layer, target=args.target, target_rms=target_rms, device=device)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "train": train_eval,
            "val": val_eval,
        }
        history.append(row)
        score = val_eval.get("hidden_recovery_vs_pert", train_eval["hidden_recovery_vs_pert"])
        print(f"epoch={epoch:03d} loss={row['train_loss']:.6g} train_rec={train_eval['hidden_recovery_vs_pert']:.4f} val_rec={val_eval.get('hidden_recovery_vs_pert', float('nan')):.4f}", flush=True)
        if score > best_score:
            best_score = float(score)
            save_checkpoint(out_dir / "best.pt", model=model, args=args, layer=layer, dim=dim, target_rms=target_rms, metrics=row)

    final_metrics = {
        "elapsed_s": time.time() - start,
        "target_layer": layer,
        "target": args.target,
        "target_rms": target_rms,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "history": history,
        "best_score": best_score,
    }
    save_checkpoint(out_dir / "final.pt", model=model, args=args, layer=layer, dim=dim, target_rms=target_rms, metrics=final_metrics)
    write_json(out_dir / "metrics.json", final_metrics)
    print(f"Saved checkpoints to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
