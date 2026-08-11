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
    LatentShiftCVAE,
    LatentShiftMLP,
    LatentShiftMoE,
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


def strip_saved_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Remove the capture-time singleton batch before DataLoader batching."""
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def sample_tensors(
    sample: dict[str, Any],
    *,
    layer: int,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = str(layer)
    if "latent_pert" in sample:
        pert = strip_saved_batch(
            select_saved_target(sample["latent_pert"]["layers"][key], target)
        )
        delta = strip_saved_batch(
            select_saved_target(sample["latent_shift"]["layers"][key], target)
        )
        return pert, delta, pert + delta
    pert = strip_saved_batch(
        select_target_from_capture(sample["hidden_pert"][key].float(), target)
    )
    clean = strip_saved_batch(
        select_target_from_capture(sample["hidden_clean"][key].float(), target)
    )
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
        preload: bool = True,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.root = root
        self.rows = rows
        self.layer = int(layer)
        self.target = target
        self.target_rms = float(target_rms)
        self.cache_dtype = cache_dtype
        self._cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if preload:
            self._cache = []
            for idx, row in enumerate(self.rows, start=1):
                self._cache.append(self._load_raw(row))
                if idx == 1 or idx == len(self.rows) or idx % 500 == 0:
                    print(f"cached samples {idx}/{len(self.rows)} from {self.root}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_raw(self, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        pert, delta, _clean = sample_tensors(
            load_sample(self.root, row),
            layer=self.layer,
            target=self.target,
        )
        # Saved latent tensors are normally fp16. Keep the cache compact and
        # cast to fp32 only when a batch is moved to the training device.
        return (
            pert.to(dtype=self.cache_dtype).contiguous(),
            delta.to(dtype=self.cache_dtype).contiguous(),
        )

    def get_raw(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._load_raw(self.rows[idx])

    @property
    def cache_bytes(self) -> int:
        if self._cache is None:
            return 0
        return sum(pert.numel() * pert.element_size() + delta.numel() * delta.element_size() for pert, delta in self._cache)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pert, delta = self.get_raw(idx)
        return pert, delta / self.target_rms


def task_key(row: dict[str, Any]) -> str:
    suite = row.get("suite", "unknown_suite")
    base_task = (
        row.get("base_task")
        or row.get("task_name")
        or row.get("clean_task_name")
        or "unknown_task"
    )
    return f"{suite}::{base_task}"


def split_rows_random(
    rows: list[dict[str, Any]],
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) <= 1 or val_frac <= 0:
        return rows, []
    n_val = max(1, min(len(rows) - 1, int(round(len(rows) * val_frac))))
    return rows[n_val:], rows[:n_val]


def split_rows_by_task(
    rows: list[dict[str, Any]],
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_task.setdefault(task_key(row), []).append(row)

    tasks = sorted(rows_by_task)
    if len(tasks) <= 1 or val_frac <= 0:
        return list(rows), []

    random.Random(seed).shuffle(tasks)
    n_val_tasks = max(1, min(len(tasks) - 1, int(round(len(tasks) * val_frac))))
    val_tasks = set(tasks[:n_val_tasks])
    train_rows = []
    val_rows = []
    for row in rows:
        if task_key(row) in val_tasks:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def state_hash_key(row: dict[str, Any]) -> str:
    state_hash = row.get("state_hash")
    if not state_hash:
        raise ValueError("--split-by state_hash requires state_hash in every manifest row")
    return f"{task_key(row)}::{state_hash}"


def split_rows_by_state_hash(
    rows: list[dict[str, Any]],
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_frac <= 0:
        return list(rows), []
    groups_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        groups_by_task.setdefault(task_key(row), {}).setdefault(state_hash_key(row), []).append(row)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for task in sorted(groups_by_task):
        groups = groups_by_task[task]
        state_keys = sorted(groups)
        if len(state_keys) <= 1:
            train_rows.extend(groups[state_keys[0]])
            continue
        rng.shuffle(state_keys)
        n_val = max(1, min(len(state_keys) - 1, int(round(len(state_keys) * val_frac))))
        val_states = set(state_keys[:n_val])
        for key, group_rows in groups.items():
            (val_rows if key in val_states else train_rows).extend(group_rows)
    return train_rows, val_rows


def split_rows(
    rows: list[dict[str, Any]],
    val_frac: float,
    seed: int,
    split_by: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_by == "sample":
        return split_rows_random(rows, val_frac, seed)
    if split_by == "task":
        return split_rows_by_task(rows, val_frac, seed)
    if split_by == "state_hash":
        return split_rows_by_state_hash(rows, val_frac, seed)
    raise ValueError(f"unknown split_by: {split_by}")


def split_summary(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    split_by: str,
) -> dict[str, Any]:
    train_tasks = sorted({task_key(row) for row in train_rows})
    val_tasks = sorted({task_key(row) for row in val_rows})
    overlap = sorted(set(train_tasks) & set(val_tasks))
    train_states = {state_hash_key(row) for row in train_rows if row.get("state_hash")}
    val_states = {state_hash_key(row) for row in val_rows if row.get("state_hash")}
    train_cameras = {tuple(row["camera_tuple"]) for row in train_rows if row.get("camera_tuple") is not None}
    val_cameras = {tuple(row["camera_tuple"]) for row in val_rows if row.get("camera_tuple") is not None}
    train_seeds = {int(row["policy_seed"]) for row in train_rows if row.get("policy_seed") is not None}
    val_seeds = {int(row["policy_seed"]) for row in val_rows if row.get("policy_seed") is not None}
    return {
        "split_by": split_by,
        "num_train_tasks": len(train_tasks),
        "num_val_tasks": len(val_tasks),
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "task_overlap": overlap,
        "num_train_states": len(train_states),
        "num_val_states": len(val_states),
        "state_overlap": sorted(train_states & val_states),
        "camera_overlap": sorted(train_cameras & val_cameras),
        "policy_seed_overlap": sorted(train_seeds & val_seeds),
    }


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


def estimate_cached_target_rms(dataset: ShiftDataset) -> float:
    """Estimate target RMS without re-reading sample files from disk."""
    total_sq = 0.0
    total_n = 0
    for idx in range(len(dataset)):
        _pert, delta = dataset.get_raw(idx)
        total_sq += float(torch.sum(delta.float() ** 2).item())
        total_n += int(delta.numel())
    if total_n == 0:
        return 1.0
    return max(math.sqrt(total_sq / total_n), 1e-8)


def prepare_training_data(args: argparse.Namespace) -> dict[str, Any]:
    """Load manifests, filter rows, resolve train/val split, and validate.

    Returns a dict with keys: root, val_root, train_rows, val_rows,
    layer, dim, split_by, split_info.
    """
    root, rows = resolve_manifest(args.input_dir)
    rows = filter_rows(rows, args.groups, args.conditions)
    if not rows:
        raise RuntimeError("no rows selected from manifest")

    first = load_sample(root, rows[0])
    layer = parse_layer_spec(args.target_layer, available_layers(first))
    pert0, _delta0, _clean0 = sample_tensors(first, layer=layer, target=args.target)
    dim = int(pert0.shape[-1])

    if args.val_input_dir:
        val_root, val_rows = resolve_manifest(args.val_input_dir)
        val_rows = filter_rows(val_rows, args.groups, args.conditions)
        if not val_rows:
            raise RuntimeError("no rows selected from validation manifest")
        train_rows, split_by = rows, "explicit"
    else:
        val_root = root
        train_rows, val_rows = split_rows(rows, args.val_frac, args.split_seed, args.split_by)
        split_by = args.split_by

    split_info = split_summary(train_rows, val_rows, split_by)

    # Validate split integrity.
    if split_by in {"explicit", "state_hash"} and split_info["state_overlap"]:
        raise RuntimeError(f"train/val state overlap: {split_info['state_overlap']}")
    if split_by == "explicit" and split_info["camera_overlap"]:
        raise RuntimeError(f"train/val camera overlap: {split_info['camera_overlap']}")
    if split_by == "explicit" and split_info["policy_seed_overlap"]:
        raise RuntimeError(f"train/val policy seed overlap: {split_info['policy_seed_overlap']}")
    if split_by == "task" and split_info["task_overlap"]:
        raise RuntimeError(f"train/val task overlap: {split_info['task_overlap']}")

    return {
        "root": root,
        "val_root": val_root,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "layer": layer,
        "dim": dim,
        "split_by": split_by,
        "split_info": split_info,
    }


def evaluate(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    target_rms: float,
    device: torch.device,
    dataset: ShiftDataset,
    batch_size: int = 32,
) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mse_values: list[float] = []
    recovery_values: list[float] = []
    router_load = None
    router_importance = None
    router_entropy = 0.0
    router_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y_scaled in loader:
            pert = x.to(device, non_blocking=True).float()
            delta = y_scaled.to(device, non_blocking=True).float() * target_rms
            if isinstance(model, LatentShiftMoE):
                pred_scaled, router_stats = model(pert, return_router_stats=True)
                batch_n = int(pert.shape[0])
                batch_load = router_stats["expert_load"].detach().cpu() * batch_n
                batch_importance = router_stats["expert_importance"].detach().cpu() * batch_n
                router_load = batch_load if router_load is None else router_load + batch_load
                router_importance = (
                    batch_importance
                    if router_importance is None
                    else router_importance + batch_importance
                )
                router_entropy += float(router_stats["router_entropy"].item()) * batch_n
                router_samples += batch_n
            else:
                pred_scaled = model(pert)
            pred = pred_scaled * target_rms
            per_sample_mse = (pred - delta).flatten(1).square().mean(dim=1)
            per_sample_base = delta.flatten(1).square().mean(dim=1)
            per_sample_recovery = 1.0 - per_sample_mse / (per_sample_base + 1e-8)
            mse_values.extend(per_sample_mse.detach().cpu().tolist())
            recovery_values.extend(per_sample_recovery.detach().cpu().tolist())
    metrics: dict[str, Any] = {
        "n": len(rows),
        "mse_delta": float(sum(mse_values) / len(mse_values)),
        "hidden_recovery_vs_pert": float(sum(recovery_values) / len(recovery_values)),
    }
    if router_samples:
        metrics["router"] = {
            "expert_load": (router_load / router_samples).tolist(),
            "expert_importance": (router_importance / router_samples).tolist(),
            "entropy": router_entropy / router_samples,
        }
    return metrics


def run_training_loop(
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    loader: DataLoader,
    train_ds: ShiftDataset,
    val_ds: ShiftDataset,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    layer: int,
    target: str,
    target_rms: float,
    out_dir: pathlib.Path,
    dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the training loop.  Returns partial metrics (elapsed_s, history, best_score)."""
    best_score = -float("inf")
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        mse_loss_sum = torch.zeros((), device=device)
        balance_loss_sum = torch.zeros((), device=device)
        z_loss_sum = torch.zeros((), device=device)
        basis_loss_sum = torch.zeros((), device=device)
        cvae_kl_sum = torch.zeros((), device=device)
        cvae_raw_kl_sum = torch.zeros((), device=device)
        cvae_active_fraction_sum = torch.zeros((), device=device)
        cvae_prior_std_sum = torch.zeros((), device=device)
        cvae_posterior_std_sum = torch.zeros((), device=device)
        expert_load_sum = None
        num_batches = 0
        if args.cvae_kl_warmup_epochs > 0:
            cvae_beta = args.cvae_beta * min(
                1.0,
                epoch / args.cvae_kl_warmup_epochs,
            )
        else:
            cvae_beta = args.cvae_beta
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            if isinstance(model, LatentShiftMoE):
                pred, router_stats = model(x, return_router_stats=True)
                mse_loss = torch.mean((pred - y) ** 2)
                balance_loss = router_stats["load_balance_loss"]
                z_loss = router_stats["router_z_loss"]
                basis_loss = model.basis_orthogonality_loss()
                loss = (
                    mse_loss
                    + args.load_balance_loss_weight * balance_loss
                    + args.router_z_loss_weight * z_loss
                    + args.basis_orthogonality_loss_weight * basis_loss
                )
                batch_load = router_stats["expert_load"].detach()
                expert_load_sum = batch_load if expert_load_sum is None else expert_load_sum + batch_load
                balance_loss_sum += balance_loss.detach()
                z_loss_sum += z_loss.detach()
                basis_loss_sum += basis_loss.detach()
            elif isinstance(model, LatentShiftCVAE):
                pred, cvae_stats = model(x, target_shift=y, return_stats=True)
                mse_loss = torch.mean((pred - y) ** 2)
                kl_per_dim = cvae_stats["kl_per_dim"]
                raw_kl_loss = kl_per_dim.mean()
                if args.cvae_free_bits > 0:
                    kl_loss = kl_per_dim.clamp_min(args.cvae_free_bits).mean()
                else:
                    kl_loss = raw_kl_loss
                loss = mse_loss + cvae_beta * kl_loss
                cvae_kl_sum += kl_loss.detach()
                cvae_raw_kl_sum += raw_kl_loss.detach()
                cvae_active_fraction_sum += (
                    kl_per_dim > args.cvae_free_bits
                ).float().mean().detach()
                cvae_prior_std_sum += cvae_stats["prior_std"].mean().detach()
                cvae_posterior_std_sum += cvae_stats["posterior_std"].mean().detach()
            else:
                pred = model(x)
                mse_loss = torch.mean((pred - y) ** 2)
                loss = mse_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_sum += loss.detach()
            mse_loss_sum += mse_loss.detach()
            num_batches += 1

        train_eval = evaluate(
            model, train_rows,
            target_rms=target_rms, device=device,
            dataset=train_ds, batch_size=args.eval_batch_size,
        )
        val_eval = evaluate(
            model, val_rows,
            target_rms=target_rms, device=device,
            dataset=val_ds, batch_size=args.eval_batch_size,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float((loss_sum / max(1, num_batches)).item()),
            "train_mse_loss": float((mse_loss_sum / max(1, num_batches)).item()),
            "train": train_eval,
            "val": val_eval,
        }
        if isinstance(model, LatentShiftMoE):
            row["router_train"] = {
                "load_balance_loss": float((balance_loss_sum / max(1, num_batches)).item()),
                "z_loss": float((z_loss_sum / max(1, num_batches)).item()),
                "basis_orthogonality_loss": float(
                    (basis_loss_sum / max(1, num_batches)).item()
                ),
                "expert_load": (expert_load_sum / max(1, num_batches)).cpu().tolist(),
            }
        elif isinstance(model, LatentShiftCVAE):
            row["cvae_train"] = {
                "beta": float(cvae_beta),
                "kl_objective": float((cvae_kl_sum / max(1, num_batches)).item()),
                "kl_raw": float((cvae_raw_kl_sum / max(1, num_batches)).item()),
                "active_fraction": float(
                    (cvae_active_fraction_sum / max(1, num_batches)).item()
                ),
                "prior_std": float((cvae_prior_std_sum / max(1, num_batches)).item()),
                "posterior_std": float(
                    (cvae_posterior_std_sum / max(1, num_batches)).item()
                ),
            }
        history.append(row)
        score = val_eval.get("hidden_recovery_vs_pert", train_eval["hidden_recovery_vs_pert"])
        route_text = ""
        if isinstance(model, LatentShiftMoE):
            load = row["router_train"]["expert_load"]
            route_text = f" route_min={min(load):.3f} route_max={max(load):.3f}"
        elif isinstance(model, LatentShiftCVAE):
            route_text = (
                f" beta={row['cvae_train']['beta']:.4g} "
                f"kl={row['cvae_train']['kl_raw']:.4g} "
                f"active={row['cvae_train']['active_fraction']:.3f}"
            )
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.6g} "
            f"mse={row['train_mse_loss']:.6g} "
            f"train_rec={train_eval['hidden_recovery_vs_pert']:.4f} "
            f"val_rec={val_eval.get('hidden_recovery_vs_pert', float('nan')):.4f}"
            f"{route_text}",
            flush=True,
        )

        if score > best_score:
            best_score = float(score)
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                layer=layer,
                dim=dim,
                target_rms=target_rms,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                metrics=row,
                train_args=vars(args),
                target_layer_spec=args.target_layer,
                target_name=args.target,
            )

    return {
        "elapsed_s": time.time() - start,
        "history": history,
        "best_score": best_score,
    }


def save_checkpoint(
    path: pathlib.Path,
    *,
    model: torch.nn.Module,
    layer: int,
    dim: int,
    target_rms: float,
    hidden_dim: int,
    dropout: float,
    metrics: dict[str, Any],
    train_args: dict[str, Any] | None = None,
    target_layer_spec: str = "",
    target_name: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, LatentShiftMoE):
        architecture = "moe"
    elif isinstance(model, LatentShiftCVAE):
        architecture = "cvae"
    else:
        architecture = "mlp"
    model_config: dict[str, Any] = {
        "architecture": architecture,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "norm": getattr(model, "norm_type", "rmsnorm"),
    }
    if isinstance(model, LatentShiftMoE):
        model_config.update(
            {
                "num_experts": model.num_experts,
                "top_k_experts": model.top_k_experts,
                "expert_mode": model.expert_mode,
                "expert_rank": model.expert_rank,
                "router_temperature": model.router_temperature,
                "router_noise": model.router_noise,
                "expert_slices": model.expert_slices,
            }
        )
    elif isinstance(model, LatentShiftCVAE):
        model_config.update(
            {
                "latent_dim": model.latent_dim,
                "min_logvar": model.min_logvar,
                "max_logvar": model.max_logvar,
            }
        )
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config,
            "target": {
                "layer": layer,
                "layer_spec": target_layer_spec,
                "target": target_name,
                "target_rms": target_rms,
            },
            "train_args": train_args if train_args is not None else {},
            "metrics": metrics,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--val-input-dir", default=None, help="independent validation manifest root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-layer", default="0", help="integer layer or 'last'")
    p.add_argument("--target", choices=["action", "video", "video_action", "wrist", "primary"], default="action")
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--split-by",
        choices=["task", "state_hash", "sample"],
        default="task",
        help="validation split unit; 'state_hash' holds out initial states within every task",
    )
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--architecture", choices=["mlp", "moe", "cvae"], default="mlp")
    p.add_argument("--cvae-latent-dim", type=int, default=32)
    p.add_argument(
        "--cvae-beta",
        type=float,
        default=1e-2,
        help="maximum KL weight; KL is averaged over samples and latent dimensions",
    )
    p.add_argument(
        "--cvae-kl-warmup-epochs",
        type=int,
        default=10,
        help="linearly warm KL weight from zero to --cvae-beta",
    )
    p.add_argument(
        "--cvae-free-bits",
        type=float,
        default=0.01,
        help="per-latent-dimension KL floor in nats; reduces posterior collapse pressure",
    )
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--top-k-experts", type=int, default=2)
    p.add_argument(
        "--expert-mode",
        choices=["partitioned", "full", "low_rank"],
        default="full",
        help="partitioned uses channel slices; full predicts all dimensions; low_rank learns dense subspaces",
    )
    p.add_argument("--expert-rank", type=int, default=32, help="basis rank for --expert-mode low_rank")
    p.add_argument("--router-temperature", type=float, default=1.0)
    p.add_argument(
        "--router-noise",
        type=float,
        default=0.1,
        help="Gaussian router-logit noise used only during training",
    )
    p.add_argument("--load-balance-loss-weight", type=float, default=1e-2)
    p.add_argument("--router-z-loss-weight", type=float, default=1e-3)
    p.add_argument("--basis-orthogonality-loss-weight", type=float, default=1e-3)
    p.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load selected latent tensors once into CPU memory; use --no-preload if RAM is limited",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--normalize-target", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.num_experts < 1:
        raise ValueError("--num-experts must be at least 1")
    if not 1 <= args.top_k_experts <= args.num_experts:
        raise ValueError("--top-k-experts must be in [1, --num-experts]")
    if args.router_temperature <= 0:
        raise ValueError("--router-temperature must be positive")
    if args.router_noise < 0:
        raise ValueError("--router-noise must be non-negative")
    if args.expert_rank < 1:
        raise ValueError("--expert-rank must be at least 1")
    if args.cvae_latent_dim < 1:
        raise ValueError("--cvae-latent-dim must be at least 1")
    if args.cvae_beta < 0:
        raise ValueError("--cvae-beta must be non-negative")
    if args.cvae_kl_warmup_epochs < 0:
        raise ValueError("--cvae-kl-warmup-epochs must be non-negative")
    if args.cvae_free_bits < 0:
        raise ValueError("--cvae-free-bits must be non-negative")
    if (
        args.load_balance_loss_weight < 0
        or args.router_z_loss_weight < 0
        or args.basis_orthogonality_loss_weight < 0
    ):
        raise ValueError("MoE auxiliary-loss weights must be non-negative")

    # ── load & prepare data ──────────────────────────────────────────
    data = prepare_training_data(args)

    out_dir = pathlib.Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── model, optimizer, datasets ───────────────────────────────────
    if args.architecture == "moe":
        model = LatentShiftMoE(
            dim=data["dim"],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_experts=args.num_experts,
            top_k_experts=args.top_k_experts,
            expert_mode=args.expert_mode,
            expert_rank=args.expert_rank,
            router_temperature=args.router_temperature,
            router_noise=args.router_noise,
        ).to(device)
    elif args.architecture == "cvae":
        model = LatentShiftCVAE(
            dim=data["dim"],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            latent_dim=args.cvae_latent_dim,
        ).to(device)
    else:
        model = LatentShiftMLP(
            dim=data["dim"], hidden_dim=args.hidden_dim, dropout=args.dropout,
        ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    train_ds = ShiftDataset(
        data["root"], data["train_rows"],
        layer=data["layer"], target=args.target, target_rms=1.0,
        preload=args.preload,
    )
    val_ds = ShiftDataset(
        data["val_root"], data["val_rows"],
        layer=data["layer"], target=args.target, target_rms=1.0,
        preload=args.preload,
    )
    target_rms = estimate_cached_target_rms(train_ds) if args.normalize_target else 1.0
    train_ds.target_rms = target_rms
    val_ds.target_rms = target_rms

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    print(
        f"rows train={len(data['train_rows'])} val={len(data['val_rows'])} "
        f"tasks train={data['split_info']['num_train_tasks']} "
        f"val={data['split_info']['num_val_tasks']} "
        f"states train={data['split_info']['num_train_states']} "
        f"val={data['split_info']['num_val_states']} "
        f"split_by={data['split_by']} layer={data['layer']} target={args.target} "
        f"rms={target_rms:.6g} architecture={args.architecture} preload={args.preload} "
        f"cache_gib={(train_ds.cache_bytes + val_ds.cache_bytes) / 2**30:.2f}",
        flush=True,
    )
    if isinstance(model, LatentShiftMoE):
        expert_detail = ""
        if model.expert_mode == "low_rank":
            expert_detail = f" expert_rank={model.expert_rank}"
        elif model.expert_mode == "partitioned":
            expert_detail = f" slices={model.expert_slices}"
        print(
            f"moe experts={model.num_experts} top_k={model.top_k_experts} "
            f"mode={model.expert_mode}{expert_detail}",
            flush=True,
        )
    elif isinstance(model, LatentShiftCVAE):
        print(
            f"cvae latent_dim={model.latent_dim} beta={args.cvae_beta:g} "
            f"warmup_epochs={args.cvae_kl_warmup_epochs} free_bits={args.cvae_free_bits:g}",
            flush=True,
        )

    # ── train ────────────────────────────────────────────────────────
    loop_metrics = run_training_loop(
        model=model, opt=opt, device=device,
        loader=loader, train_ds=train_ds, val_ds=val_ds,
        train_rows=data["train_rows"], val_rows=data["val_rows"],
        layer=data["layer"], target=args.target, target_rms=target_rms,
        out_dir=out_dir, dim=data["dim"],
        args=args,
    )

    # ── save final outputs ───────────────────────────────────────────
    final_metrics: dict[str, Any] = {
        **loop_metrics,
        "target_layer": data["layer"],
        "target": args.target,
        "target_rms": target_rms,
        "num_train": len(data["train_rows"]),
        "num_val": len(data["val_rows"]),
        "split": data["split_info"],
    }
    save_checkpoint(
        out_dir / "final.pt",
        model=model,
        layer=data["layer"],
        dim=data["dim"],
        target_rms=target_rms,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        metrics=final_metrics,
        train_args=vars(args),
        target_layer_spec=args.target_layer,
        target_name=args.target,
    )
    write_json(out_dir / "metrics.json", final_metrics)
    print(f"Saved checkpoints to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
