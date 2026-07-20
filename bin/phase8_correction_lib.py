"""Shared helpers for Phase 8 first-chunk latent correction."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch


CAPTURE_SLOT_ORDER = [2, 3, 4]
EPS = 1e-8


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_manifest(input_dir: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(input_dir).expanduser()
    if not root.is_absolute():
        root = root.resolve()
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return root, read_jsonl(manifest)


def target_model_indices(target: str) -> list[int]:
    if target == "action":
        return [4]
    if target == "video":
        return [2, 3]
    if target == "video_action":
        return [2, 3, 4]
    if target == "wrist":
        return [2]
    if target == "primary":
        return [3]
    raise ValueError(f"unknown target: {target}")


def select_target_from_capture(hidden: torch.Tensor, target: str) -> torch.Tensor:
    """Select target slots from tensors saved with CAPTURE_SLOT_ORDER=[2,3,4]."""

    if hidden.ndim < 2:
        raise ValueError(f"expected hidden with slot dimension, got {tuple(hidden.shape)}")
    slot_dim = 1 if hidden.ndim == 5 else 0
    index = {
        "wrist": [0],
        "primary": [1],
        "video": [0, 1],
        "action": [2],
        "video_action": [0, 1, 2],
    }.get(target)
    if index is None:
        raise ValueError(f"unknown target: {target}")
    idx = torch.tensor(index, dtype=torch.long, device=hidden.device)
    return torch.index_select(hidden, slot_dim, idx)


def parse_layer_spec(spec: str, available: list[int]) -> int:
    if spec == "last":
        return max(available)
    layer = int(spec)
    if layer not in available:
        raise KeyError(f"layer {layer} not available; have {available}")
    return layer


def action_rel_error(action: np.ndarray, clean: np.ndarray) -> float:
    a = action.reshape(-1).astype(np.float64)
    c = clean.reshape(-1).astype(np.float64)
    return float(np.linalg.norm(a - c) / (np.linalg.norm(c) + EPS))


def action_mse(action: np.ndarray, clean: np.ndarray) -> float:
    return float(np.mean((action.astype(np.float64) - clean.astype(np.float64)) ** 2))


def hidden_recovery(clean: torch.Tensor, pert: torch.Tensor, corrected: torch.Tensor) -> dict[str, float]:
    clean = clean.detach().float()
    pert = pert.detach().float()
    corrected = corrected.detach().float()
    base = torch.mean((pert - clean) ** 2).item()
    corr = torch.mean((corrected - clean) ** 2).item()
    return {
        "baseline_hidden_mse": float(base),
        "corrected_hidden_mse": float(corr),
        "hidden_recovery_vs_pert": 1.0 - corr / (base + EPS),
    }


class LatentShiftMLP(torch.nn.Module):
    """Small token-wise residual predictor: h_pert -> clean-minus-pert shift."""

    def __init__(self, dim: int = 2048, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def load_corrector(path: str | Path, device: torch.device | str) -> tuple[LatentShiftMLP, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = LatentShiftMLP(
        dim=int(cfg["dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


class DynamicCorrectionContext(AbstractContextManager):
    """Apply a learned correction at one DiT block entry on conditioned passes."""

    def __init__(
        self,
        model_or_blocks,
        corrector: torch.nn.Module,
        *,
        layer: int,
        target: str,
        target_rms: float,
        alpha: float = 1.0,
        condition_pass_only: bool = True,
    ) -> None:
        self.blocks = model_or_blocks.net.blocks if hasattr(model_or_blocks, "net") else model_or_blocks
        self.corrector = corrector
        self.layer = int(layer)
        self.indices = target_model_indices(target)
        self.target_rms = float(target_rms)
        self.alpha = float(alpha)
        self.condition_pass_only = bool(condition_pass_only)
        self.pass_idx = -1
        self.before: torch.Tensor | None = None
        self.after: torch.Tensor | None = None
        self.pred_delta: torch.Tensor | None = None
        self.handles = []

    def _active_pass(self) -> bool:
        return (not self.condition_pass_only) or self.pass_idx % 2 == 0

    def _hook(self, layer_idx: int):
        def hook(_module, inputs):
            if layer_idx == 0:
                self.pass_idx += 1
            if layer_idx != self.layer or not self._active_pass() or not inputs:
                return None
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or hidden.ndim != 5:
                return None
            target_hidden = hidden[:, self.indices]
            self.before = target_hidden.detach().float().cpu().clone()
            with torch.no_grad():
                pred = self.corrector(target_hidden.float()) * self.target_rms * self.alpha
                pred = pred.to(dtype=target_hidden.dtype, device=target_hidden.device)
            shifted = hidden.clone()
            shifted[:, self.indices] = target_hidden + pred
            self.after = shifted[:, self.indices].detach().float().cpu().clone()
            self.pred_delta = pred.detach().float().cpu().clone()
            return (shifted,) + inputs[1:]

        return hook

    def __enter__(self):
        hook_layers = sorted({0, self.layer})
        for layer_idx in hook_layers:
            if layer_idx < 0 or layer_idx >= len(self.blocks):
                raise ValueError(f"layer {layer_idx} out of range [0, {len(self.blocks) - 1}]")
            self.handles.append(self.blocks[layer_idx].register_forward_pre_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False
