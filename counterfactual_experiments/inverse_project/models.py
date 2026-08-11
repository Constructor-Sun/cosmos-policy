"""Small residual canonicalizers for paired Cosmos video latents."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


MODEL_NAMES = ("mlp", "transformer", "gated_mlp", "glc")


def _fixed_2d_encoding(height: int, width: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("hidden_dim must be divisible by 4")
    quarter = dim // 4
    scale = -math.log(10_000.0) / max(quarter - 1, 1)
    freq = torch.exp(torch.arange(quarter, dtype=torch.float32) * scale)
    y = torch.arange(height, dtype=torch.float32)[:, None] * freq[None]
    x = torch.arange(width, dtype=torch.float32)[:, None] * freq[None]
    y = torch.cat((y.sin(), y.cos()), dim=-1)[:, None, :].expand(-1, width, -1)
    x = torch.cat((x.sin(), x.cos()), dim=-1)[None, :, :].expand(height, -1, -1)
    return torch.cat((y, x), dim=-1)


class TokenMLPCorrection(nn.Module):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class TransformerCorrection(nn.Module):
    def __init__(
        self,
        channels: int,
        slots: int,
        height: int,
        width: int,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.slots, self.height, self.width = slots, height, width
        self.input = nn.Linear(channels, hidden_dim)
        self.slot_embedding = nn.Parameter(torch.zeros(slots, hidden_dim))
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.register_buffer(
            "spatial_encoding",
            _fixed_2d_encoding(height, width, hidden_dim),
            persistent=False,
        )
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        hidden = self.input(tokens).reshape(
            batch, self.slots, self.height, self.width, -1
        )
        hidden = hidden + self.spatial_encoding[None, None]
        hidden = hidden + self.slot_embedding[None, :, None, None]
        hidden = self.block(hidden.flatten(1, 3))
        return self.output(self.output_norm(hidden))


class SampleGate(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 32, initial_gate: float = 0.12) -> None:
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between zero and one")
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(initial_gate / (1.0 - initial_gate)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(self.norm(tokens).mean(dim=1)))


class LatentCanonicalizer(nn.Module):
    """Map ``[B, C, S, H, W]`` perturbed latents to corrected latents."""

    def __init__(
        self,
        model_name: str,
        channels: int = 16,
        slots: int = 2,
        height: int = 28,
        width: int = 28,
        hidden_dim: int = 64,
        mlp_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        gate_hidden_dim: int = 32,
        initial_gate: float = 0.12,
    ) -> None:
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_NAMES}")
        self.config = dict(
            model_name=model_name,
            channels=channels,
            slots=slots,
            height=height,
            width=width,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_hidden_dim=gate_hidden_dim,
            initial_gate=initial_gate,
        )
        self.shape = (channels, slots, height, width)
        if model_name in ("mlp", "gated_mlp"):
            self.correction = TokenMLPCorrection(channels, hidden_dim)
        else:
            self.correction = TransformerCorrection(
                channels, slots, height, width, hidden_dim, mlp_dim, num_heads, dropout
            )
        self.gate = (
            SampleGate(channels, gate_hidden_dim, initial_gate)
            if model_name in ("gated_mlp", "glc")
            else None
        )

    def forward(
        self, latent: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        if latent.ndim != 5 or tuple(latent.shape[1:]) != self.shape:
            raise ValueError(f"expected [B,{','.join(map(str, self.shape))}], got {tuple(latent.shape)}")
        tokens = latent.permute(0, 2, 3, 4, 1).flatten(1, 3)
        shift_tokens = self.correction(tokens)
        gate = self.gate(tokens) if self.gate is not None else None
        if gate is not None:
            shift_tokens = shift_tokens * gate[:, None]
        shift = shift_tokens.reshape(
            latent.shape[0], self.shape[1], self.shape[2], self.shape[3], self.shape[0]
        ).permute(0, 4, 1, 2, 3)
        corrected = latent + shift
        if return_aux:
            return corrected, {"shift": shift, "gate": gate}
        return corrected


def build_model(model_name: str, **kwargs: Any) -> LatentCanonicalizer:
    return LatentCanonicalizer(model_name=model_name, **kwargs)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> LatentCanonicalizer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = LatentCanonicalizer(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)
