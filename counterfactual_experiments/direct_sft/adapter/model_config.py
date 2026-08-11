"""Configuration for the isolated latent adapter."""

from __future__ import annotations

import attrs


@attrs.define(slots=False)
class AdapterCosmosPolicyConfig:
    adapter_model_name: str = "transformer"
    adapter_channels: int = 16
    adapter_slots: int = 2
    adapter_height: int = 28
    adapter_width: int = 28
    adapter_hidden_dim: int = 64
    adapter_mlp_dim: int = 128
    adapter_num_heads: int = 4
    adapter_dropout: float = 0.0
    adapter_gate_hidden_dim: int = 32
    adapter_initial_gate: float = 0.12
    adapter_freeze_backbone: bool = True
