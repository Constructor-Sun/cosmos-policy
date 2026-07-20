"""Layer-entry mean-shift hooks for Cosmos Policy DiT blocks."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional

import torch


class CosmosLayerCaptureIntervener:
    """Capture and optionally shift selected latent slots at DiT block entry.

    Cosmos Policy uses latent sequence layout [B, T, H, W, D].  For LIBERO,
    the current video slots are T=[2, 3] and the action chunk slot is T=4.
    This helper captures ``x[:, target_indices]`` before selected blocks and
    can add a token-wise direction at one block entrance.

    In CFG sampling, the network is called twice per denoising step:
    conditioned pass first, then unconditioned pass.  By default this helper
    captures/intervenes only on conditioned passes to match Phase 2 semantics.
    """

    def __init__(
        self,
        *,
        action_index: Optional[int] = None,
        target_indices: Optional[list[int] | tuple[int, ...]] = None,
        condition_pass_only: bool = True,
    ) -> None:
        if target_indices is None:
            if action_index is None:
                raise ValueError("either action_index or target_indices must be set")
            target_indices = [int(action_index)]
        self.target_indices = [int(index) for index in target_indices]
        if not self.target_indices:
            raise ValueError("target_indices must be non-empty")
        self.action_index = int(action_index) if action_index is not None else None
        self.condition_pass_only = bool(condition_pass_only)
        self.capture_layers: set[int] = set()
        self.intervene_layer: Optional[int] = None
        self.direction: Optional[torch.Tensor] = None
        self.alpha: float = 0.0
        self.pass_idx: int = -1
        self.captured: dict[int, torch.Tensor] = {}
        self.after: dict[int, torch.Tensor] = {}

    def reset(
        self,
        *,
        capture_layers: Optional[set[int] | list[int] | tuple[int, ...]] = None,
        intervene_layer: Optional[int] = None,
        direction: Optional[torch.Tensor] = None,
        alpha: float = 0.0,
    ) -> None:
        self.capture_layers = {int(layer) for layer in capture_layers or []}
        self.intervene_layer = None if intervene_layer is None else int(intervene_layer)
        self.direction = direction
        self.alpha = float(alpha)
        self.pass_idx = -1
        self.captured = {}
        self.after = {}

    def _active_pass(self) -> bool:
        if not self.condition_pass_only:
            return True
        return self.pass_idx % 2 == 0

    def _target_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, self.target_indices]

    def _direction_for(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if self.direction is None:
            raise RuntimeError("intervention requested without a direction tensor")
        direction = self.direction.to(device=target_hidden.device, dtype=target_hidden.dtype)
        if direction.shape == target_hidden.shape:
            return direction
        if direction.ndim == target_hidden.ndim - 1:
            return direction.unsqueeze(0)
        if direction.ndim == target_hidden.ndim - 2:
            return direction.unsqueeze(0).unsqueeze(0)
        raise RuntimeError(
            "direction shape does not match target hidden shape: "
            f"direction={tuple(direction.shape)} hidden={tuple(target_hidden.shape)}"
        )

    def pre_forward_hook(self, layer_idx: int):
        """Return a ``register_forward_pre_hook`` callback for one block."""

        layer_idx = int(layer_idx)

        def hook(_module, inputs):
            if layer_idx == 0:
                self.pass_idx += 1

            if not self._active_pass():
                return None
            if not inputs:
                return None

            hidden = inputs[0]
            if not torch.is_tensor(hidden) or hidden.ndim != 5:
                return None

            target_hidden = self._target_hidden(hidden)
            if layer_idx in self.capture_layers:
                self.captured[layer_idx] = target_hidden.detach().float().cpu().clone()

            if self.intervene_layer is None or layer_idx != self.intervene_layer:
                return None

            if float(self.alpha) == 0.0:
                self.after[layer_idx] = target_hidden.detach().float().cpu().clone()
                return None

            direction = self._direction_for(target_hidden)
            shifted = hidden.clone()
            shifted[:, self.target_indices] = target_hidden + self.alpha * direction
            self.after[layer_idx] = (
                shifted[:, self.target_indices].detach().float().cpu().clone()
            )
            return (shifted,) + inputs[1:]

        return hook


class layer_shift_context(AbstractContextManager):
    """Temporarily register DiT block-entry hooks."""

    def __init__(
        self,
        model_or_blocks,
        intervener: CosmosLayerCaptureIntervener,
        hook_layers: list[int] | tuple[int, ...] | set[int],
    ) -> None:
        if hasattr(model_or_blocks, "net") and hasattr(model_or_blocks.net, "blocks"):
            self._blocks = model_or_blocks.net.blocks
        else:
            self._blocks = model_or_blocks
        # Always hook layer 0 so pass parity is tracked for CFG.
        self._hook_layers = sorted({0, *[int(layer) for layer in hook_layers]})
        self._intervener = intervener
        self._handles: list = []

    def __enter__(self):
        num_blocks = len(self._blocks)
        for layer_idx in self._hook_layers:
            if layer_idx < 0 or layer_idx >= num_blocks:
                raise ValueError(f"layer {layer_idx} out of range [0, {num_blocks - 1}]")
            handle = self._blocks[layer_idx].register_forward_pre_hook(
                self._intervener.pre_forward_hook(layer_idx)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


def token_mean_shift_direction(
    clean_hidden: torch.Tensor,
    pert_hidden: torch.Tensor,
) -> torch.Tensor:
    """Return clean-minus-perturbation direction.

    Inputs are normally [B, T_target, H, W, D].  If B > 1, average over batch
    while keeping the selected latent-slot dimension.
    """

    diff = clean_hidden.detach().float() - pert_hidden.detach().float()
    if diff.ndim == 5 and diff.shape[0] > 1:
        return diff.mean(dim=0, keepdim=True)
    return diff
