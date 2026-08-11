"""Cosmos Policy subclass with an isolated latent canonicalizer."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch import Tensor

from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_policy._src.predict2.utils.optim_instantiate import get_base_scheduler
from cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldCondition
from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldConfig,
    CosmosPolicyVideo2WorldModel,
)

from .latent_canonicalizer import LatentCanonicalizer
from .model_config import AdapterCosmosPolicyConfig


def _indices(indices: Tensor, latent_state: Tensor) -> Tensor:
    indices = indices.to(device=latent_state.device, dtype=torch.long)
    if indices.ndim != 1 or indices.shape[0] != latent_state.shape[0]:
        raise ValueError("camera indices must have shape [B]")
    if torch.any(indices < 0) or torch.any(indices >= latent_state.shape[2]):
        raise ValueError("camera indices are outside the latent temporal dimension")
    return indices


def gather_camera_latents(
    latent_state: Tensor, wrist_indices: Tensor, image_indices: Tensor
) -> Tensor:
    """Extract current wrist/front latents as ``[B,C,2,H,W]``."""
    wrist_indices = _indices(wrist_indices, latent_state)
    image_indices = _indices(image_indices, latent_state)
    batch = torch.arange(latent_state.shape[0], device=latent_state.device)
    wrist = latent_state[batch, :, wrist_indices, :, :]
    image = latent_state[batch, :, image_indices, :, :]
    return torch.stack((wrist, image), dim=2)


def scatter_camera_latents(
    latent_state: Tensor,
    corrected: Tensor,
    wrist_indices: Tensor,
    image_indices: Tensor,
) -> Tensor:
    """Write corrected current camera latents into a cloned temporal tensor."""
    wrist_indices = _indices(wrist_indices, latent_state)
    image_indices = _indices(image_indices, latent_state)
    result = latent_state.clone()
    # The adapter may run in float32 while the Cosmos conditioning tensor is
    # bfloat16 under mixed-precision training. Advanced-index assignment does
    # not perform implicit dtype conversion, so align the corrected latents
    # with their destination before scattering them back. Tensor.to remains
    # differentiable, preserving gradients to the adapter parameters.
    corrected = corrected.to(device=result.device, dtype=result.dtype)
    batch = torch.arange(result.shape[0], device=result.device)
    result[batch, :, wrist_indices, :, :] = corrected[:, :, 0]
    result[batch, :, image_indices, :, :] = corrected[:, :, 1]
    return result


class AdapterCosmosPolicyVideo2WorldModel(CosmosPolicyVideo2WorldModel):
    """Policy model that adapts only the current wrist/front condition frames."""

    def __init__(
        self,
        config: CosmosPolicyVideo2WorldConfig,
        adapter_config: AdapterCosmosPolicyConfig | dict,
    ):
        super().__init__(config)
        if isinstance(adapter_config, dict):
            adapter_config = AdapterCosmosPolicyConfig(**adapter_config)
        self.adapter_config = adapter_config
        self.latent_adapter = LatentCanonicalizer(
            model_name=adapter_config.adapter_model_name,
            channels=adapter_config.adapter_channels,
            slots=adapter_config.adapter_slots,
            height=adapter_config.adapter_height,
            width=adapter_config.adapter_width,
            hidden_dim=adapter_config.adapter_hidden_dim,
            mlp_dim=adapter_config.adapter_mlp_dim,
            num_heads=adapter_config.adapter_num_heads,
            dropout=adapter_config.adapter_dropout,
            gate_hidden_dim=adapter_config.adapter_gate_hidden_dim,
            initial_gate=adapter_config.adapter_initial_gate,
        ).to(dtype=torch.bfloat16)
        if adapter_config.adapter_freeze_backbone:
            for parameter in super().parameters():
                parameter.requires_grad_(False)
            for parameter in self.latent_adapter.parameters():
                parameter.requires_grad_(True)

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Create the optimizer for the adapter rather than the frozen backbone."""
        optimizer = lazy_instantiate(optimizer_config, model=self.latent_adapter)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    def state_dict(self) -> dict[str, Any]:
        """Include the out-of-backbone adapter in Cosmos DCP checkpoints."""
        state = super().state_dict()
        state.update(self.latent_adapter.state_dict(prefix="latent_adapter."))
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        adapter_state = OrderedDict(
            (key.removeprefix("latent_adapter."), value)
            for key, value in state_dict.items()
            if key.startswith("latent_adapter.")
        )
        backbone_state = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if not key.startswith("latent_adapter.")
        )
        result = super().load_state_dict(backbone_state, strict=strict, assign=assign)
        if adapter_state:
            self.latent_adapter.load_state_dict(adapter_state, strict=True, assign=assign)
        elif strict:
            raise RuntimeError("checkpoint is missing latent_adapter weights")
        return result

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Video2WorldCondition]:
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        wrist_indices = data_batch["current_wrist_image_latent_idx"]
        image_indices = data_batch["current_image_latent_idx"]
        camera_latents = gather_camera_latents(latent_state, wrist_indices, image_indices)
        # The adapter parameters are BF16 so that the optimizer's FP32 master
        # weight path can manage the only trainable parameter group. Autocast
        # also converts the FP32 latent input for the BF16 adapter operations.
        autocast_device = camera_latents.device.type
        if autocast_device == "cuda":
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                corrected = self.latent_adapter(camera_latents)
        else:
            corrected = self.latent_adapter(camera_latents)
        condition.gt_frames = scatter_camera_latents(
            condition.gt_frames, corrected, wrist_indices, image_indices
        )
        return raw_state, latent_state, condition

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
