"""Shared helpers for Phase 8 first-chunk latent correction."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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


class RMSNorm(torch.nn.Module):
    """RMSNorm over the final hidden dimension.

    This local implementation avoids depending on the PyTorch version's
    ``torch.nn.RMSNorm`` availability and preserves the input dtype on output.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        hidden_float = hidden.float()
        rms_inv = torch.rsqrt(hidden_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (hidden_float * rms_inv).to(input_dtype) * self.weight


class LatentShiftMLP(torch.nn.Module):
    """Small token-wise predictor for the clean-minus-perturbed latent shift.

    The residual connection is applied when the predicted shift is injected:
    ``corrected_hidden = hidden + predicted_shift``.  Keeping the MLP output as
    a shift (rather than a full hidden state) avoids adding the input twice.
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        norm: str = "rmsnorm",
    ):
        super().__init__()
        norm = str(norm).lower()
        if norm == "rmsnorm":
            norm_layer = RMSNorm(dim)
        elif norm == "layernorm":
            # Compatibility for checkpoints produced before the RMSNorm change.
            norm_layer = torch.nn.LayerNorm(dim)
        else:
            raise ValueError(f"unknown normalization: {norm}")
        self.norm_type = norm
        self.net = torch.nn.Sequential(
            norm_layer,
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)

    def apply_residual(
        self,
        hidden: torch.Tensor,
        *,
        target_rms: float = 1.0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Return ``hidden + scaled predicted shift`` explicitly."""
        shift = self(hidden) * float(target_rms) * float(alpha)
        return hidden + shift


class LatentShiftCVAE(torch.nn.Module):
    """Conditional VAE for ``p(shift | perturbed_hidden)``.

    The conditional prior sees only the perturbed hidden state and is therefore
    available at inference time.  During training the posterior additionally
    sees the true normalized shift.  A sample-level latent is broadcast to all
    target tokens so one draw represents a coherent full-grid shift mode.

    Calling ``forward(hidden)`` decodes the conditional-prior mean for stable
    online correction.  Calling ``forward(hidden, sample=True)`` or
    ``sample_shifts`` draws stochastic candidates from the conditional prior.
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        norm: str = "rmsnorm",
        latent_dim: int = 32,
        min_logvar: float = -10.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")

        norm = str(norm).lower()
        if norm == "rmsnorm":
            input_norm = RMSNorm(dim)
        elif norm == "layernorm":
            input_norm = torch.nn.LayerNorm(dim)
        else:
            raise ValueError(f"unknown normalization: {norm}")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.norm_type = norm

        self.input_norm = input_norm
        self.condition_projection = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.target_projection = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.prior_network = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.posterior_network = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
        )

    def _validate_hidden(self, hidden: torch.Tensor) -> None:
        if hidden.ndim < 2:
            raise ValueError(f"CVAE expects a batch and hidden dimension, got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.dim:
            raise ValueError(f"expected hidden dim {self.dim}, got {hidden.shape[-1]}")

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(features.shape[0], -1, self.hidden_dim)
        mean = flat.mean(dim=1)
        rms = flat.float().square().mean(dim=1).add(1e-8).sqrt().to(mean.dtype)
        return torch.cat([mean, rms], dim=-1)

    def _distribution(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = parameters.chunk(2, dim=-1)
        return mean, logvar.clamp(self.min_logvar, self.max_logvar)

    def encode_prior(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_hidden(hidden)
        condition_features = self.condition_projection(self.input_norm(hidden))
        condition_summary = self._pool(condition_features)
        prior_mean, prior_logvar = self._distribution(self.prior_network(condition_summary))
        return condition_features, condition_summary, prior_mean, prior_logvar

    def encode_posterior(
        self,
        condition_summary: torch.Tensor,
        target_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_hidden(target_shift)
        target_summary = self._pool(self.target_projection(target_shift))
        parameters = self.posterior_network(torch.cat([condition_summary, target_summary], dim=-1))
        return self._distribution(parameters)

    @staticmethod
    def kl_divergence(
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Return KL(q || p) separately for every sample and latent dimension."""
        variance_ratio = torch.exp(posterior_logvar - prior_logvar)
        mean_term = (posterior_mean - prior_mean).square() * torch.exp(-prior_logvar)
        return 0.5 * (prior_logvar - posterior_logvar + variance_ratio + mean_term - 1.0)

    @staticmethod
    def _reparameterize(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + torch.exp(0.5 * logvar) * noise

    def decode(self, condition_features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape != (condition_features.shape[0], self.latent_dim):
            raise ValueError(
                f"expected latent [{condition_features.shape[0]}, {self.latent_dim}], "
                f"got {tuple(latent.shape)}"
            )
        broadcast_shape = (latent.shape[0],) + (1,) * (condition_features.ndim - 2) + (self.latent_dim,)
        expanded = latent.reshape(broadcast_shape).expand(*condition_features.shape[:-1], self.latent_dim)
        return self.decoder(torch.cat([condition_features, expanded], dim=-1))

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        target_shift: torch.Tensor | None = None,
        sample: bool | None = None,
        return_stats: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        condition_features, condition_summary, prior_mean, prior_logvar = self.encode_prior(hidden)
        posterior_mean = posterior_logvar = None
        if target_shift is not None:
            if target_shift.shape != hidden.shape:
                raise ValueError(
                    f"target shift shape {tuple(target_shift.shape)} does not match hidden {tuple(hidden.shape)}"
                )
            posterior_mean, posterior_logvar = self.encode_posterior(condition_summary, target_shift)
            latent_mean, latent_logvar = posterior_mean, posterior_logvar
        else:
            latent_mean, latent_logvar = prior_mean, prior_logvar

        if sample is None:
            sample = target_shift is not None
        latent = (
            self._reparameterize(latent_mean, latent_logvar, generator=generator)
            if sample
            else latent_mean
        )
        prediction = self.decode(condition_features, latent)
        if not return_stats:
            return prediction

        stats = {
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "prior_std": torch.exp(0.5 * prior_logvar),
        }
        if posterior_mean is not None and posterior_logvar is not None:
            stats.update(
                {
                    "posterior_mean": posterior_mean,
                    "posterior_logvar": posterior_logvar,
                    "posterior_std": torch.exp(0.5 * posterior_logvar),
                    "kl_per_dim": self.kl_divergence(
                        posterior_mean,
                        posterior_logvar,
                        prior_mean,
                        prior_logvar,
                    ),
                }
            )
        return prediction, stats

    def sample_shifts(
        self,
        hidden: torch.Tensor,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw prior samples with shape ``[K, *hidden.shape]``."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        condition_features, _summary, prior_mean, prior_logvar = self.encode_prior(hidden)
        predictions = []
        for _ in range(num_samples):
            latent = self._reparameterize(prior_mean, prior_logvar, generator=generator)
            predictions.append(self.decode(condition_features, latent))
        return torch.stack(predictions, dim=0)

    def apply_residual(
        self,
        hidden: torch.Tensor,
        *,
        target_rms: float = 1.0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        # Online correction uses the conditional-prior mean, not a random draw.
        shift = self(hidden, sample=False) * float(target_rms) * float(alpha)
        return hidden + shift


class LowRankShiftExpert(torch.nn.Module):
    """Predict shift coefficients and expand them through a learned dense basis."""

    def __init__(self, dim: int, hidden_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.coefficient_net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, rank),
        )
        # Weight shape is (dim, rank), so its columns are the learned dense
        # latent-shift directions. Orthogonal initialization starts with a
        # well-conditioned subspace rather than duplicated directions.
        self.basis = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.orthogonal_(self.basis.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.basis(self.coefficient_net(hidden))

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize duplicate directions while allowing basis-vector scaling."""
        directions = F.normalize(self.basis.weight.float(), dim=0)
        gram = directions.transpose(0, 1) @ directions
        identity = torch.eye(self.rank, dtype=gram.dtype, device=gram.device)
        return (gram - identity).square().mean()


class LatentShiftMoE(torch.nn.Module):
    """Sample-routed mixture of latent-shift experts.

    In ``partitioned`` mode expert ``i`` predicts only a contiguous slice of
    the final hidden dimension. The slices are disjoint and jointly cover the
    full dimension, making expert specialization explicit. ``full`` mode is a
    conventional MoE in which every expert predicts the entire shift.
    ``low_rank`` mode gives every expert a learned dense ``dim x expert_rank``
    basis and predicts only the coefficients in that subspace.

    Routing is performed once per batch sample. All token/spatial axes are
    pooled into mean and RMS features before the router selects top-k experts.
    Only selected experts are evaluated to keep activation memory bounded.
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        norm: str = "rmsnorm",
        num_experts: int = 16,
        top_k_experts: int = 2,
        expert_mode: str = "full",
        expert_rank: int = 32,
        router_temperature: float = 1.0,
        router_noise: float = 0.1,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if top_k_experts < 1 or top_k_experts > num_experts:
            raise ValueError("top_k_experts must be in [1, num_experts]")
        if expert_mode not in {"partitioned", "full", "low_rank"}:
            raise ValueError(f"unknown expert_mode: {expert_mode}")
        if expert_rank < 1 or expert_rank > dim:
            raise ValueError("expert_rank must be in [1, dim]")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")

        norm = str(norm).lower()
        if norm == "rmsnorm":
            self.norm = RMSNorm(dim)
        elif norm == "layernorm":
            self.norm = torch.nn.LayerNorm(dim)
        else:
            raise ValueError(f"unknown normalization: {norm}")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k_experts = int(top_k_experts)
        self.expert_mode = expert_mode
        self.expert_rank = int(expert_rank)
        self.router_temperature = float(router_temperature)
        self.router_noise = float(router_noise)
        self.norm_type = norm

        # Mean captures signed global structure while RMS preserves localized
        # energy that could cancel under mean pooling.
        self.router = torch.nn.Linear(2 * dim, num_experts)
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.router.bias)

        if expert_mode == "partitioned":
            base, remainder = divmod(dim, num_experts)
            if base == 0:
                raise ValueError("partitioned MoE requires num_experts <= dim")
            widths = [base + int(idx < remainder) for idx in range(num_experts)]
            self.expert_slices = []
            offset = 0
            for width in widths:
                self.expert_slices.append((offset, offset + width))
                offset += width
            self.experts = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(dim, hidden_dim),
                        torch.nn.GELU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_dim, width),
                    )
                    for width in widths
                ]
            )
        elif expert_mode == "low_rank":
            self.expert_slices = [(0, dim)] * num_experts
            self.experts = torch.nn.ModuleList(
                [
                    LowRankShiftExpert(
                        dim=dim,
                        hidden_dim=hidden_dim,
                        rank=expert_rank,
                        dropout=dropout,
                    )
                    for _ in range(num_experts)
                ]
            )
        else:
            widths = [dim] * num_experts
            self.expert_slices = [(0, dim)] * num_experts
            self.experts = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(dim, hidden_dim),
                        torch.nn.GELU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_dim, width),
                    )
                    for width in widths
                ]
            )

    def basis_orthogonality_loss(self) -> torch.Tensor:
        if self.expert_mode != "low_rank":
            return self.router.weight.new_zeros(())
        losses = [expert.orthogonality_loss() for expert in self.experts]
        return torch.stack(losses).mean()

    def _route(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if normalized.ndim < 2:
            raise ValueError(f"MoE expects a batch and hidden dimension, got {tuple(normalized.shape)}")
        pooled = normalized.reshape(normalized.shape[0], -1, self.dim)
        mean = pooled.mean(dim=1)
        rms = pooled.float().square().mean(dim=1).add(1e-8).sqrt().to(mean.dtype)
        clean_logits = self.router(torch.cat([mean, rms], dim=-1)) / self.router_temperature
        route_logits = clean_logits
        if self.training and self.router_noise > 0:
            route_logits = route_logits + torch.randn_like(route_logits) * self.router_noise

        router_probs = torch.softmax(clean_logits.float(), dim=-1)
        top_values, top_indices = torch.topk(route_logits, self.top_k_experts, dim=-1)
        if self.expert_mode == "partitioned":
            if self.top_k_experts == 1:
                # Softmax over one value is always one and gives the router no
                # reconstruction gradient; sigmoid retains that signal.
                top_gates = 2.0 * torch.sigmoid(top_values.float())
            else:
                # Disjoint subspaces need roughly unit-scale gates. Multiplying
                # by k makes a uniform top-k distribution equal to one.
                top_gates = self.top_k_experts * torch.softmax(top_values.float(), dim=-1)
        else:
            top_gates = torch.softmax(top_values.float(), dim=-1)
        top_gates = top_gates.to(normalized.dtype)

        selection = F.one_hot(top_indices, num_classes=self.num_experts).float()
        load = selection.mean(dim=(0, 1))
        importance = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(importance * load)
        router_z_loss = torch.logsumexp(clean_logits.float(), dim=-1).square().mean()
        entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        stats = {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "router_entropy": entropy.detach(),
            "expert_load": load.detach(),
            "expert_importance": importance.detach(),
        }
        return top_indices, top_gates, stats

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_router_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized = self.norm(hidden)
        top_indices, top_gates, stats = self._route(normalized)

        if self.expert_mode == "partitioned":
            chunks = []
            for expert_idx, (start, end) in enumerate(self.expert_slices):
                chunk = hidden.new_zeros((*hidden.shape[:-1], end - start))
                sample_idx, choice_idx = torch.where(top_indices == expert_idx)
                if sample_idx.numel():
                    expert_output = self.experts[expert_idx](normalized.index_select(0, sample_idx))
                    gate_shape = (sample_idx.numel(),) + (1,) * (expert_output.ndim - 1)
                    weighted = expert_output * top_gates[sample_idx, choice_idx].view(gate_shape)
                    chunk.index_add_(0, sample_idx, weighted)
                chunks.append(chunk)
            output = torch.cat(chunks, dim=-1)
        else:
            output = torch.zeros_like(hidden)
            for expert_idx, expert in enumerate(self.experts):
                sample_idx, choice_idx = torch.where(top_indices == expert_idx)
                if sample_idx.numel():
                    expert_output = expert(normalized.index_select(0, sample_idx))
                    gate_shape = (sample_idx.numel(),) + (1,) * (expert_output.ndim - 1)
                    weighted = expert_output * top_gates[sample_idx, choice_idx].view(gate_shape)
                    output.index_add_(0, sample_idx, weighted)

        if return_router_stats:
            return output, stats
        return output

    def apply_residual(
        self,
        hidden: torch.Tensor,
        *,
        target_rms: float = 1.0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        shift = self(hidden) * float(target_rms) * float(alpha)
        return hidden + shift


def load_corrector(path: str | Path, device: torch.device | str) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    common = {
        "dim": int(cfg["dim"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "dropout": float(cfg.get("dropout", 0.0)),
        "norm": str(cfg.get("norm", "layernorm")),
    }
    architecture = cfg.get("architecture", "mlp")
    if architecture == "moe":
        model = LatentShiftMoE(
            **common,
            num_experts=int(cfg["num_experts"]),
            top_k_experts=int(cfg["top_k_experts"]),
            expert_mode=str(cfg.get("expert_mode", "full")),
            expert_rank=int(cfg.get("expert_rank", 32)),
            router_temperature=float(cfg.get("router_temperature", 1.0)),
            # ``model.eval()`` below disables noise while preserving the
            # training configuration in the reconstructed module.
            router_noise=float(cfg.get("router_noise", 0.0)),
        ).to(device)
    elif architecture == "cvae":
        model = LatentShiftCVAE(
            **common,
            latent_dim=int(cfg.get("latent_dim", 32)),
            min_logvar=float(cfg.get("min_logvar", -10.0)),
            max_logvar=float(cfg.get("max_logvar", 6.0)),
        ).to(device)
    else:
        model = LatentShiftMLP(**common).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


class DynamicCorrectionContext(AbstractContextManager):
    """Apply a learned correction at one DiT block entry or exit on conditioned passes.

    Parameters
    ----------
    hook_mode : ``"pre"`` (default) or ``"forward"``
        ``"pre"`` injects at block entry (before self-attention / cross-attention /
        MLP).  ``"forward"`` injects at block exit — after all attention and MLP,
        so the correction is NOT subsequently mixed with uncorrected video tokens.
    """

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
        hook_mode: str = "pre",
    ) -> None:
        self.blocks = model_or_blocks.net.blocks if hasattr(model_or_blocks, "net") else model_or_blocks
        self.corrector = corrector
        self.layer = int(layer)
        self.indices = target_model_indices(target)
        self.target_rms = float(target_rms)
        self.alpha = float(alpha)
        self.condition_pass_only = bool(condition_pass_only)
        if hook_mode not in ("pre", "forward"):
            raise ValueError(f"hook_mode must be 'pre' or 'forward', got {hook_mode!r}")
        self.hook_mode = hook_mode
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
                if hasattr(self.corrector, "apply_residual"):
                    corrected = self.corrector.apply_residual(
                        target_hidden.float(),
                        target_rms=self.target_rms,
                        alpha=self.alpha,
                    )
                    pred = corrected - target_hidden.float()
                else:
                    # Keep compatibility with custom shift predictors.
                    pred = self.corrector(target_hidden.float()) * self.target_rms * self.alpha
                    corrected = target_hidden.float() + pred
                pred = pred.to(dtype=target_hidden.dtype, device=target_hidden.device)
                corrected = corrected.to(dtype=target_hidden.dtype, device=target_hidden.device)
            shifted = hidden.clone()
            shifted[:, self.indices] = corrected
            self.after = shifted[:, self.indices].detach().float().cpu().clone()
            self.pred_delta = pred.detach().float().cpu().clone()
            return (shifted,) + inputs[1:]

        return hook

    def _forward_hook(self, layer_idx: int):
        """Return a ``register_forward_hook`` callback — injects at block *exit*."""

        def hook(_module, _inputs, output):
            if layer_idx == 0:
                self.pass_idx += 1
            if layer_idx != self.layer or not self._active_pass():
                return None

            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 5:
                return None

            target_hidden = hidden[:, self.indices]
            self.before = target_hidden.detach().float().cpu().clone()
            with torch.no_grad():
                if hasattr(self.corrector, "apply_residual"):
                    corrected = self.corrector.apply_residual(
                        target_hidden.float(),
                        target_rms=self.target_rms,
                        alpha=self.alpha,
                    )
                    pred = corrected - target_hidden.float()
                else:
                    pred = self.corrector(target_hidden.float()) * self.target_rms * self.alpha
                    corrected = target_hidden.float() + pred
                pred = pred.to(dtype=target_hidden.dtype, device=target_hidden.device)
                corrected = corrected.to(dtype=target_hidden.dtype, device=target_hidden.device)
            shifted = hidden.clone()
            shifted[:, self.indices] = corrected
            self.after = shifted[:, self.indices].detach().float().cpu().clone()
            self.pred_delta = pred.detach().float().cpu().clone()

            if isinstance(output, tuple):
                return (shifted,) + output[1:]
            return shifted

        return hook

    def __enter__(self):
        hook_layers = sorted({0, self.layer})
        for layer_idx in hook_layers:
            if layer_idx < 0 or layer_idx >= len(self.blocks):
                raise ValueError(f"layer {layer_idx} out of range [0, {len(self.blocks) - 1}]")
            if self.hook_mode == "forward":
                self.handles.append(
                    self.blocks[layer_idx].register_forward_hook(self._forward_hook(layer_idx))
                )
            else:
                self.handles.append(
                    self.blocks[layer_idx].register_forward_pre_hook(self._hook(layer_idx))
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False
