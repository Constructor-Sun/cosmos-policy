#!/usr/bin/env python3
"""Small helper for Hessian Ritz-vector/shift alignment."""

from __future__ import annotations

import json
import math
from typing import Any

import torch


EPS = 1e-12


def ritz_shift_alignment(
    basis: list[torch.Tensor],
    tri_eigvecs: torch.Tensor,
    shift: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    """Return |cos(shift, v1)| and energy in the largest-k Ritz subspace."""
    shift = shift.to(device=basis[0].device, dtype=basis[0].dtype)
    shift_l2 = float(torch.linalg.vector_norm(shift).detach().cpu())
    shift_rms = shift_l2 / math.sqrt(shift.numel())
    k = min(top_k, tri_eigvecs.shape[1])
    if shift_l2 <= EPS or k == 0:
        return {
            "shift_l2": shift_l2,
            "shift_rms": shift_rms,
            "alignment_valid": False,
            "alignment_k": k,
            "abs_cos_top1": float("nan"),
            "max_abs_cos_topk": float("nan"),
            "alignment_energy_topk": float("nan"),
            "alignment_rms_topk": float("nan"),
            "random_energy_topk": k / shift.numel(),
            "abs_cosines_topk": "[]",
        }

    shift_hat = shift / shift_l2
    q_dot_shift = torch.tensor(
        [float(torch.dot(q.reshape(-1), shift_hat.reshape(-1)).detach().cpu()) for q in basis],
        dtype=tri_eigvecs.dtype,
    )
    # eigh is ascending; reverse its last k columns so v1 is lambda_max's vector.
    top_vectors = torch.flip(tri_eigvecs[:, -k:], dims=(1,))
    abs_cosines = torch.clamp(torch.abs(top_vectors.T @ q_dot_shift), max=1.0)
    energy = min(1.0, float(torch.sum(abs_cosines**2)))
    values = [float(value) for value in abs_cosines]
    return {
        "shift_l2": shift_l2,
        "shift_rms": shift_rms,
        "alignment_valid": True,
        "alignment_k": k,
        "abs_cos_top1": values[0],
        "max_abs_cos_topk": max(values),
        "alignment_energy_topk": energy,
        "alignment_rms_topk": math.sqrt(energy / k),
        "random_energy_topk": k / shift.numel(),
        "abs_cosines_topk": json.dumps(values),
    }
