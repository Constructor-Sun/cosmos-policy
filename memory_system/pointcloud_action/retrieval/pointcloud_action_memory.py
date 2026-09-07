"""PointCloud Action Memory retrieval by ESF shape + OBB size."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from memory_system.pointcloud_action.retrieval.descriptors import (
    chi2_distance,
    compute_esf,
    compute_obb,
    extent_key,
    normalize_points,
    size_ratio_within,
)
from memory_system.pointcloud_action.schema import ACCEPTED_MEMORY_FORMATS
class PointCloudActionMemory:
    """Index over pointcloud_action_memory.pt records."""

    def __init__(
        self,
        path: str | Path,
        cloud_key: str = "target_points_object",
    ):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") not in ACCEPTED_MEMORY_FORMATS:
            raise ValueError(f"unsupported format: {payload.get('format')!r}")
        self.cloud_key = cloud_key
        self.records = list(payload.get("records", []))
        self._index = []
        for record in self.records:
            points = record.get(self.cloud_key)
            # Fall back to the legacy visible cloud if the requested key is absent.
            if points is None or len(points) < 4:
                points = record.get("target_points_object")
            if points is None or len(points) < 4:
                continue
            try:
                transform, extent = compute_obb(points)
                normalized = normalize_points(points, transform, extent)
                esf = compute_esf(normalized)
            except Exception:
                continue
            self._index.append(
                {
                    "record": record,
                    "esf": esf,
                    "extent": extent_key(extent),
                    "skill": str(record.get("skill", "Pick")),
                }
            )

    def __len__(self) -> int:
        return len(self._index)

    def retrieve(
        self,
        points: np.ndarray,
        skill: str = "Pick",
        top_k: int = 5,
        size_threshold: float = 1.5,
        lambda_size: float = 1.0,
        item: str | None = None,
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank by shape, optionally restricted to an exact instance pair."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) < 4:
            return []

        transform, extent = compute_obb(points)
        normalized = normalize_points(points, transform, extent)
        query_esf = compute_esf(normalized)
        query_extent = extent_key(extent)

        candidates = []
        for entry in self._index:
            if skill and entry["skill"] != skill:
                continue
            arguments = entry["record"].get("arguments", {})
            if item is not None and (
                arguments.get("item") != item or arguments.get("target") != target
            ):
                continue
            if item is None and not size_ratio_within(
                query_extent, entry["extent"], size_threshold
            ):
                continue
            size_distance = float(
                np.linalg.norm(
                    np.log(query_extent / np.maximum(entry["extent"], 1e-8))
                )
            )
            distance = chi2_distance(query_esf, entry["esf"]) + lambda_size * size_distance
            candidates.append((distance, entry))

        candidates.sort(key=lambda value: value[0])
        return [
            {
                "record": entry["record"],
                "distance": float(distance),
                "key_stage": "exact" if item is not None else "shape",
            }
            for distance, entry in candidates[:top_k]
        ]
