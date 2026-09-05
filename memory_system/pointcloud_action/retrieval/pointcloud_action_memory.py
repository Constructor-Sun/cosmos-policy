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
from memory_system.pointcloud_action.schema import MEMORY_FORMAT


class PointCloudActionMemory:
    """Index over pointcloud_action_memory.pt records."""

    def __init__(
        self,
        path: str | Path,
        cloud_key: str = "target_points_object",
    ):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != MEMORY_FORMAT:
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
    ) -> list[dict[str, Any]]:
        """Return top-k memory records by ESF chi2 + OBB size distance."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) < 4:
            return []

        transform, extent = compute_obb(points)
        normalized = normalize_points(points, transform, extent)
        query_esf = compute_esf(normalized)
        query_extent = extent_key(extent)

        candidates = []
        for item in self._index:
            if skill and item["skill"] != skill:
                continue
            if not size_ratio_within(query_extent, item["extent"], size_threshold):
                continue
            size_distance = float(
                np.linalg.norm(
                    np.log(query_extent / np.maximum(item["extent"], 1e-8))
                )
            )
            distance = chi2_distance(query_esf, item["esf"]) + lambda_size * size_distance
            candidates.append((distance, item["record"]))

        candidates.sort(key=lambda value: value[0])
        return [
            {"record": record, "distance": float(distance)}
            for distance, record in candidates[:top_k]
        ]
