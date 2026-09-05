"""Shape/size descriptors for point cloud retrieval.

The ESF descriptor here is a NumPy implementation in the spirit of PCL's
640-dimensional ESF: 10 histograms of 64 bins each over normalized point cloud
shape functions.  It is self-contained and deterministic.
"""
from __future__ import annotations

import numpy as np
from trimesh.bounds import oriented_bounds


def compute_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (transform, extent) from trimesh oriented bounds."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 4:
        raise ValueError("need at least 4 points to compute OBB")
    transform, extent = oriented_bounds(points)
    return np.asarray(transform, dtype=np.float64), np.asarray(extent, dtype=np.float64)


def extent_key(extent: np.ndarray) -> np.ndarray:
    """Return a rotation/axis-invariant size key (sorted descending)."""
    return np.sort(np.asarray(extent, dtype=np.float64))[::-1]


def normalize_points(
    points: np.ndarray,
    transform: np.ndarray,
    extent: np.ndarray,
) -> np.ndarray:
    """Map points into the OBB frame and divide by the OBB extent."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    local = points @ transform[:3, :3].T + transform[:3, 3]
    scale = np.asarray(extent, dtype=np.float64).reshape(3)
    scale[scale < 1e-8] = 1.0
    return local / scale


def _histogram(values: np.ndarray, bins: int, low: float, high: float) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=(low, high))
    return hist.astype(np.float64)


def compute_esf(
    normalized_points: np.ndarray,
    seed: int = 0,
    n_pairs: int = 2048,
    n_triples: int = 1024,
) -> np.ndarray:
    """Compute a 640-D ESF-style descriptor from normalized points."""
    points = np.asarray(normalized_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 8:
        return np.zeros(640, dtype=np.float64)
    rng = np.random.default_rng(seed)
    hist = np.zeros((10, 64), dtype=np.float64)

    pair_idx = rng.integers(0, len(points), size=(n_pairs, 2))
    a = points[pair_idx[:, 0]]
    b = points[pair_idx[:, 1]]
    dist = np.linalg.norm(a - b, axis=1)
    hist[0] += _histogram(dist, 64, 0.0, np.sqrt(3.0))
    hist[1] += _histogram(np.abs(a[:, 0] - b[:, 0]), 64, 0.0, 1.0)
    hist[2] += _histogram(np.abs(a[:, 1] - b[:, 1]), 64, 0.0, 1.0)
    hist[3] += _histogram(np.abs(a[:, 2] - b[:, 2]), 64, 0.0, 1.0)

    tri_idx = rng.integers(0, len(points), size=(n_triples, 3))
    a = points[tri_idx[:, 0]]
    b = points[tri_idx[:, 1]]
    c = points[tri_idx[:, 2]]
    ab = np.linalg.norm(a - b, axis=1)
    bc = np.linalg.norm(b - c, axis=1)
    ca = np.linalg.norm(c - a, axis=1)
    semi = (ab + bc + ca) / 2.0
    area = np.sqrt(np.clip(semi * (semi - ab) * (semi - bc) * (semi - ca), 0.0, None))
    hist[4] += _histogram(area, 64, 0.0, 1.0)
    hist[5] += _histogram(ab, 64, 0.0, np.sqrt(3.0))
    hist[6] += _histogram(bc, 64, 0.0, np.sqrt(3.0))
    hist[7] += _histogram(ca, 64, 0.0, np.sqrt(3.0))

    cos_angle = np.clip((ab**2 + ca**2 - bc**2) / (2.0 * ab * ca + 1e-8), -1.0, 1.0)
    hist[8] += _histogram(np.arccos(cos_angle), 64, 0.0, np.pi)

    radius = np.linalg.norm(points, axis=1)
    hist[9] += _histogram(radius, 64, 0.0, np.sqrt(3.0))

    hist /= (hist.sum(axis=1, keepdims=True) + 1e-12)
    return hist.reshape(-1)


def chi2_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Chi-square distance between two normalized histograms."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = first + second
    mask = denominator > 0.0
    return float(0.5 * np.sum(((first[mask] - second[mask]) ** 2) / denominator[mask]))


def size_ratio_within(
    query_extent: np.ndarray,
    memory_extent: np.ndarray,
    threshold: float = 1.5,
) -> bool:
    """Check that per-dimension size ratios are within a relaxed threshold."""
    query = extent_key(query_extent)
    memory = extent_key(memory_extent)
    ratio = query / np.maximum(memory, 1e-8)
    return bool(np.all(ratio <= threshold) and np.all(ratio >= 1.0 / threshold))
