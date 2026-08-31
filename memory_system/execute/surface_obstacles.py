"""Simulator-independent RGB-D surface selection and trajectory validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from memory_system.geometry import pixel_to_world
from memory_system.types import CameraParams


@dataclass(frozen=True)
class SurfaceSelection:
    """Bounded obstacle set with accounting for each sampling tier."""

    points: np.ndarray
    global_count: int
    target_count: int
    mandatory_count: int


@dataclass(frozen=True)
class TrajectoryConflict:
    """Dense-surface points intersecting a candidate robot-sphere sweep."""

    point_indices: np.ndarray
    first_step: int
    last_step: int
    min_clearance: float
    min_clearance_point: np.ndarray
    min_clearance_step: int
    min_clearance_sphere_center: np.ndarray


def points_from_depth(depth: np.ndarray, camera: CameraParams) -> np.ndarray:
    """Back-project every valid metric depth pixel into world coordinates."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    rows, cols = np.meshgrid(
        np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij"
    )
    pixels = np.stack([rows.ravel(), cols.ravel()], axis=-1)
    valid = (np.isfinite(depth) & (depth > 0.0)).ravel()
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64)
    return pixel_to_world(pixels[valid], depth, camera)


def filter_robot_points(
    points: np.ndarray,
    robot_spheres: np.ndarray,
    padding: float = 0.005,
) -> np.ndarray:
    """Remove points belonging to the robot in its current configuration."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    spheres = np.asarray(robot_spheres, dtype=np.float64).reshape(-1, 4)
    clearance = np.linalg.norm(
        points[:, None, :] - spheres[None, :, :3], axis=2
    ) - spheres[None, :, 3]
    return points[np.all(clearance > float(padding), axis=1)]


def voxel_representatives(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one deterministic point from every occupied 3D surface voxel."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _farthest_order(points: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic spatially spread prefix of ``points``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = min(max(0, int(count)), len(points))
    if count == 0:
        return points[:0]
    if count == len(points):
        return points
    center = points.mean(axis=0)
    index = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    chosen = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    for output_index in range(count):
        chosen[output_index] = index
        distance = np.minimum(
            distance, np.sum((points - points[index]) ** 2, axis=1)
        )
        distance[chosen[: output_index + 1]] = -1.0
        index = int(np.argmax(distance))
    return points[chosen]


def _append_unique(
    selected: list[np.ndarray],
    keys: set[tuple[int, int, int]],
    candidates: np.ndarray,
    limit: int,
    spacing: float,
) -> int:
    before = len(selected)
    for point in candidates:
        key = tuple(np.floor(point / spacing).astype(np.int64).tolist())
        if key in keys:
            continue
        keys.add(key)
        selected.append(point)
        if len(selected) >= limit:
            break
    return len(selected) - before


def select_surface_points(
    points: np.ndarray,
    target: np.ndarray,
    max_points: int = 512,
    voxel_size: float = 0.02,
    target_fraction: float = 0.25,
    target_radius: float = 0.15,
    target_voxel_size: float = 0.008,
    mandatory_points: np.ndarray | None = None,
) -> SurfaceSelection:
    """Cover all visible surfaces, then add denser goal/path neighborhoods."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    max_points = max(1, int(max_points))
    local_budget = max(1, int(round(max_points * float(target_fraction))))
    mandatory = voxel_representatives(
        np.zeros((0, 3)) if mandatory_points is None else mandatory_points,
        target_voxel_size,
    )
    mandatory = _farthest_order(mandatory, local_budget)
    target_budget = max(0, local_budget - len(mandatory))
    target_distance = np.linalg.norm(points - target, axis=1)
    local = voxel_representatives(
        points[target_distance <= float(target_radius)], target_voxel_size
    )
    if len(local):
        local = local[np.argsort(np.linalg.norm(local - target, axis=1))]
    local = local[:target_budget]
    global_voxels = voxel_representatives(points, voxel_size)
    global_order = _farthest_order(global_voxels, max_points)

    selected: list[np.ndarray] = []
    keys: set[tuple[int, int, int]] = set()
    unique_spacing = min(float(voxel_size), float(target_voxel_size)) * 0.5
    mandatory_count = _append_unique(
        selected, keys, mandatory, max_points, unique_spacing
    )
    target_count = _append_unique(
        selected, keys, local, max_points, unique_spacing
    )
    global_count = _append_unique(
        selected, keys, global_order, max_points, unique_spacing
    )
    selected_array = (
        np.asarray(selected, dtype=np.float64).reshape(-1, 3)
        if selected
        else np.zeros((0, 3), dtype=np.float64)
    )
    return SurfaceSelection(
        points=selected_array,
        global_count=global_count,
        target_count=target_count,
        mandatory_count=mandatory_count,
    )


def densify_joint_path(positions: np.ndarray, max_joint_step: float = 0.02) -> np.ndarray:
    """Interpolate a joint path so swept-sphere validation has no large gaps."""
    positions = np.asarray(positions, dtype=np.float64)
    dense = [positions[0]]
    for start, end in zip(positions[:-1], positions[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_joint_step)))
        dense.extend(start + (end - start) * (index / count) for index in range(1, count + 1))
    return np.asarray(dense, dtype=np.float64)


def find_trajectory_conflict(
    surface_points: np.ndarray,
    robot_spheres: np.ndarray,
    safety_margin: float = 0.01,
    return_min_clearance: bool = False,
) -> TrajectoryConflict | None:
    """Check all trajectory robot spheres against the complete surface cloud."""
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    spheres = np.asarray(robot_spheres, dtype=np.float64)
    if len(points) == 0 or spheres.size == 0:
        return None
    spheres = spheres.reshape((-1, spheres.shape[-2], 4))
    tree = cKDTree(points)
    hit_indices: set[int] = set()
    first_step, last_step = len(spheres), -1
    min_clearance = float("inf")
    min_clearance_point = np.zeros(3, dtype=np.float64)
    min_clearance_step = -1
    min_clearance_sphere_center = np.zeros(3, dtype=np.float64)
    for step, step_spheres in enumerate(spheres):
        distance, index = tree.query(step_spheres[:, :3], k=1)
        clearance = distance - step_spheres[:, 3]
        local_min = int(np.argmin(clearance))
        if float(clearance[local_min]) < min_clearance:
            min_clearance = float(clearance[local_min])
            min_clearance_step = step
            min_clearance_point = points[int(index[local_min])].copy()
            min_clearance_sphere_center = step_spheres[local_min, :3].copy()
        colliding = np.flatnonzero(clearance <= float(safety_margin))
        if not len(colliding):
            continue
        first_step, last_step = min(first_step, step), step
        neighborhoods = tree.query_ball_point(
            step_spheres[colliding, :3],
            step_spheres[colliding, 3] + float(safety_margin),
        )
        hit_indices.update(index for group in neighborhoods for index in group)
    if last_step < 0:
        if return_min_clearance:
            return TrajectoryConflict(
                point_indices=np.array([], dtype=np.int64),
                first_step=min_clearance_step,
                last_step=min_clearance_step,
                min_clearance=min_clearance,
                min_clearance_point=min_clearance_point,
                min_clearance_step=min_clearance_step,
                min_clearance_sphere_center=min_clearance_sphere_center,
            )
        return None
    return TrajectoryConflict(
        point_indices=np.asarray(sorted(hit_indices), dtype=np.int64),
        first_step=first_step,
        last_step=last_step,
        min_clearance=min_clearance,
        min_clearance_point=min_clearance_point,
        min_clearance_step=min_clearance_step,
        min_clearance_sphere_center=min_clearance_sphere_center,
    )
