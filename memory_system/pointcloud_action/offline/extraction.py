"""Extract object point clouds and object frame from a restored simulator state."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.geometry import camera_params, depth_to_metric, flip_depth, pixel_to_world
from memory_system.offline.build_ready3d import instance_mask, resolve_instance
from memory_system.offline.build_targets import patch_numpy2_segmentation

ROOT = Path(__file__).resolve().parents[3]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"

# MuJoCo geom types used for primitive fallback sampling.
_GEOM_HFIELD = 1
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7


def _resolve_bddl(task_name: str, suite: str = "libero_90") -> Path:
    from libero.libero import get_libero_path

    sibling = LIBERO_PLUS / "libero/libero/bddl_files" / suite / f"{task_name}.bddl"
    installed = Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"
    path = sibling if sibling.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def create_env(
    task_name: str,
    resolution: int,
    suite: str = "libero_90",
    bddl_file_name: str | Path | None = None,
):
    from libero.libero.envs import SegmentationRenderEnv

    patch_numpy2_segmentation()
    if bddl_file_name is None:
        bddl_file_name = _resolve_bddl(task_name, suite)
    return SegmentationRenderEnv(
        bddl_file_name=str(bddl_file_name),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=[True, False],
    )


def visible_point_cloud(env, obs, instance_name: str, resolution: int) -> np.ndarray:
    """Back-project the visible object pixels from agentview depth."""
    mask = instance_mask(env, obs, instance_name)
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return np.empty((0, 3), dtype=np.float64)
    cam = camera_params(env.env.sim, "agentview", resolution, resolution)
    metric = depth_to_metric(obs["agentview_depth"], cam.near, cam.far)
    depth = flip_depth(metric)
    points = pixel_to_world(np.stack([ys, xs], axis=-1), depth, cam)
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def _body_id(sim, instance_name: str) -> int:
    try:
        return sim.model.body_name2id(instance_name)
    except Exception:
        pass
    # Many LIBERO objects have a root body named "<instance>_main" plus optional
    # child visual bodies. Prefer the root/main body when it exists.
    main_name = f"{instance_name}_main"
    try:
        return sim.model.body_name2id(main_name)
    except Exception:
        pass
    matches = [
        index
        for index, name in enumerate(sim.model.body_names)
        if name == instance_name or name.endswith(f"_{instance_name}") or instance_name in name
    ]
    if len(matches) == 1:
        return matches[0]
    for index in matches:
        if sim.model.body_names[index].endswith("_main"):
            return index
    raise KeyError(
        f"Cannot resolve MuJoCo body for {instance_name!r}; matches={[sim.model.body_names[i] for i in matches]}"
    )


def _mesh_vertices(sim, geom_id: int, body_id: int) -> np.ndarray:
    # Return raw mesh vertices in the geom-local/mesh frame.
    # _geom_world_points() applies geom_pos/quat and body pose afterwards,
    # so do NOT transform the mesh vertices to world coordinates here.
    mesh_id = sim.model.geom_dataid[geom_id]
    start = sim.model.mesh_vertadr[mesh_id]
    count = sim.model.mesh_vertnum[mesh_id]
    return sim.model.mesh_vert[start : start + count].copy()


def _box_points(size: np.ndarray, count: int) -> np.ndarray:
    half = np.asarray(size, dtype=np.float64).reshape(3)
    axis = np.linspace(-1.0, 1.0, max(2, int(round(count ** (1 / 3)))))
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = (np.abs(grid) < 0.999).sum(axis=1) <= 1
    return (grid[inside] * half).astype(np.float64)


def _sphere_points(radius: float, count: int) -> np.ndarray:
    indices = np.arange(count)
    golden = 0.5 * (1.0 + np.sqrt(5.0))
    theta = 2.0 * np.pi * indices / golden
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / count)
    return np.stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ],
        axis=-1,
    )


def _cylinder_points(radius: float, half_height: float, count: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, max(8, int(np.sqrt(count * 4))), endpoint=False)
    z = np.linspace(-half_height, half_height, max(2, int(np.sqrt(count * 4))))
    theta_grid, z_grid = np.meshgrid(theta, z, indexing="ij")
    points = np.stack(
        [
            radius * np.cos(theta_grid),
            radius * np.sin(theta_grid),
            z_grid,
        ],
        axis=-1,
    ).reshape(-1, 3)
    if len(points) >= count:
        points = points[:: len(points) // count][:count]
    return points


def _primitive_points(sim, geom_id: int, count: int) -> np.ndarray:
    geom_type = int(sim.model.geom_type[geom_id])
    size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float64)
    if geom_type == _GEOM_BOX:
        return _box_points(size, count)
    if geom_type == _GEOM_SPHERE:
        return _sphere_points(float(size[0]), count)
    if geom_type in (_GEOM_CYLINDER, _GEOM_CAPSULE):
        return _cylinder_points(float(size[0]), float(size[1]), count)
    return np.empty((0, 3), dtype=np.float64)


def _geom_world_points(sim, geom_id: int, body_id: int, count: int) -> np.ndarray:
    if int(sim.model.geom_type[geom_id]) == _GEOM_MESH:
        points = _mesh_vertices(sim, geom_id, body_id)
    else:
        points = _primitive_points(sim, geom_id, count)
    if len(points) == 0:
        return points
    geom_pos = sim.model.geom_pos[geom_id]
    geom_quat = sim.model.geom_quat[geom_id]
    geom_rotation = Rotation.from_quat(
        [geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]
    ).as_matrix()
    # Use the geom's actual parent body, not the object root body.  A geom can
    # be parented to a child/other body even when its name starts with the
    # object prefix.
    parent_body_id = int(sim.model.geom_bodyid[geom_id])
    body_rotation = sim.data.body_xmat[parent_body_id].reshape(3, 3)
    body_pos = sim.data.body_xpos[parent_body_id]
    return (points @ geom_rotation.T + geom_pos) @ body_rotation.T + body_pos


def complete_point_cloud(
    env,
    instance_name: str,
    max_points: int = 2048,
) -> np.ndarray:
    """Sample complete points from geoms owned by the object instance.

    A geom is considered part of the object only if:
      1. its name starts with ``<instance_name>_``, and
      2. its actual parent body also belongs to this object instance.

    This avoids including geoms that are merely name-prefixed but attached to
    other bodies (a known issue in some LIBERO/MuJoCo scenes).
    """
    sim = env.env.sim
    body_id = _body_id(sim, instance_name)
    prefix = f"{instance_name}_"
    geom_ids = []
    for index, name in enumerate(sim.model.geom_names):
        if name is None or not name.startswith(prefix):
            continue
        parent_body_id = int(sim.model.geom_bodyid[index])
        parent_body_name = sim.model.body_names[parent_body_id]
        # Only keep geoms physically attached to this object instance.
        if parent_body_name == instance_name or parent_body_name.startswith(prefix):
            geom_ids.append(index)

    all_points = []
    per_geom = max(8, max_points // max(len(geom_ids), 1))
    for geom_id in geom_ids:
        parent_body_id = int(sim.model.geom_bodyid[geom_id])
        points = _geom_world_points(sim, int(geom_id), parent_body_id, per_geom)
        if len(points):
            all_points.append(points)
    if not all_points:
        # Fall back to the previous name-only selection if the stricter filter
        # removed everything (e.g. unusual object body naming).
        geom_ids = [
            index
            for index, name in enumerate(sim.model.geom_names)
            if name is not None and name.startswith(prefix)
        ]
        all_points = []
        per_geom = max(8, max_points // max(len(geom_ids), 1))
        for geom_id in geom_ids:
            points = _geom_world_points(sim, int(geom_id), body_id, per_geom)
            if len(points):
                all_points.append(points)
    if not all_points:
        return np.empty((0, 3), dtype=np.float64)
    points = np.concatenate(all_points, axis=0)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
    return points


def object_frame(env, instance_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_world_object, translation, rotation) at the current simulator state."""
    sim = env.env.sim
    body_id = _body_id(sim, instance_name)
    translation = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy()
    rotation = np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, translation, rotation
