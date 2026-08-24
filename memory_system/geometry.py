"""Formal RGB-D geometry helpers for the memory system.

Convention (single source of truth):

* The memory system consumes images/depth in *canonical* space, i.e. after
  ``np.flipud`` (``flip_images=True``).  Camera geometry (``K``, ``T_w2c``,
  ``T_c2w``) is in *render* space.
* Back-projection from canonical pixels must mirror the row back to render
  space before applying the pinhole inverse.
* The MuJoCo camera rotation is ``cam_xmat @ diag(1, 1, -1)`` (verified by
  Stage 2); this is what ``camera_params`` returns.

This module is pure geometry: it consumes metric depth, camera parameters,
and pixel coordinates.  It does not read simulator object state.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from memory_system.types import CameraParams


def depth_to_metric(depth: Any, near: float, far: float) -> np.ndarray:
    """Convert normalized render depth to metric view-space z (m)."""
    depth = np.asarray(depth, dtype=np.float64)
    near = float(near)
    far = float(far)
    return near / (1.0 - depth * (1.0 - near / far))


def camera_params(sim: Any, camera_name: str, height: int, width: int) -> CameraParams:
    """Build CameraParams from a MuJoCo camera (LIBERO/MuJoCo adapter)."""
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64).copy()
    cam_rot = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    # Corrected camera orientation matching the actual renderer.
    R = cam_rot @ np.diag([1.0, 1.0, -1.0])
    f = 0.5 * float(height) / np.tan(np.radians(float(sim.model.cam_fovy[cam_id])) / 2)
    K = np.array(
        [[f, 0.0, float(width) / 2.0],
         [0.0, f, float(height) / 2.0],
         [0.0, 0.0, 1.0]]
    )
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R
    T_c2w[:3, 3] = cam_pos
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R.T
    T_w2c[:3, 3] = -R.T @ cam_pos
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    return CameraParams(
        K=K,
        T_w2c=T_w2c,
        T_c2w=T_c2w,
        height=int(height),
        width=int(width),
        near=near,
        far=far,
    )


def flip_depth(depth: Any) -> np.ndarray:
    """Flip depth into the same canonical space as the memory-system RGB."""
    return np.flipud(depth)


def pixel_to_world(pixels: Any, depth: Any, cam: CameraParams) -> np.ndarray:
    """Back-project canonical-space pixels with canonical-space metric depth.

    Args:
        pixels: (..., 2) array of canonical (row, col) coordinates.
        depth: canonical metric depth, shape (H, W) or (H, W, 1).
        cam: camera parameters.

    Returns:
        World XYZ points, shape (..., 3).
    """
    px = np.atleast_2d(np.asarray(pixels, dtype=np.int64).reshape(-1, 2))
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    rows_f, cols = px[:, 0], px[:, 1]
    z = d[rows_f, cols].astype(np.float64)
    rows_r = cam.height - 1 - rows_f  # unflip row back to render space
    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
    cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
    x_cam = (cols - cx) / fx * z
    y_cam = (rows_r - cy) / fy * z
    cam_pts = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
    world = (cam.T_c2w @ cam_pts.T).T[:, :3]
    if np.asarray(pixels).ndim == 1:
        return world[0]
    return world


def world_to_pixel(xyz: Any, cam: CameraParams, flip: bool = True) -> np.ndarray:
    """Project world XYZ to canonical pixel (row, col) by default."""
    pts = np.atleast_2d(np.asarray(xyz, dtype=np.float64).reshape(-1, 3))
    cam_pts = (cam.T_w2c @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    z = cam_pts[:, 2]
    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
    cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
    col = cam_pts[:, 0] / z * fx + cx
    row_r = cam_pts[:, 1] / z * fy + cy
    row = cam.height - 1 - row_r if flip else row_r
    out = np.stack([row, col], axis=-1)
    return out[0] if np.asarray(xyz).ndim == 1 else out
