"""Test-only harness for Stage 1 (depth correctness) tests.

NOT part of test-time / production code.  Formal runtime code must never
import this module: the memory system consumes metric depth + calibrated
camera parameters only.  This module exists solely so the Stage-1 tests can
create LIBERO environments with the main camera's depth enabled and measure
the depth observation chain (obs key -> normalized depth -> metric depth).

Run with the `cosmospolicy` conda env:

    conda activate cosmospolicy
    cd <cosmos-policy repo>
    pytest tests/stage1_depth/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/tests/stage1_depth -> <repo>
LIBERO_PLUS = REPO_ROOT.parent / "LIBERO-plus"

for _p in (str(REPO_ROOT), str(LIBERO_PLUS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESOLUTION = 256

# Two representative LIBERO-10 tasks (kitchen + living room) with exact bddl files.
TASKS = (
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
)


def load_task_names() -> list[str]:
    """Task names used by the memory system (from the segments manifest)."""
    import json

    manifest_path = REPO_ROOT / "skill_memory/libero_10/segments_ready_fixed16.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    return sorted(set(r["task_name"] for r in manifest["records"]))


def resolve_bddl(task_name: str) -> Path:
    """Resolve the exact LIBERO-10 bddl file for a task (test-only)."""
    candidate = LIBERO_PLUS / "libero/libero/bddl_files/libero_10" / f"{task_name}.bddl"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def create_env(task_name: str, resolution: int = RESOLUTION):
    """LIBERO env with main-camera depth enabled, wrist camera RGB only."""
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(resolve_bddl(task_name)),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=[True, False],  # agentview depth on, eye_in_hand stays RGB
    )
    return env


def metric_depth(env, obs) -> np.ndarray:
    """Normalized agentview depth -> metric (view-space z) depth, shape (H, W, 1).

    This is the sim-side conversion the Stage-1 tests measure.  On a real
    robot the sensor provides metric depth directly, so this conversion is
    LIBERO/MuJoCo-specific and lives on the sim side of the boundary.
    """
    from robosuite.utils.camera_utils import get_real_depth_map

    raw = np.asarray(obs["agentview_depth"], dtype=np.float64)
    return get_real_depth_map(env.env.sim, raw)


def flip_depth(depth: np.ndarray) -> np.ndarray:
    """Same np.flipud convention the memory system applies to RGB (flip_images)."""
    return np.flipud(depth)


def camera_params(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K, T_w2c, T_c2w built from the ACTUAL render camera.

    IMPORTANT (empirically verified in Stage 2): robosuite's
    get_camera_extrinsic_matrix axis correction diag(1,-1,-1) does NOT match
    this environment's renderer (mujoco 2.3.7 + robosuite 1.4.0); the render
    is reproduced by R = cam_xmat @ diag(1,1,-1).  Pixel-space tests are
    self-consistent under either convention, so Stage 1 passed with the
    robosuite convention, but world-space geometry (Stage 2+) must use the
    corrected one.
    """
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64).copy()
    cam_rot = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    R = cam_rot @ np.diag([1.0, 1.0, -1.0])  # corrected: matches the actual render
    f = 0.5 * RESOLUTION / np.tan(np.radians(float(sim.model.cam_fovy[cam_id])) / 2)
    K = np.array([[f, 0.0, RESOLUTION / 2], [0.0, f, RESOLUTION / 2], [0.0, 0.0, 1.0]])
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R
    T_c2w[:3, 3] = cam_pos
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R.T
    T_w2c[:3, 3] = -R.T @ cam_pos
    return K, T_w2c, T_c2w


def back_project(pixels_row_col: np.ndarray, depth: np.ndarray, K: np.ndarray,
                 T_c2w: np.ndarray) -> np.ndarray:
    """Back-project integer (row, col) pixels with metric depth to world XYZ.

    Standard pinhole inverse with view-space z:
        x_cam = (col - cx) / fx * z ;  y_cam = (row - cy) / fy * z ;  z_cam = z
        P_world = T_c2w @ [x_cam, y_cam, z_cam, 1]
    """
    d = depth[..., 0] if depth.ndim == 3 else depth
    px = np.atleast_2d(np.asarray(pixels_row_col, dtype=np.int64))
    rows, cols = px[:, 0], px[:, 1]
    z = d[rows, cols].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_cam = (cols - cx) / fx * z
    y_cam = (rows - cy) / fy * z
    cam_pts = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
    world = (T_c2w @ cam_pts.T).T
    return world[:, :3]


def back_project_flipped(pixels_row_col: np.ndarray, depth_flipped: np.ndarray,
                         K: np.ndarray, T_c2w: np.ndarray,
                         height: int = RESOLUTION) -> np.ndarray:
    """Back-project pixels given in FLIPPED image space.

    The memory system consumes the agentview image flipped (np.flipud,
    flip_images=True) and matches templates in that space, so the depth
    adapter emits a flipped metric depth map.  Pixel (r, c) of the flipped
    depth corresponds to pixel (H-1-r, c) in the unflipped camera geometry:
    the depth VALUE is sampled at the flipped pixel, but the camera-frame
    y coordinate must use the unflipped row.  This is the convention the
    production rgbd back-projection must follow.
    """
    d = depth_flipped[..., 0] if depth_flipped.ndim == 3 else depth_flipped
    px = np.atleast_2d(np.asarray(pixels_row_col, dtype=np.int64))
    rows_f, cols = px[:, 0], px[:, 1]
    z = d[rows_f, cols].astype(np.float64)
    rows_u = height - 1 - rows_f  # unflip the row for camera geometry
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_cam = (cols - cx) / fx * z
    y_cam = (rows_u - cy) / fy * z
    cam_pts = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
    world = (T_c2w @ cam_pts.T).T
    return world[:, :3]
