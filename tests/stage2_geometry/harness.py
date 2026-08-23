"""Test-only harness for Stage 2: per-pixel object-surface depth.

NOT part of test-time / production code.  Same constraints as
tests/stage1_depth: the formal path consumes metric depth + camera params +
robot proprioception only; simulator object state / segmentation appear here
only as test Oracle or as the "given target region" input.

Convention: reuses the flip-aware pixel-space convention validated in Stage 1
(see tests/stage1_depth/README.md): camera geometry in render space, memory
system in canonical (np.flipud) space, back-projection unflips the row first.

The depth oracle uses MuJoCo's built-in mj_ray() on the simulator geometry,
so it supports mesh geoms and is independent of get_real_depth_map().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/tests/stage2_geometry -> <repo>
LIBERO_PLUS = REPO_ROOT.parent / "LIBERO-plus"

for _p in (str(REPO_ROOT), str(LIBERO_PLUS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESOLUTION = 256

# All 10 LIBERO-10 tasks with valid Pick segments.
TASKS = (
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
)

MANIFEST = REPO_ROOT / "skill_memory/libero_10/segments_ready_fixed16.json"
DEMO_DIR = REPO_ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"


def patch_numpy2_segmentation() -> None:
    """Fix robosuite's uint8 segmentation decode under NumPy 2 (test-only)."""
    from robosuite.utils import binding_utils as binding

    original = binding.MjRenderContext.read_pixels
    if getattr(original, "_cosmos_numpy2_safe", False):
        return

    def read_pixels(self, width, height, depth=False, segmentation=False):
        if not segmentation:
            return original(self, width, height, depth=depth, segmentation=False)
        viewport = binding.mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        binding.mujoco.mjr_readPixels(rgb=rgb, depth=depth_img, viewport=viewport, con=self.con)
        rgb32 = rgb.astype(np.int32)
        encoded = rgb32[:, :, 0] + rgb32[:, :, 1] * 256 + rgb32[:, :, 2] * 65536
        encoded[encoded >= self.scn.ngeom + 1] = 0
        ids = np.full((self.scn.ngeom + 1, 2), -1, dtype=np.int32)
        for index in range(self.scn.ngeom):
            geom = self.scn.geoms[index]
            if geom.segid != -1:
                ids[geom.segid + 1] = (geom.objtype, geom.objid)
        result = ids[encoded]
        return (result, depth_img) if depth else result

    read_pixels._cosmos_numpy2_safe = True
    binding.MjRenderContext.read_pixels = read_pixels


def resolve_bddl(task_name: str) -> Path:
    candidate = LIBERO_PLUS / "libero/libero/bddl_files/libero_10" / f"{task_name}.bddl"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def create_seg_env(task_name: str, resolution: int = RESOLUTION):
    """SegmentationRenderEnv with main depth on, wrist RGB only (single env at a time)."""
    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(resolve_bddl(task_name)),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=[True, False],
    )
    return env


def load_manifest() -> dict:
    import json

    with open(MANIFEST) as f:
        return json.load(f)


def demo_hdf5(task_name: str) -> Path:
    return DEMO_DIR / f"{task_name}_demo.hdf5"


def phase_frames(segment: dict, length: int) -> list[int]:
    """Same phase frames the offline builder samples: start/middle/ready."""
    start = min(max(int(segment["start"]), 0), length - 1)
    end = min(max(int(segment["end"]), start + 1), length)
    middle = min(start + max((end - start) // 2, 1), end - 1)
    ready = segment.get("ready_frame")
    ready = min(max(int(ready), start), end - 1) if ready is not None else end - 1
    return sorted(set((start, middle, ready)))


def metric_depth(env, obs) -> np.ndarray:
    """Metric (view-space z) depth aligned with the raw agentview image, (H, W, 1)."""
    from robosuite.utils.camera_utils import get_real_depth_map

    raw = np.asarray(obs["agentview_depth"], dtype=np.float64)
    return get_real_depth_map(env.env.sim, raw)


def camera_params(env):
    """K, T_w2c, T_c2w built from the ACTUAL render camera.

    IMPORTANT (empirically verified): robosuite's get_camera_extrinsic_matrix
    applies an axis correction diag(1,-1,-1) that does NOT match this
    environment's renderer (mujoco 2.3.7 + robosuite 1.4.0).  The render is
    reproduced by R = cam_xmat @ diag(1,1,-1): with it, back-projecting the
    rendered instance pixels lands on the true simulator positions, while
    robosuite's convention is off by ~13 cm in world z.  Stage 2 (world
    geometry) requires the corrected convention.
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


def instance_mask(env, obs, instance_name: str) -> np.ndarray:
    """Instance segmentation mask for an object, in CANONICAL (flipped) space.

    The segmentation observation is in raw render space (robosuite
    IMAGE_CONVENTION=opengl applies no flip), while the memory system's
    canonical image space is np.flipud'd (flip_images=True).  This mirrors
    the offline builder: ``mask = np.flipud(raw_mask)``.  The returned mask
    aligns with the canonical depth map.
    """
    seg = np.asarray(obs["agentview_segmentation_instance"])[..., 0]
    inst_id = env.instance_to_id.get(instance_name)
    if inst_id is None:
        # try anchor-style matching (owner instance for region arguments)
        for name, iid in env.instance_to_id.items():
            if name.startswith(instance_name.split("_")[0]) or instance_name in name:
                inst_id = iid
                break
    if inst_id is None:
        raise KeyError(f"instance {instance_name!r} not in {sorted(env.instance_to_id)}")
    return np.flipud(seg == inst_id)


# ---------------------------------------------------------------------------
# Per-pixel surface-depth oracle using MuJoCo mj_ray()
# ---------------------------------------------------------------------------

def ray_surface_depth(env, obs, mask: np.ndarray, instance_name: str,
                      erosion_k: int = 3) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Per-pixel simulator-geometry depth oracle for the target object's mask.

    mask: canonical (flipped) space.  For each (eroded) mask pixel, casts a
    ray from the camera through the (unflipped) pixel and uses
    ``mujoco.mj_ray`` to get the first surface intersection.  The returned
    distance is the camera-frame Z because the ray direction is constructed
    with camera z = 1.  Returns (obs_metric_depth_at_pixels,
    oracle_depth_at_pixels, n_excluded) where n_excluded counts mask pixels
    whose first hit is not on the target object (background bleed / occlusion)
    and are therefore excluded from metrics.
    """
    import cv2
    import mujoco

    K, _T_w2c, T_c2w = camera_params(env)
    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data

    eroded = cv2.erode(mask.astype(np.uint8), np.ones((erosion_k, erosion_k), np.uint8)).astype(bool)
    ys_c, xs = np.nonzero(eroded)
    if len(ys_c) == 0:
        return None, np.asarray([]), 0

    d = np.flipud(metric_depth(env, obs))
    obs_z = d[ys_c, xs, 0].astype(np.float64)

    cam_pos = T_c2w[:3, 3]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    rows_r = 255 - ys_c  # unflip rows back to render space
    # camera-frame ray directions through the PIXEL CENTERS (z component = 1).
    # Empirically the renderer samples depth at (u+0.5, v+0.5): using the
    # integer pixel index adds a half-pixel bias (up to ~9 mm on sloped
    # surfaces in the measured cases).
    u = xs.astype(np.float64) + 0.5
    v = rows_r.astype(np.float64) + 0.5
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    R = T_c2w[:3, :3]
    d_world = d_cam @ R.T  # (N, 3)

    oracle = np.full(len(ys_c), np.inf)
    keep = np.zeros(len(ys_c), dtype=bool)
    geomid = np.zeros(1, dtype=np.int32)
    for i in range(len(ys_c)):
        dist = mujoco.mj_ray(model, data, cam_pos.astype(np.float64),
                             d_world[i].astype(np.float64), None, 1, -1, geomid)
        if dist >= 0 and 0 <= geomid[0] < sim.model.ngeom:
            name = sim.model.geom_id2name(geomid[0])
            if name and name.startswith(instance_name):
                oracle[i] = dist
                keep[i] = True

    n_excluded = int((~keep).sum())
    return obs_z[keep], oracle[keep], n_excluded
