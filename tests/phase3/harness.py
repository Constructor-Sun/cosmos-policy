"""Test-only harness for Phase 3 (RGB-D Pick approach discrimination).

This module is NOT part of runtime.  It uses simulator state / segmentation
only as test Oracle or to build candidate templates.  The per-candidate
matching path reuses ``PhaseVerifier.match_current``, and the Phase normal-end
check reuses ``FeasibleVerifier`` exactly like the existing 2D runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS = REPO_ROOT.parent / "LIBERO-plus"
for _p in (str(REPO_ROOT), str(LIBERO_PLUS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESOLUTION = 256
PHASE_TARGETS = REPO_ROOT / "skill_memory_test/libero_10/phase_targets.pt"
SEGMENTS_MANIFEST = REPO_ROOT / "skill_memory_test/libero_10/segments_ready_fixed16.json"
DEMO_DIR = REPO_ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"

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


def patch_numpy2_segmentation() -> None:
    """Fix robosuite uint8 segmentation decode under NumPy 2 (test-only)."""
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


def create_env(task_name: str, resolution: int = RESOLUTION):
    """SegmentationRenderEnv with main depth on, wrist RGB only."""
    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(resolve_bddl(task_name)),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=[True, False],
    )
    return env


def load_manifest(path=SEGMENTS_MANIFEST) -> dict:
    import json

    return json.loads(Path(path).read_text())


def demo_hdf5(task_name: str) -> Path:
    return DEMO_DIR / f"{task_name}_demo.hdf5"


def instance_mask(env, obs, instance_name: str) -> np.ndarray:
    """Canonical (flipped) binary mask for an instance."""
    seg = np.asarray(obs["agentview_segmentation_instance"])[..., 0]
    inst_id = env.instance_to_id.get(instance_name)
    if inst_id is None:
        raise KeyError(f"instance {instance_name!r} not found")
    return np.flipud(seg == inst_id)


def bbox_from_mask(mask: np.ndarray, padding: int = 4):
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    h, w = mask.shape
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(w, int(xs.max()) + padding + 1),
        min(h, int(ys.max()) + padding + 1),
    )


def make_template(rgb: np.ndarray, mask: np.ndarray, demo_id: str) -> dict | None:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    crop, crop_mask = rgb[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    return {
        "demo_id": str(demo_id),
        "frame": 0,
        "crop_rgb": crop.copy(),
        "crop_mask": crop_mask.copy(),
        "bbox_xyxy": np.asarray(bbox, dtype=np.int16),
        "target_center_xy": np.asarray([xs.mean(), ys.mean()], dtype=np.float32),
        "visible_pixels": int(mask.sum()),
    }


def object_positions(env) -> dict[str, np.ndarray]:
    base = env.env
    out = {}
    for name in env.instance_to_id:
        if name.startswith("MountedPanda") or "robot" in name.lower():
            continue
        try:
            pos = base.object_states_dict[name].get_geom_state()["pos"]
            out[name] = np.asarray(pos, dtype=np.float64).copy()
        except Exception:
            continue
    return out


def load_item_objects(manifest: dict) -> set[str]:
    """All objects that ever appear as ``arguments.item`` in LIBERO-10."""
    items = set()
    for record in manifest.get("records", []):
        for segment in record.get("segments", []):
            item = segment.get("arguments", {}).get("item")
            if item:
                items.add(str(item))
    return items


def load_target_arguments(manifest: dict) -> set[str]:
    """All raw ``arguments.target`` strings in LIBERO-10."""
    targets = set()
    for record in manifest.get("records", []):
        for segment in record.get("segments", []):
            target = segment.get("arguments", {}).get("target")
            if target:
                targets.add(str(target))
    return targets


def underlying_instance(env, name: str) -> str | None:
    """Map a region/object argument to a segmentable instance name."""
    if name in env.instance_to_id:
        return name
    for inst in env.instance_to_id:
        if name.startswith(inst) or inst in name:
            return inst
    return None


def target_underlying_objects(env, manifest: dict) -> set[str]:
    """All segmentable instances that can appear as a target in LIBERO-10."""
    out = set()
    for target in load_target_arguments(manifest):
        inst = underlying_instance(env, target)
        if inst is not None:
            out.add(inst)
    return out


def select_candidates(
    env,
    correct_item: str,
    max_total: int = 5,
    allowed_objects: set[str] | None = None,
) -> list[str]:
    """Return [correct_item] + up to max_total-1 nearest allowed objects."""
    pos = object_positions(env)
    if correct_item not in pos:
        pos[correct_item] = np.zeros(3)
    candidates = [
        name for name in pos
        if name != correct_item
        and (allowed_objects is None or name in allowed_objects)
    ]
    others = sorted(
        ((np.linalg.norm(pos[correct_item] - pos[name]), name)
         for name in candidates)
    )
    return [correct_item] + [name for _, name in others[: max_total - 1]]


def select_target_candidates(
    env,
    correct_target: str,
    target_objects: set[str],
    max_total: int = 5,
) -> list[str]:
    """Return [correct_target] + up to max_total-1 nearest target-like objects."""
    pos = object_positions(env)
    if correct_target not in pos:
        pos[correct_target] = np.zeros(3)
    candidates = [
        name for name in pos
        if name != correct_target and name in target_objects
    ]
    others = sorted(
        ((np.linalg.norm(pos[correct_target] - pos[name]), name)
         for name in candidates)
    )
    return [correct_target] + [name for _, name in others[: max_total - 1]]


def prepare_matcher(templates: list[dict], min_demo_votes: int = 1):
    """Build a PhaseVerifier-like matcher using the same matching pipeline."""
    from memory_system.execute.phase import PhaseVerifier

    verifier = PhaseVerifier(PHASE_TARGETS, min_demo_votes=min_demo_votes)
    verifier.templates = templates
    verifier.prepared_templates = [
        (tpl, verifier._prepare_template(tpl)) for tpl in templates
    ]
    return verifier
