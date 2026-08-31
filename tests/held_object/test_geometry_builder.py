"""Unit tests for HeldObjectGeometryBuilder with synthetic RGB-D."""
import numpy as np

from memory_system.execute.planner.held_object.geometry_builder import (
    HeldObjectGeometryBuilder,
)
from memory_system.execute.planner.held_object.mask_matcher import MemoryTemplateMatcher
from memory_system.types import CameraParams


def _template():
    rng = np.random.default_rng(0)
    obj = rng.integers(0, 255, size=(60, 60, 3), dtype=np.uint8)
    return {
        "crop_rgb": obj.copy(),
        "crop_mask": np.ones((60, 60), dtype=np.uint8),
        "task_name": "synthetic",
        "demo_id": "demo_0",
        "planner_step_id": 0,
        "skill": "Pick",
    }


def _camera():
    k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    eye = np.eye(4)
    return CameraParams(
        K=k,
        T_w2c=eye,
        T_c2w=eye,
        height=100,
        width=100,
        near=0.1,
        far=5.0,
    )


def test_geometry_builder_accumulates_hand_points():
    template = _template()
    current = np.full((100, 100, 3), 220, dtype=np.uint8)
    current[20:80, 30:90] = template["crop_rgb"]
    depth = np.ones((100, 100), dtype=np.float64)
    camera = _camera()
    hand_to_world = np.eye(4)

    builder = HeldObjectGeometryBuilder(MemoryTemplateMatcher(), "item", voxel_size=0.01)
    assert builder.update(current, depth, camera, template, hand_to_world)
    obs = builder.build()
    assert obs is not None
    assert obs.item == "item"
    assert len(obs.points_hand) >= 4
    assert np.isfinite(obs.points_hand).all()
    # With identity hand pose, points should keep the camera z around 1.0.
    assert np.allclose(obs.points_hand[:, 2], 1.0, atol=1e-6)
