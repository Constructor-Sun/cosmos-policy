"""Unit tests for HeldObjectPlanner helpers and input validation."""
import numpy as np
import pytest

from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
from memory_system.execute.planner.held_object.types import (
    HeldObjectObservation,
    HeldObjectPlannerInput,
)
from memory_system.types import CameraParams


def test_build_spheres_caps_slots():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3)) * 0.05
    spheres = HeldObjectPlanner._build_spheres(points, max_slots=32, voxel_size=0.01, padding=0.005)
    assert spheres is not None
    assert spheres.shape[1] == 4
    assert len(spheres) <= 32
    assert (spheres[:, 3] > 0).all()


def test_default_attachment_uses_64_slots():
    planner = HeldObjectPlanner()
    assert planner.attachment_slots == 64
    assert planner.removal_sphere_slots == 32


def test_build_attachment_spheres_covers_all_points_and_is_deterministic():
    rng = np.random.default_rng(7)
    points = rng.uniform(
        low=[-0.06, -0.025, -0.05],
        high=[0.07, 0.03, 0.055],
        size=(1000, 3),
    )
    padding = 0.002
    first = HeldObjectPlanner._build_attachment_spheres(
        points, max_slots=64, padding=padding
    )
    second = HeldObjectPlanner._build_attachment_spheres(
        points, max_slots=64, padding=padding
    )

    assert first is not None
    assert len(first) <= 64
    assert np.array_equal(first, second)
    clearance = (
        np.linalg.norm(points[:, None, :] - first[None, :, :3], axis=2)
        - first[None, :, 3]
    )
    assert np.all(np.min(clearance, axis=1) <= -padding + 1e-12)


def test_build_attachment_spheres_skips_empty_cells():
    points = np.array(
        [
            [-0.100, 0.0, 0.0],
            [-0.099, 0.0, 0.0],
            [0.099, 0.0, 0.0],
            [0.100, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    spheres = HeldObjectPlanner._build_attachment_spheres(
        points, max_slots=64, padding=0.002
    )

    assert spheres is not None
    assert len(spheres) == 2


def test_build_attachment_spheres_is_tighter_than_removal_cover():
    rng = np.random.default_rng(11)
    left = rng.normal(loc=[-0.05, -0.015, 0.0], scale=0.002, size=(200, 3))
    right = rng.normal(loc=[0.05, 0.015, 0.0], scale=0.002, size=(200, 3))
    points = np.concatenate([left, right], axis=0)
    removal = HeldObjectPlanner._build_spheres(
        points, max_slots=32, voxel_size=0.02, padding=0.005
    )
    attachment = HeldObjectPlanner._build_attachment_spheres(
        points, max_slots=64, padding=0.002
    )

    assert removal is not None
    assert attachment is not None
    removal_extent = HeldObjectPlanner._sphere_extent(removal)
    attachment_extent = HeldObjectPlanner._sphere_extent(attachment)
    assert np.prod(attachment_extent) < np.prod(removal_extent)


def test_build_attachment_spheres_returns_none_for_empty_points():
    spheres = HeldObjectPlanner._build_attachment_spheres(
        np.empty((0, 3)), max_slots=64, padding=0.002
    )
    assert spheres is None


def test_base_pose_parts_identity():
    rotation, translation = HeldObjectPlanner._base_pose_parts(None)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, np.zeros(3))


def test_plan_returns_none_without_depth():
    planner = HeldObjectPlanner()
    camera = CameraParams(
        K=np.eye(3),
        T_w2c=np.eye(4),
        T_c2w=np.eye(4),
        height=10,
        width=10,
        near=0.1,
        far=1.0,
    )
    inp = HeldObjectPlannerInput(
        joint_positions=np.zeros(7),
        ee_states=np.zeros(6),
        depth=None,
        camera_params=camera,
        held_object=HeldObjectObservation(item="item", points_hand=np.zeros((3, 3))),
        ready_pose=np.zeros(6),
    )
    assert planner.plan(inp) is None


def test_plan_rejects_invalid_held_object_before_planning():
    planner = HeldObjectPlanner()
    camera = CameraParams(
        K=np.eye(3),
        T_w2c=np.eye(4),
        T_c2w=np.eye(4),
        height=10,
        width=10,
        near=0.1,
        far=1.0,
    )
    held = HeldObjectObservation(
        item="item",
        points_hand=np.zeros((3, 3)),
        valid=False,
        rejection_reason="test failure",
    )
    inp = HeldObjectPlannerInput(
        joint_positions=np.zeros(7),
        ee_states=np.zeros(6),
        depth=np.ones((10, 10), dtype=np.float64),
        camera_params=camera,
        held_object=held,
        ready_pose=np.zeros(6),
    )
    assert planner.plan(inp) is None
