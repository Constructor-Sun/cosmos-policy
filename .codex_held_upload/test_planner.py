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
