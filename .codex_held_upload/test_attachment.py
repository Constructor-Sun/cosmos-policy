"""Tests for multi-frame RGB-D attachment estimation."""
import numpy as np

from memory_system.execute.planner.held_object.attachment import (
    HeldObjectAttachmentEstimator,
)


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def test_consensus_keeps_hand_fixed_points_and_rejects_background():
    rng = np.random.default_rng(4)
    object_points = rng.normal(size=(80, 3)) * 0.012
    estimator = HeldObjectAttachmentEstimator(
        "cup",
        voxel_size=0.01,
        min_frames=3,
        min_consensus_frames=2,
        consensus_ratio=0.5,
        min_hand_translation=0.01,
        min_hand_rotation=0.0,
        min_points=16,
    )

    for frame in range(4):
        # The object is fixed in hand coordinates.  The background candidate
        # moves as the hand moves and should not survive the consensus.
        background = np.array(
            [[0.25 + 0.06 * frame, 0.10, 0.10 + 0.02 * i] for i in range(30)],
            dtype=np.float64,
        )
        assert estimator.add_hand_points(
            np.concatenate([object_points, background], axis=0), _pose(0.02 * frame)
        )

    result = estimator.finalize()
    assert result.valid
    points = result.observation.points_hand
    assert len(points) >= 16
    assert np.max(np.linalg.norm(points, axis=1)) < 0.08
    assert result.observation.capture_frames == 4
    assert result.observation.source == "rgbd_hand_consensus"


def test_consensus_rejects_capture_without_hand_motion():
    estimator = HeldObjectAttachmentEstimator(
        "cup",
        voxel_size=0.01,
        min_frames=3,
        min_hand_translation=0.01,
        min_hand_rotation=0.03,
    )
    points = np.zeros((32, 3), dtype=np.float64)
    for _ in range(3):
        assert estimator.add_hand_points(points, _pose(0.0))

    result = estimator.finalize()
    assert not result.valid
    assert "insufficient hand motion" in result.reason
    assert not result.observation.valid
