"""Tests for the rule-based 3D Pick completion checker."""
from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.execute.skill_completion.pick import PickCompletionChecker

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
CLOUD_OFFSETS = np.array(
    [
        [-0.005, 0.0, 0.0],
        [0.005, 0.0, 0.0],
        [0.0, -0.005, 0.0],
        [0.0, 0.005, 0.0],
        [0.0, 0.0, 0.0],
    ]
)


def cloud(center) -> np.ndarray:
    return np.asarray(center, dtype=np.float64) + CLOUD_OFFSETS


def update(
    checker: PickCompletionChecker,
    *,
    eef=(0.0, 0.0, 0.0),
    center=(0.1, 0.0, 0.0),
    quat=IDENTITY_QUAT,
    closed=True,
    qpos=(0.002, -0.002),
    points=True,
) -> bool:
    return checker.update(
        target_points=cloud(center) if points else None,
        eef_pos=np.asarray(eef),
        eef_quat=np.asarray(quat),
        gripper_closed=closed,
        gripper_qpos=np.asarray(qpos) if qpos is not None else None,
    )


class PickCompletionCheckerTest(unittest.TestCase):
    def test_default_requires_five_stable_frames(self) -> None:
        checker = PickCompletionChecker()
        self.assertEqual(checker.stable_frames, 5)

    def test_empty_close_never_starts_candidate(self) -> None:
        checker = PickCompletionChecker(stable_frames=1)
        self.assertFalse(update(checker, qpos=(0.0015, -0.0015)))
        self.assertFalse(
            update(
                checker,
                eef=(0.05, 0.0, 0.0),
                center=(0.15, 0.0, 0.0),
                qpos=(0.0015, -0.0015),
            )
        )
        self.assertIsNone(checker._previous_eef_pos)

    def test_missing_qpos_is_not_treated_as_nonempty(self) -> None:
        checker = PickCompletionChecker(stable_frames=1)
        self.assertFalse(update(checker, qpos=None))
        self.assertIsNone(checker._previous_eef_pos)

    def test_rigid_comovement_completes_after_stable_frames(self) -> None:
        checker = PickCompletionChecker(stable_frames=3)
        self.assertFalse(update(checker))
        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.02), center=(0.1, 0.0, 0.02)))
        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.03), center=(0.1, 0.0, 0.03)))
        self.assertTrue(update(checker, eef=(0.0, 0.0, 0.04), center=(0.1, 0.0, 0.04)))
        self.assertTrue(checker.completed)
        self.assertEqual(checker.confirmation_count, 3)
        self.assertAlmostEqual(checker.last_rigid_error, 0.0)

    def test_stationary_object_does_not_follow_moving_gripper(self) -> None:
        checker = PickCompletionChecker(stable_frames=1)
        self.assertFalse(update(checker))
        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.04)))
        self.assertEqual(checker.confirmation_count, 0)
        self.assertAlmostEqual(checker.vertical_progress, 0.0)
        self.assertGreater(checker.last_rigid_error, checker.max_rigid_error)

    def test_horizontal_comovement_does_not_count_as_lift(self) -> None:
        checker = PickCompletionChecker(stable_frames=1)
        self.assertFalse(update(checker))
        self.assertFalse(
            update(checker, eef=(0.1, 0.0, 0.0), center=(0.2, 0.0, 0.0))
        )
        self.assertAlmostEqual(checker.vertical_progress, 0.0)

    def test_relative_position_is_measured_in_rotating_eef_frame(self) -> None:
        checker = PickCompletionChecker(stable_frames=1, max_rigid_error=1e-8)
        self.assertFalse(update(checker))
        quat = Rotation.from_euler("z", 90, degrees=True).as_quat()
        self.assertTrue(
            update(
                checker,
                eef=(0.0, 0.0, 0.02),
                center=(0.0, 0.1, 0.02),
                quat=quat,
            )
        )
        self.assertAlmostEqual(checker.last_rigid_error, 0.0)

    def test_open_gripper_resets_candidate_and_confirmation(self) -> None:
        checker = PickCompletionChecker(stable_frames=2)
        self.assertFalse(update(checker))
        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.02), center=(0.1, 0.0, 0.02)))
        self.assertEqual(checker.confirmation_count, 1)

        self.assertFalse(update(checker, closed=False))
        self.assertIsNone(checker._previous_eef_pos)
        self.assertEqual(checker.confirmation_count, 0)

        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.02), center=(0.1, 0.0, 0.02)))
        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.04), center=(0.1, 0.0, 0.04)))
        self.assertTrue(update(checker, eef=(0.0, 0.0, 0.05), center=(0.1, 0.0, 0.05)))

    def test_missing_cloud_breaks_motion_chain(self) -> None:
        checker = PickCompletionChecker(stable_frames=2)
        self.assertFalse(update(checker))
        self.assertFalse(update(checker, eef=(0.0, 0.0, 0.02), center=(0.1, 0.0, 0.02)))
        self.assertFalse(update(checker, points=False))
        self.assertIsNone(checker._previous_eef_pos)
        self.assertEqual(checker.confirmation_count, 0)

    def test_downward_motion_cancels_accumulated_lift(self) -> None:
        checker = PickCompletionChecker(stable_frames=1)
        self.assertFalse(update(checker))
        self.assertFalse(
            update(checker, eef=(0.0, 0.0, 0.015), center=(0.1, 0.0, 0.015))
        )
        self.assertFalse(
            update(checker, eef=(0.0, 0.0, 0.005), center=(0.1, 0.0, 0.005))
        )
        self.assertFalse(
            update(checker, eef=(0.0, 0.0, 0.015), center=(0.1, 0.0, 0.015))
        )
        self.assertAlmostEqual(checker.vertical_progress, 0.015)

    def test_reset_clears_latched_completion(self) -> None:
        checker = PickCompletionChecker(stable_frames=1)
        self.assertFalse(update(checker))
        self.assertTrue(update(checker, eef=(0.0, 0.0, 0.02), center=(0.1, 0.0, 0.02)))
        checker.reset()
        self.assertFalse(checker.completed)
        self.assertEqual(checker.confirmation_count, 0)


if __name__ == "__main__":
    unittest.main()
