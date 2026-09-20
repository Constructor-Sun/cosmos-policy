"""Tests for the TurnOn rotation rule."""
from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.execute.skill_completion import (
    COMPLETION_REGISTRY,
    DEFAULT_MIN_ROTATION_DEG,
    TurnOnCompletion,
)


def _quat(deg: float) -> np.ndarray:
    return Rotation.from_euler("x", deg, degrees=True).as_quat()


class TurnOnCompletionTest(unittest.TestCase):
    def test_default_threshold_matches_calibration(self) -> None:
        # Measured from the grasp: 29-34 deg when the knob turns on,
        # 49-55 deg ten frames later.
        self.assertEqual(DEFAULT_MIN_ROTATION_DEG, 35.0)

    def test_registered_as_the_turnon_rule(self) -> None:
        self.assertIs(COMPLETION_REGISTRY["TurnOn"], TurnOnCompletion)

    def test_advances_once_rotation_passes_threshold(self) -> None:
        rule = TurnOnCompletion(min_rotation_deg=40.0)
        self.assertFalse(rule.observe_frame(eef_quat=_quat(0), gripper_closed=True).advance)
        self.assertFalse(rule.observe_frame(eef_quat=_quat(39), gripper_closed=True).advance)
        decision = rule.observe_frame(eef_quat=_quat(41), gripper_closed=True)
        self.assertTrue(decision.advance)
        self.assertEqual(decision.reason, "rule")
        self.assertTrue(decision.semantic_completed)

    def test_open_gripper_never_fires(self) -> None:
        # The approach rotates the wrist too; without a grasp there is no
        # baseline and no completion.
        rule = TurnOnCompletion(min_rotation_deg=40.0)
        for deg in (0, 20, 60, 180):
            self.assertFalse(
                rule.observe_frame(eef_quat=_quat(deg), gripper_closed=False).advance
            )

    def test_baseline_is_taken_at_the_grasp_not_the_phase_start(self) -> None:
        rule = TurnOnCompletion(min_rotation_deg=40.0)
        # Approach: gripper open, wrist already rotated 25 deg.
        self.assertFalse(
            rule.observe_frame(eef_quat=_quat(25), gripper_closed=False).advance
        )
        # Grasp at 25 deg, then turn to 55: 30 deg from the grasp, below 40.
        self.assertFalse(
            rule.observe_frame(eef_quat=_quat(25), gripper_closed=True).advance
        )
        self.assertFalse(
            rule.observe_frame(eef_quat=_quat(55), gripper_closed=True).advance
        )
        # Turn further to 70: 45 deg from the grasp, fires.
        self.assertTrue(
            rule.observe_frame(eef_quat=_quat(70), gripper_closed=True).advance
        )

    def test_missing_orientation_waits_for_timeout(self) -> None:
        rule = TurnOnCompletion(min_rotation_deg=40.0, max_action_chunks=2)
        self.assertFalse(rule.observe_frame(eef_quat=None).advance)
        self.assertFalse(rule.finish_action_chunk().advance)
        decision = rule.finish_action_chunk()
        self.assertTrue(decision.advance)
        self.assertEqual(decision.reason, "timeout")
        self.assertFalse(decision.semantic_completed)

    def test_reset_restarts_from_the_new_orientation(self) -> None:
        rule = TurnOnCompletion(min_rotation_deg=40.0)
        rule.observe_frame(eef_quat=_quat(100), gripper_closed=True)
        rule.reset()
        # The pre-reset orientation is gone; the new baseline starts at 100.
        self.assertFalse(rule.observe_frame(eef_quat=_quat(100), gripper_closed=True).advance)
        self.assertTrue(rule.observe_frame(eef_quat=_quat(141), gripper_closed=True).advance)


if __name__ == "__main__":
    unittest.main()
