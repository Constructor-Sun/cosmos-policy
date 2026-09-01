"""Tests for PlaceIn/PlaceOn VLA closed->open completion."""
from __future__ import annotations

import unittest

from memory_system.execute.plan import PhaseSpec
from memory_system.execute.skill_completion.place import (
    DEFAULT_OPEN_FRAMES,
    ReleaseSkillCompletion,
)
from memory_system.execute.vla_skill_runtime import make_completion


class ReleaseSkillCompletionTest(unittest.TestCase):
    def test_default_open_frames_is_reasonable(self) -> None:
        self.assertEqual(DEFAULT_OPEN_FRAMES, 5)

    def test_without_closed_confirmation_waits_for_closed_then_open(self) -> None:
        skill = ReleaseSkillCompletion(required_open_frames=2)
        # Not yet holding: wait for a command-closed frame to establish baseline.
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.08, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=True, gripper_qpos=[0.02, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.02, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.08, 0.0]).advance
        )
        decision = skill.observe_frame(
            gripper_closed=False, gripper_qpos=[0.08, 0.0]
        )
        self.assertTrue(decision.advance)
        self.assertTrue(decision.semantic_completed)
        self.assertEqual(decision.reason, "rule")

    def test_with_closed_confirmation_starts_in_wait_open(self) -> None:
        skill = ReleaseSkillCompletion(
            required_open_frames=2,
            closed_confirmed=True,
        )
        # First frame records the holding baseline.
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.02, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.08, 0.0]).advance
        )
        decision = skill.observe_frame(
            gripper_closed=False, gripper_qpos=[0.08, 0.0]
        )
        self.assertTrue(decision.advance)
        self.assertTrue(decision.semantic_completed)

    def test_open_jitter_resets_consecutive_count(self) -> None:
        skill = ReleaseSkillCompletion(
            required_open_frames=3,
            closed_confirmed=True,
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.02, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.08, 0.0]).advance
        )
        # A closed command in the middle resets the open streak.
        self.assertFalse(
            skill.observe_frame(gripper_closed=True, gripper_qpos=[0.02, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.08, 0.0]).advance
        )
        self.assertFalse(
            skill.observe_frame(gripper_closed=False, gripper_qpos=[0.08, 0.0]).advance
        )
        decision = skill.observe_frame(
            gripper_closed=False, gripper_qpos=[0.08, 0.0]
        )
        self.assertTrue(decision.advance)
        self.assertEqual(decision.reason, "rule")

    def test_timeout_still_advances_as_backstop(self) -> None:
        skill = ReleaseSkillCompletion(
            max_action_chunks=2,
            required_open_frames=10,
            closed_confirmed=True,
        )
        self.assertFalse(skill.finish_action_chunk().advance)
        decision = skill.finish_action_chunk()
        self.assertTrue(decision.advance)
        self.assertFalse(decision.semantic_completed)
        self.assertEqual(decision.reason, "timeout")

    def test_make_completion_uses_release_for_place(self) -> None:
        completion = make_completion(PhaseSpec(1, "PlaceIn", {"item": "a"}))
        self.assertIsInstance(completion, ReleaseSkillCompletion)
        self.assertFalse(completion.closed_confirmed)

    def test_make_completion_passes_initially_holding_to_place(self) -> None:
        completion = make_completion(
            PhaseSpec(1, "PlaceOn", {"item": "a"}),
            initially_holding=True,
        )
        self.assertIsInstance(completion, ReleaseSkillCompletion)
        self.assertTrue(completion.closed_confirmed)

    def test_make_completion_rejects_unknown_skill(self) -> None:
        with self.assertRaises(KeyError):
            make_completion(PhaseSpec(1, "UnknownSkill", {}))


if __name__ == "__main__":
    unittest.main()
