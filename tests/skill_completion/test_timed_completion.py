"""Tests for the shared action-chunk skill-completion lifecycle."""
from __future__ import annotations

import unittest

import numpy as np

from memory_system.execute.skill_completion.base import (
    TimedSkillCompletion,
)
from memory_system.execute.skill_completion.pick import PickSkillCompletion


class RuleSkill(TimedSkillCompletion):
    def _check_rule(self, *, satisfied: bool = False) -> bool:
        return satisfied


class TimedSkillCompletionTest(unittest.TestCase):
    def test_timeout_advances_after_budget(self) -> None:
        skill = RuleSkill(max_action_chunks=2)
        self.assertFalse(skill.finish_action_chunk().advance)
        decision = skill.finish_action_chunk()
        self.assertTrue(decision.advance)
        self.assertFalse(decision.semantic_completed)
        self.assertEqual(decision.reason, "timeout")
        self.assertEqual(decision.action_chunks, 2)

    def test_rule_completion_advances_before_timeout(self) -> None:
        skill = RuleSkill(max_action_chunks=2)
        decision = skill.observe_frame(satisfied=True)
        self.assertTrue(decision.advance)
        self.assertTrue(decision.semantic_completed)
        self.assertEqual(decision.reason, "rule")

    def test_reset_clears_budget_and_decision(self) -> None:
        skill = RuleSkill(max_action_chunks=1)
        skill.finish_action_chunk()
        skill.reset()
        self.assertFalse(skill.decision.advance)
        self.assertEqual(skill.decision.reason, "running")
        self.assertEqual(skill.decision.action_chunks, 0)

    def test_pick_child_reuses_existing_rule(self) -> None:
        skill = PickSkillCompletion(max_action_chunks=2, stable_frames=1)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        points = np.array(
            [[0.1, 0.0, 0.0], [0.11, 0.0, 0.0], [0.1, 0.01, 0.0], [0.1, 0.0, 0.01]]
        )
        skill.observe_frame(
            target_points=points,
            eef_pos=np.array([0.0, 0.0, 0.0]),
            eef_quat=quat,
            gripper_closed=True,
            gripper_qpos=np.array([0.002, -0.002]),
        )
        decision = skill.observe_frame(
            target_points=points + np.array([0.0, 0.0, 0.02]),
            eef_pos=np.array([0.0, 0.0, 0.02]),
            eef_quat=quat,
            gripper_closed=True,
            gripper_qpos=np.array([0.002, -0.002]),
        )
        self.assertTrue(decision.advance)
        self.assertTrue(decision.semantic_completed)
        self.assertEqual(decision.reason, "rule")


if __name__ == "__main__":
    unittest.main()
