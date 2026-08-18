from pathlib import Path
import sys
import tempfile
import unittest

import torch


EXECUTE_BIN = Path(__file__).resolve().parents[1] / "bin" / "execute"
sys.path.insert(0, str(EXECUTE_BIN))

from libero_skill_completion_verifier import (  # noqa: E402
    COMPLETION_UNKNOWN,
    SKILL_COMPLETE,
    LiberoSkillCompletionVerifier,
    SkillSuccessMemory,
)


class SkillCompletionVerifierTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        for index, vector in enumerate(((1.0, 0.0), (0.99, 0.1), (0.98, -0.05))):
            torch.save({
                "format": "libero_skill_memory_v2",
                "task_name": "task",
                "demo_id": f"demo_{index}",
                "segments": [
                    {"planner_step_id": 1, "skill": "Pick",
                     "arguments": {"item": "object"},
                     "success_vae": torch.tensor(vector),
                     "success_embedding_source": "recorded_success_window"},
                    {"planner_step_id": 2, "skill": "PlaceIn",
                     "arguments": {"item": "object", "target": "basket"},
                     "success_vae": torch.tensor(tuple(reversed(vector))),
                     "success_embedding_source": "recorded_success_window"},
                ],
            }, root / f"skill_memory_demo_{index}.pt")
        self.verifier = LiberoSkillCompletionVerifier(SkillSuccessMemory(root))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pick_requires_closed_gripper_and_two_confirmations(self):
        self.verifier.reset("task", 1, "Pick", {"item": "object"})
        blocked = self.verifier.update(torch.tensor((1.0, 0.0)), gripper_closed=False)
        first = self.verifier.update(torch.tensor((1.0, 0.0)), gripper_closed=True)
        second = self.verifier.update(torch.tensor((1.0, 0.0)), gripper_closed=True)
        self.assertEqual(blocked.reason, "gripper_gate_not_met")
        self.assertEqual(first.status, COMPLETION_UNKNOWN)
        self.assertEqual(first.reason, "completion_candidate")
        self.assertEqual(second.status, SKILL_COMPLETE)

    def test_place_requires_open_gripper(self):
        self.verifier.reset(
            "task", 2, "PlaceIn", {"item": "object", "target": "basket"}
        )
        blocked = self.verifier.update(torch.tensor((0.0, 1.0)), gripper_closed=True)
        first = self.verifier.update(torch.tensor((0.0, 1.0)), gripper_closed=False)
        second = self.verifier.update(torch.tensor((0.0, 1.0)), gripper_closed=False)
        self.assertEqual(blocked.reason, "gripper_gate_not_met")
        self.assertEqual(first.status, COMPLETION_UNKNOWN)
        self.assertEqual(second.status, SKILL_COMPLETE)

    def test_distant_observation_is_only_unknown(self):
        self.verifier.reset("task", 1, "Pick", {"item": "object"})
        result = self.verifier.update(torch.tensor((0.0, 1.0)), gripper_closed=True)
        self.assertEqual(result.status, COMPLETION_UNKNOWN)
        self.assertEqual(result.reason, "success_memory_not_matched")


if __name__ == "__main__":
    unittest.main()
