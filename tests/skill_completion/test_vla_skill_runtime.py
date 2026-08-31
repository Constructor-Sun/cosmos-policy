from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory_system.execute.plan import PhaseSpec, load_phase_sequences
from memory_system.execute.vla_skill_runtime import (
    SKILL_MAX_ACTION_CHUNKS,
    VLASkillRuntime,
    make_completion,
)


class PhaseSequenceLoaderTest(unittest.TestCase):
    def test_preserves_record_order_and_skips_only_satisfied_segments(self) -> None:
        payload = {
            "records": [
                {
                    "valid": True,
                    "task_name": "task",
                    "demo_id": "demo_0",
                    "segments": [
                        {"planner_step_id": 3, "skill": "Pick", "arguments": {"item": "a"}},
                        {"planner_step_id": 1, "skill": "PlaceOn", "arguments": {"item": "a"}},
                        {"planner_step_id": 2, "skill": "Open", "status": "already_satisfied"},
                        {"planner_step_id": 2, "skill": "Close", "arguments": {"target": "x"}},
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.json"
            path.write_text(json.dumps(payload))
            sequence = load_phase_sequences(path)[("task", "demo_0")]
        self.assertEqual([phase.planner_step_id for phase in sequence], [3, 1, 2])
        self.assertEqual([phase.skill for phase in sequence], ["Pick", "PlaceOn", "Close"])


class VLASkillRuntimeTest(unittest.TestCase):
    def test_pick_times_out_on_seventh_chunk(self) -> None:
        phase = PhaseSpec(3, "Pick", {"item": "a"})
        runtime = VLASkillRuntime((phase,), demo_id="demo_0")
        runtime.begin_vla(frame=10)
        self.assertEqual(runtime.completion.action_chunks, 0)
        for _ in range(6):
            decision = runtime.finish_action_chunk(frame=11)
            self.assertFalse(decision.advance)
        decision = runtime.finish_action_chunk(frame=12)
        self.assertTrue(decision.advance)
        self.assertEqual(decision.reason, "timeout")
        self.assertFalse(runtime.active)
        self.assertTrue(runtime.exhausted)
        self.assertEqual(runtime.summaries[0]["action_chunks"], 7)

    def test_non_pick_uses_three_chunk_budget_and_resets_on_phase_switch(self) -> None:
        phases = (
            PhaseSpec(3, "PlaceOn", {"item": "a"}),
            PhaseSpec(1, "TurnOn", {"target": "stove"}),
        )
        runtime = VLASkillRuntime(phases)
        runtime.begin_vla(frame=0)
        runtime.finish_action_chunk(frame=1)
        runtime.finish_action_chunk(frame=2)
        decision = runtime.finish_action_chunk(frame=3)
        self.assertEqual(decision.reason, "timeout")
        self.assertEqual(runtime.phase_index, 1)
        self.assertIsNone(runtime.completion)
        runtime.begin_vla(frame=3)
        self.assertEqual(runtime.completion.action_chunks, 0)
        self.assertEqual(runtime.active_phase.skill, "TurnOn")

    def test_registry_matches_documented_budgets(self) -> None:
        self.assertEqual(SKILL_MAX_ACTION_CHUNKS["Pick"], 7)
        self.assertEqual(SKILL_MAX_ACTION_CHUNKS["PlaceOn"], 3)
        self.assertEqual(make_completion(PhaseSpec(1, "Open", {})).max_action_chunks, 3)


if __name__ == "__main__":
    unittest.main()
