import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


EXECUTE_BIN = Path(__file__).resolve().parents[1] / "bin" / "execute"
sys.path.insert(0, str(EXECUTE_BIN))

from libero_feasible_region_verifier import (  # noqa: E402
    FEASIBLE,
    FEASIBLE_UNKNOWN,
    NOT_FEASIBLE,
    LiberoFeasibleRegionVerifier,
    ReadyDistanceMemory,
)


def template(demo_id: str, distance_px: float) -> dict:
    return {
        "task_name": "task",
        "demo_id": demo_id,
        "planner_step_id": 1,
        "skill": "Pick",
        "arguments": {"item": "object_1"},
        "frame": 10,
        "target_center_xy": np.asarray([0.0, 0.0], dtype=np.float32),
        "gripper_xy": np.asarray([distance_px, 0.0], dtype=np.float32),
        "bbox_xyxy": np.asarray([0, 0, 6, 8], dtype=np.int16),
    }


class FeasibleRegionVerifierTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.targets_path = root / "phase_targets.pt"
        self.manifest_path = root / "segments.json"
        demos = (("demo_0", 5.0), ("demo_1", 6.0), ("demo_2", 7.0))
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [template(demo_id, distance) for demo_id, distance in demos],
            },
            self.targets_path,
        )
        records = []
        for demo_id, _distance in demos:
            records.append(
                {
                    "task_name": "task",
                    "demo_id": demo_id,
                    "valid": True,
                    "segments": [
                        {
                            "planner_step_id": 1,
                            "skill": "Pick",
                            "arguments": {"item": "object_1"},
                            "start": 0,
                            "end": 26,
                            "terminal_start": 10,
                            "ready_frame": 10,
                        }
                    ],
                }
            )
        self.manifest_path.write_text(json.dumps({"records": records}))
        self.memory = ReadyDistanceMemory(self.targets_path, self.manifest_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def verifier(self) -> LiberoFeasibleRegionVerifier:
        result = LiberoFeasibleRegionVerifier(self.memory)
        result.reset("task", 1, "Pick", {"item": "object_1"})
        return result

    def test_two_memory_votes_enter_and_latch_ready_region(self):
        verifier = self.verifier()
        outside = verifier.update(
            target_xy=(0, 0),
            matched_bbox_xyxy=(0, 0, 6, 8),
            gripper_xy=(6.5, 0),
        )
        self.assertEqual(outside.status, FEASIBLE_UNKNOWN)
        self.assertEqual(outside.ready_votes, 1)

        entered = verifier.update(
            target_xy=(0, 0),
            matched_bbox_xyxy=(0, 0, 6, 8),
            gripper_xy=(5.5, 0),
        )
        self.assertEqual(entered.status, FEASIBLE)
        self.assertEqual(entered.ready_votes, 2)
        self.assertEqual(entered.ready_distance, 0.6)

        moved_away = verifier.update(
            target_xy=(0, 0),
            matched_bbox_xyxy=(0, 0, 6, 8),
            gripper_xy=(20, 0),
        )
        self.assertEqual(moved_away.status, FEASIBLE)
        self.assertEqual(moved_away.reason, "latched_feasible")

    def test_two_reversals_produce_not_feasible(self):
        verifier = self.verifier()
        verifier.update((0, 0), (0, 0, 6, 8), (10, 0))
        candidate = verifier.update((0, 0), (0, 0, 6, 8), (16, 0))
        rejected = verifier.update((0, 0), (0, 0, 6, 8), (22, 0))
        self.assertEqual(candidate.reason, "reversal_candidate")
        self.assertEqual(rejected.status, NOT_FEASIBLE)
        self.assertEqual(rejected.reason, "persistent_reversal")

    def test_two_stationary_updates_produce_not_feasible(self):
        verifier = self.verifier()
        verifier.update((0, 0), (0, 0, 6, 8), (10, 0))
        candidate = verifier.update((0, 0), (0, 0, 6, 8), (10, 0))
        rejected = verifier.update((0, 0), (0, 0, 6, 8), (10, 0))
        self.assertEqual(candidate.reason, "stagnation_candidate")
        self.assertEqual(rejected.status, NOT_FEASIBLE)
        self.assertEqual(rejected.reason, "persistent_stagnation")

    def test_leave_one_demo_out_changes_vote_memory(self):
        verifier = LiberoFeasibleRegionVerifier(self.memory)
        verifier.reset(
            "task", 1, "Pick", {"item": "object_1"}, exclude_demo_ids=("demo_2",)
        )
        result = verifier.update((0, 0), (0, 0, 6, 8), (5.5, 0))
        self.assertEqual(result.memory_count, 2)
        self.assertEqual(result.ready_votes, 1)
        self.assertEqual(result.status, FEASIBLE_UNKNOWN)

    def test_successful_approach_never_requests_correction(self):
        verifier = self.verifier()
        results = [
            verifier.update((0, 0), (0, 0, 6, 8), gripper)
            for gripper in ((20, 0), (14, 0), (5.5, 0), (18, 0))
        ]
        self.assertNotIn(NOT_FEASIBLE, [result.status for result in results])
        self.assertEqual(results[2].status, FEASIBLE)
        self.assertEqual(results[3].reason, "latched_feasible")


if __name__ == "__main__":
    unittest.main()
