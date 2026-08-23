import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


EXECUTE_BIN = Path(__file__).resolve().parents[1] / "bin" / "execute"
sys.path.insert(0, str(EXECUTE_BIN))

from libero_execution_monitor import (  # noqa: E402
    COMPLETION_CHECK,
    FEASIBLE_CHECK,
    PHASE_CHECK,
    PLAN_COMPLETE,
    LiberoExecutionMonitor,
)
from libero_feasible_region_verifier import FEASIBLE, FEASIBLE_UNKNOWN  # noqa: E402
from libero_phase_monitor import PhaseSpec, load_phase_plans  # noqa: E402
from libero_phase_verifier import PHASE_OK, PHASE_UNKNOWN  # noqa: E402
from libero_skill_completion_verifier_geo import (  # noqa: E402
    COMPLETION_UNKNOWN,
    SKILL_COMPLETE,
)


class FakeVerifier:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.resets = 0

    def reset(self, *args):
        self.resets += 1

    def calibrate(self, *args, **kwargs):
        pass

    def freeze_baseline(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        self.calls += 1
        return self.results.pop(0)


def phase(progress, status=PHASE_OK, reason=None):
    details = {"matched_bbox_xyxy": (0, 0, 10, 10)}
    if reason:
        details = {"reason": reason}
    return SimpleNamespace(
        status=status, progress_px=progress, target_xy=(5, 5), details=details,
    )


class ExecutionMonitorTest(unittest.TestCase):
    def test_phase_plan_skips_only_steps_that_are_always_already_satisfied(self):
        records = [
            {"task_name": "task", "demo_id": "demo_0", "valid": True,
             "segments": [
                 {"planner_step_id": 1, "skill": "Open", "arguments": {},
                  "status": "already_satisfied"},
                 {"planner_step_id": 2, "skill": "Pick", "arguments": {},
                  "status": "executed"},
             ]},
            {"task_name": "task", "demo_id": "demo_1", "valid": True,
             "segments": [
                 {"planner_step_id": 1, "skill": "Open", "arguments": {},
                  "status": "already_satisfied"},
                 {"planner_step_id": 2, "skill": "Pick", "arguments": {},
                  "status": "executed"},
             ]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "segments.json"
            manifest.write_text(json.dumps({"records": records}))
            self.assertEqual(
                [phase.planner_step_id for phase in load_phase_plans(manifest)["task"]],
                [2],
            )
            records.append({
                "task_name": "task", "demo_id": "demo_2", "valid": True,
                "segments": [{"planner_step_id": 1, "skill": "Open",
                              "arguments": {}, "status": "executed"}],
            })
            manifest.write_text(json.dumps({"records": records}))
            self.assertEqual(
                [phase.planner_step_id for phase in load_phase_plans(manifest)["task"]],
                [1, 2],
            )

    def test_runs_one_stage_at_a_time_and_advances_on_completion(self):
        phase_verifier = FakeVerifier([phase(None), phase(3.0)])
        feasible = FakeVerifier([
            SimpleNamespace(status=FEASIBLE_UNKNOWN, reason="approaching"),
            SimpleNamespace(status=FEASIBLE, reason="entered_ready_radius"),
        ])
        feasible.prototypes, feasible.min_demo_votes = (1, 2), 2
        completion = FakeVerifier([
            SimpleNamespace(status=COMPLETION_UNKNOWN, reason="completion_candidate"),
            SimpleNamespace(status=SKILL_COMPLETE, reason="confirmed_complete"),
        ])
        plans = {"task": (
            PhaseSpec(1, "Pick", {"item": "object"}),
            PhaseSpec(2, "PlaceIn", {"item": "object", "target": "basket"}),
        )}
        monitor = LiberoExecutionMonitor(plans, phase_verifier, feasible, completion)
        monitor.start_episode("task")
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        self.assertEqual(monitor.observe(image, (0, 0)).stage_after, PHASE_CHECK)
        self.assertEqual(monitor.observe(image, (1, 1)).stage_after, FEASIBLE_CHECK)
        self.assertEqual(monitor.observe(image, (2, 2)).stage_after, FEASIBLE_CHECK)
        self.assertEqual(monitor.observe(image, (3, 3)).stage_after, COMPLETION_CHECK)
        self.assertEqual(monitor.observe_vae(None).stage_after, COMPLETION_CHECK)
        advanced = monitor.observe_vae(None)
        self.assertTrue(advanced.phase_advanced)
        self.assertEqual(advanced.completed_step_id, 1)
        self.assertEqual(advanced.active_step_id, 2)
        self.assertEqual(advanced.stage_after, PHASE_CHECK)
        self.assertEqual((phase_verifier.calls, feasible.calls, completion.calls), (2, 2, 2))

    def test_no_phase_templates_goes_to_completion_precheck(self):
        phase_verifier = FakeVerifier([phase(None, PHASE_UNKNOWN, "no_templates")])
        feasible = FakeVerifier([])
        feasible.prototypes, feasible.min_demo_votes = (), 2
        completion = FakeVerifier([
            SimpleNamespace(status=SKILL_COMPLETE, reason="confirmed_complete")
        ])
        plans = {"task": (PhaseSpec(1, "Open", {"target": "drawer"}),)}
        monitor = LiberoExecutionMonitor(plans, phase_verifier, feasible, completion)
        monitor.start_episode("task")
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        routed = monitor.observe(image, (0, 0))
        done = monitor.observe_vae(None)
        self.assertEqual(routed.stage_after, COMPLETION_CHECK)
        self.assertEqual(done.stage_after, PLAN_COMPLETE)
        self.assertTrue(done.plan_complete)


if __name__ == "__main__":
    unittest.main()
