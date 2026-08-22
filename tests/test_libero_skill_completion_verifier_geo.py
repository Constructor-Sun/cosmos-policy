#!/usr/bin/env python3
"""Unit tests for the parameter-free geometric completion verifier."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

EXECUTE_BIN = Path(__file__).resolve().parents[1] / "bin" / "execute"
sys.path.insert(0, str(EXECUTE_BIN))

from libero_skill_completion_verifier_geo import (  # noqa: E402
    COMPLETION_UNKNOWN,
    SKILL_COMPLETE,
    LiberoSkillCompletionVerifierGeo,
)


def make_phase_targets(task="T", demo_ids=("d0", "d1", "d2")):
    """Templates: step2 Pick (lifted cluster ~(108-110,95-97)) and step4 Close."""
    templates = []
    lifted = {demo: (108.0 + i, 95.0 + i) for i, demo in enumerate(demo_ids)}
    closed = {demo: (100.0 + i, 150.0 + i) for i, demo in enumerate(demo_ids)}
    for demo in demo_ids:
        for step, skill, args, frames, grippers in (
            (2, "Pick", {"item": "obj"}, (0, 40, 80),
             ((129.0, 189.0), (112.0, 144.0), lifted[demo])),
            (4, "Close", {"target": "t"}, (0, 40, 80),
             ((180.0, 100.0), (120.0, 120.0), closed[demo])),
        ):
            for frame, gxy in zip(frames, grippers):
                templates.append({
                    "task_name": task,
                    "demo_id": demo,
                    "planner_step_id": step,
                    "skill": skill,
                    "arguments": args,
                    "frame": frame,
                    "gripper_xy": np.asarray(gxy, dtype=np.float32),
                    "bbox_xyxy": np.asarray([80, 140, 120, 190], dtype=np.float32),
                    "target_center_xy": np.asarray([100, 165], dtype=np.float32),
                })
    return {"templates": templates}


class GeoVerifierTest(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
            self.path = fh.name
        torch.save(make_phase_targets(), self.path)
        self.v = LiberoSkillCompletionVerifierGeo(self.path)

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def _step(self, closed, xy, bbox=None, **kw):
        return self.v.update(None, gripper_closed=closed, gripper_xy=xy,
                             target_bbox=bbox, timestep=kw.get("t"))

    def test_pick_rejects_hover_outside_anchor_cluster(self):
        # Pick anchors are (108,95); hovering at x=116 is outside the cluster.
        self.v.reset("T", 2, "Pick", {"item": "obj"})
        r = self._step(True, np.asarray([116.0, 106.0]))
        self.assertEqual(r.status, COMPLETION_UNKNOWN)
        self.assertEqual(r.reason, "outside_anchor_cluster")

    def test_pick_latches_at_lifted_pose(self):
        self.v.reset("T", 2, "Pick", {"item": "obj"})
        r1 = self._step(True, np.asarray([108.0, 95.0]))
        self.assertEqual(r1.reason, "completion_candidate")
        r2 = self._step(True, np.asarray([109.0, 96.0]))
        self.assertEqual(r2.status, SKILL_COMPLETE)

    def test_pick_latches_by_position_without_closed(self):
        # Anchor skills fire on pose alone: the drawer push is done open.
        self.v.reset("T", 2, "Pick", {"item": "obj"})
        r1 = self._step(False, np.asarray([108.0, 95.0]))
        self.assertEqual(r1.reason, "completion_candidate")
        r2 = self._step(False, np.asarray([109.0, 96.0]))
        self.assertEqual(r2.status, SKILL_COMPLETE)

    def test_pick_resets_on_gap(self):
        self.v.reset("T", 2, "Pick", {"item": "obj"})
        self._step(True, np.asarray([108.0, 95.0]))
        self._step(True, np.asarray([116.0, 106.0]))
        r = self._step(True, np.asarray([108.0, 95.0]))
        self.assertEqual(r.reason, "completion_candidate")
        self.assertEqual(r.confirmation_count, 1)

    def test_place_completes_on_release(self):
        self.v.reset("T", 3, "PlaceIn", {"item": "obj", "target": "t"})
        bbox = (130, 80, 250, 200)
        self._step(True, np.asarray([160.0, 100.0]), bbox=bbox)
        r1 = self._step(False, np.asarray([161.0, 101.0]), bbox=bbox)
        self.assertEqual(r1.status, SKILL_COMPLETE)
        self.assertEqual(r1.reason, "confirmed_complete")

    def test_close_latches_inside_anchor_cluster(self):
        # Close anchors are (100,150).
        self.v.reset("T", 4, "Close", {"target": "t"})
        r1 = self._step(True, np.asarray([100.0, 150.0]))
        self.assertEqual(r1.reason, "completion_candidate")
        r2 = self._step(True, np.asarray([101.0, 151.0]))
        self.assertEqual(r2.status, SKILL_COMPLETE)

    def test_close_rejects_far_from_anchors(self):
        self.v.reset("T", 4, "Close", {"target": "t"})
        r = self._step(True, np.asarray([180.0, 100.0]))
        self.assertEqual(r.reason, "outside_anchor_cluster")

    def test_unknown_skill_is_unknown(self):
        self.v.reset("T", 1, "TurnOn", {"target": "t"})
        r = self._step(True, np.asarray([100.0, 150.0]))
        self.assertEqual(r.status, COMPLETION_UNKNOWN)
        self.assertEqual(r.reason, "no_geometric_completion_rule")

    def test_pick_moved_uses_cumulative_eef_displacement(self):
        # Movement is latched once cumulative 3D EEF displacement reaches 0.04.
        self.v.reset("T", 2, "Pick", {"item": "obj"})
        dummy_wrist = np.zeros((16, 16, 3), dtype=np.uint8)
        self.v.update(
            None, gripper_closed=True, gripper_xy=np.asarray([0.0, 0.0]),
            wrist_image=dummy_wrist, eef_pos=np.asarray([0.0, 0.0, 0.0]),
        )
        self.assertFalse(self.v.pick_moved)
        self.v.update(
            None, gripper_closed=True, gripper_xy=np.asarray([0.0, 0.0]),
            wrist_image=dummy_wrist, eef_pos=np.asarray([0.02, 0.0, 0.0]),
        )
        self.assertFalse(self.v.pick_moved)
        self.v.update(
            None, gripper_closed=True, gripper_xy=np.asarray([0.0, 0.0]),
            wrist_image=dummy_wrist, eef_pos=np.asarray([0.05, 0.0, 0.0]),
        )
        self.assertTrue(self.v.pick_moved)


if __name__ == "__main__":
    unittest.main()
