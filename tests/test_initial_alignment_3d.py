"""Tests for first-Pick 3-D retrieval in Initial Alignment."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from memory_system.execute.initial_alignment import InitialAlignmentSelector


class InitialAlignment3DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        phase = {
            "planner_step_id": 1,
            "skill": "Pick",
            "arguments": {"item": "object_1"},
        }
        manifest = root / "segments.json"
        demos = (
            ("demo_0", [1.0, 0.0], 0.0, 0.0),
            ("demo_1", [0.0, 1.0], 1.0, 1.0),
            ("demo_2", [-1.0, 0.0], 4.0, 2.0),
        )
        manifest.write_text(json.dumps({"records": [
            {
                "task_name": "task",
                "demo_id": demo_id,
                "valid": True,
                "segments": [phase],
            }
            for demo_id, _vae, _ee, _x in demos
        ]}))

        feasible = root / "feasible.pt"
        torch.save({
            "format": "libero_feasible_recovery_targets_v1",
            "targets": [
                {
                    "task_name": "task",
                    "demo_id": demo_id,
                    **phase,
                    "ready_frame": 10,
                    "ready_vae_main": np.array(vae, dtype=np.float32),
                    "ee_states": np.array([ee, ee, ee, 0.0, 0.0, 0.0]),
                }
                for demo_id, vae, ee, _x in demos
            ],
        }, feasible)

        ready3d = root / "ready3d.pt"
        torch.save({
            "format": "libero_ready3d_targets_v1",
            "prototypes": [
                {
                    "task_name": "task",
                    "demo_id": demo_id,
                    **phase,
                    "target_xyz_world": np.array([x, 0.0, 0.0]),
                }
                for demo_id, _vae, _ee, x in demos
            ],
        }, ready3d)
        self.selector = InitialAlignmentSelector(
            manifest, feasible, ready3d_targets=ready3d
        )
        self.observation = {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}
        self.env = SimpleNamespace(instance_to_id={"object_1": 1})

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def select(self, points):
        with patch(
            "memory_system.execute.initial_alignment.PickTargetPointCloud"
        ) as point_cloud:
            point_cloud.return_value.points.return_value = points
            return self.selector.select(
                "task",
                np.array([1.0, 0.0]),
                np.zeros(6),
                observation=self.observation,
                env=self.env,
            )

    def test_pick_prefers_nearest_3d_demo_over_vae_similarity(self) -> None:
        points = np.repeat([[0.9, 0.0, 0.0]], 4, axis=0)
        result = self.select(points)

        self.assertEqual(result.demo_ids, ("demo_1", "demo_0", "demo_2"))
        np.testing.assert_allclose(
            result.target_ee_states[:3],
            [5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0 + 0.02],
        )
        self.assertEqual(self.selector.last_spatial_match[1][0][0]["demo_id"], "demo_1")

    def test_missing_3d_falls_back_to_existing_vae_selection(self) -> None:
        result = self.select(None)

        self.assertEqual(result.demo_ids, ("demo_0",))
        self.assertIsNone(self.selector.last_spatial_match)


if __name__ == "__main__":
    unittest.main()
