"""Characterization tests for current LIBERO recovery selectors.

These tests lock the retrieval/clustering behavior of the phase and feasible
recovery modules before the memory_system migration.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = REPO_ROOT / "bin"
EXECUTE_BIN = BIN_DIR / "execute"
sys.path.insert(0, str(BIN_DIR))
sys.path.insert(0, str(EXECUTE_BIN))

from libero_feasible_recovery import LiberoFeasibleRecovery  # noqa: E402
from libero_pose_recovery import LiberoPoseRecovery  # noqa: E402


SAME_TOKEN = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def pose_target(n: int, step: int = 1, same_token: bool = True) -> dict:
    if same_token:
        token = SAME_TOKEN
    else:
        onehot = [0.0, 0.0, 0.0, 0.0]
        onehot[(n % 3) + 1] = 1.0
        token = torch.tensor(onehot, dtype=torch.float32)
    return {
        "task_name": "task",
        "demo_id": f"demo_{n}",
        "planner_step_id": step,
        "skill": "Pick",
        "arguments": {"item": "object_1"},
        "recovery_vae_main": token,
        "ee_states": np.array([0.05 * n, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "recovery_frame": 20 + n,
        "action_scale": np.array(
            [0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float32
        ),
    }


def feasible_target(n: int, step: int = 1, same_token: bool = True) -> dict:
    if same_token:
        token = SAME_TOKEN
    else:
        onehot = [0.0, 0.0, 0.0, 0.0]
        onehot[(n % 3) + 1] = 1.0
        token = torch.tensor(onehot, dtype=torch.float32)
    return {
        "task_name": "task",
        "demo_id": f"demo_{n}",
        "planner_step_id": step,
        "skill": "Pick",
        "arguments": {"item": "object_1"},
        "ready_vae_main": token,
        "ready_vae_wrist": token,
        "ee_states": np.array([0.05 * n, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "ready_frame": 30 + n,
        "action_scale": np.array(
            [0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float32
        ),
    }


class PoseRecoverySelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "recovery_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write(self, targets: list[dict]) -> LiberoPoseRecovery:
        torch.save(
            {"format": "libero_recovery_targets_v1", "targets": targets},
            self.path,
        )
        return LiberoPoseRecovery(
            self.path,
            min_demo_votes=2,
            similarity_threshold=0.2,
            position_radius=0.1,
            rotation_radius=0.5,
        )

    def test_select_prefers_exact_step_over_fallback(self) -> None:
        recovery = self._write(
            [pose_target(0, step=1), pose_target(1, step=2), pose_target(2, step=2)]
        )
        exact = recovery._select("task", 2, "Pick", {"item": "object_1"})
        self.assertEqual({item["planner_step_id"] for item in exact}, {2})

        fallback = recovery._select("task", 99, "Pick", {"item": "object_1"})
        self.assertEqual({item["planner_step_id"] for item in fallback}, {1, 2})

    def test_compute_returns_averaged_target_from_cluster(self) -> None:
        recovery = self._write([pose_target(0), pose_target(1), pose_target(2)])
        result = recovery.compute(
            "task",
            1,
            "Pick",
            {"item": "object_1"},
            current_vae_main=SAME_TOKEN,
            current_ee_states=np.zeros(6, dtype=np.float32),
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.demo_ids), 2)
        self.assertEqual(result.target_ee_states.shape, (6,))
        # Average of first two cluster members is 0.025 in x.
        self.assertAlmostEqual(float(result.target_ee_states[0]), 0.025, places=5)
        self.assertEqual(result.correction_per_step.shape, (1, 6))

    def test_compute_returns_none_when_insufficient_votes(self) -> None:
        recovery = self._write([pose_target(0)])
        result = recovery.compute(
            "task",
            1,
            "Pick",
            {"item": "object_1"},
            current_vae_main=SAME_TOKEN,
            current_ee_states=np.zeros(6, dtype=np.float32),
        )
        self.assertIsNone(result)

    def test_compute_returns_none_when_similarity_below_threshold(self) -> None:
        recovery = self._write(
            [pose_target(0, same_token=False), pose_target(1, same_token=False)]
        )
        result = recovery.compute(
            "task",
            1,
            "Pick",
            {"item": "object_1"},
            current_vae_main=SAME_TOKEN,
            current_ee_states=np.zeros(6, dtype=np.float32),
        )
        self.assertIsNone(result)


class FeasibleRecoverySelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "feasible_recovery_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write(self, targets: list[dict]) -> LiberoFeasibleRecovery:
        torch.save(
            {
                "format": "libero_feasible_recovery_targets_v1",
                "targets": targets,
            },
            self.path,
        )
        return LiberoFeasibleRecovery(
            self.path,
            min_demo_votes=2,
            similarity_threshold=0.2,
            position_radius=0.1,
            rotation_radius=0.5,
            z_lift=0.02,
        )

    def test_compute_returns_result_using_main_plus_wrist_tokens(self) -> None:
        recovery = self._write(
            [feasible_target(0), feasible_target(1), feasible_target(2)]
        )
        result = recovery.compute(
            "task",
            1,
            "Pick",
            {"item": "object_1"},
            current_vae_main=SAME_TOKEN,
            current_vae_wrist=SAME_TOKEN,
            current_ee_states=np.zeros(6, dtype=np.float32),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.target_ee_states.shape, (6,))
        self.assertEqual(result.controller.z_lift, 0.02)

    def test_compute_nearest_ready_returns_closest_pose(self) -> None:
        recovery = self._write(
            [feasible_target(0), feasible_target(1), feasible_target(2)]
        )
        result = recovery.compute_nearest_ready(
            "task",
            1,
            "Pick",
            {"item": "object_1"},
            current_ee_states=np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.demo_ids, ("demo_1",))
        self.assertAlmostEqual(float(result.target_ee_states[0]), 0.05, places=5)


if __name__ == "__main__":
    unittest.main()
