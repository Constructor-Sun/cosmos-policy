"""Tests for memory_system artifact loaders.

These tests mirror the bin artifact-loader characterization tests so the new
self-contained package is kept behaviorally consistent with the frozen bin
implementation.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from memory_system.artifacts import (
    FeasibleRecoveryMemory,
    PhaseTargetMemory,
    PoseRecoveryMemory,
    Ready3DMemory,
    ReadyDistanceMemory,
    WristCompletionMemory,
    WristFeasibleMemory,
)


def phase_template(
    demo_id: str,
    step: int = 1,
    skill: str = "Pick",
    arguments: dict[str, str] | None = None,
    frame: int = 10,
) -> dict:
    return {
        "task_name": "task",
        "demo_id": demo_id,
        "planner_step_id": step,
        "skill": skill,
        "arguments": dict(arguments or {"item": "object_1"}),
        "frame": frame,
        "target_center_xy": np.array([0.0, 0.0], dtype=np.float32),
        "gripper_xy": np.array([5.0, 0.0], dtype=np.float32),
        "bbox_xyxy": np.array([0, 0, 6, 8], dtype=np.int16),
    }


class MemorySystemArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_phase_target_memory_loads_and_selects(self) -> None:
        path = self.root / "phase_targets.pt"
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [
                    phase_template("demo_0"),
                    phase_template("demo_1"),
                    phase_template("demo_2", arguments={"item": "other"}),
                ],
            },
            path,
        )
        memory = PhaseTargetMemory(path)
        selected = memory.select("task", 1, "Pick", {"item": "object_1"})
        self.assertEqual(len(selected), 2)

    def test_phase_target_memory_rejects_unknown_format(self) -> None:
        path = self.root / "bad.pt"
        torch.save({"format": "libero_phase_targets_v2", "templates": []}, path)
        with self.assertRaises(ValueError):
            PhaseTargetMemory(path)

    def test_ready_distance_memory_builds_prototypes(self) -> None:
        phase_path = self.root / "phase_targets.pt"
        manifest_path = self.root / "segments.json"
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [
                    phase_template("demo_0"),
                    phase_template("demo_1"),
                ],
            },
            phase_path,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "task_name": "task",
                            "demo_id": demo_id,
                            "valid": True,
                            "segments": [
                                {
                                    "planner_step_id": 1,
                                    "skill": "Pick",
                                    "arguments": {"item": "object_1"},
                                    "ready_frame": 10,
                                }
                            ],
                        }
                        for demo_id in ("demo_0", "demo_1")
                    ]
                }
            )
        )
        memory = ReadyDistanceMemory(phase_path, manifest_path)
        prototypes = memory.select("task", 1, "Pick", {"item": "object_1"})
        self.assertEqual(len(prototypes), 2)
        self.assertEqual(prototypes[0].normalized_distance, 0.5)

    def test_ready3d_memory_returns_nearest_exact_phase(self) -> None:
        path = self.root / "ready3d_targets.pt"
        prototypes = [
            {
                "task_name": "task",
                "demo_id": demo_id,
                "planner_step_id": step,
                "skill": "PlaceOn",
                "arguments": {"item": "mug", "target": "plate"},
                "target_xyz_world": np.array(xyz),
            }
            for demo_id, step, xyz in (
                ("demo_0", 2, [0.0, 0.0, 0.0]),
                ("demo_1", 2, [0.2, 0.0, 0.0]),
                ("wrong_step", 3, [0.19, 0.0, 0.0]),
            )
        ]
        torch.save(
            {"format": "libero_ready3d_targets_v1", "prototypes": prototypes},
            path,
        )

        ranked = Ready3DMemory(path).nearest(
            "task",
            2,
            "PlaceOn",
            {"item": "mug", "target": "plate"},
            [0.19, 0.0, 0.0],
        )

        self.assertEqual([item[0]["demo_id"] for item in ranked], ["demo_1", "demo_0"])
        self.assertAlmostEqual(ranked[0][1], 0.01)

    def test_pose_recovery_memory_loads_and_selects(self) -> None:
        path = self.root / "recovery_targets.pt"
        torch.save(
            {
                "format": "libero_recovery_targets_v1",
                "targets": [
                    {
                        "task_name": "task",
                        "demo_id": "demo_0",
                        "planner_step_id": 1,
                        "skill": "Pick",
                        "arguments": {"item": "object_1"},
                    }
                ],
            },
            path,
        )
        memory = PoseRecoveryMemory(path)
        self.assertEqual(len(memory.select("task", 1, "Pick", {"item": "object_1"})), 1)

    def test_feasible_recovery_memory_loads_and_selects(self) -> None:
        path = self.root / "feasible_recovery_targets.pt"
        torch.save(
            {
                "format": "libero_feasible_recovery_targets_v1",
                "targets": [
                    {
                        "task_name": "task",
                        "demo_id": "demo_0",
                        "planner_step_id": 1,
                        "skill": "Pick",
                        "arguments": {"item": "object_1"},
                    }
                ],
            },
            path,
        )
        memory = FeasibleRecoveryMemory(path)
        self.assertEqual(len(memory.select("task", 1, "Pick", {"item": "object_1"})), 1)

    def test_wrist_completion_memory_loads_and_selects(self) -> None:
        path = self.root / "wrist_completion_targets.pt"
        torch.save(
            {
                "format": "libero_wrist_completion_targets_v1",
                "templates": [
                    {
                        "task_name": "task",
                        "demo_id": "demo_0",
                        "planner_step_id": 2,
                        "skill": "Pick",
                        "arguments": {"item": "object_1"},
                        "frame": 10,
                    }
                ],
            },
            path,
        )
        memory = WristCompletionMemory(path)
        self.assertEqual(len(memory.select("task", 2, "Pick", {"item": "object_1"})), 1)

    def test_wrist_feasible_memory_loads_and_selects(self) -> None:
        path = self.root / "feasible_wrist_targets.pt"
        torch.save(
            {
                "format": "libero_wrist_feasible_targets_v1",
                "templates": [
                    {
                        "task_name": "task",
                        "demo_id": "demo_0",
                        "planner_step_id": 2,
                        "skill": "Pick",
                        "arguments": {"item": "object_1"},
                        "frame": 10,
                    }
                ],
            },
            path,
        )
        memory = WristFeasibleMemory(path)
        self.assertEqual(len(memory.select("task", 2, "Pick", {"item": "object_1"})), 1)

    def test_recovery_memories_reject_unknown_format(self) -> None:
        for cls, name, fmt in (
            (PoseRecoveryMemory, "recovery.pt", "libero_recovery_targets_v2"),
            (FeasibleRecoveryMemory, "feasible.pt", "libero_feasible_recovery_targets_v2"),
        ):
            path = self.root / name
            torch.save({"format": fmt, "targets": []}, path)
            with self.assertRaises(ValueError):
                cls(path)


if __name__ == "__main__":
    unittest.main()
