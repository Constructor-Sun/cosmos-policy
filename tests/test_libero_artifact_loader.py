"""Characterization tests for current LIBERO memory artifact loaders.

These tests lock the existing artifact formats and error behavior before the
memory_system migration.  They use small synthetic payloads, not the full
LIBERO memories.
"""
from __future__ import annotations

import json
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
from libero_feasible_region_verifier import (  # noqa: E402
    LiberoFeasibleRegionVerifier,
    ReadyDistanceMemory,
)
from libero_phase_verifier import PhaseTargetMemory  # noqa: E402
from libero_pose_recovery import LiberoPoseRecovery  # noqa: E402
from libero_skill_completion_verifier_geo import LiberoSkillCompletionVerifierGeo  # noqa: E402


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


class PhaseTargetMemoryArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "phase_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_templates_and_selects(self) -> None:
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [phase_template("demo_0"), phase_template("demo_1")],
            },
            self.path,
        )
        memory = PhaseTargetMemory(self.path)
        selected = memory.select("task", 1, "Pick", {"item": "object_1"})
        self.assertEqual(len(selected), 2)

    def test_unsupported_format_raises(self) -> None:
        torch.save(
            {"format": "libero_phase_targets_v2", "templates": []},
            self.path,
        )
        with self.assertRaises(ValueError):
            PhaseTargetMemory(self.path)


class ReadyDistanceMemoryArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.targets_path = self.root / "phase_targets.pt"
        self.manifest_path = self.root / "segments.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _save_valid(self) -> None:
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [
                    phase_template("demo_0", frame=10),
                    phase_template("demo_1", frame=10),
                ],
            },
            self.targets_path,
        )
        self.manifest_path.write_text(
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

    def test_builds_prototypes_from_manifest(self) -> None:
        self._save_valid()
        memory = ReadyDistanceMemory(self.targets_path, self.manifest_path)
        prototypes = memory.select("task", 1, "Pick", {"item": "object_1"})
        self.assertEqual(len(prototypes), 2)
        self.assertEqual(prototypes[0].normalized_distance, 0.5)

    def test_unsupported_phase_format_raises(self) -> None:
        torch.save({"format": "not_a_phase_format", "templates": []}, self.targets_path)
        self.manifest_path.write_text(json.dumps({"records": []}))
        with self.assertRaises(ValueError):
            ReadyDistanceMemory(self.targets_path, self.manifest_path)


class PoseRecoveryArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "recovery_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_targets(self) -> None:
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
            self.path,
        )
        recovery = LiberoPoseRecovery(self.path)
        self.assertEqual(len(recovery.targets), 1)

    def test_unsupported_format_raises(self) -> None:
        torch.save({"format": "libero_recovery_targets_v2", "targets": []}, self.path)
        with self.assertRaises(ValueError):
            LiberoPoseRecovery(self.path)


class FeasibleRecoveryArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "feasible_recovery_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_targets(self) -> None:
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
            self.path,
        )
        recovery = LiberoFeasibleRecovery(self.path)
        self.assertEqual(len(recovery.targets), 1)

    def test_unsupported_format_raises(self) -> None:
        torch.save(
            {"format": "libero_feasible_recovery_targets_v2", "targets": []},
            self.path,
        )
        with self.assertRaises(ValueError):
            LiberoFeasibleRecovery(self.path)


class CompletionArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.phase_path = self.root / "phase_targets.pt"
        self.wrist_path = self.root / "wrist_completion_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_phase_anchors_and_wrist_templates(self) -> None:
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [
                    phase_template("demo_0", step=2, skill="Pick"),
                    phase_template("demo_1", step=2, skill="Pick"),
                ],
            },
            self.phase_path,
        )
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
            self.wrist_path,
        )
        verifier = LiberoSkillCompletionVerifierGeo(self.phase_path, self.wrist_path)
        self.assertTrue(any(key[0] == "task" and key[2] == "Pick"
                            for key in verifier._ready_anchors))
        self.assertTrue(any(key[0] == "task" and key[2] == "Pick"
                            for key in verifier._wrist_memory))


class FeasibleWristArtifactTest(unittest.TestCase):
    """Lock the optional feasible_wrist_targets.pt loader in the feasible verifier."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.phase_path = self.root / "phase_targets.pt"
        self.manifest_path = self.root / "segments.json"
        self.wrist_path = self.root / "feasible_wrist_targets.pt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _save_valid(self) -> None:
        torch.save(
            {
                "format": "libero_phase_targets_v1",
                "templates": [
                    phase_template("demo_0", step=2, skill="Pick", frame=10),
                    phase_template("demo_1", step=2, skill="Pick", frame=10),
                ],
            },
            self.phase_path,
        )
        self.manifest_path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "task_name": "task",
                            "demo_id": demo_id,
                            "valid": True,
                            "segments": [
                                {
                                    "planner_step_id": 2,
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
        torch.save(
            {
                "format": "libero_wrist_feasible_targets_v1",
                "templates": [
                    {
                        "task_name": "task",
                        "demo_id": demo_id,
                        "planner_step_id": 2,
                        "skill": "Pick",
                        "arguments": {"item": "object_1"},
                        "frame": 10,
                    }
                    for demo_id in ("demo_0", "demo_1")
                ],
            },
            self.wrist_path,
        )

    def test_loads_wrist_feasible_templates(self) -> None:
        self._save_valid()
        memory = ReadyDistanceMemory(self.phase_path, self.manifest_path)
        verifier = LiberoFeasibleRegionVerifier(
            memory=memory,
            wrist_feasible_targets=self.wrist_path,
        )
        verifier.reset("task", 2, "Pick", {"item": "object_1"})
        self.assertEqual(len(verifier.wrist_templates), 2)
        self.assertEqual(
            len(verifier.wrist_templates_by_phase), 1
        )

    def test_unsupported_wrist_format_raises(self) -> None:
        self._save_valid()
        # Overwrite the wrist artifact with an unsupported format after the
        # valid phase/manifest files have been created.
        torch.save({"format": "bad_wrist_format", "templates": []}, self.wrist_path)
        memory = ReadyDistanceMemory(self.phase_path, self.manifest_path)
        with self.assertRaises(ValueError):
            LiberoFeasibleRegionVerifier(
                memory=memory,
                wrist_feasible_targets=self.wrist_path,
            )




if __name__ == "__main__":
    unittest.main()
