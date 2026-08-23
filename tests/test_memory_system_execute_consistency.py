"""Old bin/execute vs new memory_system/execute consistency tests.

These tests lock Step 3 migration behavior: the new execute package should
produce the same decisions/state transitions as the frozen bin/execute
implementations on synthetic inputs.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = REPO_ROOT / "bin"
EXECUTE_BIN = BIN_DIR / "execute"
sys.path.insert(0, str(BIN_DIR))
sys.path.insert(0, str(EXECUTE_BIN))

from libero_feasible_region_verifier import (  # noqa: E402
    LiberoFeasibleRegionVerifier,
)
from libero_phase_verifier import (  # noqa: E402
    LiberoPhaseVerifier,
)
from libero_skill_completion_verifier_geo import (  # noqa: E402
    LiberoSkillCompletionVerifierGeo,
)

from memory_system.execute.feasible import FeasibleVerifier  # noqa: E402
from memory_system.execute.phase import PhaseVerifier  # noqa: E402
from memory_system.execute.skill_completion import SkillCompletionVerifier  # noqa: E402
from memory_system.types import VerifierObservation  # noqa: E402


def make_phase_template(
    demo_id: str,
    step: int = 1,
    skill: str = "Pick",
    arguments: dict[str, str] | None = None,
    color: tuple[int, int, int] = (80, 120, 200),
    frame: int = 0,
    gripper_xy: tuple[float, float] = (16.0, 16.0),
) -> dict:
    crop = np.full((32, 32, 3), 200, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(crop, (8, 8), (23, 23), color, -1)
    cv2.rectangle(mask, (8, 8), (23, 23), 255, -1)
    return {
        "task_name": "task",
        "demo_id": demo_id,
        "planner_step_id": step,
        "skill": skill,
        "arguments": dict(arguments or {"item": "object_1"}),
        "frame": frame,
        "crop_rgb": crop,
        "crop_mask": mask,
        "target_center_xy": np.array([16.0, 16.0], dtype=np.float32),
        "gripper_xy": np.array(gripper_xy, dtype=np.float32),
        "bbox_xyxy": np.array([8, 8, 23, 23], dtype=np.int16),
    }


def save_phase_targets(path: Path, templates: list[dict]) -> None:
    torch.save(
        {"format": "libero_phase_targets_v1", "templates": templates},
        path,
    )


def make_image(rect_xy: tuple[int, int] = (28, 18), size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = rect_xy
    cv2.rectangle(image, (x, y), (x + 15, y + 15), (80, 120, 200), -1)
    return image


class PhaseConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "phase_targets.pt"
        save_phase_targets(
            self.path,
            [
                make_phase_template("demo_0"),
                make_phase_template("demo_1"),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_phase_update_matches_old(self) -> None:
        old = LiberoPhaseVerifier(self.path, min_similarity=0.01, min_demo_votes=2)
        old.reset("task", 1, "Pick", {"item": "object_1"})
        new = PhaseVerifier(self.path, min_similarity=0.01, min_demo_votes=2)
        new.reset("task", 1, "Pick", {"item": "object_1"})

        image = make_image()
        old_result = old.update(image, (20, 20))
        new_result = new.update(
            VerifierObservation(third_view_rgb=image, gripper_xy=(20, 20))
        )

        self.assertEqual(new_result.status, old_result.status)
        self.assertEqual(new_result.target_xy, old_result.target_xy)
        self.assertEqual(new_result.match_count, old_result.match_count)
        self.assertEqual(new_result.template_demo_id, old_result.template_demo_id)
        self.assertAlmostEqual(
            new_result.progress_px if new_result.progress_px is not None else -1,
            old_result.progress_px if old_result.progress_px is not None else -1,
            places=4,
        )


class FeasibleConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.targets_path = self.root / "phase_targets.pt"
        self.manifest_path = self.root / "segments.json"
        demos = (("demo_0", 5.0), ("demo_1", 6.0), ("demo_2", 7.0))
        save_phase_targets(
            self.targets_path,
            [
                make_phase_template(
                    demo_id,
                    frame=10,
                    gripper_xy=(distance, 0.0),
                )
                for demo_id, distance in demos
            ],
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

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_feasible_sequence_matches_old(self) -> None:
        old = LiberoFeasibleRegionVerifier(
            self.targets_path, self.manifest_path
        )
        old.reset("task", 1, "Pick", {"item": "object_1"})
        new = FeasibleVerifier(self.targets_path, self.manifest_path)
        new.reset("task", 1, "Pick", {"item": "object_1"})

        grippers = [(10, 0), (14, 0), (5.5, 0), (18, 0)]
        for gripper in grippers:
            old_result = old.update(
                (0, 0), (0, 0, 6, 8), gripper
            )
            new_result = new.update(
                VerifierObservation(gripper_xy=gripper, timestep=0),
                (0, 0),
                (0, 0, 6, 8),
            )
            self.assertEqual(new_result.status, old_result.status)
            self.assertEqual(new_result.reason, old_result.reason)
            self.assertEqual(new_result.ready_votes, old_result.ready_votes)
            self.assertEqual(new_result.entered_feasible, old_result.entered_feasible)


class CompletionConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "phase_targets.pt"
        # Anchors around x=16 for Open/Close and Pick; bbox target is supplied at runtime.
        save_phase_targets(
            self.path,
            [
                make_phase_template("demo_0", skill="Open", frame=10, gripper_xy=(16, 16)),
                make_phase_template("demo_1", skill="Open", frame=10, gripper_xy=(17, 16)),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_open_completion_matches_old(self) -> None:
        old = LiberoSkillCompletionVerifierGeo(self.path)
        old.reset("task", 1, "Open", {"target": "drawer"})
        new = SkillCompletionVerifier(self.path)
        new.reset("task", 1, "Open", {"target": "drawer"})

        old_result = old.update(gripper_closed=True, gripper_xy=(16, 16))
        new_result = new.update(gripper_closed=True, gripper_xy=(16, 16))
        self.assertEqual(new_result.status, old_result.status)
        self.assertEqual(new_result.reason, old_result.reason)
        self.assertEqual(new_result.confirmation_count, old_result.confirmation_count)

        old_result = old.update(gripper_closed=True, gripper_xy=(16, 16))
        new_result = new.update(gripper_closed=True, gripper_xy=(16, 16))
        self.assertEqual(new_result.status, old_result.status)
        self.assertEqual(new_result.reason, old_result.reason)
        self.assertEqual(new_result.confirmation_count, old_result.confirmation_count)

    def test_place_completion_matches_old(self) -> None:
        old = LiberoSkillCompletionVerifierGeo(self.path)
        old.reset("task", 1, "PlaceIn", {"item": "object_1", "target": "basket"})
        new = SkillCompletionVerifier(self.path)
        new.reset("task", 1, "PlaceIn", {"item": "object_1", "target": "basket"})

        bbox = (5, 5, 20, 20)
        old_result = old.update(
            gripper_closed=False, gripper_xy=(10, 10), target_bbox=bbox
        )
        new_result = new.update(
            gripper_closed=False, gripper_xy=(10, 10), target_bbox=bbox
        )
        self.assertEqual(new_result.status, old_result.status)
        self.assertEqual(new_result.reason, old_result.reason)
        self.assertEqual(new_result.confirmation_count, old_result.confirmation_count)

        old_result = old.update(
            gripper_closed=False, gripper_xy=(10, 10), target_bbox=bbox
        )
        new_result = new.update(
            gripper_closed=False, gripper_xy=(10, 10), target_bbox=bbox
        )
        self.assertEqual(new_result.status, old_result.status)
        self.assertEqual(new_result.reason, old_result.reason)
        self.assertEqual(new_result.confirmation_count, old_result.confirmation_count)


class ExecutionMonitorConsistencyTest(unittest.TestCase):
    def test_monitor_state_machine_matches_old(self) -> None:
        from libero_execution_monitor import LiberoExecutionMonitor
        from libero_phase_monitor import PhaseSpec as OldPhaseSpec

        from memory_system.execute.execution_monitor import ExecutionMonitor
        from memory_system.execute.plan import PhaseSpec

        old_plans = {
            "task": (
                OldPhaseSpec(1, "Pick", {"item": "object"}),
                OldPhaseSpec(2, "PlaceIn", {"item": "object", "target": "basket"}),
            )
        }
        new_plans = {
            "task": (
                PhaseSpec(1, "Pick", {"item": "object"}),
                PhaseSpec(2, "PlaceIn", {"item": "object", "target": "basket"}),
            )
        }

        def old_phase(status, progress, reason=None):
            details = {"matched_bbox_xyxy": (0, 0, 10, 10)}
            if reason:
                details = {"reason": reason}
            return type("R", (), {
                "status": status, "progress_px": progress, "target_xy": (5, 5),
                "details": details,
            })()

        def new_phase(status, progress, reason=None):
            return old_phase(status, progress, reason)

        old_feasible = type("F", (), {
            "prototypes": (1, 2), "min_demo_votes": 2,
            "entered_feasible": False, "stall_count": 0, "wrong_way_count": 0,
            "reset": lambda *a, **k: None,
            "update": lambda *a, **k: type("R", (), {
                "status": "FEASIBLE_UNKNOWN", "reason": "approaching",
            })(),
        })()
        new_feasible = type("F", (), {
            "prototypes": (1, 2), "min_demo_votes": 2,
            "entered_feasible": False, "stall_count": 0, "wrong_way_count": 0,
            "reset": lambda *a, **k: None,
            "update": lambda *a, **k: type("R", (), {
                "status": "FEASIBLE_UNKNOWN", "reason": "approaching",
            })(),
        })()

        def make_completion(status):
            return type("C", (), {
                "reset": lambda *a, **k: None,
                "calibrate": lambda *a, **k: None,
                "freeze_baseline": lambda *a, **k: None,
                "observe_wrist": lambda *a, **k: None,
                "evaluate_close_edge": lambda *a, **k: None,
                "wrong_grasp_pending": False,
                "update": lambda *a, **k: type("R", (), {
                    "status": status, "reason": "r",
                })(),
            })()

        # Use a phase sequence that confirms after the second PHASE observation.
        old_phase_verifier = type("P", (), {
            "reset": lambda *a, **k: None,
            "calls": 0,
            "update": lambda self, img, gxy: (
                old_phase("PHASE_UNKNOWN", None, "no_templates")
                if self.calls == 0 else old_phase("PHASE_OK", 3.0)
            ),
        })()
        new_phase_verifier = type("P", (), {
            "reset": lambda *a, **k: None,
            "calls": 0,
            "update": lambda self, obs: (
                new_phase("PHASE_UNKNOWN", None, "no_templates")
                if self.calls == 0 else new_phase("PHASE_OK", 3.0)
            ),
        })()
        # The lambda above does not mutate calls; use a small wrapper via manual patching.
        old_phase_verifier.calls = [0]
        new_phase_verifier.calls = [0]
        old_phase_verifier.update = lambda img, gxy: (
            old_phase("PHASE_UNKNOWN", None, "no_templates")
            if old_phase_verifier.calls[0] == 0
            else old_phase("PHASE_OK", 3.0)
        )
        new_phase_verifier.update = lambda obs: (
            new_phase("PHASE_UNKNOWN", None, "no_templates")
            if new_phase_verifier.calls[0] == 0
            else new_phase("PHASE_OK", 3.0)
        )

        old_monitor = LiberoExecutionMonitor(
            old_plans, old_phase_verifier, old_feasible,
            make_completion("COMPLETION_UNKNOWN"),
        )
        new_monitor = ExecutionMonitor(
            new_plans, new_phase_verifier, new_feasible,
            make_completion("COMPLETION_UNKNOWN"),
        )
        old_monitor.start_episode("task")
        new_monitor.start_episode("task")
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        old_first = old_monitor.observe(image, (0, 0))
        new_first = new_monitor.observe(
            VerifierObservation(third_view_rgb=image, gripper_xy=(0, 0))
        )
        self.assertEqual(new_first.stage_after, old_first.stage_after)
        self.assertEqual(new_first.reason, old_first.reason)

        old_phase_verifier.calls[0] += 1
        new_phase_verifier.calls[0] += 1
        old_second = old_monitor.observe(image, (1, 1))
        new_second = new_monitor.observe(
            VerifierObservation(third_view_rgb=image, gripper_xy=(1, 1))
        )
        self.assertEqual(new_second.stage_after, old_second.stage_after)
        self.assertEqual(new_second.reason, old_second.reason)


class RecoverySelectorConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.phase_path = self.root / "recovery_targets.pt"
        self.feasible_path = self.root / "feasible_recovery_targets.pt"
        self.same_token = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        torch.save(
            {
                "format": "libero_recovery_targets_v1",
                "targets": [self._pose_target(n) for n in range(3)],
            },
            self.phase_path,
        )
        torch.save(
            {
                "format": "libero_feasible_recovery_targets_v1",
                "targets": [self._feasible_target(n) for n in range(3)],
            },
            self.feasible_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _pose_target(self, n: int) -> dict:
        return {
            "task_name": "task",
            "demo_id": f"demo_{n}",
            "planner_step_id": 1,
            "skill": "Pick",
            "arguments": {"item": "object_1"},
            "recovery_vae_main": self.same_token,
            "ee_states": np.array(
                [0.05 * n, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "recovery_frame": 20 + n,
            "action_scale": np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float32),
        }

    def _feasible_target(self, n: int) -> dict:
        return {
            "task_name": "task",
            "demo_id": f"demo_{n}",
            "planner_step_id": 1,
            "skill": "Pick",
            "arguments": {"item": "object_1"},
            "ready_vae_main": self.same_token,
            "ready_vae_wrist": self.same_token,
            "ee_states": np.array(
                [0.05 * n, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "ready_frame": 30 + n,
            "action_scale": np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float32),
        }

    def test_phase_recovery_matches_old(self) -> None:
        from libero_pose_recovery import LiberoPoseRecovery

        from memory_system.execute.recovery import PhaseRecoverySelector

        old = LiberoPoseRecovery(
            self.phase_path, min_demo_votes=2, similarity_threshold=0.2,
            position_radius=0.1, rotation_radius=0.5,
        )
        new = PhaseRecoverySelector(
            self.phase_path, min_demo_votes=2, similarity_threshold=0.2,
            position_radius=0.1, rotation_radius=0.5,
        )
        current_ee = np.zeros(6, dtype=np.float32)
        old_result = old.compute(
            "task", 1, "Pick", {"item": "object_1"},
            current_vae_main=self.same_token, current_ee_states=current_ee,
        )
        new_result = new.compute(
            "task", 1, "Pick", {"item": "object_1"},
            current_vae_main=self.same_token, current_ee_states=current_ee,
        )
        self.assertIsNotNone(old_result)
        self.assertIsNotNone(new_result)
        self.assertEqual(new_result.demo_ids, old_result.demo_ids)
        np.testing.assert_allclose(new_result.target_ee_states, old_result.target_ee_states, atol=1e-6)
        self.assertAlmostEqual(new_result.similarity, old_result.similarity, places=6)
        self.assertEqual(new_result.correction_per_step.shape, old_result.correction_per_step.shape)

    def test_feasible_recovery_matches_old(self) -> None:
        from libero_feasible_recovery import LiberoFeasibleRecovery

        from memory_system.execute.recovery import FeasibleRecoverySelector

        old = LiberoFeasibleRecovery(
            self.feasible_path, min_demo_votes=2, similarity_threshold=0.2,
            position_radius=0.1, rotation_radius=0.5, z_lift=0.02,
        )
        new = FeasibleRecoverySelector(
            self.feasible_path, min_demo_votes=2, similarity_threshold=0.2,
            position_radius=0.1, rotation_radius=0.5, z_lift=0.02,
        )
        current_ee = np.zeros(6, dtype=np.float32)
        old_result = old.compute(
            "task", 1, "Pick", {"item": "object_1"},
            current_vae_main=self.same_token, current_vae_wrist=self.same_token,
            current_ee_states=current_ee,
        )
        new_result = new.compute(
            "task", 1, "Pick", {"item": "object_1"},
            current_vae_main=self.same_token, current_vae_wrist=self.same_token,
            current_ee_states=current_ee,
        )
        self.assertIsNotNone(old_result)
        self.assertIsNotNone(new_result)
        self.assertEqual(new_result.demo_ids, old_result.demo_ids)
        np.testing.assert_allclose(new_result.target_ee_states, old_result.target_ee_states, atol=1e-6)
        self.assertEqual(new_result.controller.z_lift, old_result.controller.z_lift)


if __name__ == "__main__":
    unittest.main()
