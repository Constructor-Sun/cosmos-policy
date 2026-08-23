"""Characterization tests for the current 2D phase matcher / target memory.

These tests lock the existing bin/execute behavior before the memory_system
migration.  They use synthetic templates so they do not depend on the full
LIBERO memory artifacts.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

EXECUTE_BIN = Path(__file__).resolve().parents[1] / "bin" / "execute"
sys.path.insert(0, str(EXECUTE_BIN))

from libero_phase_verifier import (  # noqa: E402
    PHASE_ERROR,
    PHASE_OK,
    PHASE_UNKNOWN,
    LiberoPhaseVerifier,
    PhaseTargetMemory,
)


def make_template(
    demo_id: str,
    step: int = 1,
    skill: str = "Pick",
    arguments: dict[str, str] | None = None,
    color: tuple[int, int, int] = (80, 120, 200),
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
        "frame": 0,
        "crop_rgb": crop,
        "crop_mask": mask,
        "target_center_xy": np.array([16.0, 16.0], dtype=np.float32),
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


class PhaseTargetMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "phase_targets.pt"
        save_phase_targets(
            self.path,
            [
                make_template("demo_0", step=1, arguments={"item": "object_1"}),
                make_template("demo_1", step=1, arguments={"item": "object_1"}),
                make_template("demo_2", step=1, arguments={"item": "other"}),
                make_template("demo_3", step=2, arguments={"item": "object_1"}),
            ],
        )
        self.memory = PhaseTargetMemory(self.path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_select_exact_matches_step_and_arguments(self) -> None:
        selected = self.memory.select("task", 1, "Pick", {"item": "object_1"})
        self.assertEqual(
            {(item["demo_id"], item["planner_step_id"]) for item in selected},
            {("demo_0", 1), ("demo_1", 1)},
        )

    def test_select_falls_back_to_same_skill_and_arguments_when_step_missing(self) -> None:
        selected = self.memory.select("task", 99, "Pick", {"item": "object_1"})
        self.assertEqual(
            {(item["demo_id"], item["planner_step_id"]) for item in selected},
            {("demo_0", 1), ("demo_1", 1), ("demo_3", 2)},
        )

    def test_select_excludes_demo_ids(self) -> None:
        selected = self.memory.select(
            "task", 1, "Pick", {"item": "object_1"}, exclude_demo_ids=("demo_1",)
        )
        self.assertEqual([item["demo_id"] for item in selected], ["demo_0"])

    def test_unsupported_format_raises(self) -> None:
        bad = self.root / "bad.pt"
        torch.save({"format": "libero_phase_targets_v2", "templates": []}, bad)
        with self.assertRaises(ValueError):
            PhaseTargetMemory(bad)


class PhaseVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "phase_targets.pt"
        save_phase_targets(
            self.path,
            [
                make_template("demo_0", step=1, arguments={"item": "object_1"}),
                make_template("demo_1", step=1, arguments={"item": "object_1"}),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_update_without_templates_returns_unknown(self) -> None:
        empty = self.root / "empty.pt"
        save_phase_targets(empty, [])
        verifier = LiberoPhaseVerifier(
            empty, min_similarity=0.01, min_demo_votes=2
        )
        verifier.reset("task", 1, "Pick", {"item": "object_1"})
        result = verifier.update(make_image(), (0, 0))
        self.assertEqual(result.status, PHASE_UNKNOWN)
        self.assertEqual(result.details["reason"], "no_templates")

    def test_update_returns_ok_when_two_demos_vote_for_same_region(self) -> None:
        verifier = LiberoPhaseVerifier(
            self.path, min_similarity=0.01, min_demo_votes=2
        )
        verifier.reset("task", 1, "Pick", {"item": "object_1"})
        result = verifier.update(make_image(), (0, 0))
        self.assertEqual(result.status, PHASE_OK)
        self.assertEqual(result.match_count, 2)
        # The template rectangle is placed at x=28..43, y=18..33, center ~35,25.
        self.assertAlmostEqual(result.target_xy[0], 35.0, delta=1.0)
        self.assertAlmostEqual(result.target_xy[1], 25.0, delta=1.0)

    def test_update_requires_min_demo_votes(self) -> None:
        verifier = LiberoPhaseVerifier(
            self.path, min_similarity=0.01, min_demo_votes=3
        )
        verifier.reset("task", 1, "Pick", {"item": "object_1"})
        result = verifier.update(make_image(), (0, 0))
        self.assertEqual(result.status, PHASE_UNKNOWN)
        self.assertEqual(result.details["reason"], "no_visual_consensus")

    def test_update_detects_wrong_way_motion_as_phase_error(self) -> None:
        verifier = LiberoPhaseVerifier(
            self.path, min_similarity=0.01, min_demo_votes=2
        )
        verifier.reset("task", 1, "Pick", {"item": "object_1"})
        # First observation warms up the previous gripper position.
        first = verifier.update(make_image(), (60, 0))
        self.assertEqual(first.status, PHASE_OK)
        # Moving from x=60 back to x=0 is away from the target at x~35.
        second = verifier.update(make_image(), (0, 0))
        self.assertEqual(second.status, PHASE_ERROR)
        self.assertEqual(second.details["wrong_way_count"], 1)


if __name__ == "__main__":
    unittest.main()
