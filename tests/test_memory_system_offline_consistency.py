"""Consistency tests between bin/memory and memory_system/offline.

These tests cover the pure/offline logic that can be checked without running
MuJoCo/HDF5 builds.  Heavy end-to-end comparison tests live in
``test_memory_system_offline_migration.py`` and are opt-in via environment
variables.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_BIN = REPO_ROOT / "bin" / "memory"
if str(MEMORY_BIN) not in sys.path:
    sys.path.insert(0, str(MEMORY_BIN))

from label_libero_skill_ready_boundaries import (  # noqa: E402
    fixed_terminal_boundary as bin_fixed_terminal_boundary,
    shared_terminal_boundaries as bin_shared_terminal_boundaries,
    sustained_event_start as bin_sustained_event_start,
)
from libero_skill_skeleton import build_skeleton as bin_build_skeleton  # noqa: E402

from memory_system.offline.label_boundaries import (  # noqa: E402
    fixed_terminal_boundary as ms_fixed_terminal_boundary,
    shared_terminal_boundaries as ms_shared_terminal_boundaries,
    sustained_event_start as ms_sustained_event_start,
)
from memory_system.offline.planner import build_skeleton as ms_build_skeleton  # noqa: E402


class BoundaryConsistencyTest(unittest.TestCase):
    def test_fixed_terminal_boundary_matches(self) -> None:
        for length, horizon in ((30, 16), (10, 16), (5, 16), (100, 8)):
            old = bin_fixed_terminal_boundary(length, horizon)
            new = ms_fixed_terminal_boundary(length, horizon)
            self.assertEqual(new.terminal_start, old.terminal_start)
            self.assertEqual(new.method, old.method)

    def test_sustained_event_start_matches(self) -> None:
        cases = [
            np.array([-1] * 4 + [1] * 2 + [-1] * 5 + [1] * 7, dtype=float),
            np.array([1] * 8 + [-1] * 6, dtype=float),
            np.ones(12, dtype=float),
            np.array([-1, 1, -1, 1, 1, 1], dtype=float),
        ]
        for gripper in cases:
            for target_value, min_run in ((1.0, 3), (-1.0, 3), (1.0, 2)):
                self.assertEqual(
                    ms_sustained_event_start(gripper, target_value, min_run),
                    bin_sustained_event_start(gripper, target_value, min_run),
                )

    def test_shared_terminal_boundaries_matches(self) -> None:
        rng = np.random.default_rng(0)
        sequences = [
            np.concatenate(
                [
                    rng.normal(0.0, 0.3, size=(20, 7)),
                    np.tile(np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]), (8, 1)),
                ],
                axis=0,
            )
            for _ in range(3)
        ]
        old = bin_shared_terminal_boundaries(
            sequences, fallback_length=16, max_suffix=32, min_terminal=4
        )
        new = ms_shared_terminal_boundaries(
            sequences, fallback_length=16, max_suffix=32, min_terminal=4
        )
        self.assertEqual(len(new), len(old))
        for old_b, new_b in zip(old, new):
            self.assertEqual(new_b.terminal_start, old_b.terminal_start)
            self.assertEqual(new_b.method, old_b.method)


class PlannerConsistencyTest(unittest.TestCase):
    def test_skeleton_matches(self) -> None:
        goals = [
            ("open", "drawer"),
            ("in", "bowl", "drawer"),
            ("close", "drawer"),
        ]
        old = bin_build_skeleton(goals, "open the drawer and put the bowl inside", {"drawer"})
        new = ms_build_skeleton(goals, "open the drawer and put the bowl inside", {"drawer"})
        self.assertEqual(
            [(step.skill, step.arguments, step.execution_mode) for step in new],
            [(step.skill, step.arguments, step.execution_mode) for step in old],
        )


if __name__ == "__main__":
    unittest.main()
