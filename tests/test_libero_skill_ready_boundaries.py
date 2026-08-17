from pathlib import Path
import sys
import unittest

import numpy as np


MEMORY_BIN = Path(__file__).resolve().parents[1] / "bin" / "memory"
sys.path.insert(0, str(MEMORY_BIN))

from label_libero_skill_ready_boundaries import (  # noqa: E402
    fixed_terminal_boundary,
    shared_terminal_boundaries,
    sustained_event_start,
)


def action_sequence(prefix_length: int, tail: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prefix = rng.normal(0.0, 0.35, size=(prefix_length, 7))
    prefix[:, -1] = -1.0
    return np.concatenate([prefix, tail], axis=0)


class ReadyBoundaryTest(unittest.TestCase):
    def test_fixed_boundary_uses_exact_tail_horizon(self):
        self.assertEqual(fixed_terminal_boundary(30, 16).terminal_start, 14)
        self.assertEqual(fixed_terminal_boundary(10, 16).terminal_start, 0)
        self.assertEqual(fixed_terminal_boundary(30, 16).method, "fixed_last_chunk")

    def test_final_sustained_close_edge_ignores_failed_attempt(self):
        gripper = np.array([-1] * 4 + [1] * 2 + [-1] * 5 + [1] * 7, dtype=float)
        self.assertEqual(
            sustained_event_start(gripper, target_value=1.0, min_run=3), 11
        )

    def test_final_sustained_open_edge(self):
        gripper = np.array([1] * 8 + [-1] * 6, dtype=float)
        self.assertEqual(
            sustained_event_start(gripper, target_value=-1.0, min_run=3), 8
        )

    def test_initial_target_run_is_not_an_event(self):
        self.assertIsNone(
            sustained_event_start(np.ones(12), target_value=1.0, min_run=3)
        )

    def test_shared_suffix_finds_common_terminal_region(self):
        tail = np.zeros((10, 7), dtype=float)
        tail[:3, 6] = 1.0
        tail[3:, 2] = 0.45
        tail[3:, 6] = 1.0
        sequences = [
            action_sequence(21, tail, 1),
            action_sequence(25, tail + 0.005, 2),
            action_sequence(18, tail - 0.005, 3),
        ]
        boundaries = shared_terminal_boundaries(
            sequences, fallback_length=8, max_suffix=20, min_terminal=4
        )
        expected = [21, 25, 18]
        self.assertTrue(
            all(
                abs(result.terminal_start - target) <= 2
                for result, target in zip(boundaries, expected)
            )
        )
        self.assertTrue(
            all(result.method == "same_task_shared_suffix" for result in boundaries)
        )

    def test_single_sequence_uses_last_chunk_fallback(self):
        sequence = np.zeros((30, 7), dtype=float)
        result = shared_terminal_boundaries([sequence], fallback_length=16)[0]
        self.assertEqual(result.terminal_start, 14)
        self.assertEqual(result.method, "fallback_last_chunk")

    def test_shared_suffix_cannot_absorb_a_long_common_approach(self):
        tail = np.zeros((10, 7), dtype=float)
        sequences = [action_sequence(40, tail, seed) for seed in range(3)]
        results = shared_terminal_boundaries(
            sequences, fallback_length=8, max_suffix=32, min_terminal=4
        )
        self.assertTrue(all(len(sequence) - result.terminal_start <= 16 for sequence, result in zip(sequences, results)))


if __name__ == "__main__":
    unittest.main()
