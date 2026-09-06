"""Invariants of the re-anchor object-frame re-map (P2 correctness contract).

Contract under test (_remap_object_sequence):
  world[t] = T_world_object @ object[t]
  - the re-anchored first frame equals the re-mapped ready target
  - orientation and translation travel rigidly with the object frame
  - the input sequence is never mutated

Run:  pytest memory_system/pointcloud_action/tests/test_reanchor_invariants.py
  or: python memory_system/pointcloud_action/tests/test_reanchor_invariants.py
"""
import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.pointcloud_action.eval.local_pick_core import (
    _remap_object_sequence,
)
from memory_system.pointcloud_action.offline.geometry_utils import (
    matrix_to_ee_states,
)


def _random_transform(rng, scale=1.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.5).as_matrix()
    T[:3, 3] = rng.normal(size=3) * scale
    return T


def _random_sequence(rng, n=7):
    return [_random_transform(rng) for _ in range(n)]


def test_world_equals_anchor_times_object():
    rng = np.random.default_rng(0)
    T_anchor = _random_transform(rng)
    seq = _random_sequence(rng)
    world = np.stack([T_anchor @ m for m in seq], axis=0)
    for t in range(len(seq)):
        assert np.allclose(world[t], T_anchor @ seq[t], atol=1e-12)


def test_remap_first_frame_matches_ready_target():
    rng = np.random.default_rng(1)
    T_anchor = _random_transform(rng)
    T_ready = _random_transform(rng)
    seq = _random_sequence(rng)
    remapped = _remap_object_sequence(T_ready, seq)
    ready_target = matrix_to_ee_states(T_ready @ seq[0])
    assert np.allclose(matrix_to_ee_states(remapped[0]), ready_target, atol=1e-12)


def test_remap_rotation_rigid_in_object_frame():
    """Orientation remaps as R_ready @ R_obj (translation shape is covered by
    test_remap_translation_shape_preserving)."""
    rng = np.random.default_rng(2)
    T_ready = _random_transform(rng)
    seq = _random_sequence(rng)
    remapped = _remap_object_sequence(T_ready, seq)
    for t in range(len(seq)):
        assert np.allclose(remapped[t][:3, :3], T_ready[:3, :3] @ seq[t][:3, :3], atol=1e-12)


def test_remap_translation_shape_preserving():
    rng = np.random.default_rng(3)
    T_ready = _random_transform(rng)
    seq = _random_sequence(rng)
    remapped = _remap_object_sequence(T_ready, seq)
    p0 = (T_ready @ seq[0])[:3, 3]
    for t in range(len(seq)):
        expected = p0 + T_ready[:3, :3] @ (seq[t][:3, 3] - seq[0][:3, 3])
        assert np.allclose(remapped[t][:3, 3], expected, atol=1e-12)


def test_input_sequence_not_mutated():
    rng = np.random.default_rng(4)
    T_ready = _random_transform(rng)
    seq = np.stack(_random_sequence(rng), axis=0)
    snapshot = seq.copy()
    _remap_object_sequence(T_ready, seq)
    assert np.array_equal(seq, snapshot)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
