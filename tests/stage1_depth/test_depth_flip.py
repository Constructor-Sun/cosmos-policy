"""Stage 1, check 3: flip alignment of the metric depth map.

The memory system consumes the agentview image flipped (np.flipud,
flip_images=True) and matches templates in that flipped space, so the depth
adapter must emit a metric depth map flipped with the same convention.

Check A (exact, array identity): flip_depth(D) == np.flipud(D), i.e. the
adapter flips depth exactly the way the RGB pipeline flips images, so pixel
(r, c) of the flipped depth corresponds to pixel (r, c) of the flipped RGB.

Check B (exact, back-projection consistency): back-projecting the FLIPPED
depth at the flipped coordinate (H-1-r, c) -- using the flip-aware camera
convention (unflip the row before the pinhole math, see harness.back_project
_flipped) -- must recover the same world point as back-projecting the
unflipped depth at (r, c).  This pins down the row-mirror convention the
production rgbd back-projection must follow.

Check C (world-anchored sanity, pixel level): project the EEF world pose
(robot proprioception, real-robot compatible) to the agentview pixel, flip
the row, back-project the flipped depth there, and require the recovered
point to re-project to the SAME flipped pixel (within rounding).  The depth
at the gripper pixel measures the finger *surface*, which is offset from the
wrist EEF frame along the ray by a few cm, so we assert pixel consistency
(1-2 px) rather than 5 mm 3D distance; the 5 mm claim is Stage 2's job.
"""
import numpy as np
import pytest

from harness import RESOLUTION, back_project, back_project_flipped, camera_params, flip_depth, metric_depth
from robosuite.utils.camera_utils import project_points_from_world_to_camera

PX_TOL = 2  # pixels (rounding + K principal-point half-pixel)


@pytest.mark.parametrize("task", [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
])
def test_flip_array_identity(env_ctx, task):
    _env, obs = env_ctx(task)
    d = metric_depth(_env, obs)
    d_f = flip_depth(d)
    assert d_f.shape == d.shape
    assert np.array_equal(d_f, np.flipud(d))
    # per-pixel: flipped depth at flipped coord == unflipped depth at original coord
    r = 100
    c = 200
    assert d_f[RESOLUTION - 1 - r, c] == d[r, c]


@pytest.mark.parametrize("task", [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
])
def test_flip_backproject_consistency(env_ctx, task):
    env, obs = env_ctx(task)
    K, _T_w2c, T_c2w = camera_params(env)
    d = metric_depth(env, obs)
    d_f = flip_depth(d)

    rng = np.random.default_rng(1)
    rows = rng.integers(0, RESOLUTION, 200)
    cols = rng.integers(0, RESOLUTION, 200)
    pix = np.stack([rows, cols], axis=-1)
    pix_f = np.stack([RESOLUTION - 1 - rows, cols], axis=-1)

    P = back_project(pix, d, K, T_c2w)
    P_f = back_project_flipped(pix_f, d_f, K, T_c2w)
    assert np.allclose(P, P_f, atol=1e-9), (
        "flip-aware back-projection at flipped coords != unflipped back-projection"
    )


@pytest.mark.parametrize("task", [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
])
def test_eef_roundtrip_flipped_pixel(env_ctx, task):
    env, obs = env_ctx(task)
    K, T_w2c, T_c2w = camera_params(env)
    d_f = flip_depth(metric_depth(env, obs))

    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    row_col = project_points_from_world_to_camera(eef, T_w2c, RESOLUTION, RESOLUTION)
    row_f = int(RESOLUTION - 1 - row_col[0])
    col = int(row_col[1])

    P = back_project_flipped(np.array([[row_f, col]]), d_f, K, T_c2w)[0]
    # forward-project the recovered point; it must land on the same flipped pixel
    r1 = project_points_from_world_to_camera(P, T_w2c, RESOLUTION, RESOLUTION)
    row_f1 = int(RESOLUTION - 1 - r1[0])
    col1 = int(r1[1])
    assert abs(row_f1 - row_f) <= PX_TOL and abs(col1 - col) <= PX_TOL, (
        f"recovered point re-projects to flipped pixel ({row_f1},{col1}) "
        f"instead of ({row_f},{col}); P={P.round(4)} eef={eef.round(4)}"
    )
