"""Stage 1, check 1: depth observation plumbing.

Purpose (3D.md Stage 1): confirm the LIBERO RGB-D observation yields the same
normalized depth the renderer produces for the same camera/resolution.

Under test: obs["agentview_depth"] (key, shape, dtype, values).
Reference: sim.render("agentview", depth=True) -- same renderer, so this is a
plumbing check only, NOT an independent oracle for the metric conversion.

Each test creates its OWN env (fresh, single-env-at-a-time) because keeping
two LIBERO envs alive in one process corrupts rendering (see README).
"""
import numpy as np
import pytest

from harness import RESOLUTION


@pytest.mark.parametrize("task", ["KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"])
def test_depth_obs_shape_dtype_range(env_ctx, task):
    _env, obs = env_ctx(task)
    d = np.asarray(obs["agentview_depth"])
    assert d.shape == (RESOLUTION, RESOLUTION, 1)
    assert d.dtype == np.float32
    assert d.min() >= 0.0 and d.max() <= 1.0


@pytest.mark.parametrize("task", ["KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"])
def test_depth_obs_matches_sim_render(env_ctx, task):
    env, obs = env_ctx(task)
    _rgb, rendered = env.env.sim.render(
        camera_name="agentview", width=RESOLUTION, height=RESOLUTION, depth=True
    )
    rendered = np.asarray(rendered, dtype=np.float64)
    obs_d = np.asarray(obs["agentview_depth"], dtype=np.float64)[..., 0]
    assert rendered.shape == obs_d.shape
    assert np.allclose(obs_d, rendered, atol=1e-6), (
        f"obs depth differs from sim.render depth: max abs diff = "
        f"{np.abs(obs_d - rendered).max():.3e}"
    )
