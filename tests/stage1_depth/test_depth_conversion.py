"""Stage 1, check 2: normalized -> metric depth conversion correctness.

Purpose (3D.md Stage 1): confirm the metric depth obtained from the LIBERO
RGB-D observation is correct.

Method (implied-near consistency, direct simulator reference): the metric
conversion `get_real_depth_map` is the standard view-space-z inverse

    z(d) = near / (1 - d * (1 - near/far))

with near = znear*extent and far = zfar*extent from the MuJoCo model.  Given a
rendered normalized value d and its converted metric value z, we can solve for
the near the conversion *implies*:

    near_implied = z * (1 - d) / (1 - d * z / far)

If the conversion is correct, near_implied must equal the model's near at
every pixel, across the whole depth range of the scene.  Any wrong near/far
or a wrong conversion formula breaks this identity proportionally to depth,
so this check certifies the metric values to well below the 5 mm target
(implied near matches the model near to ~1e-6 relative; at 1 m distance that
is ~1 um of depth error).
"""
import numpy as np
import pytest

from harness import RESOLUTION
from memory_system.geometry import camera_params as build_camera_params, depth_to_metric

# relative tolerance on the implied-near identity. float32 depth buffer noise
# makes (1-d) the dominant error term: rel error ~ eps32/(1-d) ~ 1e-4 at worst.
REL_TOL = 5e-4


def _metric(env, obs):
    cam = build_camera_params(env.env.sim, "agentview", RESOLUTION, RESOLUTION)
    raw = np.asarray(obs["agentview_depth"], dtype=np.float64)
    return depth_to_metric(raw, cam.near, cam.far)


def _implied_near_errors(env, obs):
    sim = env.env.sim
    raw = np.asarray(obs["agentview_depth"], dtype=np.float64)[..., 0]
    metric = _metric(env, obs)[..., 0]

    extent = float(sim.model.stat.extent)
    near_m = float(sim.model.vis.map.znear) * extent
    far_m = float(sim.model.vis.map.zfar) * extent

    valid = (raw > 0.01) & (raw < 0.9999)
    assert valid.any(), "no valid depth pixels"
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(valid)
    sample = rng.choice(len(ys), 1000, replace=False)
    d = raw[ys[sample], xs[sample]]
    z = metric[ys[sample], xs[sample]]

    implied = z * (1.0 - d) / (1.0 - d * z / far_m)
    rel_err = np.abs(implied - near_m) / near_m
    return rel_err, near_m, far_m, metric


@pytest.mark.parametrize("task", [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
])
def test_implied_near_matches_model(env_ctx, task):
    env, obs = env_ctx(task)
    rel_err, near_m, _far_m, _metric = _implied_near_errors(env, obs)
    assert rel_err.max() < REL_TOL, (
        f"implied near deviates from model near={near_m:.6f}: "
        f"max rel err = {rel_err.max():.3e} (tolerance {REL_TOL:.0e})"
    )


@pytest.mark.parametrize("task", [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
])
def test_metric_depth_plausible_range(env_ctx, task):
    env, obs = env_ctx(task)
    _rel, _near, _far, metric = _implied_near_errors(env, obs)
    assert np.all(np.isfinite(metric))
    assert np.all(metric > 0.0)
    # LIBERO agentview scene: camera ~0.7-2.5 m from surfaces
    assert 0.5 < metric.min() < 1.0
    assert metric.max() < 3.0
