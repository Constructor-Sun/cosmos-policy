"""Shared fixtures for Stage-1 depth tests.

IMPORTANT (empirically verified): keeping two LIBERO offscreen envs alive in
one process corrupts the first env's rendering (EGL context bleed) -- the
first env's depth renders return wrong values once a second env is created.
Therefore these fixtures create exactly ONE env at a time: a fresh env per
test (function scope), closed at test teardown.  Sequential pytest execution
guarantees only one env exists at any moment.
"""
import pytest

from harness import TASKS, create_env


@pytest.fixture()
def env_ctx():
    """Create one env per call; close all created envs at test teardown.

    Only one env is alive at a time across the whole session because each
    test creates (and the fixture closes) its own env.
    """
    envs = []

    def _make(task: str):
        env = create_env(task)
        obs = env.reset()
        envs.append(env)
        return env, obs

    yield _make
    for env in envs:
        env.close()


@pytest.fixture()
def tasks():
    return TASKS
