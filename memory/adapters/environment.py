"""Environment creation, case initialisation, target identity and geometry.

One adapter serves one process, and therefore one flavor: whichever libero
package PYTHONPATH resolves plus LIBERO_CONFIG_PATH decide which task tree a
suite name means, and the plus and PRO trees must not share a process.

``make`` builds the environment for one case and returns it with the task's
real BDDL instruction (never the benchmark's name-derived one); ``target``
resolves a phase's item or destination to a name the simulator knows;
``geometry`` returns that target's world pose and its visible point cloud.

Nothing here judges, schedules, retrieves, executes or records.  Moving to a
different robot means replacing this module.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from memory.types import CaseSpec, PhaseSpec

DEFAULT_RESOLUTION = 256
DEFAULT_MODEL_FAMILY = "cosmos"


class EnvironmentAdapter:
    """Create cases and resolve what the memory machinery needs about them."""

    def __init__(
        self,
        *,
        resolution: int = DEFAULT_RESOLUTION,
        camera_depths: list[bool] | None = None,
        model_family: str = DEFAULT_MODEL_FAMILY,
        suite_kwargs: dict[str, Any] | None = None,
        env_factory: Any | None = None,
    ) -> None:
        self.resolution = int(resolution)
        self.camera_depths = camera_depths
        self.model_family = str(model_family)
        # Passed to the suite constructor; the entry point decides them (the
        # plus flavors take a perturbation category, the PRO suites take none).
        self.suite_kwargs = dict(suite_kwargs or {})
        # Injected so a caller that monkeypatches its module-level env factory
        # (the smoke harness does) keeps that patch in effect.
        self.env_factory = env_factory
        self._env: Any | None = None
        self._suites: dict[str, Any] = {}

    def make(self, case: CaseSpec) -> tuple[Any, str]:
        """Create the environment at the case's initial state; return it and
        the task's real instruction."""
        get_libero_env = self.env_factory
        if get_libero_env is None:
            from cosmos_policy.experiments.robot.libero.libero_utils import (
                get_libero_env as _default_env_factory,
            )

            get_libero_env = _default_env_factory

        suite = self._suite(case.suite)
        task_id = _task_id(suite, case.task)
        task = suite.get_task(task_id)
        env, _ = get_libero_env(
            task,
            self.model_family,
            resolution=self.resolution,
            camera_depths=self.camera_depths,
        )
        env.reset()
        env.set_init_state(suite.get_task_init_states(task_id)[case.init_index])
        self._env = env
        return env, _bddl_language(task)

    def target(self, phase: PhaseSpec) -> str:
        """The phase's item/destination as a name the simulator knows.

        Segmentation instances come first when the environment has them; with
        them absent (the eval's unsegmented render env) the BDDL argument is
        resolved to its anchor body instead, which is what the geometry side
        needs.  An argument that resolves to neither is passed through.
        """
        from memory_system.execute.skill_completion.shadow import resolve_target_instance

        argument = str(phase.arguments.get("item" if phase.skill == "Pick" else "target", ""))
        if not argument:
            raise KeyError(f"no reference argument for {phase.skill} {dict(phase.arguments)}")
        instance = resolve_target_instance(self._env, phase.arguments, phase.skill)
        if instance is not None:
            return instance
        return self._anchor(argument)

    def geometry(
        self, observation: Any, target: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """The target's world frame and its visible point cloud."""
        from memory_system.pointcloud_action.offline.extraction import (
            object_frame,
            visible_point_cloud,
        )

        T_world_target, _, _ = object_frame(self._env, target)
        points = visible_point_cloud(self._env, observation, target, self.resolution)
        return T_world_target, points

    # ---- internals

    def _suite(self, name: str) -> Any:
        # One suite object per name, as the eval loop keeps one for the run.
        if name not in self._suites:
            from libero.libero import benchmark

            suites = benchmark.get_benchmark_dict()
            if name not in suites:
                raise KeyError(f"unknown suite {name!r}; have {sorted(suites)}")
            self._suites[name] = suites[name](**self.suite_kwargs)
        return self._suites[name]

    def _anchor(self, argument: str) -> str:
        """Region name -> anchor body, matching the memory build's rule."""
        from memory_system.offline.build_targets import match_region_anchor

        sim = self._env.env.sim
        body_names: dict[str, None] = {}
        for index in range(sim.model.nbody):
            name = sim.model.body_id2name(index)
            if not name:
                continue
            body_names[name] = None
            if name.endswith("_main"):
                # Root body names carry the suffix; the instance-level keys the
                # memory build used are the stripped ones.
                body_names[name[: -len("_main")]] = None
        if argument in body_names:
            return argument
        anchor = match_region_anchor(argument, body_names)
        return argument if anchor is None else anchor


def _task_id(suite: Any, name: str) -> int:
    """Index of ``name`` in ``suite``; the name must match exactly once."""
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == str(name)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one exact task match for {name!r}, found {len(matches)}"
        )
    return matches[0]


def _bddl_path(task: Any) -> Path:
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _bddl_language(task: Any) -> str:
    """The BDDL's own (:language ...); falls back to the benchmark's."""
    try:
        match = re.search(r"\(:language\s+(.+?)\)", _bddl_path(task).read_text())
        if match:
            return match.group(1).strip()
    except Exception:  # noqa: BLE001 - fall back rather than block evaluation
        pass
    return task.language
