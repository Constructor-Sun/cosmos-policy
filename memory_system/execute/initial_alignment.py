"""One-shot initial alignment selector for LIBERO episodes.

This module implements the standalone online intervention path.  It only:

* loads phase plans and feasible/ready-pose memory,
* maps a LIBERO-plus perturbed task back to its base task,
* selects the most visually similar first-phase ready pose using the main
  camera VAE only,
* returns one alignment trajectory/controller for ``run_episode`` to execute.

No skill verifier state is kept and no online recovery is triggered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from memory_system.artifacts import FeasibleRecoveryMemory
from memory_system.execute.plan import PhaseSpec, load_phase_plans
from memory_system.execute.recovery.controller import PoseController
from memory_system.execute.recovery.retrieval import similarity, token
from memory_system.types import RecoveryTarget


@dataclass(frozen=True)
class InitialAlignmentResult:
    """Result of a one-shot Initial Alignment target selection."""

    target: RecoveryTarget
    correction_steps: int
    controller: Any = None
    joint_trajectory: Any = None

    @property
    def demo_ids(self) -> tuple[str, ...]:
        return self.target.demo_ids

    @property
    def target_ee_states(self) -> np.ndarray:
        return self.target.target_ee_states

    @property
    def similarity(self) -> float:
        return self.target.similarity

    @property
    def frame(self) -> int:
        return self.target.frame

    @property
    def z_lift(self) -> float:
        return self.target.z_lift


class InitialAlignmentSelector:
    """Select a single first-phase ready pose using main-camera VAE similarity.

    The selector loads the skill plan and ready-pose memory itself.  By default
    the selected target is raised 2 cm in z so the robot moves to the ready
    x/y pose slightly above the final height.
    """

    def __init__(
        self,
        segments_manifest: str | Path,
        feasible_recovery_targets: str | Path,
        correction_steps: int = 48,
        z_offset: float = 0.02,
        planner: Any | None = None,
    ):
        self.plans = load_phase_plans(segments_manifest)
        self.memory = FeasibleRecoveryMemory(feasible_recovery_targets)
        self.correction_steps = max(2, int(correction_steps))
        # The initial alignment moves to the ready pose but keeps the end
        # effector 2 cm above the final height by default.
        self.z_offset = float(z_offset)
        self.planner = planner

        # ``load_phase_plans`` sorts by ``planner_step_id``, which is not always
        # the same as demonstration execution order (e.g. KITCHEN_SCENE8 picks
        # moka_pot_2 before moka_pot_1 even though its step ids are 3/4 then 1/2).
        # For initial alignment we need the actual first phase in the demos.
        self.first_phases = self._build_first_phases(segments_manifest)

    def _build_first_phases(self, segments_manifest: str | Path) -> dict[str, PhaseSpec]:
        """Return the most common first phase per task in demo order."""
        payload = json.loads(Path(segments_manifest).read_text())
        votes: dict[str, dict[tuple, int]] = {}
        for record in payload.get("records", []):
            if not record.get("valid"):
                continue
            task_name = str(record["task_name"])
            segments = [
                segment
                for segment in record.get("segments", [])
                if segment.get("status") != "already_satisfied"
            ]
            if not segments:
                continue
            first = segments[0]
            args = tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in first.get("arguments", {}).items()
                )
            )
            key = (
                int(first["planner_step_id"]),
                str(first["skill"]),
                args,
            )
            task_votes = votes.setdefault(task_name, {})
            task_votes[key] = task_votes.get(key, 0) + 1

        first_phases: dict[str, PhaseSpec] = {}
        for task_name, task_votes in votes.items():
            if not task_votes:
                continue
            (step_id, skill, args), _ = max(
                task_votes.items(), key=lambda item: item[1]
            )
            first_phases[task_name] = PhaseSpec(
                step_id,
                skill,
                {str(key): str(value) for key, value in args},
            )
        return first_phases

    def resolve_task_name(self, task_name: str) -> str | None:
        """Map a perturbed task name back to a base phase-plan task."""
        task_name = str(task_name)
        if task_name in self.plans:
            return task_name
        matches = [
            base_task
            for base_task in self.plans
            if task_name.startswith(f"{base_task}_")
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def select(
        self,
        task_name: str,
        current_vae_main: Any,
        current_ee_states: Any,
        main_depth: Any = None,
        camera_params: Any = None,
        joint_positions: Any = None,
        gripper_joint_positions: Any = None,
        robot_base_pose: Any = None,
    ) -> InitialAlignmentResult | None:
        """Return the best-matching first-phase ready pose, or ``None``.

        The match is based only on main-camera VAE cosine similarity.  There is
        intentionally no similarity threshold: if any candidate exists, the best
        one is returned.
        """
        base_task = self.resolve_task_name(task_name)
        if base_task is None:
            return None
        if base_task not in self.plans or not self.plans[base_task]:
            return None

        first_phase = self.first_phases.get(base_task)
        if first_phase is None:
            if base_task not in self.plans or not self.plans[base_task]:
                return None
            first_phase = self.plans[base_task][0]
        candidates = [
            item
            for item in self.memory.select(
                base_task,
                first_phase.planner_step_id,
                first_phase.skill,
                first_phase.arguments,
            )
            if item.get("ready_vae_main") is not None
            and item.get("ee_states") is not None
        ]
        if not candidates:
            return None

        current_token = token(current_vae_main)
        best_item = None
        best_sim = -float("inf")
        for item in candidates:
            try:
                item_token = token(item["ready_vae_main"])
            except Exception:
                continue
            sim = similarity(current_token, item_token)
            if sim > best_sim:
                best_sim = sim
                best_item = item

        if best_item is None:
            return None

        # Copy before modifying: the memory items are reused across episodes and
        # must not be mutated in-place.
        target_ee = np.array(best_item["ee_states"], dtype=np.float32).reshape(6)
        # Keep the final target 2 cm above the stored ready pose.
        target_ee[2] += self.z_offset
        if self.planner is not None:
            try:
                plan = self.planner.plan(
                    current_ee_states=current_ee_states,
                    target_ee_states=target_ee,
                    depth=main_depth,
                    camera_params=camera_params,
                    joint_positions=joint_positions,
                    gripper_joint_positions=gripper_joint_positions,
                    robot_base_pose=robot_base_pose,
                )
                if plan is not None:
                    planned_target = (
                        plan.target_ee_states
                        if plan.target_ee_states is not None
                        else target_ee
                    )
                    return InitialAlignmentResult(
                        target=RecoveryTarget(
                            demo_ids=(str(best_item["demo_id"]),),
                            target_ee_states=planned_target,
                            similarity=float(best_sim),
                            frame=int(best_item.get("ready_frame", 0)),
                            z_lift=0.0,
                        ),
                        correction_steps=plan.correction_steps,
                        controller=plan.controller,
                        joint_trajectory=plan.joint_trajectory,
                    )
            except Exception:
                # The strict joint-execution path is rejected immediately below.
                pass
            if getattr(self.planner, "joint_execution", False):
                return None
        controller = PoseController(target_ee_states=target_ee, z_lift=0.0)
        return InitialAlignmentResult(
            target=RecoveryTarget(
                demo_ids=(str(best_item["demo_id"]),),
                target_ee_states=target_ee,
                similarity=float(best_sim),
                frame=int(best_item.get("ready_frame", 0)),
                z_lift=0.0,
            ),
            correction_steps=self.correction_steps,
            controller=controller,
        )
