"""Recovery selectors for phase and feasible errors.

This module replaces the old ``LiberoPoseRecovery`` and
``LiberoFeasibleRecovery`` with shared retrieval helpers and memory_system
recovery result types.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from memory_system.artifacts import (
    FeasibleRecoveryMemory,
    PoseRecoveryMemory,
)
from memory_system.execute.recovery.controller import PoseController
from memory_system.execute.recovery.retrieval import (
    combined_token,
    mean_ee_states,
    retrieve_cluster,
    select_targets,
    token,
)
from memory_system.types import RecoveryResult, RecoveryTarget


class PhaseRecoverySelector:
    """Retrieve a phase-recovery target using the main VAE token."""

    def __init__(
        self,
        recovery_targets: str | Path | PoseRecoveryMemory,
        min_demo_votes: int = 2,
        similarity_threshold: float = 0.2,
        position_radius: float = 0.1,
        rotation_radius: float = 0.5,
        correction_steps: int = 48,
        target_average_count: int = 3,
    ):
        if min_demo_votes <= 0:
            raise ValueError("min_demo_votes must be positive")
        if not np.isfinite(similarity_threshold):
            raise ValueError("similarity_threshold must be finite")
        self.memory = (
            recovery_targets
            if isinstance(recovery_targets, PoseRecoveryMemory)
            else PoseRecoveryMemory(recovery_targets)
        )
        self.min_demo_votes = int(min_demo_votes)
        self.similarity_threshold = float(similarity_threshold)
        self.position_radius = float(position_radius)
        self.rotation_radius = float(rotation_radius)
        self.correction_steps = int(correction_steps)
        self.target_average_count = max(1, int(target_average_count))

    def compute(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        current_vae_main: Any,
        current_ee_states: Any,
    ) -> RecoveryResult | None:
        candidates = select_targets(
            self.memory.targets, task_name, planner_step_id, skill, arguments
        )
        if not candidates:
            return None
        cluster = retrieve_cluster(
            candidates,
            token(current_vae_main),
            lambda item: token(item["recovery_vae_main"]),
            self.similarity_threshold,
            self.position_radius,
            self.rotation_radius,
            self.min_demo_votes,
            self.target_average_count,
        )
        if cluster is None:
            return None
        target_items, sim, frame = cluster
        target = mean_ee_states(target_items)
        controller = PoseController(target_ee_states=target, z_lift=0.0)
        return RecoveryResult(
            target=RecoveryTarget(
                demo_ids=tuple(sorted({str(item["demo_id"]) for item in target_items})),
                target_ee_states=target,
                similarity=sim,
                frame=int(frame),
                z_lift=0.0,
            ),
            correction_steps=self.correction_steps,
            controller=controller,
            correction_per_step=controller.preview_action(current_ee_states).reshape(1, 6),
        )


class FeasibleRecoverySelector:
    """Retrieve a feasible-recovery target using main + wrist VAE tokens."""

    def __init__(
        self,
        feasible_recovery_targets: str | Path | FeasibleRecoveryMemory,
        min_demo_votes: int = 2,
        similarity_threshold: float = 0.2,
        position_radius: float = 0.1,
        rotation_radius: float = 0.5,
        correction_steps: int = 48,
        z_lift: float = 0.02,
        target_average_count: int = 3,
    ):
        if min_demo_votes <= 0:
            raise ValueError("min_demo_votes must be positive")
        if not np.isfinite(similarity_threshold):
            raise ValueError("similarity_threshold must be finite")
        self.memory = (
            feasible_recovery_targets
            if isinstance(feasible_recovery_targets, FeasibleRecoveryMemory)
            else FeasibleRecoveryMemory(feasible_recovery_targets)
        )
        self.min_demo_votes = int(min_demo_votes)
        self.similarity_threshold = float(similarity_threshold)
        self.position_radius = float(position_radius)
        self.rotation_radius = float(rotation_radius)
        self.correction_steps = int(correction_steps)
        self.z_lift = float(z_lift)
        self.target_average_count = max(1, int(target_average_count))

    def compute(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        current_vae_main: Any,
        current_vae_wrist: Any,
        current_ee_states: Any,
    ) -> RecoveryResult | None:
        candidates = select_targets(
            self.memory.targets, task_name, planner_step_id, skill, arguments
        )
        candidates = [item for item in candidates if item.get("ready_vae_main") is not None]
        if not candidates:
            return None
        cluster = retrieve_cluster(
            candidates,
            combined_token(current_vae_main, current_vae_wrist),
            lambda item: combined_token(item.get("ready_vae_main"), item.get("ready_vae_wrist")),
            self.similarity_threshold,
            self.position_radius,
            self.rotation_radius,
            self.min_demo_votes,
            self.target_average_count,
        )
        if cluster is None:
            return None
        target_items, sim, frame = cluster
        target = mean_ee_states(target_items)
        controller = PoseController(target_ee_states=target, z_lift=self.z_lift)
        return RecoveryResult(
            target=RecoveryTarget(
                demo_ids=tuple(sorted({str(item["demo_id"]) for item in target_items})),
                target_ee_states=target,
                similarity=sim,
                frame=int(frame),
                z_lift=self.z_lift,
            ),
            correction_steps=max(int(self.correction_steps), 2),
            controller=controller,
            correction_per_step=controller.preview_action(current_ee_states).reshape(1, 6),
        )

    def compute_nearest_ready(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        current_ee_states: Any,
    ) -> RecoveryResult | None:
        candidates = [
            item for item in select_targets(
                self.memory.targets, task_name, planner_step_id, skill, arguments
            )
            if item.get("ee_states") is not None
        ]
        if not candidates:
            return None
        current = np.asarray(current_ee_states, dtype=np.float32).reshape(6)

        def ee_distance(item: dict[str, Any]) -> float:
            ee = np.asarray(item["ee_states"], dtype=np.float32).reshape(6)
            return float(np.linalg.norm(ee[:3] - current[:3])) + 0.2 * float(
                np.linalg.norm(ee[3:] - current[3:])
            )

        best = min(candidates, key=ee_distance)
        target = np.asarray(best["ee_states"], dtype=np.float32).reshape(6)
        controller = PoseController(target_ee_states=target, z_lift=self.z_lift)
        return RecoveryResult(
            target=RecoveryTarget(
                demo_ids=(str(best["demo_id"]),),
                target_ee_states=target,
                similarity=0.0,
                frame=int(best.get("ready_frame", 0)),
                z_lift=self.z_lift,
            ),
            correction_steps=max(int(self.correction_steps), 2),
            controller=controller,
            correction_per_step=controller.preview_action(current_ee_states).reshape(1, 6),
        )
