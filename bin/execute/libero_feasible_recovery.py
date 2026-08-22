#!/usr/bin/env python3
"""Retrieval-based recovery for LIBERO FEASIBLE_ERROR.

When the feasible-region verifier reports that the robot is persistently
stagnating or reversing before entering the ready region, retrieve the most
similar successful demo ready-frame state using main + wrist VAE tokens, then
drive the end effector to that demo's ready-frame 6D pose with a step-level
closed-loop controller (see closed_loop.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - scipy is expected in the eval env
    Rotation = None

from execute.closed_loop import ClosedLoopPoseController


# Fixed global action calibration (kept for reference and the unused
# _compute_global_action_scale fallback; the closed-loop controller uses its
# own fixed scales, so this no longer affects the correction trajectory).
POS_ACTION_SCALE = np.array([0.011, 0.012, 0.013], dtype=np.float32)
ROT_ACTION_SCALE = np.array([0.12, 0.10, 0.12], dtype=np.float32)
ACTION_SCALE = np.concatenate([POS_ACTION_SCALE, ROT_ACTION_SCALE]).astype(np.float32)


@dataclass(frozen=True)
class FeasibleRecoveryResult:
    demo_ids: tuple[str, ...]
    target_ee_states: np.ndarray
    similarity: float
    ready_frame: int
    correction_steps: int
    controller: ClosedLoopPoseController
    # First-step action preview (kept for the eval log line); the actual
    # trajectory is generated online by the closed-loop controller.
    correction_per_step: np.ndarray
    offset_gripper_action: float | None = None


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


class LiberoFeasibleRecovery:
    """Match current main+wrist tokens to successful demo ready-frame states."""

    def __init__(
        self,
        feasible_recovery_targets: str | Path,
        min_demo_votes: int = 2,
        similarity_threshold: float = 0.2,
        position_radius: float = 0.1,
        rotation_radius: float = 0.5,
        correction_steps: int = 48,
        z_lift: float = 0.02,
        lift_steps: int = 4,  # kept for backward compatibility; the closed-loop controller handles the lift itself
        target_average_count: int = 3,
    ):
        if min_demo_votes <= 0:
            raise ValueError("min_demo_votes must be positive")
        if not np.isfinite(similarity_threshold):
            raise ValueError("similarity_threshold must be finite")
        payload = torch.load(
            Path(feasible_recovery_targets), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "libero_feasible_recovery_targets_v1":
            raise ValueError(
                f"Unsupported feasible recovery target format: {payload.get('format')!r}"
            )
        self.targets = list(payload.get("targets", []))
        self.min_demo_votes = int(min_demo_votes)
        self.similarity_threshold = float(similarity_threshold)
        self.position_radius = float(position_radius)
        self.rotation_radius = float(rotation_radius)
        self.correction_steps = int(correction_steps)
        self.z_lift = float(z_lift)
        self.lift_steps = int(lift_steps)
        self.target_average_count = max(1, int(target_average_count))
        self.action_scale = self._compute_global_action_scale(self.targets)

    @staticmethod
    def _compute_global_action_scale(targets: list[dict[str, Any]]) -> np.ndarray:
        """Aggregate per-demo action_scale into one fixed global scale."""
        scales = []
        for item in targets:
            s = np.asarray(item.get("action_scale"), dtype=np.float64)
            if s.shape != (6,):
                continue
            if not np.isfinite(s).all():
                continue
            # Keep only positive, physically plausible scales. Negative values
            # here are usually numerical artifacts from near-zero net motion and
            # would flip the correction direction if used directly.
            pos_ok = (s[:3] > 0.001) & (s[:3] < 0.1)
            rot_ok = (s[3:] > 0.01) & (s[3:] < 1.0)
            if pos_ok.all() and rot_ok.all():
                scales.append(s)
        if len(scales) >= 10:
            return np.median(np.asarray(scales), axis=0).astype(np.float32)
        return ACTION_SCALE.copy()

    def _select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expected = _arguments_key(arguments)
        exact = [
            item for item in self.targets
            if item["task_name"] == task_name
            and int(item["planner_step_id"]) == int(planner_step_id)
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected
        ]
        if exact:
            return exact
        return [
            item for item in self.targets
            if item["task_name"] == task_name
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected
        ]

    @staticmethod
    def _token(vae: Any) -> torch.Tensor:
        tensor = torch.as_tensor(vae)
        if tensor.dim() == 4 and tensor.shape[1] == 2:
            tensor = tensor[:, 1:2, :, :]
        flat = tensor.float().reshape(1, -1)
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return flat / norm

    @classmethod
    def _combined_token(
        cls, vae_main: Any, vae_wrist: Any | None
    ) -> torch.Tensor:
        main = cls._token(vae_main)
        if vae_wrist is None:
            return main
        wrist = cls._token(vae_wrist)
        combined = torch.cat([main, wrist], dim=1)
        return combined / combined.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left @ right.T).reshape(-1)[0])

    @staticmethod
    def _close(left: np.ndarray, right: np.ndarray,
               position_radius: float, rotation_radius: float) -> bool:
        left = np.asarray(left, dtype=np.float32).reshape(6)
        right = np.asarray(right, dtype=np.float32).reshape(6)
        if float(np.linalg.norm(left[:3] - right[:3])) > position_radius:
            return False
        rot_dist = float(np.linalg.norm(left[3:] - right[3:]))
        return rot_dist <= rotation_radius

    @staticmethod
    def _mean_ee_states(items: list[dict[str, Any]]) -> np.ndarray:
        """Average absolute target poses, not per-step deltas.

        Positions are averaged in Euclidean space. Rotations are averaged in
        quaternion space to avoid the component-wise rotvec averaging problem.
        """
        positions = np.mean(
            [np.asarray(item["ee_states"], dtype=np.float64)[:3] for item in items],
            axis=0,
        )
        if Rotation is None:
            rotvecs = np.mean(
                [np.asarray(item["ee_states"], dtype=np.float64)[3:] for item in items],
                axis=0,
            )
            return np.concatenate([positions, rotvecs]).astype(np.float32)

        quats = []
        for item in items:
            q = Rotation.from_rotvec(
                np.asarray(item["ee_states"], dtype=np.float64)[3:]
            ).as_quat()
            if quats and np.dot(quats[0], q) < 0:
                q = -q
            quats.append(q)
        avg_q = np.mean(np.asarray(quats, dtype=np.float64), axis=0)
        avg_q = avg_q / np.linalg.norm(avg_q)
        avg_rotvec = Rotation.from_quat(avg_q).as_rotvec()
        return np.concatenate([positions, avg_rotvec]).astype(np.float32)

    def _delta(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=np.float32).reshape(6)
        target = np.asarray(target, dtype=np.float32).reshape(6)
        delta = np.zeros(6, dtype=np.float32)
        delta[:3] = target[:3] - current[:3]
        if Rotation is not None:
            rot = (
                Rotation.from_rotvec(target[3:])
                * Rotation.from_rotvec(current[3:]).inv()
            ).as_rotvec()
            delta[3:] = np.asarray(rot, dtype=np.float32)
        else:
            delta[3:] = target[3:] - current[3:]
        return delta

    def compute_nearest_ready(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        current_ee_states: Any,
    ) -> FeasibleRecoveryResult | None:
        """Fallback: use the nearest ready-frame EE pose, ignoring VAE similarity."""
        candidates = [
            item for item in self._select(task_name, planner_step_id, skill, arguments)
            if item.get("ee_states") is not None
        ]
        if not candidates:
            return None
        current = np.asarray(current_ee_states, dtype=np.float32).reshape(6)
        def ee_distance(item):
            ee = np.asarray(item["ee_states"], dtype=np.float32).reshape(6)
            return float(np.linalg.norm(ee[:3] - current[:3])) + 0.2 * float(
                np.linalg.norm(ee[3:] - current[3:])
            )
        best = min(candidates, key=ee_distance)
        target = np.asarray(best["ee_states"], dtype=np.float32).reshape(6)
        controller = ClosedLoopPoseController(
            target_ee_states=target,
            z_lift=self.z_lift,
        )
        return FeasibleRecoveryResult(
            demo_ids=(str(best["demo_id"]),),
            target_ee_states=target,
            similarity=0.0,
            ready_frame=int(best["ready_frame"]),
            correction_steps=max(int(self.correction_steps), 2),
            controller=controller,
            correction_per_step=controller.preview_action(
                current_ee_states
            ).reshape(1, 6),
        )

    def compute(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        current_vae_main: Any,
        current_vae_wrist: Any,
        current_ee_states: Any,
    ) -> FeasibleRecoveryResult | None:
        candidates = self._select(task_name, planner_step_id, skill, arguments)
        if not candidates:
            return None

        current = self._combined_token(current_vae_main, current_vae_wrist)
        best_by_demo: dict[str, tuple[float, dict[str, Any]]] = {}
        for item in candidates:
            item_main = item.get("ready_vae_main")
            item_wrist = item.get("ready_vae_wrist")
            if item_main is None:
                continue
            sim = self._similarity(
                current, self._combined_token(item_main, item_wrist)
            )
            demo = str(item["demo_id"])
            if demo not in best_by_demo or sim > best_by_demo[demo][0]:
                best_by_demo[demo] = (sim, item)

        ranked = sorted(best_by_demo.values(), key=lambda pair: pair[0], reverse=True)
        if len(ranked) < self.min_demo_votes:
            return None
        if ranked[0][0] < self.similarity_threshold:
            return None

        best_item = ranked[0][1]
        cluster = [
            item for sim, item in ranked
            if sim >= self.similarity_threshold
            and self._close(
                item["ee_states"], best_item["ee_states"],
                self.position_radius, self.rotation_radius,
            )
        ]
        if len(cluster) < self.min_demo_votes:
            return None

        # Do NOT replay the demo's raw action chunk here. The raw chunk is only
        # valid from that demo's own starting pose; it does not know where the
        # current end-effector is. Instead, average the first few absolute
        # target poses (Do NOT average per-dimension deltas) and drive the EE
        # there with a step-level closed-loop controller, lifting first to
        # avoid collision. The fixed action scale used by the controller is
        # approximate by design: it only affects convergence speed, not whether
        # the target is reached.
        target_items = cluster[: self.target_average_count]
        target = self._mean_ee_states(target_items)
        controller = ClosedLoopPoseController(
            target_ee_states=target,
            z_lift=self.z_lift,  # lift first (only z), to avoid collision/stall
        )
        return FeasibleRecoveryResult(
            demo_ids=tuple(sorted({str(item["demo_id"]) for item in target_items})),
            target_ee_states=target,
            similarity=float(ranked[0][0]),
            ready_frame=int(best_item["ready_frame"]),
            correction_steps=max(int(self.correction_steps), 2),
            controller=controller,
            correction_per_step=controller.preview_action(
                current_ee_states
            ).reshape(1, 6),
        )