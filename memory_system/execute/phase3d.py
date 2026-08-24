"""Formal Phase 3D verifier for the memory system.

This module implements the RGB-D Phase judgment that was validated offline in
``tests/phase3/``.  It reuses the existing template matcher and Feasible
verifier while adding a metric 3D distance trend.

Close is disabled by default: Phase 3D correction is not applied to Close.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from memory_system.execute.feasible import FEASIBLE, FeasibleVerifier
from memory_system.execute.phase import (
    PHASE_ERROR,
    PHASE_OK,
    PHASE_UNKNOWN,
    PhaseVerifier,
)
from memory_system.geometry import pixel_to_world
from memory_system.types import PhaseResult, VerifierObservation

PHASE_DONE = "PHASE_DONE"


class Phase3DVerifier:
    """Phase verifier that tracks a metric 3D distance to the target."""

    def __init__(
        self,
        phase_targets: str | Path,
        segments_manifest: str | Path,
        scales: tuple[float, ...] = (0.8, 1.0, 1.2),
        min_similarity: float = 0.45,
        min_demo_votes: int = 2,
        cluster_radius_px: float = 32.0,
        ambiguity_ratio: float = 0.9,
        progress_tolerance_m: float = 0.001,
        evidence_updates: int = 2,
        disabled_skills: tuple[str, ...] = ("Close",),
    ):
        self.matcher = PhaseVerifier(
            phase_targets,
            scales=scales,
            min_similarity=min_similarity,
            min_demo_votes=min_demo_votes,
            cluster_radius_px=cluster_radius_px,
            ambiguity_ratio=ambiguity_ratio,
        )
        self.feasible = FeasibleVerifier(phase_targets, segments_manifest)
        self.progress_tolerance_m = float(progress_tolerance_m)
        self.evidence_updates = int(evidence_updates)
        self.disabled_skills = set(disabled_skills)
        self.task_name: str | None = None
        self.planner_step_id: int | None = None
        self.skill = ""
        self.arguments: dict[str, str] = {}
        self.previous_distance: float | None = None
        self.away_count = 0
        self.stall_count = 0
        self.abnormal_count = 0

    def reset(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        self.task_name = str(task_name)
        self.planner_step_id = int(planner_step_id)
        self.skill = str(skill)
        self.arguments = {str(k): str(v) for k, v in arguments.items()}
        self.matcher.reset(
            self.task_name,
            self.planner_step_id,
            self.skill,
            self.arguments,
            exclude_demo_ids,
        )
        self.feasible.reset(
            self.task_name,
            self.planner_step_id,
            self.skill,
            self.arguments,
            exclude_demo_ids,
        )
        self.previous_distance = None
        self.away_count = 0
        self.stall_count = 0
        self.abnormal_count = 0

    def _target_xyz(self, mask: np.ndarray, observation: VerifierObservation):
        if observation.main_depth is None or observation.camera_params is None:
            return None
        ys, xs = np.nonzero(mask)
        if len(ys) < 4:
            return None
        depth = np.asarray(observation.main_depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        try:
            pts = pixel_to_world(
                np.stack([ys, xs], axis=-1), depth, observation.camera_params
            )
        except Exception:
            return None
        valid = np.isfinite(pts).all(axis=1)
        if int(valid.sum()) < 4:
            return None
        return np.median(pts[valid], axis=0)

    def _trend_error(self, current_distance: float) -> str | None:
        if self.previous_distance is None:
            return None
        progress = self.previous_distance - current_distance
        tol = self.progress_tolerance_m
        if progress > tol:
            # Normal approach: reset all anomaly evidence.
            self.away_count = 0
            self.stall_count = 0
            self.abnormal_count = 0
        elif progress < -tol:
            self.away_count += 1
            self.stall_count = 0
            self.abnormal_count += 1
            if self.abnormal_count >= self.evidence_updates:
                return "away"
        else:
            self.stall_count += 1
            self.away_count = 0
            self.abnormal_count += 1
            if self.abnormal_count >= self.evidence_updates:
                return "stall"
        return None

    def update(self, observation: VerifierObservation) -> PhaseResult:
        if not self.matcher.templates:
            return PhaseResult(
                PHASE_UNKNOWN, 0.0, None, None, 0, None, {"reason": "no_templates"}
            )
        match = self.matcher.match_current(observation.third_view_rgb)
        if match is None:
            return PhaseResult(
                PHASE_UNKNOWN,
                0.0,
                None,
                None,
                0,
                None,
                {"reason": "no_visual_consensus"},
            )

        bbox = match["bbox_xyxy"]
        target_xy = match["target_xy"]
        template_demo_id = match["template"]["demo_id"]
        support = match["support"]
        confidence = match["confidence"]

        # Close is intentionally excluded from Phase 3D correction.
        if self.skill in self.disabled_skills:
            return PhaseResult(
                PHASE_DONE,
                confidence,
                tuple(float(x) for x in target_xy),
                None,
                support,
                template_demo_id,
                {"matched_bbox_xyxy": bbox, "disabled_skill": True},
            )

        target_xyz = self._target_xyz(match["mask"], observation)
        eef = getattr(observation, "eef_pos", None)
        if target_xyz is None or eef is None:
            return PhaseResult(
                PHASE_UNKNOWN,
                confidence,
                tuple(float(x) for x in target_xy),
                None,
                support,
                template_demo_id,
                {"matched_bbox_xyxy": bbox, "reason": "rgbd_geometry_unavailable"},
            )

        current_distance = float(
            np.linalg.norm(np.asarray(eef, dtype=np.float64) - target_xyz)
        )
        error = self._trend_error(current_distance)
        self.previous_distance = current_distance
        if error is not None:
            return PhaseResult(
                PHASE_ERROR,
                confidence,
                tuple(float(x) for x in target_xy),
                None,
                support,
                template_demo_id,
                {
                    "matched_bbox_xyxy": bbox,
                    "error_type": error,
                    "distance_m": current_distance,
                    "away_count": self.away_count,
                    "stall_count": self.stall_count,
                },
            )

        feasible_result = self.feasible.update(
            observation,
            target_xy,
            bbox,
            confidence=confidence,
        )
        if feasible_result.status == FEASIBLE:
            return PhaseResult(
                PHASE_DONE,
                confidence,
                tuple(float(x) for x in target_xy),
                None,
                support,
                template_demo_id,
                {
                    "matched_bbox_xyxy": bbox,
                    "distance_m": current_distance,
                    "feasible_reason": feasible_result.reason,
                },
            )

        return PhaseResult(
            PHASE_OK,
            confidence,
            tuple(float(x) for x in target_xy),
            None,
            support,
            template_demo_id,
            {
                "matched_bbox_xyxy": bbox,
                "distance_m": current_distance,
                "away_count": self.away_count,
                "stall_count": self.stall_count,
            },
        )
