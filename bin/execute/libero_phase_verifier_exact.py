#!/usr/bin/env python3
"""Training-free LIBERO phase verifier based on target templates and 2-D motion."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


PHASE_OK = "PHASE_OK"
PHASE_ERROR = "PHASE_ERROR"
PHASE_UNKNOWN = "PHASE_UNKNOWN"


@dataclass(frozen=True)
class PhaseResult:
    status: str
    confidence: float
    target_xy: tuple[float, float] | None
    progress_px: float | None
    match_count: int
    template_demo_id: str | None
    details: dict[str, Any] = field(default_factory=dict)


def _arguments_key(arguments: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class PhaseTargetMemory:
    """Read-only index over the sidecar produced by build_libero_phase_targets.py."""

    def __init__(self, path: str | Path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != "libero_phase_targets_v1":
            raise ValueError(f"Unsupported phase target format: {payload.get('format')!r}")
        self.path = Path(path)
        self.templates = list(payload.get("templates", []))

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        exclude_demo_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        excluded = set(exclude_demo_ids)
        expected_args = _arguments_key(arguments)
        exact = [
            item
            for item in self.templates
            if item["task_name"] == task_name
            and int(item["planner_step_id"]) == int(planner_step_id)
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected_args
            and item["demo_id"] not in excluded
        ]
        if exact:
            return exact
        return [
            item
            for item in self.templates
            if item["task_name"] == task_name
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected_args
            and item["demo_id"] not in excluded
        ]


class LiberoPhaseVerifier:
    """Bind a target from third-view RGB and reject sustained motion away from it."""

    def __init__(
        self,
        memory: PhaseTargetMemory | str | Path,
        ratio_test: float = 0.75,
        ransac_px: float = 5.0,
        min_inliers: int = 4,
        ambiguity_ratio: float = 0.85,
        cluster_radius_px: float = 24.0,
        wrong_way_tolerance_px: float = 2.0,
        wrong_way_updates: int = 2,
    ):
        self.memory = memory if isinstance(memory, PhaseTargetMemory) else PhaseTargetMemory(memory)
        self.ratio_test = ratio_test
        self.ransac_px = ransac_px
        self.min_inliers = min_inliers
        self.ambiguity_ratio = ambiguity_ratio
        self.cluster_radius_px = cluster_radius_px
        self.wrong_way_tolerance_px = wrong_way_tolerance_px
        self.wrong_way_updates = wrong_way_updates
        self.sift = cv2.SIFT_create(nfeatures=1024)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.templates: list[dict[str, Any]] = []
        self.previous_gripper: np.ndarray | None = None
        self.wrong_way_count = 0

    def reset(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        self.templates = self.memory.select(
            task_name, planner_step_id, skill, arguments, exclude_demo_ids
        )
        self.previous_gripper = None
        self.wrong_way_count = 0

    def _match_template(
        self,
        template: dict[str, Any],
        current_keypoints: list[cv2.KeyPoint],
        current_descriptors: np.ndarray,
    ) -> tuple[np.ndarray, float, int] | None:
        descriptors = template.get("descriptors")
        points = template.get("keypoints_xy")
        if descriptors is None or points is None:
            return None
        descriptors = _as_numpy(descriptors, np.float32)
        points = _as_numpy(points, np.float32)
        if len(descriptors) < 2 or len(current_descriptors) < 2:
            return None
        pairs = self.matcher.knnMatch(descriptors, current_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < self.ratio_test * second.distance]
        if len(good) < self.min_inliers:
            return None
        source = np.float32([points[item.queryIdx] for item in good]).reshape(-1, 1, 2)
        target = np.float32([current_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        transform, mask = cv2.findHomography(source, target, cv2.RANSAC, self.ransac_px)
        if transform is None or mask is None:
            return None
        inliers = int(mask.ravel().sum())
        if inliers < self.min_inliers:
            return None
        center = _as_numpy(template["crop_center_xy"], np.float32).reshape(1, 1, 2)
        projected = cv2.perspectiveTransform(center, transform).reshape(2)
        if not np.isfinite(projected).all():
            return None
        score = inliers / max(float(min(len(descriptors), 64)), 1.0)
        return projected, score, inliers

    def _cluster_matches(
        self, matches: list[tuple[np.ndarray, float, int, dict[str, Any]]]
    ) -> list[tuple[np.ndarray, float, int, dict[str, Any]]]:
        clusters: list[list[tuple[np.ndarray, float, int, dict[str, Any]]]] = []
        for match in sorted(matches, key=lambda item: item[1], reverse=True):
            for cluster in clusters:
                center = np.average(
                    [item[0] for item in cluster], weights=[item[1] for item in cluster], axis=0
                )
                if np.linalg.norm(match[0] - center) <= self.cluster_radius_px:
                    cluster.append(match)
                    break
            else:
                clusters.append([match])
        ranked = []
        for cluster in clusters:
            weights = np.asarray([item[1] for item in cluster], dtype=np.float32)
            center = np.average([item[0] for item in cluster], weights=weights, axis=0)
            score = float(weights.sum())
            best = max(cluster, key=lambda item: item[1])
            ranked.append((center, score, sum(item[2] for item in cluster), best[3]))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def update(
        self,
        third_view_rgb: np.ndarray,
        gripper_xy: tuple[float, float] | np.ndarray | None = None,
    ) -> PhaseResult:
        if not self.templates:
            return PhaseResult(PHASE_UNKNOWN, 0.0, None, None, 0, None, {"reason": "no_templates"})
        image = _as_numpy(third_view_rgb, np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
        keypoints, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
        )
        if descriptors is None or len(keypoints) < self.min_inliers:
            return PhaseResult(PHASE_UNKNOWN, 0.0, None, None, 0, None, {"reason": "few_image_features"})
        matches = []
        for template in self.templates:
            result = self._match_template(template, keypoints, descriptors)
            if result is not None:
                matches.append((*result, template))
        ranked = self._cluster_matches(matches)
        if not ranked:
            return PhaseResult(PHASE_UNKNOWN, 0.0, None, None, 0, None, {"reason": "no_geometric_match"})
        target_xy, score, inliers, template = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = score / max(score + runner_up, 1e-8)
        if runner_up > 0 and runner_up / score >= self.ambiguity_ratio:
            return PhaseResult(
                PHASE_UNKNOWN, confidence, tuple(target_xy), None, inliers,
                template["demo_id"], {"reason": "ambiguous", "runner_up": runner_up},
            )
        progress = None
        status = PHASE_OK
        if gripper_xy is not None:
            gripper = _as_numpy(gripper_xy, np.float32).reshape(2)
            if self.previous_gripper is not None:
                displacement = gripper - self.previous_gripper
                direction = target_xy - self.previous_gripper
                norm = float(np.linalg.norm(direction))
                progress = float(np.dot(displacement, direction / norm)) if norm > 1e-6 else 0.0
                if progress < -self.wrong_way_tolerance_px:
                    self.wrong_way_count += 1
                else:
                    self.wrong_way_count = 0
                if self.wrong_way_count >= self.wrong_way_updates:
                    status = PHASE_ERROR
            self.previous_gripper = gripper.copy()
        return PhaseResult(
            status, confidence, tuple(float(x) for x in target_xy), progress, inliers,
            template["demo_id"], {"cluster_score": score, "wrong_way_count": self.wrong_way_count},
        )
