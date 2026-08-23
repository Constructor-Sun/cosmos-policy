"""Positive-only 2D feasible verifier migrated into memory_system.

This module keeps the original ``bin/execute/libero_feasible_region_verifier.py``
behavior while using the shared memory_system artifacts/types.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from memory_system.artifacts import ReadyDistanceMemory, WristFeasibleMemory
from memory_system.types import FeasibleResult, VerifierObservation


FEASIBLE = "FEASIBLE"
NOT_FEASIBLE = "NOT_FEASIBLE"
FEASIBLE_UNKNOWN = "FEASIBLE_UNKNOWN"


def _as_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _bbox_diagonal(bbox_xyxy: Any) -> float:
    x0, y0, x1, y1 = _as_numpy(bbox_xyxy).reshape(4)
    width, height = float(x1 - x0), float(y1 - y0)
    diagonal = float(np.hypot(width, height))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError(f"Invalid target bounding box: {bbox_xyxy!r}")
    return diagonal


class FeasibleVerifier:
    """Latch entry into the memory-defined ready radius and flag pre-entry drift."""

    def __init__(
        self,
        memory: ReadyDistanceMemory | str | Path,
        segments_manifest: str | Path | None = None,
        min_demo_votes: int = 2,
        progress_tolerance_px: float = 5.0,
        evidence_updates: int = 2,
        wrist_feasible_targets: str | Path | None = None,
        wrist_center_tolerance_ratio: float = 0.2,
        wrist_scale_tolerance_ratio: float = 0.5,
        wrist_min_inliers: int = 5,
    ):
        if isinstance(memory, ReadyDistanceMemory):
            self.memory = memory
        else:
            if segments_manifest is None:
                raise ValueError("segments_manifest is required with a phase-target path")
            self.memory = ReadyDistanceMemory(memory, segments_manifest)
        if min_demo_votes <= 0 or evidence_updates <= 0:
            raise ValueError("vote and update counts must be positive")
        if not np.isfinite(progress_tolerance_px) or progress_tolerance_px < 0:
            raise ValueError("progress_tolerance_px must be finite and non-negative")
        self.min_demo_votes = int(min_demo_votes)
        self.progress_tolerance_px = float(progress_tolerance_px)
        self.evidence_updates = int(evidence_updates)
        self.wrist_center_tolerance_ratio = float(wrist_center_tolerance_ratio)
        self.wrist_scale_tolerance_ratio = float(wrist_scale_tolerance_ratio)
        self.wrist_min_inliers = int(wrist_min_inliers)
        self._wrist_memory: WristFeasibleMemory | None = None
        if wrist_feasible_targets is not None:
            self._wrist_memory = WristFeasibleMemory(wrist_feasible_targets)
        self.sift = cv2.SIFT_create(nfeatures=1024)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.task_name: str | None = None
        self.planner_step_id: int | None = None
        self.skill = ""
        self.arguments: dict[str, str] = {}
        self.prototypes: tuple[Any, ...] = ()
        self.wrist_templates: tuple[dict[str, Any], ...] = ()
        self.previous_gripper: np.ndarray | None = None
        self.entered_feasible = False
        self.stall_count = 0
        self.wrong_way_count = 0
        self.history: list[FeasibleResult] = []

    def reset(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        self.task_name = str(task_name)
        self.planner_step_id = int(planner_step_id)
        self.skill = str(skill)
        self.arguments = {str(key): str(value) for key, value in arguments.items()}
        self.prototypes = self.memory.select(
            self.task_name,
            self.planner_step_id,
            self.skill,
            self.arguments,
            exclude_demo_ids,
        )
        self.wrist_templates = (
            self._wrist_memory.select(
                self.task_name,
                self.planner_step_id,
                self.skill,
                self.arguments,
                exclude_demo_ids,
            )
            if self._wrist_memory is not None
            else ()
        )
        self.previous_gripper = None
        self.entered_feasible = False
        self.stall_count = 0
        self.wrong_way_count = 0

    def _wrist_object_center(self, wrist_image: Any) -> tuple[bool, dict[str, Any]]:
        if not self.wrist_templates:
            return True, {"reason": "no_wrist_memory"}
        if wrist_image is None:
            return False, {"reason": "wrist_image_unavailable"}
        image = np.asarray(wrist_image)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            return False, {"reason": "invalid_wrist_image"}
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            return False, {"reason": "few_wrist_features"}
        best = None
        for tpl in self.wrist_templates:
            tdes = tpl.get("descriptors")
            if tdes is None or len(tdes) < 2:
                continue
            pairs = self.matcher.knnMatch(
                np.asarray(tdes, dtype=np.float32), descriptors, k=2
            )
            good = [
                first for first, second in pairs
                if first.distance < 0.75 * second.distance
            ]
            if len(good) < self.wrist_min_inliers:
                continue
            src = np.float32(
                [tpl["keypoints_xy"][m.queryIdx] for m in good]
            ).reshape(-1, 1, 2)
            dst = np.float32(
                [keypoints[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            homography, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            bbox = None
            if homography is not None:
                height, width = tpl["crop_rgb"].shape[:2]
                corners = np.float32(
                    [[0, 0], [width, 0], [width, height], [0, height]]
                ).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
                x0 = max(0, int(projected[:, 0].min()))
                y0 = max(0, int(projected[:, 1].min()))
                x1 = min(gray.shape[1], int(projected[:, 0].max()) + 1)
                y1 = min(gray.shape[0], int(projected[:, 1].max()) + 1)
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    bbox = (x0, y0, x1, y1)
            if bbox is None:
                xs = [keypoints[m.trainIdx].pt[0] for m in good]
                ys = [keypoints[m.trainIdx].pt[1] for m in good]
                x0 = max(0, int(min(xs)))
                y0 = max(0, int(min(ys)))
                x1 = min(gray.shape[1], int(max(xs)) + 1)
                y1 = min(gray.shape[0], int(max(ys)) + 1)
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    bbox = (x0, y0, x1, y1)
            score = len(good)
            if best is None or score > best[0]:
                best = (score, bbox, tpl)
        if best is None or best[1] is None:
            return False, {"reason": "wrist_object_not_found"}
        score, bbox, tpl = best
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return True, {
            "reason": "wrist_object_found",
            "score": score,
            "center": (float(cx), float(cy)),
            "bbox": bbox,
            "template_demo_id": tpl.get("demo_id"),
            "template_center": tuple(float(v) for v in tpl.get("target_center_xy", [0, 0])),
        }

    def _wrist_ready(self, wrist_image: Any) -> tuple[bool, dict[str, Any]]:
        if not self.wrist_templates:
            return True, {"reason": "no_wrist_memory"}
        found, info = self._wrist_object_center(wrist_image)
        if not found:
            return False, info
        tpl = None
        for item in self.wrist_templates:
            if item.get("demo_id") == info["template_demo_id"]:
                tpl = item
                break
        if tpl is None:
            tpl = self.wrist_templates[0]
        bbox = np.asarray(tpl["bbox_xyxy"], dtype=np.float32).reshape(4)
        diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        tpl_center = np.asarray(info["template_center"], dtype=np.float32)
        cur_center = np.asarray(info["center"], dtype=np.float32)
        center_dist = float(np.linalg.norm(cur_center - tpl_center))
        center_ok = center_dist <= self.wrist_center_tolerance_ratio * max(diag, 1.0)
        cur_bbox = np.asarray(info["bbox"], dtype=np.float32).reshape(4)
        cur_diag = float(np.hypot(cur_bbox[2] - cur_bbox[0], cur_bbox[3] - cur_bbox[1]))
        scale_ratio = cur_diag / max(diag, 1e-6)
        scale_ok = abs(scale_ratio - 1.0) <= self.wrist_scale_tolerance_ratio
        return (center_ok and scale_ok), {
            **info,
            "center_dist": center_dist,
            "center_tolerance": self.wrist_center_tolerance_ratio * max(diag, 1.0),
            "scale_ratio": scale_ratio,
            "center_ok": center_ok,
            "scale_ok": scale_ok,
        }

    def _result(self, **kwargs: Any) -> FeasibleResult:
        result = FeasibleResult(**kwargs)
        self.history.append(result)
        return result

    def update(
        self,
        observation: VerifierObservation,
        target_xy: tuple[float, float] | np.ndarray | None,
        matched_bbox_xyxy: tuple[float, float, float, float] | np.ndarray | None,
        *,
        confidence: float = 0.0,
    ) -> FeasibleResult:
        if self.task_name is None:
            raise RuntimeError("reset() must be called before update()")
        gripper_xy = observation.gripper_xy
        timestep = observation.timestep
        wrist_image = observation.wrist_image
        common = {
            "timestep": timestep,
            "entered_feasible": self.entered_feasible,
            "current_distance": None,
            "current_distance_px": None,
            "ready_distance": None,
            "ready_votes": 0,
            "memory_count": len(self.prototypes),
            "progress": None,
            "progress_px": None,
            "stall_count": self.stall_count,
            "wrong_way_count": self.wrong_way_count,
            "confidence": float(confidence),
            "details": {},
        }
        if len(self.prototypes) < self.min_demo_votes:
            return self._result(
                status=FEASIBLE_UNKNOWN,
                reason="insufficient_ready_memory",
                **common,
            )
        if target_xy is None or matched_bbox_xyxy is None or gripper_xy is None:
            return self._result(
                status=FEASIBLE_UNKNOWN,
                reason="target_geometry_unknown",
                **common,
            )

        target = _as_numpy(target_xy).reshape(2)
        gripper = _as_numpy(gripper_xy).reshape(2)
        target_scale = _bbox_diagonal(matched_bbox_xyxy)
        current_distance_px = float(np.linalg.norm(gripper - target))
        current_distance = current_distance_px / target_scale
        ready_distances = sorted(
            (item.normalized_distance for item in self.prototypes), reverse=True
        )
        ready_distance = ready_distances[self.min_demo_votes - 1]
        ready_votes = sum(
            current_distance <= item.normalized_distance for item in self.prototypes
        )
        progress_px = None
        progress = None
        if self.previous_gripper is not None:
            previous_distance_px = float(np.linalg.norm(self.previous_gripper - target))
            progress_px = previous_distance_px - current_distance_px
            progress = progress_px / target_scale

        wrist_details: dict[str, Any] = {}
        if ready_votes >= self.min_demo_votes:
            if self.wrist_templates:
                wrist_ready, wrist_details = self._wrist_ready(wrist_image)
                if not wrist_ready:
                    status, reason = NOT_FEASIBLE, "wrist_not_ready"
                else:
                    newly_entered = not self.entered_feasible
                    self.entered_feasible = True
                    self.stall_count = 0
                    self.wrong_way_count = 0
                    status, reason = FEASIBLE, (
                        "entered_ready_radius" if newly_entered else "latched_feasible"
                    )
            else:
                newly_entered = not self.entered_feasible
                self.entered_feasible = True
                self.stall_count = 0
                self.wrong_way_count = 0
                status, reason = FEASIBLE, (
                    "entered_ready_radius" if newly_entered else "latched_feasible"
                )
        elif self.entered_feasible:
            status, reason = FEASIBLE, "latched_feasible"
        elif progress_px is None:
            status, reason = FEASIBLE_UNKNOWN, "warmup"
        elif progress_px < -self.progress_tolerance_px:
            self.wrong_way_count += 1
            self.stall_count = 0
            status = (
                NOT_FEASIBLE
                if self.wrong_way_count >= self.evidence_updates
                else FEASIBLE_UNKNOWN
            )
            reason = (
                "persistent_reversal"
                if status == NOT_FEASIBLE
                else "reversal_candidate"
            )
        elif abs(progress_px) <= self.progress_tolerance_px:
            self.stall_count += 1
            self.wrong_way_count = 0
            status = (
                NOT_FEASIBLE
                if self.stall_count >= self.evidence_updates
                else FEASIBLE_UNKNOWN
            )
            reason = (
                "persistent_stagnation"
                if status == NOT_FEASIBLE
                else "stagnation_candidate"
            )
        else:
            self.stall_count = 0
            self.wrong_way_count = 0
            status, reason = FEASIBLE_UNKNOWN, "approaching"

        self.previous_gripper = gripper.copy()
        return self._result(
            status=status,
            reason=reason,
            timestep=timestep,
            entered_feasible=self.entered_feasible,
            current_distance=current_distance,
            current_distance_px=current_distance_px,
            ready_distance=ready_distance,
            ready_votes=ready_votes,
            memory_count=len(self.prototypes),
            progress=progress,
            progress_px=progress_px,
            stall_count=self.stall_count,
            wrong_way_count=self.wrong_way_count,
            confidence=float(confidence),
            details={
                "target_scale_px": target_scale,
                "progress_tolerance": self.progress_tolerance_px / target_scale,
                "wrist_details": wrist_details,
                "ready_demo_ids": [
                    item.demo_id
                    for item in self.prototypes
                    if current_distance <= item.normalized_distance
                ],
            },
        )
