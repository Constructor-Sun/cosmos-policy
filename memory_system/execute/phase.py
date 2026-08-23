"""2D phase verifier migrated into memory_system.

This module keeps the original bin/execute/libero_phase_verifier.py
behavior while consuming the shared memory_system types/artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from memory_system.artifacts import PhaseTargetMemory
from memory_system.types import PhaseResult, VerifierObservation


PHASE_OK = "PHASE_OK"
PHASE_ERROR = "PHASE_ERROR"
PHASE_UNKNOWN = "PHASE_UNKNOWN"


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class PhaseVerifier:
    """Locate a coarse target region and measure gripper distance reduction."""

    def __init__(
        self,
        memory: PhaseTargetMemory | str | Path,
        scales: tuple[float, ...] = (0.8, 1.0, 1.2),
        min_similarity: float = 0.45,
        min_demo_votes: int = 2,
        cluster_radius_px: float = 32.0,
        ambiguity_ratio: float = 0.9,
        wrong_way_tolerance_px: float = 5.0,
        wrong_way_updates: int = 1,
    ):
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")
        if min_demo_votes <= 0 or wrong_way_updates <= 0:
            raise ValueError("vote and update counts must be positive")
        self.memory = memory if isinstance(memory, PhaseTargetMemory) else PhaseTargetMemory(memory)
        self.scales = tuple(float(scale) for scale in scales)
        self.min_similarity = float(min_similarity)
        self.min_demo_votes = int(min_demo_votes)
        self.cluster_radius_px = float(cluster_radius_px)
        self.ambiguity_ratio = float(ambiguity_ratio)
        self.wrong_way_tolerance_px = float(wrong_way_tolerance_px)
        self.wrong_way_updates = int(wrong_way_updates)
        self.templates: list[dict[str, Any]] = []
        self.prepared_templates: list[tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]] = []
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
        self.prepared_templates = [
            (template, self._prepare_template(template)) for template in self.templates
        ]
        self.previous_gripper = None
        self.wrong_way_count = 0

    def _prepare_template(self, template: dict[str, Any]):
        crop, mask = template.get("crop_rgb"), template.get("crop_mask")
        if crop is None or mask is None:
            return []
        crop = _as_numpy(crop, np.uint8)
        mask = _as_numpy(mask, np.uint8)
        if crop.ndim != 3 or mask.ndim != 2 or np.count_nonzero(mask) < 16:
            return []
        prepared = []
        for scale in self.scales:
            width = max(6, int(round(crop.shape[1] * scale)))
            height = max(6, int(round(crop.shape[0] * scale)))
            resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
            resized = cv2.GaussianBlur(resized, (3, 3), 0)
            resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_rgb = np.repeat((resized_mask > 0)[..., None].astype(np.uint8), 3, axis=2)
            prepared.append((resized, mask_rgb))
        return prepared

    def _match_template(
        self, image: np.ndarray, prepared: list[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, float, tuple[int, int, int, int]] | None:
        best = None
        for resized, mask_rgb in prepared:
            height, width = resized.shape[:2]
            if width > image.shape[1] or height > image.shape[0]:
                continue
            response = cv2.matchTemplate(
                image, resized, cv2.TM_CCORR_NORMED, mask=mask_rgb
            )
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, score, _, location = cv2.minMaxLoc(response)
            x0, y0 = location
            candidate = (
                np.asarray([x0 + width / 2, y0 + height / 2], dtype=np.float32),
                float(score), (x0, y0, x0 + width, y0 + height),
            )
            if best is None or candidate[1] > best[1]:
                best = candidate
        return best if best is not None and best[1] >= self.min_similarity else None

    def _cluster_votes(self, votes):
        clusters = []
        for vote in sorted(votes, key=lambda item: item[1], reverse=True):
            for cluster in clusters:
                center = np.average(
                    [item[0] for item in cluster], weights=[item[1] for item in cluster], axis=0
                )
                if np.linalg.norm(vote[0] - center) <= self.cluster_radius_px:
                    cluster.append(vote)
                    break
            else:
                clusters.append([vote])
        ranked = []
        for cluster in clusters:
            weights = np.asarray([item[1] for item in cluster], dtype=np.float32)
            center = np.average([item[0] for item in cluster], weights=weights, axis=0)
            best = max(cluster, key=lambda item: item[1])
            support, mean_score = len(cluster), float(weights.mean())
            ranked.append((center, support * mean_score, support, mean_score, best))
        return sorted(ranked, key=lambda item: (item[2], item[1]), reverse=True)

    def update(self, observation: VerifierObservation) -> PhaseResult:
        third_view_rgb = observation.third_view_rgb
        gripper_xy = observation.gripper_xy
        if not self.templates:
            return PhaseResult(PHASE_UNKNOWN, 0.0, None, None, 0, None, {"reason": "no_templates"})
        image = _as_numpy(third_view_rgb, np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
        image = cv2.GaussianBlur(image, (3, 3), 0)
        by_demo = {}
        for template, prepared in self.prepared_templates:
            match = self._match_template(image, prepared)
            demo_id = template["demo_id"]
            if match is not None and (demo_id not in by_demo or match[1] > by_demo[demo_id][1]):
                by_demo[demo_id] = (*match, template)
        ranked = self._cluster_votes(list(by_demo.values()))
        if not ranked or ranked[0][2] < self.min_demo_votes:
            return PhaseResult(
                PHASE_UNKNOWN, 0.0, None, None, ranked[0][2] if ranked else 0,
                None, {"reason": "no_visual_consensus"},
            )
        target, strength, support, similarity, best = ranked[0]
        runner_strength = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = strength / max(strength + runner_strength, 1e-8)
        if runner_strength > 0 and runner_strength / strength >= self.ambiguity_ratio:
            return PhaseResult(
                PHASE_UNKNOWN, confidence, tuple(float(x) for x in target), None,
                support, best[3]["demo_id"], {"reason": "ambiguous_visual_region"},
            )
        progress, status = None, PHASE_OK
        if gripper_xy is not None:
            gripper = _as_numpy(gripper_xy, np.float32).reshape(2)
            if self.previous_gripper is not None:
                previous_distance = float(np.linalg.norm(self.previous_gripper - target))
                current_distance = float(np.linalg.norm(gripper - target))
                progress = previous_distance - current_distance
                if progress < -self.wrong_way_tolerance_px:
                    self.wrong_way_count += 1
                else:
                    self.wrong_way_count = 0
                if self.wrong_way_count >= self.wrong_way_updates:
                    status = PHASE_ERROR
            self.previous_gripper = gripper.copy()
        template = best[3]
        return PhaseResult(
            status, confidence, tuple(float(x) for x in target), progress, support,
            template["demo_id"], {
                "similarity": similarity,
                "demo_votes": support,
                "cluster_strength": strength,
                "matched_bbox_xyxy": best[2],
                "template_frame": int(template["frame"]),
                "wrong_way_count": self.wrong_way_count,
            },
        )
