"""Shared helpers for the memory_system skill-completion verifiers.

This module is internal to ``memory_system.execute.skill_completion``.  It
contains the common state/helpers from the original
``bin/execute/libero_skill_completion_verifier_geo.py`` so the public Pick,
Place and OpenClose verifiers stay focused on their skill-specific update
rules.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from memory_system.artifacts import PhaseTargetMemory, WristCompletionMemory
from memory_system.types import CompletionResult


SKILL_COMPLETE = "SKILL_COMPLETE"
COMPLETION_UNKNOWN = "COMPLETION_UNKNOWN"

TOLERANCE_PX = 5.0
CONFIRMATIONS = 2
MIN_DEMO_VOTES = 2

PICK_SKILLS = {"Pick"}
PLACE_SKILLS = {"PlaceIn", "PlaceOn"}
ANCHOR_SKILLS = {"Pick", "Open", "Close"}

WRIST_HISTORY = 5
WRIST_DIFF_THRESHOLD = 6.0
PICK_MOVE_DIST = 0.04
PLACE_STABLE_DROP = 0.15
PLACE_MOTION_THRESHOLD = 8.0
FLOW_ROI_HALF = 40
FLOW_MIN_POINTS = 8
FLOW_INLIER_THRESHOLD = 0.5

EMPTY_CLOSED_GAP = 0.003
CENTER_JUMP_THRESHOLD = 10.0
PICK_N_OF_M = 4
PICK_M_OF_M = 5

CLOSE_GRACE_STEPS = 5
ANCHOR_TOL_PX = 25.0
WRONG_GRASP_MAX = 2


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def _phase_key(task: str, step: int, skill: str, arguments: dict[str, Any]):
    return str(task), int(step), str(skill), _arguments_key(arguments)


class BaseCompletionVerifier:
    """Shared completion state and helper methods."""

    def __init__(
        self,
        phase_targets: str | Path,
        wrist_completion_targets: str | Path | None = None,
    ):
        self._ready_anchors: dict[tuple, tuple[tuple[str, np.ndarray], ...]] = {}
        self._wrist_memory: dict[tuple, list[dict[str, Any]]] = {}
        self._load_ready_anchors(phase_targets)
        if wrist_completion_targets is not None:
            self._load_wrist_completion_targets(wrist_completion_targets)
        self.skill = ""
        self.ready_anchors: tuple[tuple[str, np.ndarray], ...] = ()
        self.wrist_templates: tuple[dict[str, Any], ...] = ()
        self.target_bbox: tuple[float, float, float, float] | None = None
        self._prev_closed: bool | None = None
        self.confirmation_count = 0
        self.completed = False
        self.wrist_history: list[np.ndarray] = []
        self.prev_stable_ratio: float | None = None
        self.pick_close_xyz: np.ndarray | None = None
        self.pick_moved = False
        self.pick_last_center = None
        self.flow_prev_gray: np.ndarray | None = None
        self.flow_prev_pts: np.ndarray | None = None
        self.flow_inlier_ratio = 0.0
        self.pick_confirmations: deque[bool] = deque(maxlen=PICK_M_OF_M)
        self._edge_prev_closed: bool | None = None
        self.pick_edge_t: int | None = None
        self.pick_close_xy2d: np.ndarray | None = None
        self._edge_resolved = False
        self.wrong_grasp_pending = False
        self.wrong_grasp_count = 0

    def _load_ready_anchors(self, phase_targets: str | Path) -> None:
        memory = PhaseTargetMemory(phase_targets)
        by_key: dict[tuple, dict[str, list]] = {}
        for tpl in memory.templates:
            key = _phase_key(
                tpl["task_name"], tpl["planner_step_id"], tpl["skill"],
                tpl.get("arguments", {}),
            )
            by_key.setdefault(key, {}).setdefault(tpl["demo_id"], []).append(tpl)
        for key, demos in by_key.items():
            anchors = []
            for demo_id, tpls in demos.items():
                last = max(tpls, key=lambda t: t["frame"])
                anchors.append((
                    str(demo_id),
                    np.asarray(last["gripper_xy"], dtype=np.float32),
                ))
            self._ready_anchors[key] = tuple(anchors)

    def _load_wrist_completion_targets(self, path: str | Path) -> None:
        memory = WristCompletionMemory(path)
        for tpl in memory.templates:
            key = _phase_key(
                tpl["task_name"], tpl["planner_step_id"], tpl["skill"],
                tpl.get("arguments", {}),
            )
            self._wrist_memory.setdefault(key, []).append(tpl)

    def _select_wrist_templates(
        self,
        task: str,
        step: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str],
    ) -> tuple[dict[str, Any], ...]:
        excluded = {str(item) for item in exclude_demo_ids}
        exact = self._wrist_memory.get(
            _phase_key(task, step, skill, arguments), []
        )
        selected = [t for t in exact if t["demo_id"] not in excluded]
        if not selected:
            args_key = _arguments_key(arguments)
            selected = [
                tpl
                for (name, _step, kind, args), items in self._wrist_memory.items()
                if name == task and kind == skill and args == args_key
                for tpl in items if tpl["demo_id"] not in excluded
            ]
        by_demo = {tpl["demo_id"]: tpl for tpl in selected}
        return tuple(by_demo.values())

    def reset(
        self,
        task: str,
        step: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        self.skill = str(skill)
        excluded = {str(item) for item in exclude_demo_ids}
        key = _phase_key(task, step, skill, arguments)
        self.ready_anchors = tuple(
            item for item in self._ready_anchors.get(key, ())
            if item[0] not in excluded
        )
        if not self.ready_anchors:
            args_key = _arguments_key(arguments)
            self.ready_anchors = tuple(
                item for (name, _s, kind, args), items in self._ready_anchors.items()
                if name == task and kind == skill and args == args_key
                for item in items if item[0] not in excluded
            )
        self.wrist_templates = self._select_wrist_templates(
            task, step, skill, arguments, exclude_demo_ids
        )
        self.target_bbox = None
        self._prev_closed = None
        self.confirmation_count = 0
        self.completed = False
        self.wrist_history.clear()
        self.prev_stable_ratio = None
        self.pick_close_xyz = None
        self.pick_moved = False
        self.pick_last_center = None
        self.flow_prev_gray = None
        self.flow_prev_pts = None
        self.flow_inlier_ratio = 0.0
        self.pick_confirmations.clear()
        self._edge_prev_closed = None
        self.pick_edge_t = None
        self.pick_close_xy2d = None
        self._edge_resolved = False
        self.wrong_grasp_pending = False
        self.wrong_grasp_count = 0

    def calibrate(self, current_vae: Any | None = None) -> None:
        pass

    def freeze_baseline(self) -> None:
        pass

    def _inside(self, gripper_xy: np.ndarray, bbox) -> bool:
        return (
            bbox[0] - TOLERANCE_PX <= gripper_xy[0] <= bbox[2] + TOLERANCE_PX
            and bbox[1] - TOLERANCE_PX <= gripper_xy[1] <= bbox[3] + TOLERANCE_PX
        )

    def _gate_met(
        self,
        gripper_closed: bool | None,
        gripper_xy: np.ndarray,
    ) -> tuple[bool, str, dict[str, Any]]:
        if gripper_xy is None:
            return False, "gripper_xy_unknown", {}
        if self.skill in ANCHOR_SKILLS:
            if len(self.ready_anchors) < MIN_DEMO_VOTES:
                return False, "insufficient_anchors", {}
            xs = [anchor[0] for _demo, anchor in self.ready_anchors]
            ys = [anchor[1] for _demo, anchor in self.ready_anchors]
            cluster = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            inside = (
                cluster[0] <= gripper_xy[0] <= cluster[2]
                and cluster[1] <= gripper_xy[1] <= cluster[3]
            )
            return inside, ("inside_anchor_cluster" if inside else "outside_anchor_cluster"), {
                "anchor_cluster": cluster,
                "anchor_count": len(self.ready_anchors),
            }
        if self.skill in PLACE_SKILLS:
            if self.target_bbox is None:
                return False, "target_bbox_unknown", {}
            if gripper_closed is not False:
                return False, "gripper_still_closed", {}
            inside = self._inside(gripper_xy, self.target_bbox)
            return inside, ("open_inside_region" if inside else "gripper_outside_region"), {}
        return False, "no_geometric_completion_rule", {}

    def _inside_anchor_cluster(self, gripper_xy) -> bool:
        if len(self.ready_anchors) < MIN_DEMO_VOTES:
            return True
        xs = [float(anchor[0]) for _demo, anchor in self.ready_anchors]
        ys = [float(anchor[1]) for _demo, anchor in self.ready_anchors]
        return (
            min(xs) - ANCHOR_TOL_PX <= float(gripper_xy[0]) <= max(xs) + ANCHOR_TOL_PX
            and min(ys) - ANCHOR_TOL_PX <= float(gripper_xy[1]) <= max(ys) + ANCHOR_TOL_PX
        )

    def evaluate_close_edge(
        self,
        gripper_closed: bool | None,
        gripper_xy: Any = None,
        timestep: int | None = None,
    ) -> None:
        if self.skill not in PICK_SKILLS:
            return
        edge = gripper_closed is True and self._edge_prev_closed is not True
        self._edge_prev_closed = bool(gripper_closed)
        if edge and self.pick_edge_t is None:
            self.pick_edge_t = timestep
            if gripper_xy is not None:
                self.pick_close_xy2d = np.asarray(gripper_xy, dtype=np.float32).reshape(2)
        if (
            self.pick_edge_t is not None
            and not self._edge_resolved
            and timestep is not None
            and timestep - self.pick_edge_t >= CLOSE_GRACE_STEPS
        ):
            self._edge_resolved = True
            if (
                self.wrong_grasp_count < WRONG_GRASP_MAX
                and self.pick_close_xy2d is not None
                and not self._inside_anchor_cluster(self.pick_close_xy2d)
            ):
                self.wrong_grasp_pending = True

    def begin_wrong_grasp_recovery(self) -> None:
        self._edge_prev_closed = None
        self.pick_edge_t = None
        self.pick_close_xy2d = None
        self._edge_resolved = False
        self.wrong_grasp_pending = False
        self.pick_close_xyz = None
        self.pick_moved = False
        self.pick_last_center = None
        self.pick_confirmations.clear()

    def _update_wrist_metrics(self, wrist_image: Any) -> dict[str, float] | None:
        image = np.asarray(wrist_image)
        if image.ndim == 3:
            gray = image.astype(np.float32).mean(axis=2)
        elif image.ndim == 2:
            gray = image.astype(np.float32)
        else:
            return None
        self.wrist_history.append(gray)
        self.wrist_history = self.wrist_history[-WRIST_HISTORY:]
        if len(self.wrist_history) < 2:
            return None
        diffs = [
            np.abs(self.wrist_history[i + 1] - self.wrist_history[i])
            for i in range(len(self.wrist_history) - 1)
        ]
        diff = np.mean(np.stack(diffs), axis=0)
        return {
            "stable_ratio": float(np.mean(diff < WRIST_DIFF_THRESHOLD)),
            "mean_diff": float(np.mean(diff)),
        }

    def _pick_center(self, wrist_image):
        gray = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        if self.wrist_templates:
            tpl = self.wrist_templates[0]
            cx = int(round(float(tpl["target_center_xy"][0])))
            cy = int(round(float(tpl["target_center_xy"][1])))
        else:
            cx, cy = w // 2, h // 2
        x0 = max(0, cx - FLOW_ROI_HALF)
        y0 = max(0, cy - FLOW_ROI_HALF)
        x1 = min(w, cx + FLOW_ROI_HALF)
        y1 = min(h, cy + FLOW_ROI_HALF)

        if self.flow_prev_pts is None or len(self.flow_prev_pts) < FLOW_MIN_POINTS:
            mask = np.zeros_like(gray)
            mask[y0:y1, x0:x1] = 255
            pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.01,
                minDistance=10, mask=mask,
            )
            self.flow_prev_pts = pts
            self.flow_prev_gray = gray
            self.flow_inlier_ratio = 0.0
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.flow_prev_gray, gray, self.flow_prev_pts, None
        )
        if curr_pts is None:
            self.flow_prev_pts = None
            self.flow_inlier_ratio = 0.0
            return None

        prev_ok = self.flow_prev_pts[status.flatten() == 1]
        curr_ok = curr_pts[status.flatten() == 1]
        if len(prev_ok) < FLOW_MIN_POINTS:
            self.flow_prev_pts = None
            self.flow_inlier_ratio = 0.0
            return None

        H, mask = cv2.findHomography(prev_ok, curr_ok, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            self.flow_prev_pts = None
            self.flow_inlier_ratio = 0.0
            return None

        inliers = curr_ok[mask.flatten() == 1]
        self.flow_inlier_ratio = float(len(inliers) / len(curr_ok))
        if self.flow_inlier_ratio < FLOW_INLIER_THRESHOLD or len(inliers) < FLOW_MIN_POINTS:
            self.flow_prev_pts = None
            self.flow_inlier_ratio = 0.0
            return None

        center = inliers.reshape(-1, 2).mean(axis=0)
        self.flow_prev_pts = inliers.reshape(-1, 1, 2)
        self.flow_prev_gray = gray
        return (int(round(center[0])), int(round(center[1])))

    def _wrist_place_gate(
        self,
        gripper_closed: bool | None,
        metrics: dict[str, float],
    ) -> tuple[bool, str, dict[str, Any]]:
        if gripper_closed is not False:
            return False, "gripper_still_closed", dict(metrics)
        stable_drop = (
            (self.prev_stable_ratio - metrics["stable_ratio"])
            if self.prev_stable_ratio is not None else 0.0
        )
        details = dict(metrics)
        details["stable_drop"] = stable_drop
        moving = (
            metrics["mean_diff"] > PLACE_MOTION_THRESHOLD
            or stable_drop > PLACE_STABLE_DROP
        )
        details["moving"] = moving
        return True, ("wrist_object_released" if moving else "wrist_open_after_release"), details

    def observe_wrist(self, wrist_image: Any = None) -> None:
        if wrist_image is None:
            return
        metrics = self._update_wrist_metrics(wrist_image)
        if metrics is not None:
            self.prev_stable_ratio = metrics["stable_ratio"]

    def _finish(
        self,
        reason: str,
        timestep: int | None,
        details: dict[str, Any],
        gripper_xy: Any = None,
    ) -> CompletionResult:
        gxy = (
            np.asarray(gripper_xy, dtype=np.float32).tolist()
            if gripper_xy is not None else None
        )
        return CompletionResult(
            SKILL_COMPLETE if self.completed else COMPLETION_UNKNOWN,
            reason,
            timestep,
            None,
            None,
            len(self.ready_anchors),
            self.confirmation_count,
            {
                "rule": self.skill,
                "gripper_closed": self._prev_closed,
                "gripper_xy": gxy,
                "target_bbox": self.target_bbox,
                "pick_confirmations": list(self.pick_confirmations),
                **details,
            },
        )
