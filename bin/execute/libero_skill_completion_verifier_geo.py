#!/usr/bin/env python3
"""Wrist/geometric skill-completion verifier for LIBERO."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


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

# Wrong-object grasp guard: a Pick close command whose gripper is outside the
# demo ready-anchor cluster is treated as an invalid (wrong-object) grasp.
CLOSE_GRACE_STEPS = 5      # steps after the close edge before judging (let feasible latch first)
ANCHOR_TOL_PX = 25.0       # tolerance outside the demo anchor cluster (data: ok<=12px, wrong>=58px)
WRONG_GRASP_MAX = 2        # per-phase budget for wrong-grasp interventions (separate from phase corrections)


@dataclass(frozen=True)
class SkillCompletionResult:
    status: str
    reason: str
    timestep: int | None
    distance: float | None
    success_radius: float | None
    memory_count: int
    confirmation_count: int
    details: dict[str, Any] = field(default_factory=dict)


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def _phase_key(task: str, step: int, skill: str, arguments: dict[str, Any]):
    return str(task), int(step), str(skill), _arguments_key(arguments)


class LiberoSkillCompletionVerifierGeo:
    """Per-skill completion verifier.

    The execution monitor decides when COMPLETION_CHECK starts.  This class only
    decides whether an observation is completion evidence and whether enough
    evidence has accumulated.
    """

    def __init__(self, phase_targets: str | Path,
                 wrist_completion_targets: str | Path | None = None):
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
        # Wrong-object grasp guard state (reset per phase).
        self._edge_prev_closed: bool | None = None
        self.pick_edge_t: int | None = None
        self.pick_close_xy2d: np.ndarray | None = None
        self._edge_resolved = False
        self.wrong_grasp_pending = False
        self.wrong_grasp_count = 0

    # ------------------------------------------------------------------ #
    def _load_ready_anchors(self, phase_targets: str | Path) -> None:
        payload = torch.load(phase_targets, map_location="cpu", weights_only=False)
        templates = (payload["templates"] if isinstance(payload, dict)
                     and "templates" in payload else payload)
        by_key: dict[tuple, dict[str, list]] = {}
        for tpl in templates:
            key = _phase_key(tpl["task_name"], tpl["planner_step_id"],
                             tpl["skill"], tpl.get("arguments", {}))
            by_key.setdefault(key, {}).setdefault(tpl["demo_id"], []).append(tpl)
        for key, demos in by_key.items():
            anchors = []
            for demo_id, tpls in demos.items():
                last = max(tpls, key=lambda t: t["frame"])
                anchors.append((str(demo_id),
                                np.asarray(last["gripper_xy"], dtype=np.float32)))
            self._ready_anchors[key] = tuple(anchors)

    def _load_wrist_completion_targets(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for tpl in payload.get("templates", []):
            key = _phase_key(tpl["task_name"], tpl["planner_step_id"],
                             tpl["skill"], tpl.get("arguments", {}))
            self._wrist_memory.setdefault(key, []).append(tpl)

    def _select_wrist_templates(
        self, task: str, step: int, skill: str, arguments: dict[str, Any],
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

    def reset(self, task: str, step: int, skill: str, arguments: dict[str, Any],
              exclude_demo_ids: Iterable[str] = ()) -> None:
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

    # ------------------------------------------------------------------ #
    def _inside(self, gripper_xy: np.ndarray, bbox) -> bool:
        return (bbox[0] - TOLERANCE_PX <= gripper_xy[0] <= bbox[2] + TOLERANCE_PX
                and bbox[1] - TOLERANCE_PX <= gripper_xy[1] <= bbox[3] + TOLERANCE_PX)

    def _gate_met(self, gripper_closed: bool | None,
                  gripper_xy: np.ndarray) -> tuple[bool, str, dict[str, Any]]:
        if gripper_xy is None:
            return False, "gripper_xy_unknown", {}
        if self.skill in ANCHOR_SKILLS:
            if len(self.ready_anchors) < MIN_DEMO_VOTES:
                return False, "insufficient_anchors", {}
            xs = [anchor[0] for _demo, anchor in self.ready_anchors]
            ys = [anchor[1] for _demo, anchor in self.ready_anchors]
            cluster = (float(min(xs)), float(min(ys)),
                       float(max(xs)), float(max(ys)))
            inside = (cluster[0] <= gripper_xy[0] <= cluster[2]
                      and cluster[1] <= gripper_xy[1] <= cluster[3])
            return inside, ("inside_anchor_cluster" if inside
                            else "outside_anchor_cluster"), {
                "anchor_cluster": cluster,
                "anchor_count": len(self.ready_anchors),
            }
        if self.skill in PLACE_SKILLS:
            if self.target_bbox is None:
                return False, "target_bbox_unknown", {}
            if gripper_closed is not False:
                return False, "gripper_still_closed", {}
            inside = self._inside(gripper_xy, self.target_bbox)
            return inside, ("open_inside_region" if inside
                            else "gripper_outside_region"), {}
        return False, "no_geometric_completion_rule", {}

    # ------------------------------------------------------------------ #
    def _inside_anchor_cluster(self, gripper_xy) -> bool:
        """True if the (projected) gripper is inside the demo ready-anchor
        cluster inflated by ANCHOR_TOL_PX.  The anchors come from the demos
        (last gripper pose of the phase), so this check is immune to the
        runtime phase-verifier match quality."""
        if len(self.ready_anchors) < MIN_DEMO_VOTES:
            return True
        xs = [float(anchor[0]) for _demo, anchor in self.ready_anchors]
        ys = [float(anchor[1]) for _demo, anchor in self.ready_anchors]
        return (min(xs) - ANCHOR_TOL_PX <= float(gripper_xy[0]) <= max(xs) + ANCHOR_TOL_PX
                and min(ys) - ANCHOR_TOL_PX <= float(gripper_xy[1]) <= max(ys) + ANCHOR_TOL_PX)

    def evaluate_close_edge(self, gripper_closed: bool | None,
                            gripper_xy: Any = None,
                            timestep: int | None = None) -> None:
        """Step-level close-edge monitor, run in every stage (not only during
        COMPLETION_CHECK).

        On the rising edge of the close command, the projected gripper
        position is remembered.  After CLOSE_GRACE_STEPS (during which the
        normal feasible/wrist state machine may latch), if the gripper is
        outside the demo ready-anchor cluster of the current Pick phase, the
        close is flagged as a wrong-object grasp (wrong_grasp_pending) so the
        caller can intervene before the object is lifted away.
        """
        if self.skill not in PICK_SKILLS:
            return
        edge = gripper_closed is True and self._edge_prev_closed is not True
        self._edge_prev_closed = bool(gripper_closed)
        if edge and self.pick_edge_t is None:
            self.pick_edge_t = timestep
            if gripper_xy is not None:
                self.pick_close_xy2d = np.asarray(gripper_xy, dtype=np.float32).reshape(2)
        if (self.pick_edge_t is not None and not self._edge_resolved
                and timestep is not None and timestep - self.pick_edge_t >= CLOSE_GRACE_STEPS):
            self._edge_resolved = True
            if (self.wrong_grasp_count < WRONG_GRASP_MAX
                    and self.pick_close_xy2d is not None
                    and not self._inside_anchor_cluster(self.pick_close_xy2d)):
                self.wrong_grasp_pending = True

    def begin_wrong_grasp_recovery(self) -> None:
        """Clear grasp + edge state when a wrong-object grasp correction starts."""
        self._edge_prev_closed = None
        self.pick_edge_t = None
        self.pick_close_xy2d = None
        self._edge_resolved = False
        self.wrong_grasp_pending = False
        self.pick_close_xyz = None
        self.pick_moved = False
        self.pick_last_center = None
        self.pick_confirmations.clear()

    # ------------------------------------------------------------------ #
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

    def _wrist_place_gate(self, gripper_closed: bool | None,
                          metrics: dict[str, float]) -> tuple[bool, str, dict[str, Any]]:
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

    def update(self, current_vae: Any | None = None, *,
               gripper_closed: bool | None = None, gripper_xy: Any = None,
               target_bbox: Any = None, timestep: int | None = None,
               wrist_image: Any = None, gripper_qpos: Any = None,
               eef_pos: Any = None) -> SkillCompletionResult:
        gxy = (np.asarray(gripper_xy, dtype=np.float32)
               if gripper_xy is not None else None)
        details: dict[str, Any] = {}
        if self.completed:
            reason = "latched_complete"
        else:
            if target_bbox is not None:
                self.target_bbox = tuple(float(v) for v in target_bbox)
            wrist_metrics = (
                self._update_wrist_metrics(wrist_image)
                if wrist_image is not None else None
            )
            if gripper_closed is True:
                if eef_pos is not None:
                    eef = np.asarray(eef_pos, dtype=np.float32).reshape(3)
                    if self._prev_closed is not True:
                        self.pick_close_xyz = eef.copy()
                        self.pick_moved = False
                        self.pick_last_center = None
                        self.pick_confirmations.clear()
                    elif not self.pick_moved and self.pick_close_xyz is not None:
                        if np.linalg.norm(eef - self.pick_close_xyz) >= PICK_MOVE_DIST:
                            self.pick_moved = True
            else:
                self.pick_close_xyz = None
                self.pick_moved = False
                self.pick_last_center = None
                self.pick_confirmations.clear()
            specific = False
            if wrist_metrics is not None and self.skill in PICK_SKILLS:
                center = self._pick_center(wrist_image)
                jump = float("inf")
                if center is not None:
                    if self.pick_last_center is not None:
                        jump = float(np.hypot(
                            center[0] - self.pick_last_center[0],
                            center[1] - self.pick_last_center[1],
                        ))
                    self.pick_last_center = center
                gap = None
                if gripper_qpos is not None:
                    try:
                        gap = float(gripper_qpos[0]) - float(gripper_qpos[1])
                    except Exception:
                        gap = None
                met = bool(
                    gripper_closed is True
                    and center is not None
                    and self.pick_moved
                    and jump <= CENTER_JUMP_THRESHOLD
                    and self.flow_inlier_ratio >= FLOW_INLIER_THRESHOLD
                    and (gap is None or gap > EMPTY_CLOSED_GAP)
                    and (self.pick_close_xy2d is None
                         or self._inside_anchor_cluster(self.pick_close_xy2d))
                )
                details = dict(
                    wrist_metrics,
                    object_center=center,
                    pick_moved=self.pick_moved,
                    gap=gap,
                    flow_inlier_ratio=self.flow_inlier_ratio,
                    close_xy2d=(
                        self.pick_close_xy2d.tolist()
                        if self.pick_close_xy2d is not None else None
                    ),
                )
                reason = (
                    "object_center_stable_with_object" if met
                    else "gripper_not_closed" if gripper_closed is not True
                    else "gripper_empty_close" if gap is not None and gap <= EMPTY_CLOSED_GAP
                    else "object_center_unknown" if center is None
                    else "gripper_not_moved_since_close" if not self.pick_moved
                    else "gripper_outside_target_anchor" if (
                        self.pick_close_xy2d is not None
                        and not self._inside_anchor_cluster(self.pick_close_xy2d)
                    )
                    else "object_center_jump"
                )
                self.pick_confirmations.append(met)
                self.confirmation_count = sum(self.pick_confirmations)
            elif wrist_metrics is not None and self.skill in PLACE_SKILLS:
                released = self._prev_closed is True and gripper_closed is False
                met, reason, details = self._wrist_place_gate(
                    gripper_closed, wrist_metrics
                )
                # We intentionally do not try to repair a failed release in the
                # current Place phase. Treat any release as completion.
                if released:
                    self.confirmation_count = max(self.confirmation_count, CONFIRMATIONS)
                    reason = "released_wrist_object"
                    specific = True
                elif met and self.confirmation_count >= 1:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 0
            else:
                met, reason, details = self._gate_met(gripper_closed, gxy)
                if self.skill in PLACE_SKILLS:
                    released = self._prev_closed is True and gripper_closed is False
                    if released:
                        self.confirmation_count = max(self.confirmation_count, CONFIRMATIONS)
                        reason = "released_inside_region"
                        specific = True
                    elif met and self.confirmation_count >= 1:
                        self.confirmation_count += 1
                    else:
                        self.confirmation_count = 0
                elif met:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 0
            if wrist_metrics is not None:
                self.prev_stable_ratio = wrist_metrics["stable_ratio"]
            self._prev_closed = (None if gripper_closed is None
                                 else bool(gripper_closed))
            required_confirmations = (
                PICK_N_OF_M
                if wrist_metrics is not None and self.skill in PICK_SKILLS
                else CONFIRMATIONS
            )
            self.completed = self.confirmation_count >= required_confirmations
            if self.completed:
                reason = "confirmed_complete"
            elif self.confirmation_count > 0 and not specific:
                reason = "completion_candidate"
        return SkillCompletionResult(
            SKILL_COMPLETE if self.completed else COMPLETION_UNKNOWN,
            reason, timestep, None, None, len(self.ready_anchors),
            self.confirmation_count,
            {"rule": self.skill,
             "gripper_closed": gripper_closed,
             "gripper_xy": (gxy.tolist() if gxy is not None else None),
             "target_bbox": self.target_bbox,
             "pick_confirmations": list(self.pick_confirmations),
             **details},
        )
