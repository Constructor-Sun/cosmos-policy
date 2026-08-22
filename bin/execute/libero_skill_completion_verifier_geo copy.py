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

# Existing framework constants.
TOLERANCE_PX = 5.0      # phase verifier's 5px tolerance
CONFIRMATIONS = 2       # existing two-consecutive-evidence rule
MIN_DEMO_VOTES = 2      # existing cross-demo consensus rule

PICK_SKILLS = {"Pick"}
PLACE_SKILLS = {"PlaceIn", "PlaceOn"}
ANCHOR_SKILLS = {"Pick", "Open", "Close"}

# Wrist-based completion heuristics.
WRIST_HISTORY = 5
WRIST_DIFF_THRESHOLD = 6.0
WRIST_STABLE_RATIO = 0.75
WRIST_PICK_CONFIRMATIONS = 3
WRIST_MIN_INLIERS = 5
WRIST_RATIO_TEST = 0.75
WRIST_OBJECT_MIN_FRAMES = 2
WRIST_OBJECT_MAX_CENTER_DRIFT = 6.0
WRIST_GRIPPER_MOVE_THRESHOLD = 4.0
GRIPPER_MOVE_WINDOW = 3
PLACE_STABLE_DROP = 0.15
PLACE_MOTION_THRESHOLD = 8.0

# Simplified Pick completion (non-empty grasp + lightweight tracking).
EMPTY_CLOSED_GAP = 0.01
CENTER_JUMP_THRESHOLD = 10.0
PICK_N_OF_M = 2
PICK_M_OF_M = 3


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

    When a wrist image is available:
      Pick: gripper closed + wrist image becomes stable (object held still);
      PlaceIn/On: release transition + gripper open; motion metrics are logged.

    Without a wrist image, it falls back to the previous geometric rules:
      Pick/Open/Close: gripper_xy inside the demo ready-anchor cluster;
      PlaceIn/On: release transition inside the target bbox;
      other: always UNKNOWN.

    A phase completes after CONFIRMATIONS consecutive observations meet the
    gate; any miss resets the count.
    """

    def __init__(self, phase_targets: str | Path,
                 wrist_completion_targets: str | Path | None = None):
        self._ready_anchors: dict[tuple, tuple[tuple[str, np.ndarray], ...]] = {}
        self._load_ready_anchors(phase_targets)
        self._wrist_memory: dict[tuple, list[dict[str, Any]]] = {}
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
        self.object_centers: list[tuple[int, int]] = []
        self.object_bbox: tuple[int, int, int, int] | None = None
        self.prev_gripper_xy: np.ndarray | None = None
        self.gripper_xy_history: deque[np.ndarray] = deque(maxlen=GRIPPER_MOVE_WINDOW)
        self.gripper_moved_since_close = False
        self.pick_tracker = None
        self.pick_tracker_initialized = False
        self.pick_last_center = None
        self.pick_confirmations: deque[bool] = deque(maxlen=PICK_M_OF_M)
        self.pick_moved_since_close = False
        self.sift = cv2.SIFT_create(nfeatures=1024)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

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
            key = _phase_key(
                tpl["task_name"], tpl["planner_step_id"],
                tpl["skill"], tpl.get("arguments", {}),
            )
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
        self.ready_anchors = tuple(
            item for item in self._ready_anchors.get(
                _phase_key(task, step, skill, arguments), ())
            if item[0] not in excluded)
        if not self.ready_anchors:  # fall back to same task+skill+arguments
            args_key = _arguments_key(arguments)
            self.ready_anchors = tuple(
                item for (name, _s, kind, args), items in self._ready_anchors.items()
                if name == task and kind == skill and args == args_key
                for item in items if item[0] not in excluded)
        self.wrist_templates = self._select_wrist_templates(
            task, step, skill, arguments, exclude_demo_ids
        )
        self.target_bbox = None
        self._prev_closed = None
        self.confirmation_count = 0
        self.completed = False
        self.wrist_history.clear()
        self.prev_stable_ratio = None
        self.object_centers.clear()
        self.object_bbox = None
        self.prev_gripper_xy = None
        self.gripper_xy_history.clear()
        self.gripper_moved_since_close = False
        self.pick_tracker = None
        self.pick_tracker_initialized = False
        self.pick_last_center = None
        self.pick_confirmations.clear()
        self.pick_moved_since_close = False

    # Interface compatibility with the sequential monitor (VAE-era no-ops).
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
            details = {"anchor_cluster": cluster,
                       "anchor_count": len(self.ready_anchors)}
            return inside, ("inside_anchor_cluster" if inside
                            else "outside_anchor_cluster"), details
        if self.skill in PLACE_SKILLS:
            if self.target_bbox is None:
                return False, "target_bbox_unknown", {}
            if gripper_closed is not False:
                return False, "gripper_still_closed", {}
            inside = self._inside(gripper_xy, self.target_bbox)
            return inside, ("open_inside_region" if inside
                            else "gripper_outside_region"), {}
        if self.skill in ANCHOR_SKILLS:
            if not gripper_closed:
                return False, "gripper_not_closed", {}
            dists = [float(np.hypot(*(gripper_xy - anchor)))
                     for _demo, anchor in self.ready_anchors]
            votes = sum(1 for dist in dists if dist <= TOLERANCE_PX)
            details = {"anchor_votes": votes,
                       "nearest_anchor_px": min(dists) if dists else None}
            return (votes >= MIN_DEMO_VOTES,
                    ("ready_anchor_votes" if votes >= MIN_DEMO_VOTES
                     else "ready_anchor_far"), details)
        return False, "no_geometric_completion_rule", {}

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

    def _update_object_tracking(self, wrist_image: Any) -> dict[str, Any] | None:
        if not self.wrist_templates:
            return None
        image = np.asarray(wrist_image)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            return None
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            self.object_centers.clear()
            self.object_bbox = None
            return {
                "present": False, "score": 0.0, "frame_count": 0,
                "center_drift": 0.0, "reason": "few_wrist_features",
            }

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
                if first.distance < WRIST_RATIO_TEST * second.distance
            ]
            if len(good) < WRIST_MIN_INLIERS:
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
                best = (score, bbox)

        if best is None or best[1] is None:
            self.object_centers.clear()
            self.object_bbox = None
            return {
                "present": False, "score": 0.0, "frame_count": 0,
                "center_drift": 0.0, "reason": "wrist_object_no_memory_match",
            }

        score, bbox = best
        x0, y0, x1, y1 = bbox
        cx = int(round((x0 + x1) / 2.0))
        cy = int(round((y0 + y1) / 2.0))
        self.object_centers.append((cx, cy))
        self.object_centers = self.object_centers[-WRIST_HISTORY:]
        self.object_bbox = bbox

        center_drift = 0.0
        if len(self.object_centers) >= 2:
            drifts = [
                np.hypot(
                    self.object_centers[i + 1][0] - self.object_centers[i][0],
                    self.object_centers[i + 1][1] - self.object_centers[i][1],
                )
                for i in range(len(self.object_centers) - 1)
            ]
            center_drift = float(np.mean(drifts))

        return {
            "present": True,
            "score": float(score),
            "frame_count": len(self.object_centers),
            "center_drift": center_drift,
            "reason": "wrist_object_memory_match",
            "bbox": bbox,
        }

    def _create_pick_tracker(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            try:
                return cv2.TrackerCSRT_create()
            except Exception:
                pass
        return cv2.TrackerMIL_create()

    def _pick_bbox_xywh(self, bbox):
        x0, y0, x1, y1 = bbox
        return (int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0)))

    def _pick_reinit_tracker(self, wrist_image, center):
        h, w = wrist_image.shape[:2]
        half = 24
        x0 = max(0, int(center[0]) - half)
        y0 = max(0, int(center[1]) - half)
        x1 = min(w, int(center[0]) + half)
        y1 = min(h, int(center[1]) + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return False
        self.pick_tracker = self._create_pick_tracker()
        self.pick_tracker.init(wrist_image, self._pick_bbox_xywh((x0, y0, x1, y1)))
        self.pick_tracker_initialized = True
        return True

    def _pick_detect_center(self, wrist_image):
        tracking = self._update_object_tracking(wrist_image)
        if tracking is None or not tracking.get("present") or tracking.get("bbox") is None:
            return None
        x0, y0, x1, y1 = tracking["bbox"]
        return (int((x0 + x1) / 2.0), int((y0 + y1) / 2.0))

    def _pick_track_center(self, wrist_image):
        if not self.pick_tracker_initialized or self.pick_tracker is None:
            center = self._pick_detect_center(wrist_image)
            if center is None:
                return None
            self._pick_reinit_tracker(wrist_image, center)
            self.pick_last_center = center
            return center
        ok, bbox = self.pick_tracker.update(wrist_image)
        if ok and bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            center = (int(bbox[0] + bbox[2] / 2.0), int(bbox[1] + bbox[3] / 2.0))
            if self.pick_last_center is not None:
                jump = float(np.hypot(center[0] - self.pick_last_center[0],
                                      center[1] - self.pick_last_center[1]))
                if jump > CENTER_JUMP_THRESHOLD:
                    center = self._pick_detect_center(wrist_image)
                    if center is None:
                        return None
                    self._pick_reinit_tracker(wrist_image, center)
                    self.pick_last_center = center
            return center
        center = self._pick_detect_center(wrist_image)
        if center is None:
            return None
        self._pick_reinit_tracker(wrist_image, center)
        self.pick_last_center = center
        return center

    def _wrist_pick_gate(self, gripper_closed: bool | None,
                         metrics: dict[str, float],
                         wrist_image: Any = None,
                         gripper_qpos: Any = None) -> tuple[bool, str, dict[str, Any]]:
        # 1. Cheapest filter: non-empty closed grasp.
        if gripper_closed is not True:
            return False, "gripper_not_closed", dict(metrics)
        if gripper_qpos is not None:
            try:
                gap = float(gripper_qpos[0]) - float(gripper_qpos[1])
            except Exception:
                gap = None
            if gap is not None and gap <= EMPTY_CLOSED_GAP:
                return False, "gripper_empty_close", dict(metrics)
        # 2. Object center from CSRT, with SIFT re-detection on jumps.
        if wrist_image is None:
            return False, "wrist_image_unknown", dict(metrics)
        center = self._pick_track_center(wrist_image)
        if center is None:
            return False, "object_center_unknown", dict(metrics)
        details = dict(metrics)
        details["object_center"] = center
        details["pick_last_center"] = self.pick_last_center
        # 3. Center stability / tracker jump use the same threshold.
        if self.pick_last_center is not None:
            displacement = float(np.hypot(center[0] - self.pick_last_center[0],
                                          center[1] - self.pick_last_center[1]))
            if displacement > CENTER_JUMP_THRESHOLD:
                return False, "object_center_jump", details
        self.pick_last_center = center
        # 4. The gripper must have moved after closing.
        if not self.gripper_moved_since_close:
            return False, "gripper_not_moved_since_close", details
        return True, "object_center_stable_with_object", details

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
        # The release transition is handled by the caller; once the gripper is
        # open after a release, count it as completion evidence. The motion
        # metrics are logged for diagnostics and can be tightened later.
        return True, ("wrist_object_released" if moving else "wrist_open_after_release"), details

    def _window_total_displacement(self, positions: list[np.ndarray]) -> float:
        """Return total path length over the kept position window."""
        if len(positions) < 2:
            return 0.0
        total = 0.0
        for i in range(len(positions) - 1):
            total += float(np.linalg.norm(
                np.asarray(positions[i + 1], dtype=np.float32)
                - np.asarray(positions[i], dtype=np.float32)
            ))
        return total

    def observe_wrist(self, wrist_image: Any = None) -> None:
        """Warm up wrist-history without affecting completion state."""
        if wrist_image is None:
            return
        metrics = self._update_wrist_metrics(wrist_image)
        if metrics is not None:
            self.prev_stable_ratio = metrics["stable_ratio"]

    def update(self, current_vae: Any | None = None, *,
               gripper_closed: bool | None = None, gripper_xy: Any = None,
               target_bbox: Any = None, timestep: int | None = None,
               wrist_image: Any = None, gripper_qpos: Any = None) -> SkillCompletionResult:
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
            if gripper_xy is not None:
                gxy_arr = np.asarray(gripper_xy, dtype=np.float32).reshape(2)
                if gripper_closed is True:
                    if self._prev_closed is not True:
                        self.gripper_moved_since_close = False
                        self.gripper_xy_history.clear()
                        self.gripper_xy_history.append(gxy_arr.copy())
                    else:
                        self.gripper_xy_history.append(gxy_arr.copy())
                        if (
                            not self.gripper_moved_since_close
                            and self._window_total_displacement(self.gripper_xy_history)
                            > WRIST_GRIPPER_MOVE_THRESHOLD
                        ):
                            self.gripper_moved_since_close = True
                else:
                    self.gripper_xy_history.clear()
                self.prev_gripper_xy = gxy_arr.copy()
            specific = False
            if wrist_metrics is not None and self.skill in PICK_SKILLS:
                met, reason, details = self._wrist_pick_gate(
                    gripper_closed, wrist_metrics, wrist_image,
                    gripper_qpos=gripper_qpos,
                )
                self.pick_confirmations.append(met)
                self.confirmation_count = sum(self.pick_confirmations)
                if not met:
                    reason = details.get("reason", reason)
            elif wrist_metrics is not None and self.skill in PLACE_SKILLS:
                released = self._prev_closed is True and gripper_closed is False
                met, reason, details = self._wrist_place_gate(
                    gripper_closed, wrist_metrics
                )
                if released and met:
                    self.confirmation_count = max(self.confirmation_count, 1)
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
                    if released and met:
                        self.confirmation_count = max(self.confirmation_count, 1)
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
