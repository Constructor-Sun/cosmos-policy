"""Accumulate hand-local object geometry from masked RGB-D frames."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from memory_system.execute.planner.held_object.mask_matcher import MemoryTemplateMatcher
from memory_system.execute.planner.held_object.attachment import (
    HeldObjectAttachmentEstimate,
    HeldObjectAttachmentEstimator,
)
from memory_system.execute.planner.held_object.observation_store import (
    HeldObjectObservationStore,
)
from memory_system.execute.planner.held_object.types import HeldObjectObservation
from memory_system.execute.surface_obstacles import voxel_representatives
from memory_system.geometry import pixel_to_world
from memory_system.types import CameraParams


class HeldObjectGeometryBuilder:
    """Collect object points while it is visible and convert them to hand frame."""

    def __init__(
        self,
        matcher: MemoryTemplateMatcher,
        item: str,
        voxel_size: float = 0.005,
    ) -> None:
        self.matcher = matcher
        self.item = item
        self.voxel_size = float(voxel_size)
        self._points: list[np.ndarray] = []
        self._hand_poses: list[np.ndarray] = []
        self._frames = 0
        self._matched_frames = 0

    def reset(self) -> None:
        self._points = []
        self._hand_poses = []
        self._frames = 0
        self._matched_frames = 0

    @property
    def matched_frames(self) -> int:
        return self._matched_frames

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: CameraParams,
        template: dict[str, Any],
        hand_to_world: np.ndarray,
    ) -> bool:
        """Add one frame. ``hand_to_world`` maps hand-local points to world."""
        self._frames += 1
        match = self.matcher.match(rgb, template)
        if match is None:
            return False
        mask = np.asarray(match.mask)
        if mask.shape != depth.shape[:2]:
            return False
        pixels = np.stack(np.nonzero(mask), axis=-1)
        if len(pixels) < 16:
            return False
        world = pixel_to_world(pixels, depth, camera)
        world = world[np.isfinite(world).all(axis=1)]
        if len(world) < 8:
            return False
        hand_to_world = np.asarray(hand_to_world, dtype=np.float64).reshape(4, 4)
        rotation = hand_to_world[:3, :3]
        translation = hand_to_world[:3, 3]
        hand = (rotation.T @ (world - translation).T).T
        self._points.append(hand)
        self._hand_poses.append(hand_to_world.copy())
        self._matched_frames += 1
        return True

    @property
    def frame_points(self) -> tuple[np.ndarray, ...]:
        """Candidate hand-frame clouds retained for diagnostics/consensus."""
        return tuple(self._points)

    def build_stable(
        self,
        estimator: HeldObjectAttachmentEstimator | None = None,
    ) -> HeldObjectAttachmentEstimate:
        """Finalize a short multi-frame hand-frame attachment estimate."""
        estimator = estimator or HeldObjectAttachmentEstimator(
            self.item, voxel_size=self.voxel_size
        )
        estimator.reset()
        for points, pose in zip(self._points, self._hand_poses):
            estimator.add_hand_points(points, pose)
        estimate = estimator.finalize()
        if estimate.valid and self._frames > 0 and self._matched_frames < self._frames:
            observation = estimate.observation
            observation = HeldObjectObservation(
                item=observation.item,
                points_hand=observation.points_hand,
                confidence=observation.confidence
                * self._matched_frames
                / self._frames,
                source=observation.source,
                valid=observation.valid,
                capture_frames=observation.capture_frames,
                stable_fraction=observation.stable_fraction,
                hand_translation_span=observation.hand_translation_span,
                hand_rotation_span=observation.hand_rotation_span,
                rejection_reason=observation.rejection_reason,
            )
            estimate = HeldObjectAttachmentEstimate(
                observation=observation,
                valid=estimate.valid,
                reason=estimate.reason,
                frame_count=estimate.frame_count,
                stable_voxels=estimate.stable_voxels,
                hand_translation_span=estimate.hand_translation_span,
                hand_rotation_span=estimate.hand_rotation_span,
            )
        return estimate

    def build(self) -> HeldObjectObservation | None:
        if not self._points:
            return None
        points = np.concatenate(self._points, axis=0)
        points = voxel_representatives(points, self.voxel_size)
        if len(points) < 4:
            return None
        confidence = min(1.0, self._matched_frames / max(1, self._frames))
        return HeldObjectObservation(
            item=self.item,
            points_hand=points.astype(np.float32),
            confidence=float(confidence),
            source="memory_template_rgbd",
        )


def _tighten_observation(
    observation: HeldObjectObservation,
    eps: float = 0.01,
    min_samples: int = 10,
) -> HeldObjectObservation:
    """Remove background/outlier points from a hand-local observation.

    The memory template mask is often not a tight object mask, so the raw
    hand-local points can include nearby table/background points.  This is a
    one-time Pick-stage cleanup: keep only the largest 3D connected component
    of the hand-local point cloud.
    """
    points = np.asarray(observation.points_hand, dtype=np.float64).reshape(-1, 3)
    if len(points) < min_samples:
        return observation
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    if not (labels >= 0).any():
        return observation
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    best_label = unique[int(np.argmax(counts))]
    kept = points[labels == best_label]
    if len(kept) < 4:
        return observation
    return HeldObjectObservation(
        item=observation.item,
        points_hand=kept.astype(np.float32),
        confidence=observation.confidence,
        source=observation.source,
    )


def build_and_save_held_object_observation(
    *,
    matcher: MemoryTemplateMatcher,
    item: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: CameraParams,
    template: dict[str, Any],
    hand_to_world: np.ndarray,
    store: HeldObjectObservationStore,
    task_name: str,
    demo_id: str,
    voxel_size: float = 0.005,
) -> HeldObjectObservation | None:
    """Build a hand-local observation from one memory-mask frame and save it.

    This is the Pick-stage entry point.  The saved observation is later loaded
    and passed directly to ``HeldObjectPlanner``; no per-frame re-matching or
    connected-component re-extraction is needed at planning time.
    """
    builder = HeldObjectGeometryBuilder(matcher, item, voxel_size=voxel_size)
    if not builder.update(rgb, depth, camera, template, hand_to_world):
        return None
    observation = builder.build()
    if observation is None:
        return None
    observation = _tighten_observation(observation)
    store.save(observation, task_name, demo_id, item)
    return observation
