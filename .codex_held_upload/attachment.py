"""One-time RGB-D attachment estimation in the hand frame.

The estimator is deliberately independent of Pick execution.  A caller feeds
candidate object points already expressed in the hand frame.  Points belonging
to the held object stay fixed in that frame while static background points move
when the hand moves.  A voxel consensus across a short capture window removes
the latter before the geometry is frozen for planning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memory_system.execute.planner.held_object.types import HeldObjectObservation
from memory_system.execute.surface_obstacles import voxel_representatives


@dataclass(frozen=True)
class HeldObjectAttachmentEstimate:
    """Result of one post-grasp attachment estimation window."""

    observation: HeldObjectObservation
    valid: bool
    reason: str
    frame_count: int
    stable_voxels: int
    hand_translation_span: float
    hand_rotation_span: float


class HeldObjectAttachmentEstimator:
    """Build a hand-local held-object cloud from a short RGB-D window."""

    def __init__(
        self,
        item: str,
        *,
        voxel_size: float = 0.005,
        min_frames: int = 3,
        min_consensus_frames: int = 2,
        consensus_ratio: float = 0.5,
        min_hand_translation: float = 0.005,
        min_hand_rotation: float = 0.03,
        min_points: int = 16,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if min_frames < 1:
            raise ValueError("min_frames must be positive")
        if min_consensus_frames < 1:
            raise ValueError("min_consensus_frames must be positive")
        if not 0.0 < consensus_ratio <= 1.0:
            raise ValueError("consensus_ratio must be in (0, 1]")
        self.item = str(item)
        self.voxel_size = float(voxel_size)
        self.min_frames = int(min_frames)
        self.min_consensus_frames = int(min_consensus_frames)
        self.consensus_ratio = float(consensus_ratio)
        self.min_hand_translation = float(min_hand_translation)
        self.min_hand_rotation = float(min_hand_rotation)
        self.min_points = int(min_points)
        self.reset()

    def reset(self) -> None:
        self._frame_points: list[np.ndarray] = []
        self._hand_poses: list[np.ndarray] = []

    @property
    def frame_count(self) -> int:
        return len(self._frame_points)

    @property
    def frame_points(self) -> tuple[np.ndarray, ...]:
        return tuple(self._frame_points)

    @staticmethod
    def _pose_parts(hand_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(hand_to_world, dtype=np.float64).reshape(4, 4)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        if not np.isfinite(pose).all():
            raise ValueError("hand_to_world must be finite")
        return rotation, translation

    def add_hand_points(
        self,
        points_hand: np.ndarray,
        hand_to_world: np.ndarray,
    ) -> bool:
        """Add one candidate cloud and its hand pose.

        ``points_hand`` may contain mask/background outliers.  They are not
        trusted until ``finalize`` finds a cross-frame consensus.
        """
        try:
            points = np.asarray(points_hand, dtype=np.float64).reshape(-1, 3)
            pose = np.asarray(hand_to_world, dtype=np.float64).reshape(4, 4)
            self._pose_parts(pose)
        except (TypeError, ValueError):
            return False
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return False
        self._frame_points.append(points)
        self._hand_poses.append(pose.copy())
        return True

    def _motion_span(self) -> tuple[float, float]:
        if not self._hand_poses:
            return 0.0, 0.0
        first = self._hand_poses[0]
        first_rotation, first_translation = self._pose_parts(first)
        translation_span = 0.0
        rotation_span = 0.0
        for pose in self._hand_poses:
            rotation, translation = self._pose_parts(pose)
            translation_span = max(
                translation_span, float(np.linalg.norm(translation - first_translation))
            )
            relative = first_rotation.T @ rotation
            cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            rotation_span = max(rotation_span, float(np.arccos(cosine)))
        return translation_span, rotation_span

    def _voxel_keys(self, points: np.ndarray) -> np.ndarray:
        return np.floor(points / self.voxel_size).astype(np.int64)

    def _invalid(
        self,
        reason: str,
        translation_span: float,
        rotation_span: float,
        stable_voxels: int = 0,
    ) -> HeldObjectAttachmentEstimate:
        observation = HeldObjectObservation(
            item=self.item,
            points_hand=np.empty((0, 3), dtype=np.float32),
            confidence=0.0,
            source="rgbd_hand_consensus",
            valid=False,
            capture_frames=self.frame_count,
            stable_fraction=0.0,
            hand_translation_span=translation_span,
            hand_rotation_span=rotation_span,
            rejection_reason=reason,
        )
        return HeldObjectAttachmentEstimate(
            observation=observation,
            valid=False,
            reason=reason,
            frame_count=self.frame_count,
            stable_voxels=stable_voxels,
            hand_translation_span=translation_span,
            hand_rotation_span=rotation_span,
        )

    def finalize(self) -> HeldObjectAttachmentEstimate:
        """Return a frozen observation or an explicit fail-closed result."""
        translation_span, rotation_span = self._motion_span()
        if self.frame_count < self.min_frames:
            return self._invalid(
                f"need at least {self.min_frames} matched frames",
                translation_span,
                rotation_span,
            )
        if (
            translation_span < self.min_hand_translation
            and rotation_span < self.min_hand_rotation
        ):
            return self._invalid(
                "insufficient hand motion to reject static background",
                translation_span,
                rotation_span,
            )

        frame_keys = [
            {tuple(key) for key in self._voxel_keys(points)}
            for points in self._frame_points
        ]
        counts: dict[tuple[int, int, int], int] = {}
        for keys in frame_keys:
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        required = max(
            self.min_consensus_frames,
            int(np.ceil(self.consensus_ratio * self.frame_count)),
        )
        required = min(required, self.frame_count)
        stable_keys = {key for key, count in counts.items() if count >= required}
        if not stable_keys:
            return self._invalid(
                "no cross-frame stable object voxels",
                translation_span,
                rotation_span,
            )

        stable_points = []
        for points in self._frame_points:
            keys = self._voxel_keys(points)
            keep = np.asarray([tuple(key) in stable_keys for key in keys], dtype=bool)
            if keep.any():
                stable_points.append(points[keep])
        if not stable_points:
            return self._invalid(
                "stable voxels contain no finite points",
                translation_span,
                rotation_span,
                len(stable_keys),
            )
        points = voxel_representatives(
            np.concatenate(stable_points, axis=0), self.voxel_size
        )
        if len(points) < self.min_points:
            return self._invalid(
                f"stable cloud has only {len(points)} points",
                translation_span,
                rotation_span,
                len(stable_keys),
            )
        stable_fraction = len(stable_keys) / max(1, len(counts))
        confidence = float(stable_fraction)
        observation = HeldObjectObservation(
            item=self.item,
            points_hand=points.astype(np.float32),
            confidence=confidence,
            source="rgbd_hand_consensus",
            valid=True,
            capture_frames=self.frame_count,
            stable_fraction=float(stable_fraction),
            hand_translation_span=translation_span,
            hand_rotation_span=rotation_span,
        )
        return HeldObjectAttachmentEstimate(
            observation=observation,
            valid=True,
            reason="ok",
            frame_count=self.frame_count,
            stable_voxels=len(stable_keys),
            hand_translation_span=translation_span,
            hand_rotation_span=rotation_span,
        )


__all__ = [
    "HeldObjectAttachmentEstimate",
    "HeldObjectAttachmentEstimator",
]
