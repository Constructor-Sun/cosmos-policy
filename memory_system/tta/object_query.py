"""RGB-D interaction-object queries for observation-only TTA diagnostics."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from memory_system.execute.skill_completion.shadow import resolve_target_instance
from memory_system.geometry import (
    camera_params,
    depth_to_metric,
    flip_depth,
    pixel_to_world,
)

CONTROL_WORDS = {
    "button",
    "knob",
    "handle",
    "switch",
    "dial",
    "burner",
    "faucet",
    "lever",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in str(value).lower().replace("-", "_").split("_")
        if token
    }


def _related_control_name(expected: str, actual: str) -> bool:
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(actual)
    shared = expected_tokens & actual_tokens
    numeric = {token for token in expected_tokens if token.isdigit()}
    return (
        bool(actual_tokens & CONTROL_WORDS)
        and len(shared) >= 1
        and numeric.issubset(actual_tokens)
    )


def interaction_aliases(
    env: Any, phase: Any
) -> tuple[str, ...]:
    """Use the existing instance resolver plus TurnOn control aliases."""
    arguments = dict(getattr(phase, "arguments", {}) or {})
    skill = str(getattr(phase, "skill", ""))
    expected = (
        arguments.get("item")
        if skill in {"Pick", "PlaceIn", "PlaceOn"}
        else arguments.get("target")
    )
    if not expected:
        return ()
    aliases = {str(expected)}
    try:
        resolved = resolve_target_instance(env, arguments, skill)
    except Exception:
        resolved = None
    if resolved:
        aliases.add(str(resolved))
    if skill == "TurnOn":
        for name in getattr(env, "instance_to_id", {}):
            if _related_control_name(str(expected), str(name)):
                aliases.add(str(name))
    return tuple(sorted(aliases))


class SceneObjectQuery:
    """Find all visible simulator instances close to the EEF.

    The query does not select the expected target first. It back-projects every
    visible instance mask and ranks candidates by the median of each object's
    point cloud relative to the current EEF position. The phase recorder then
    applies the common interaction-distance gate and target-name comparison.
    """

    def __init__(
        self,
        env: Any,
        *,
        resolution: int = 256,
        max_candidates: int = 16,
        min_points: int = 4,
    ) -> None:
        self.env = env
        self.resolution = int(resolution)
        self.max_candidates = int(max_candidates)
        self.min_points = int(min_points)
        self.camera = camera_params(
            env.sim, "agentview", self.resolution, self.resolution
        )

    def __call__(self, obs: Mapping[str, Any], eef_pos: Any, phase: Any):
        if eef_pos is None:
            return []
        segmentation = obs.get("agentview_segmentation_instance")
        depth = obs.get("agentview_depth")
        if segmentation is None or depth is None:
            return []
        try:
            eef = np.asarray(eef_pos, dtype=np.float64).reshape(3)
            if not np.isfinite(eef).all():
                return []
            segmentation = np.asarray(segmentation)
            if segmentation.ndim == 3:
                segmentation = segmentation[..., 0]
            if segmentation.ndim != 2:
                return []
            metric = depth_to_metric(
                depth, self.camera.near, self.camera.far
            )
            canonical_depth = flip_depth(metric)
        except (TypeError, ValueError):
            return []

        aliases = interaction_aliases(self.env, phase)
        candidates = []
        for name, instance_id in getattr(self.env, "instance_to_id", {}).items():
            lowered = str(name).lower()
            # The mounted arm is always within the interaction radius of its
            # own EEF and must never be an interaction candidate.
            if "panda" in lowered or "robot" in lowered:
                continue
            mask = np.flipud(segmentation == instance_id)
            pixels = np.stack(np.nonzero(mask), axis=-1)
            if len(pixels) < self.min_points:
                continue
            try:
                points = pixel_to_world(pixels, canonical_depth, self.camera)
            except (IndexError, TypeError, ValueError):
                continue
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) < self.min_points:
                continue
            # Rank by the object's median center on the horizontal plane,
            # matching how memory represents object positions; min-over-points
            # is dominated by segmentation noise on sparse clouds.
            center = np.median(points, axis=0)
            eef_xy = np.asarray(eef, dtype=np.float64).reshape(-1)[:2]
            nearest = float(np.linalg.norm(center[:2] - eef_xy))
            candidates.append(
                {
                    "name": str(name),
                    "distance": nearest,
                    "point_count": int(len(points)),
                    "center": center,
                    "aliases": aliases,
                    "source": "agentview_segmentation_rgbd",
                }
            )
        return sorted(candidates, key=lambda item: item["distance"])[
            : self.max_candidates
        ]
