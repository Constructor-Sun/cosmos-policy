"""Test-time, read-only skill-completion telemetry for LIBERO rollouts."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from memory_system.execute.skill_completion.pick import PickSkillCompletion
from memory_system.geometry import camera_params, depth_to_metric, flip_depth, pixel_to_world


def patch_numpy2_segmentation() -> None:
    """Fix robosuite's uint8 segmentation decode without editing site-packages."""
    from robosuite.utils import binding_utils as binding

    original = binding.MjRenderContext.read_pixels
    if getattr(original, "_cosmos_numpy2_safe", False):
        return

    def read_pixels(self, width, height, depth=False, segmentation=False):
        if not segmentation:
            return original(self, width, height, depth=depth, segmentation=False)
        viewport = binding.mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        binding.mujoco.mjr_readPixels(
            rgb=rgb, depth=depth_img, viewport=viewport, con=self.con
        )
        rgb32 = rgb.astype(np.int32)
        encoded = rgb32[:, :, 0] + rgb32[:, :, 1] * 256 + rgb32[:, :, 2] * 65536
        encoded[encoded >= self.scn.ngeom + 1] = 0
        ids = np.full((self.scn.ngeom + 1, 2), -1, dtype=np.int32)
        for index in range(self.scn.ngeom):
            geom = self.scn.geoms[index]
            if geom.segid != -1:
                ids[geom.segid + 1] = (geom.objtype, geom.objid)
        result = ids[encoded]
        return (result, depth_img) if depth else result

    read_pixels._cosmos_numpy2_safe = True
    binding.MjRenderContext.read_pixels = read_pixels


def resolve_target_instance(
    env: Any, arguments: dict[str, Any], skill: str
) -> str | None:
    """Resolve a Pick item or Place target to a visible instance name."""
    argument = str(arguments.get("item" if skill == "Pick" else "target", ""))
    instances = getattr(env, "instance_to_id", {})
    if argument in instances:
        return argument
    normalized = argument.lower().replace("_", "")
    matches = [
        name for name in instances
        if name.lower().replace("_", "") == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    matches = [
        name for name in instances
        if normalized in name.lower().replace("_", "")
    ]
    if len(matches) == 1:
        return matches[0]
    anchors = []
    for name in instances:
        stem, suffix = name.rsplit("_", 1) if "_" in name else (name, "")
        token = (stem if suffix.isdigit() else name).lower().replace("_", "")
        if len(token) >= 4 and token in normalized:
            anchors.append((len(token), name))
    return max(anchors)[1] if anchors else None


class PickTargetPointCloud:
    """Extract the visible 3-D cloud for a memory Pick item."""

    def __init__(self, env: Any, resolution: int = 256) -> None:
        self.env = env
        self.camera = camera_params(env.sim, "agentview", resolution, resolution)

    def target_instance_id(self, item: str) -> Any:
        return getattr(self.env, "instance_to_id", {}).get(str(item))

    def points(self, obs: dict[str, Any], item: str) -> np.ndarray | None:
        target_instance_id = self.target_instance_id(item)
        if target_instance_id is None:
            return None
        segmentation = obs.get("agentview_segmentation_instance")
        depth = obs.get("agentview_depth")
        if segmentation is None or depth is None:
            return None
        segmentation = np.asarray(segmentation)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            return None
        mask = np.flipud(segmentation == target_instance_id)
        pixels = np.stack(np.nonzero(mask), axis=-1)
        if len(pixels) < 4:
            return None
        metric = depth_to_metric(depth, self.camera.near, self.camera.far)
        points = pixel_to_world(pixels, flip_depth(metric), self.camera)
        points = points[np.isfinite(points).all(axis=1)]
        return points if len(points) >= 4 else None


class PickCompletionShadow:
    """Feed live observations to PickSkillCompletion without owning actions."""

    def __init__(
        self,
        env: Any,
        item: str,
        log_file: Any,
        resolution: int = 256,
        max_action_chunks: int = 7,
    ):
        self.env = env
        self.item = str(item)
        self.log_file = log_file
        self.completion = PickSkillCompletion(max_action_chunks=max_action_chunks)
        # Keep the rule checker available for existing diagnostics.
        self.checker = self.completion.checker
        self.completed_frame: int | None = None
        self.frame_count = 0
        self.point_cloud = PickTargetPointCloud(env, resolution)
        self.target_instance_id = self.point_cloud.target_instance_id(self.item)

    def _write(self, payload: dict[str, Any]) -> None:
        if self.log_file is None:
            return
        self.log_file.write("[SKILL_COMPLETION] " + json.dumps(payload) + "\n")
        self.log_file.flush()

    def _points(self, obs: dict[str, Any]) -> np.ndarray | None:
        return self.point_cloud.points(obs, self.item)

    def observe(self, obs: dict[str, Any], action: Any, frame: int) -> bool:
        """Record one post-action frame; return value must not control rollout."""
        points = self._points(obs)
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        gripper_command = bool(action_array[-1] > 0.0) if len(action_array) else False
        decision = self.completion.observe_frame(
            target_points=points,
            eef_pos=obs.get("robot0_eef_pos"),
            eef_quat=obs.get("robot0_eef_quat"),
            gripper_closed=gripper_command,
            gripper_qpos=obs.get("robot0_gripper_qpos"),
        )
        completed = decision.semantic_completed
        if completed and self.completed_frame is None:
            self.completed_frame = int(frame)
        self.frame_count += 1
        self._write(
            {
                "frame": int(frame),
                "item": self.item,
                "target_instance_id": self.target_instance_id,
                "target_points": 0 if points is None else int(len(points)),
                "gripper_command_closed": gripper_command,
                "completed": bool(completed),
                "advance": bool(decision.advance),
                "advance_reason": decision.reason,
                "action_chunks": decision.action_chunks,
                "completed_frame": self.completed_frame,
                "confirmation_count": int(self.checker.confirmation_count),
                "vertical_progress": float(self.checker.vertical_progress),
                "rigid_error": self.checker.last_rigid_error,
                "gripper_gap": self.checker.last_gripper_gap,
            }
        )
        return bool(completed)

    def finish_action_chunk(self) -> Any:
        """Apply the VLA chunk budget and return the current decision."""
        decision = self.completion.finish_action_chunk()
        self._write(
            {
                "chunk_summary": True,
                "item": self.item,
                "action_chunks": decision.action_chunks,
                "advance": bool(decision.advance),
                "semantic_completed": bool(decision.semantic_completed),
                "advance_reason": decision.reason,
            }
        )
        return decision

    def close(self) -> None:
        decision = self.completion.decision
        self._write(
            {
                "summary": True,
                "item": self.item,
                "frames": self.frame_count,
                "completed": bool(decision.semantic_completed),
                "advance": bool(decision.advance),
                "advance_reason": decision.reason,
                "action_chunks": decision.action_chunks,
                "completed_frame": self.completed_frame,
            }
        )
