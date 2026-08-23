"""Skill-completion registry and facade for memory_system.execute.

The public ``SkillCompletionVerifier`` preserves the orchestration API of the
old ``LiberoSkillCompletionVerifierGeo`` while delegating each skill to a
dedicated public verifier class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from memory_system.execute.skill_completion._common import (
    COMPLETION_UNKNOWN,
    SKILL_COMPLETE,
    BaseCompletionVerifier,
)
from memory_system.execute.skill_completion.open_close import OpenCloseCompletionVerifier
from memory_system.execute.skill_completion.pick import PickCompletionVerifier
from memory_system.execute.skill_completion.place import PlaceCompletionVerifier
from memory_system.types import CompletionResult


SKILL_REGISTRY: dict[str, type] = {
    "Pick": PickCompletionVerifier,
    "PlaceIn": PlaceCompletionVerifier,
    "PlaceOn": PlaceCompletionVerifier,
    "Open": OpenCloseCompletionVerifier,
    "Close": OpenCloseCompletionVerifier,
}


class SkillCompletionVerifier:
    """Facade that dispatches to the per-skill completion verifier."""

    def __init__(
        self,
        phase_targets: str | Path,
        wrist_completion_targets: str | Path | None = None,
    ):
        self.phase_targets = phase_targets
        self.wrist_completion_targets = wrist_completion_targets
        self._inner: BaseCompletionVerifier | None = None

    def reset(
        self,
        task: str,
        step: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> None:
        cls = SKILL_REGISTRY.get(skill, OpenCloseCompletionVerifier)
        self._inner = cls(self.phase_targets, self.wrist_completion_targets)
        self._inner.reset(task, step, skill, arguments, exclude_demo_ids)

    def calibrate(self, current_vae: Any | None = None) -> None:
        if self._inner is not None:
            self._inner.calibrate(current_vae)

    def freeze_baseline(self) -> None:
        if self._inner is not None:
            self._inner.freeze_baseline()

    def observe_wrist(self, wrist_image: Any = None) -> None:
        if self._inner is not None:
            self._inner.observe_wrist(wrist_image)

    def evaluate_close_edge(
        self,
        gripper_closed: bool | None,
        gripper_xy: Any = None,
        timestep: int | None = None,
    ) -> None:
        if self._inner is not None:
            self._inner.evaluate_close_edge(gripper_closed, gripper_xy, timestep)

    def begin_wrong_grasp_recovery(self) -> None:
        if self._inner is not None:
            self._inner.begin_wrong_grasp_recovery()

    def update(
        self,
        current_vae: Any | None = None,
        *,
        gripper_closed: bool | None = None,
        gripper_xy: Any = None,
        target_bbox: Any = None,
        timestep: int | None = None,
        wrist_image: Any = None,
        gripper_qpos: Any = None,
        eef_pos: Any = None,
    ) -> CompletionResult:
        if self._inner is None:
            raise RuntimeError("reset() must be called before update()")
        return self._inner.update(
            current_vae,
            gripper_closed=gripper_closed,
            gripper_xy=gripper_xy,
            target_bbox=target_bbox,
            timestep=timestep,
            wrist_image=wrist_image,
            gripper_qpos=gripper_qpos,
            eef_pos=eef_pos,
        )

    def __getattr__(self, name: str):
        # Only called when normal lookup fails.
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


__all__ = [
    "COMPLETION_UNKNOWN",
    "SKILL_COMPLETE",
    "OpenCloseCompletionVerifier",
    "PickCompletionVerifier",
    "PlaceCompletionVerifier",
    "SKILL_REGISTRY",
    "SkillCompletionVerifier",
]
