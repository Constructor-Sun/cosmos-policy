"""Artifact loading and indexing for the memory_system package.

Step 1 keeps the current v1 artifact formats unchanged.  The loaders below are
self-contained replacements for the ad-hoc loading currently spread across
``bin/memory`` and ``bin/execute``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in arguments.items()))


def _phase_key(
    task_name: str,
    planner_step_id: int,
    skill: str,
    arguments: dict[str, Any],
) -> tuple[str, int, str, tuple[tuple[str, str], ...]]:
    return (
        str(task_name),
        int(planner_step_id),
        str(skill),
        _arguments_key(arguments),
    )


def _load_artifact(path: str | Path, expected_format: str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != expected_format:
        raise ValueError(
            f"Unsupported {expected_format} format: "
            f"{payload.get('format') if isinstance(payload, dict) else type(payload)!r}"
        )
    return payload


def _bbox_diagonal(bbox_xyxy: Any) -> float:
    x0, y0, x1, y1 = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    diagonal = float(np.hypot(x1 - x0, y1 - y0))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError(f"Invalid target bounding box: {bbox_xyxy!r}")
    return diagonal


@dataclass(frozen=True)
class ReadyDistancePrototype:
    demo_id: str
    ready_frame: int
    distance_px: float
    normalized_distance: float


class PhaseTargetMemory:
    """Index over ``libero_phase_targets_v1`` templates."""

    def __init__(self, path: str | Path):
        payload = _load_artifact(path, "libero_phase_targets_v1")
        self.path = Path(path)
        self.templates = list(payload.get("templates", []))

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        exclude_demo_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        excluded = set(exclude_demo_ids)
        expected_args = _arguments_key(arguments)
        exact = [
            item for item in self.templates
            if item["task_name"] == task_name
            and int(item["planner_step_id"]) == int(planner_step_id)
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected_args
            and item["demo_id"] not in excluded
        ]
        if exact:
            return exact
        return [
            item for item in self.templates
            if item["task_name"] == task_name
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected_args
            and item["demo_id"] not in excluded
        ]


class ReadyDistanceMemory:
    """Build ready-distance prototypes from phase targets + segment manifest."""

    def __init__(self, phase_targets: str | Path, segments_manifest: str | Path):
        target_payload = _load_artifact(phase_targets, "libero_phase_targets_v1")
        manifest = json.loads(Path(segments_manifest).read_text())

        ready_frames: dict[
            tuple[str, str, int, str, tuple[tuple[str, str], ...]], int
        ] = {}
        for record in manifest.get("records", []):
            if not record.get("valid"):
                continue
            for segment in record.get("segments", []):
                ready_frame = segment.get("ready_frame")
                if ready_frame is None:
                    continue
                key = (
                    str(record["task_name"]),
                    str(record["demo_id"]),
                    int(segment["planner_step_id"]),
                    str(segment["skill"]),
                    _arguments_key(segment.get("arguments", {})),
                )
                ready_frames[key] = int(ready_frame)

        per_phase: dict[
            tuple[str, int, str, tuple[tuple[str, str], ...]],
            dict[str, ReadyDistancePrototype],
        ] = {}
        self.templates = list(target_payload.get("templates", []))
        for template in self.templates:
            segment_key = (
                str(template["task_name"]),
                str(template["demo_id"]),
                int(template["planner_step_id"]),
                str(template["skill"]),
                _arguments_key(template.get("arguments", {})),
            )
            ready_frame = ready_frames.get(segment_key)
            if ready_frame is None or int(template["frame"]) != ready_frame:
                continue
            center = np.asarray(template["target_center_xy"], dtype=np.float32).reshape(2)
            gripper = np.asarray(template["gripper_xy"], dtype=np.float32).reshape(2)
            distance_px = float(np.linalg.norm(gripper - center))
            normalized = distance_px / _bbox_diagonal(template["bbox_xyxy"])
            phase_key = _phase_key(
                template["task_name"],
                template["planner_step_id"],
                template["skill"],
                template.get("arguments", {}),
            )
            per_phase.setdefault(phase_key, {})[str(template["demo_id"])] = (
                ReadyDistancePrototype(
                    demo_id=str(template["demo_id"]),
                    ready_frame=ready_frame,
                    distance_px=distance_px,
                    normalized_distance=normalized,
                )
            )
        self._per_phase = {
            key: tuple(sorted(values.values(), key=lambda item: item.demo_id))
            for key, values in per_phase.items()
        }

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, Any],
        exclude_demo_ids: Iterable[str] = (),
    ) -> tuple[ReadyDistancePrototype, ...]:
        excluded = set(str(item) for item in exclude_demo_ids)
        exact_key = _phase_key(task_name, planner_step_id, skill, arguments)
        exact = tuple(
            item for item in self._per_phase.get(exact_key, ())
            if item.demo_id not in excluded
        )
        if exact:
            return exact

        arguments_key = _arguments_key(arguments)
        fallback = []
        for (task, _step, candidate_skill, candidate_args), prototypes in self._per_phase.items():
            if task == task_name and candidate_skill == skill and candidate_args == arguments_key:
                fallback.extend(item for item in prototypes if item.demo_id not in excluded)
        by_demo = {item.demo_id: item for item in fallback}
        return tuple(sorted(by_demo.values(), key=lambda item: item.demo_id))


class PoseRecoveryMemory:
    """Index over ``libero_recovery_targets_v1``."""

    def __init__(self, path: str | Path):
        payload = _load_artifact(path, "libero_recovery_targets_v1")
        self.path = Path(path)
        self.targets = list(payload.get("targets", []))

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
    ) -> list[dict[str, Any]]:
        expected = _arguments_key(arguments)
        exact = [
            item for item in self.targets
            if item["task_name"] == task_name
            and int(item["planner_step_id"]) == int(planner_step_id)
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected
        ]
        if exact:
            return exact
        return [
            item for item in self.targets
            if item["task_name"] == task_name
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected
        ]


class FeasibleRecoveryMemory:
    """Index over ``libero_feasible_recovery_targets_v1``."""

    def __init__(self, path: str | Path):
        payload = _load_artifact(path, "libero_feasible_recovery_targets_v1")
        self.path = Path(path)
        self.targets = list(payload.get("targets", []))

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
    ) -> list[dict[str, Any]]:
        expected = _arguments_key(arguments)
        exact = [
            item for item in self.targets
            if item["task_name"] == task_name
            and int(item["planner_step_id"]) == int(planner_step_id)
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected
        ]
        if exact:
            return exact
        return [
            item for item in self.targets
            if item["task_name"] == task_name
            and item["skill"] == skill
            and _arguments_key(item.get("arguments", {})) == expected
        ]


class WristCompletionMemory:
    """Index over ``libero_wrist_completion_targets_v1``."""

    def __init__(self, path: str | Path):
        payload = _load_artifact(path, "libero_wrist_completion_targets_v1")
        self.path = Path(path)
        self.templates = list(payload.get("templates", []))
        self._by_phase: dict[
            tuple[str, int, str, tuple[tuple[str, str], ...]], list[dict[str, Any]]
        ] = {}
        for tpl in self.templates:
            key = _phase_key(
                tpl["task_name"], tpl["planner_step_id"], tpl["skill"], tpl.get("arguments", {})
            )
            self._by_phase.setdefault(key, []).append(tpl)

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        exclude_demo_ids: Iterable[str] = (),
    ) -> tuple[dict[str, Any], ...]:
        excluded = set(exclude_demo_ids)
        exact = self._by_phase.get(
            _phase_key(task_name, planner_step_id, skill, arguments), []
        )
        selected = [t for t in exact if t["demo_id"] not in excluded]
        if not selected:
            args_key = _arguments_key(arguments)
            selected = [
                tpl
                for (name, _step, kind, args), items in self._by_phase.items()
                if name == task_name and kind == skill and args == args_key
                for tpl in items if tpl["demo_id"] not in excluded
            ]
        by_demo = {tpl["demo_id"]: tpl for tpl in selected}
        return tuple(by_demo.values())


class WristFeasibleMemory:
    """Index over ``libero_wrist_feasible_targets_v1``."""

    def __init__(self, path: str | Path):
        payload = _load_artifact(path, "libero_wrist_feasible_targets_v1")
        self.path = Path(path)
        self.templates = list(payload.get("templates", []))
        self._by_phase: dict[
            tuple[str, int, str, tuple[tuple[str, str], ...]], list[dict[str, Any]]
        ] = {}
        for tpl in self.templates:
            key = _phase_key(
                tpl["task_name"], tpl["planner_step_id"], tpl["skill"], tpl.get("arguments", {})
            )
            self._by_phase.setdefault(key, []).append(tpl)

    def select(
        self,
        task_name: str,
        planner_step_id: int,
        skill: str,
        arguments: dict[str, str],
        exclude_demo_ids: Iterable[str] = (),
    ) -> tuple[dict[str, Any], ...]:
        excluded = set(exclude_demo_ids)
        exact = self._by_phase.get(
            _phase_key(task_name, planner_step_id, skill, arguments), []
        )
        selected = [t for t in exact if t["demo_id"] not in excluded]
        if not selected:
            args_key = _arguments_key(arguments)
            selected = [
                tpl
                for (name, _step, kind, args), items in self._by_phase.items()
                if name == task_name and kind == skill and args == args_key
                for tpl in items if tpl["demo_id"] not in excluded
            ]
        by_demo = {tpl["demo_id"]: tpl for tpl in selected}
        return tuple(by_demo.values())
