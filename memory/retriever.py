"""Memory candidate retrieval for one phase.

Selection is pure: the phase's skill decides which library to read and which
ranking policy to apply, and the phase's arguments are the exact-match key.
Everything else the intervention needs -- mapping the pose into the current
scene, planning, executing -- belongs to the executor.

Policies carried over from scripts/libero_pro/oracle_ready_eval.py: Pick keeps
the same-type fallback and prefers the smallest lift above the threshold, Place
prefers the ready pose whose tool axis is closest to vertical, and TurnOn takes
a fixed record index because its library holds 40 identical-argument records
whose first entry is a five-frame empty segment.  The first candidate returned
is the record that policy used to pick.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from memory.types import MemoryCandidate, PhaseSpec, ReplaySegment

DEFAULT_MEMORY = (
    Path(__file__).resolve().parents[1]
    / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
)
# TurnOn 记录的选择索引：库里 40 条参数完全相同，第一条是 5 帧/8.6° 的空段。
TURNON_RECORD_INDEX = 1
# Pick 记录按"合爪后抬升"挑：规则要求 ≥2cm，而库里 22% 的记录不到 2cm。
PICK_MIN_LIFT = 0.02
# TEMP experiment (2026-09-25): pin the Pick record to one memory_id, to trial a
# ready pose farther from the object (demo_0::step2: 4.82 cm to the nearest
# visible-cloud point, vs 1.68 cm for the lift-ranked head demo_6::step2).
# Empty string restores the normal lift-based ranking.  Remove after the trial.
PICK_RECORD_ID_OVERRIDE = {
    # item -> memory_id, forced to the head of that item's Pick candidates.
    "chefmate_8_frypan_1": (
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it::demo_0::step2"
    ),
    "ketchup_1": (
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket::demo_2::step1"
    ),
}
_SELECTED_SKILLS = {"TurnOn", "Pick", "PlaceIn", "PlaceOn"}


def _object_type(name: str) -> str:
    """moka_pot_2 -> moka_pot。

    memory 存的是【物体系】ready 位姿，实例编号不参与匹配；真系统的检索按点云
    形状走，同型物体天然命中。所以 Pick 也用同型回退，而不是要求实例名精确一致
    （否则 KITCHEN_SCENE8 的 moka_pot_2 在 memory 里找不到，直接崩）。
    """
    return re.sub(r"_\d+$", "", str(name))


def _place_tilt(record: dict) -> float:
    """ready pose 的工具 z 轴在物体系里偏离竖直多少（越小越"俯视放下"）。"""
    try:
        matrix = np.asarray(record["T_object_ee_ready"], dtype=np.float64).reshape(4, 4)
        tool_z = matrix[:3, :3][:, 2]
        return float(np.hypot(tool_z[0], tool_z[1]))
    except Exception:
        return float("inf")


def _post_grasp_lift(record: dict) -> float:
    """合爪之后末端还能抬高多少（物体系、米）——Pick 完成规则要的信号。"""
    try:
        seq = np.asarray(record["ee_pose_object_sequence"], dtype=np.float64)
        grip = np.asarray(record["gripper_sequence"], dtype=np.float64).reshape(-1)
        count = min(len(seq), len(grip))
        closed = np.nonzero(grip[:count] > 0)[0]
        if len(closed) == 0 or int(closed[0]) >= count - 1:
            return 0.0
        z = seq[:count, 2, 3]
        return float(z[int(closed[0]):].max() - z[int(closed[0])])
    except Exception:
        return 0.0


# 已加载 memory 记录的缓存：place 库 ~100MB，select() 每个 episode 都会调，不能反复读盘。
_RECORD_CACHE: dict[Path, list] = {}


def _records(memory_path: Path) -> list:
    import torch

    if memory_path not in _RECORD_CACHE:
        _RECORD_CACHE[memory_path] = torch.load(
            memory_path, map_location="cpu", weights_only=False
        )["records"]
    return _RECORD_CACHE[memory_path]


def _memory_paths(pick: Path) -> dict[str, Path]:
    """每个技能一条库：Pick 用主库，Place 共用 place 库，TurnOn 用 turnon 库。"""
    return {
        "Pick": pick,
        "PlaceIn": pick.with_name("pointcloud_action_memory_place.pt"),
        "PlaceOn": pick.with_name("pointcloud_action_memory_place.pt"),
        "TurnOn": pick.with_name("pointcloud_action_memory_turnon.pt"),
    }


@dataclass(frozen=True)
class RetrieverConfig:
    """Calibrated values in one place; the defaults are the current ones."""

    pick_memory: Path = DEFAULT_MEMORY
    turnon_record_index: int = TURNON_RECORD_INDEX
    pick_min_lift: float = PICK_MIN_LIFT


class MemoryRetriever:
    """Select and rank the memory records for one phase."""

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = RetrieverConfig() if config is None else config
        self.memory_paths = _memory_paths(self.config.pick_memory)

    def retrieve(self, phase: PhaseSpec) -> list[MemoryCandidate]:
        """Return the candidates for ``phase``, best first.

        Raises KeyError when the phase has no library (Open/Close) and when no
        record matches: the caller decides whether that phase falls back to the
        policy.
        """
        skill = phase.skill
        memory_path = self.memory_paths[skill]
        want = {str(key): str(value) for key, value in phase.arguments.items()}
        want_type = _object_type(want.get("item", ""))
        exact: list[dict] = []
        same_type: list[dict] = []
        for record in _records(memory_path):
            got = {
                str(key): str(value)
                for key, value in (record.get("arguments") or {}).items()
            }
            if got == want:
                if skill not in _SELECTED_SKILLS:
                    return [self._candidate(record, "exact")]
                exact.append(record)
                continue
            if skill == "Pick" and _object_type(got.get("item", "")) == want_type:
                same_type.append(record)
        if exact:
            return [self._candidate(record, "exact") for record in self._rank(skill, exact)]
        if same_type:
            return [self._candidate(record, "same_type") for record in same_type]
        raise KeyError(f"no {skill} record for {want} in {memory_path}")

    def _rank(self, skill: str, exact: list[dict]) -> list[dict]:
        """Order exact matches by the skill's policy; the head is today's pick."""
        if skill == "TurnOn":
            index = min(self.config.turnon_record_index, len(exact) - 1)
            return exact[index:] + exact[:index]
        if skill == "Pick":
            # Every entry in ``exact`` carries the same arguments, so the item
            # can be read off any of them.
            item = str((exact[0].get("arguments") or {}).get("item", "")) if exact else ""
            pinned_id = PICK_RECORD_ID_OVERRIDE.get(item, "")
            if pinned_id:
                pinned = [
                    record
                    for record in exact
                    if str(record.get("memory_id")) == pinned_id
                ]
                if pinned:
                    return pinned + [r for r in exact if r is not pinned[0]]
            ranked = sorted(exact, key=_post_grasp_lift)
            above = [
                record
                for record in ranked
                if _post_grasp_lift(record) >= self.config.pick_min_lift
            ]
            if above:
                # Exclude by identity: the records hold numpy arrays, so dict
                # equality between two different records is not usable here.
                selected = {id(record) for record in above}
                return above + [
                    record for record in ranked if id(record) not in selected
                ]
            return ranked[::-1]
        return sorted(exact, key=_place_tilt)

    @staticmethod
    def _candidate(record: dict, match: str) -> MemoryCandidate:
        return MemoryCandidate(
            record=record,
            match=match,
            rank=None,
            target_ee=np.asarray(record["T_object_ee_ready"], dtype=np.float64).reshape(4, 4),
            replay=ReplaySegment(
                ready_frame=int(record["ready_frame"]),
                ee_pose_object_sequence=np.asarray(
                    record["ee_pose_object_sequence"], dtype=np.float64
                ),
                gripper_sequence=np.asarray(
                    record["gripper_sequence"], dtype=np.float64
                ),
                action_sequence_object_physical=np.asarray(
                    record["action_sequence_object_physical"], dtype=np.float64
                ),
            ),
        )
