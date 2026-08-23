"""Skill declarations shared by offline and online memory_system components.

The argument roles follow the original LIBERO semantics:
- ``item`` is the object being manipulated (e.g. picked/placed);
- ``target`` is the destination or other target object (e.g. a plate, drawer,
  stove, cabinet).
"""
from __future__ import annotations

SKILLS: dict[str, dict[str, str | None]] = {
    "Pick": {"item_argument": "item", "target_argument": None},
    "PlaceOn": {"item_argument": "item", "target_argument": "target"},
    "PlaceIn": {"item_argument": "item", "target_argument": "target"},
    "Open": {"item_argument": None, "target_argument": "target"},
    "Close": {"item_argument": None, "target_argument": "target"},
    "TurnOn": {"item_argument": None, "target_argument": "target"},
}

def item_argument(skill: str) -> str | None:
    roles = SKILLS.get(skill)
    if roles is None:
        return None
    return roles["item_argument"]


def target_argument(skill: str) -> str | None:
    roles = SKILLS.get(skill)
    if roles is None:
        return None
    return roles["target_argument"]


def is_pick(skill: str) -> bool:
    roles = SKILLS.get(skill)
    return bool(roles and roles["item_argument"] and skill not in ("PlaceIn", "PlaceOn"))


def is_place(skill: str) -> bool:
    return skill in ("PlaceIn", "PlaceOn")
