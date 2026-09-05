"""Discover Pick-target objects from LIBERO BDDL suites.

The goal is to enumerate (suite, object type) pairs that are actually intended
to be picked/moved in at least one task of the four LIBERO suites.  We keep the
original BDDL as the source so each object can be tested in its native scene.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from memory_system.pointcloud_action.offline.single_object_scene import (
    _fixtures,
    _parse_section_entries,
    _section_text,
)

# LIBERO benchmark suites considered here.
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Object types that are containers/furniture rather than Pick targets.
NON_PICKABLE_TYPES = {
    "basket",
    "plate",
    "tray",
    "desk_caddy",
    "caddy",
    "flat_stove",
    "stove",
    "cabinet",
    "drawer",
    "top_drawer",
    "bottom_drawer",
    "middle_drawer",
    "microwave",
    "shelf",
    "wine_rack",
}


@dataclass
class PickObject:
    """A (suite, object type) pair with a representative original BDDL."""

    object_type: str
    object_name: str
    suite: str
    task_name: str
    bddl_path: Path

    def to_dict(self) -> dict:
        return {
            "object_type": self.object_type,
            "object_name": self.object_name,
            "suite": self.suite,
            "task_name": self.task_name,
            "bddl_path": str(self.bddl_path),
        }


def _object_section_mapping(text: str) -> dict[str, str]:
    """Return {object_name: object_type} from a BDDL's (:objects ...)."""
    mapping: dict[str, str] = {}
    object_text = _section_text(text, "objects")
    for line in object_text.splitlines():
        line = line.strip()
        if " - " not in line:
            continue
        names, obj_type = line.split(" - ", 1)
        obj_type = obj_type.strip()
        for name in names.split():
            mapping[name] = obj_type
    return mapping


def _obj_of_interest_names(text: str) -> set[str]:
    names: set[str] = set()
    obj_text = _section_text(text, "obj_of_interest")
    for token in re.findall(r"[\w]+", obj_text):
        if token not in {"obj_of_interest"}:
            names.add(token)
    return names


def _object_support_fixture(text: str, object_name: str) -> str | None:
    """Return the support fixture if this object starts directly on it."""
    init_text = _section_text(text, "init")
    for block in _parse_section_entries(init_text):
        m = re.match(
            rf"\(\s*On\s+{re.escape(object_name)}\s+([\w]+)\s*\)", block.strip()
        )
        if not m:
            continue
        full_region = m.group(1)
        region_text = _section_text(text, "regions")
        for region_block in _parse_section_entries(region_text):
            region_name_match = re.match(r"\(\s*([\w]+)", region_block)
            if not region_name_match:
                continue
            region_name = region_name_match.group(1)
            if full_region == region_name or full_region.endswith(
                "_" + region_name
            ):
                target_match = re.search(r"\(\s*:target\s+([\w]+)\s*\)", region_block)
                if target_match:
                    return target_match.group(1)
    return None


def discover_pick_objects(
    bddl_root: str | Path | None = None,
    suites: tuple[str, ...] = SUITES,
) -> list[PickObject]:
    """Discover one representative BDDL per (suite, object type)."""
    if bddl_root is None:
        bddl_root = Path(
            "/data1/liu/exp/counterfactual/external/LIBERO-plus/libero/libero/bddl_files"
        )
    bddl_root = Path(bddl_root)
    discovered: dict[tuple[str, str], PickObject] = {}

    for suite in suites:
        suite_dir = bddl_root / suite
        if not suite_dir.is_dir():
            continue
        # Sort for deterministic selection.
        for bddl_path in sorted(suite_dir.glob("*.bddl")):
            try:
                text = bddl_path.read_text()
                objects = _object_section_mapping(text)
                fixtures = _fixtures(text)
                interests = _obj_of_interest_names(text)
            except Exception:
                continue

            for object_name in interests:
                if object_name in fixtures:
                    continue
                obj_type = objects.get(object_name)
                if obj_type is None or obj_type in NON_PICKABLE_TYPES:
                    continue
                if _object_support_fixture(text, object_name) is None:
                    continue
                key = (suite, obj_type)
                if key not in discovered:
                    discovered[key] = PickObject(
                        object_type=obj_type,
                        object_name=object_name,
                        suite=suite,
                        task_name=bddl_path.stem,
                        bddl_path=bddl_path,
                    )

    # Sort by suite then object type for readability.
    return [
        discovered[key]
        for key in sorted(discovered, key=lambda item: (item[0], item[1]))
    ]


if __name__ == "__main__":
    for obj in discover_pick_objects():
        print(obj.to_dict())
