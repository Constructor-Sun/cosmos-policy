"""Resolve LIBERO-plus task variants from the canonical classification file.

Task names, rather than filename suffixes, are the source of truth.  This
keeps diagnosis and repair on the same path for robot-init, background,
camera, lighting, and future perturbation categories.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class VariantSpec:
    suite: str
    task_name: str
    category: str
    condition: str
    task_id: int | None = None


def _condition_name(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _classification_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("LIBERO_TASK_CLASSIFICATION", "").strip()
    if explicit:
        candidates.append(Path(explicit))

    libero_root = os.environ.get("LIBERO_ROOT", "").strip()
    if libero_root:
        candidates.append(
            Path(libero_root) / "libero" / "libero" / "benchmark" / "task_classification.json"
        )

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(
        repo_root.parent
        / "LIBERO-plus"
        / "libero"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )
    for entry in sys.path:
        if entry:
            candidates.append(
                Path(entry) / "libero" / "libero" / "benchmark" / "task_classification.json"
            )
    return candidates


def classification_path() -> Path:
    seen: set[Path] = set()
    for candidate in _classification_candidates():
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in seen)
    raise FileNotFoundError(
        "LIBERO-plus task_classification.json was not found. Searched:\n  " + searched
    )


@lru_cache(maxsize=None)
def _load_classification(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_variant(task_name: str, suite: str = "libero_10") -> VariantSpec:
    """Return the unique classification entry for an exact task name."""
    path = classification_path()
    classification = _load_classification(str(path))
    if suite not in classification:
        raise ValueError(f"Suite {suite!r} is absent from {path}")

    matches = [entry for entry in classification[suite] if entry.get("name") == task_name]
    if not matches:
        raise ValueError(
            f"Task {task_name!r} is absent from suite {suite!r} in {path}; "
            "refusing to fall back to a different variant"
        )
    if len(matches) != 1:
        categories = [entry.get("category") for entry in matches]
        raise ValueError(
            f"Task {task_name!r} has {len(matches)} classification entries: {categories}"
        )

    entry = matches[0]
    category = str(entry["category"])
    task_id = entry.get("id")
    return VariantSpec(
        suite=suite,
        task_name=task_name,
        category=category,
        condition=_condition_name(category),
        task_id=int(task_id) if task_id is not None else None,
    )
