"""Build explicit LIBERO-Plus variant indexes from authoritative task data."""

from __future__ import annotations

import pathlib
from typing import Any


def index_variants_by_base(base_tasks: list[str], variants: list[Any]) -> dict[str, list[Any]]:
    """Index classified variants against the authoritative clean-task names."""
    index = {base_task: [] for base_task in base_tasks}
    for variant in variants:
        matches = [
            base_task for base_task in base_tasks
            if variant.name == base_task or variant.name.startswith(base_task + "_")
        ]
        if not matches:
            raise RuntimeError(f"classified variant has no clean base task: {variant.name}")
        base_task = max(matches, key=len)
        index[base_task].append(variant)
    return index


def clean_base_tasks(suite: str) -> list[str]:
    from libero.libero import get_libero_path

    init_dir = pathlib.Path(get_libero_path("init_states")) / suite
    base_tasks = sorted(path.name.removesuffix(".pruned_init") for path in init_dir.glob("*.pruned_init"))
    if not base_tasks:
        raise RuntimeError(f"no clean task definitions found for {suite} in {init_dir}")
    return base_tasks
