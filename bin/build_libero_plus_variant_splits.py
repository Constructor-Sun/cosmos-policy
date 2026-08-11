#!/usr/bin/env python3
"""Split non-registered LIBERO-Plus resources into deterministic train/val pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONDITIONS = ("camera", "background", "light", "noise", "language")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_order(ids: list[str], seed: int, task: str, condition: str) -> list[str]:
    return sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{task}:{condition}:{value}".encode()).digest())


def build(catalog: dict[str, Any], seed: int, val_fraction: float) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    tasks = {}
    for task_name, task in catalog["tasks"].items():
        tasks[task_name] = {}
        for condition in CONDITIONS:
            records = task["variants"][condition]
            external = [record["variant_id"] for record in records if record["official"]]
            candidates = stable_order(
                [record["variant_id"] for record in records if not record["official"]],
                seed, task_name, condition,
            )
            if not candidates:
                raise RuntimeError(f"no non-official {condition} variants for {task_name}")
            if len(candidates) == 1:
                train = val = candidates
            else:
                val_count = max(1, min(round(len(candidates) * val_fraction), len(candidates) - 1))
                train, val = candidates[:-val_count], candidates[-val_count:]
            if (set(train) | set(val)) & set(external):
                raise RuntimeError(f"split overlap for {task_name}/{condition}")
            tasks[task_name][condition] = {
                "train": train,
                "val": val,
                "external_test": external,
                "train_val_variant_overlap": bool(set(train) & set(val)),
            }
    return {
        "schema_version": 1,
        "catalog_sha256": canonical_sha256(catalog),
        "split_seed": seed,
        "val_fraction": val_fraction,
        "official_variants_reserved_for_external_test": True,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(ROOT / "configs/libero_plus_variant_catalog.json"))
    parser.add_argument("--output", default=str(ROOT / "configs/libero_plus_variant_splits.json"))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    catalog = json.loads(pathlib.Path(args.catalog).read_text(encoding="utf-8"))
    splits = build(catalog, args.split_seed, args.val_fraction)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
