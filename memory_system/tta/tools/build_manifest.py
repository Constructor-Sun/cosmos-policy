"""One-time data prep: build the chosen/rejected pair manifest.

Draft mode emits pair identity + labels from the screening meta and the repair
summary. With --chosen-dir/--rejected-dir it attaches the actual hdf5 files and
VALIDATES each pair: required datasets, aligned stream lengths, finite values,
expected dims, and the success label in the h5 attrs (chosen must have
succeeded, rejected must have failed — the attr is ground truth, the manifest
label is cross-checked). The obs[k]-before-action[k] timing property is NOT
checkable from the h5; it is guaranteed by the capture loops' write order.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def load_json(path: Path):
    return json.load(open(path))


def build_pairs(screening_meta: dict, repair_summary: dict) -> tuple[list[dict], dict]:
    stats = {"repair_cases": len(repair_summary.get("cases", [])), "paired": 0, "repair_failed": 0}
    pairs = []
    for case in repair_summary.get("cases", []):
        tag = case["tag"]
        meta = screening_meta.get(tag)
        if meta is None:
            continue
        if not case.get("task_success", False):
            stats["repair_failed"] += 1
            continue
        pairs.append({
            "pair_id": tag,
            "task": meta.get("task"),
            "init": meta.get("init"),
            "chosen_success": True,
            "rejected_success": False,
            "chosen_path": None,
            "rejected_path": None,
        })
        stats["paired"] += 1
    return pairs, stats


def attach_paths(pairs: list[dict], chosen_dir: Path | None, rejected_dir: Path | None) -> None:
    for pair in pairs:
        tag = pair["pair_id"]
        if chosen_dir and pair["chosen_path"] is None:
            m = sorted(chosen_dir.glob(f"*--{tag}--*success=True*.hdf5"))
            pair["chosen_path"] = str(m[0]) if m else None
        if rejected_dir and pair["rejected_path"] is None:
            m = sorted(rejected_dir.glob(f"*{tag}*success=False*.hdf5"))
            pair["rejected_path"] = str(m[0]) if m else None


def check_episode_file(path: str, expect_success: bool) -> list[str]:
    """Manifest-level validation: required fields, aligned stream lengths,
    shapes, finite values, and the success attr (must exist and match)."""
    problems = []
    with h5py.File(path, "r") as f:
        def count(*names):
            for name in names:
                if name in f:
                    return len(f[name])
            return None

        n_primary = count("primary_images_jpeg", "primary_images")
        n_wrist = count("wrist_images_jpeg", "wrist_images")
        if n_primary is None:
            problems.append("missing primary images dataset")
        if n_wrist is None:
            problems.append("missing wrist images dataset")
        if "actions" not in f or "proprio" not in f:
            problems.append("missing actions/proprio dataset")
            return problems
        actions, proprio = f["actions"][:], f["proprio"][:]
        lengths = {"primary": n_primary, "wrist": n_wrist,
                   "actions": len(actions), "proprio": len(proprio)}
        if len(set(lengths.values())) != 1:
            problems.append(f"stream lengths differ: {lengths}")
        if actions.ndim != 2 or actions.shape[1] != 7:
            problems.append(f"actions shape {actions.shape} != (T, 7)")
        elif not np.isfinite(actions).all():
            problems.append("actions contain non-finite values")
        if proprio.ndim != 2 or proprio.shape[1] != 9:
            problems.append(f"proprio shape {proprio.shape} != (T, 9)")
        elif not np.isfinite(proprio).all():
            problems.append("proprio contains non-finite values")
        if "success" not in f.attrs:
            problems.append("missing success attr")
            return problems
        success = bool(f.attrs["success"])
    if expect_success != success:
        problems.append(f"h5 success={success} but manifest expects {expect_success}: {path}")
    return problems


def validate(pairs: list[dict]) -> list[str]:
    errors = []
    for pair in pairs:
        if pair.get("status") == "repair_failed_no_chosen":
            continue
        missing = [key for key in ("chosen_path", "rejected_path") if not pair.get(key)]
        if missing:
            # an incomplete pair must NOT slip through: the dataset would try
            # to open None (review finding 4, 2026-09-09)
            errors.append(f"{pair['pair_id']}: incomplete pair, missing {missing}")
            continue
        errors += [f"{pair['pair_id']}: {p}" for p in check_episode_file(pair["chosen_path"], True)]
        errors += [f"{pair['pair_id']}: {p}" for p in check_episode_file(pair["rejected_path"], False)]
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/validate the TTA pair manifest")
    parser.add_argument("--screening-meta", default=str(REPO / "memory_system/pointcloud_action/results/tta_screening_meta.json"))
    parser.add_argument("--repair-summary", default=str(REPO / "memory_system/pointcloud_action/results/tta_repair_68_summary.json"))
    parser.add_argument("--chosen-dir", default="")
    parser.add_argument("--rejected-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--validate-only", default="")
    args = parser.parse_args()

    if args.validate_only:
        errors = validate(load_json(Path(args.validate_only))["pairs"])
        print(json.dumps({"errors": errors}, indent=1, ensure_ascii=False))
        sys.exit(1 if errors else 0)

    out = Path(args.out)
    if not out or out.exists():
        raise FileExistsError("--out required and must not already exist")

    pairs, stats = build_pairs(load_json(Path(args.screening_meta)), load_json(Path(args.repair_summary)))
    attach_paths(pairs, Path(args.chosen_dir) if args.chosen_dir else None,
                 Path(args.rejected_dir) if args.rejected_dir else None)
    complete = sum(1 for p in pairs if p.get("chosen_path") and p.get("rejected_path"))
    manifest = {
        "created": datetime.now().isoformat(),
        "stats": {**stats, "complete_pairs": complete},
        "pairs": pairs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[build_manifest] {out}: {stats}, complete_pairs={complete}")


if __name__ == "__main__":
    main()
