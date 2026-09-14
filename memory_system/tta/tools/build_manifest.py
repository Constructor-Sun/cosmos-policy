"""One-time data prep: build the chosen/rejected pair manifest.

Pairs are formed from the FILES present in --chosen-dir/--rejected-dir
(chosen ∩ rejected by tag); the h5 success attrs are the ground truth and
are validated per pair: required datasets, aligned stream lengths, finite
values, expected dims, and the success label (chosen must have succeeded,
rejected must have failed). The obs[k]-before-action[k] timing property is
NOT checkable from the h5; it is guaranteed by the capture loops' write
order.
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


import re

TAG_RE = re.compile(r"--([A-Za-z0-9]+-init\d+)--")


def tags_from_dir(dir_path: Path, expected_success: str) -> dict:
    """tag -> file path, from flat episode filenames (…--<tag>--success=<bool>…)."""
    out = {}
    for p in sorted(dir_path.glob("*.hdf5")):
        m = TAG_RE.search(p.name)
        if not m or f"success={expected_success}" not in p.name:
            continue
        out.setdefault(m.group(1), str(p))
    return out


def build_pairs(screening_meta: dict, chosen_dir: Path, rejected_dir: Path,
                repair_summary: dict, diagnosis_root: Path) -> tuple[list[dict], dict]:
    """Pair by FILE AVAILABILITY (chosen ∩ rejected), not by the repair
    summary. The summary's success set drifts from the files on disk (6
    summary-success cases have no chosen file after the v2 re-collection
    drifted; 5 summary-failure cases DO have one — accounting bug found
    2026-09-13). The h5 success attrs stay the ground truth and are
    re-checked per-file by validate().

    Each pair also carries:
      t_star    — repair cut-in (repair summary; documented fallback t*=10).
      phase_end — end of the failing phase on the baseline timeline
                  (phase_record.json). The DPO contrast window is
                  [t_star, phase_end): inside it, chosen = memory repair
                  execution and rejected = the failure unfolding — the causal
                  intervention. AFTER phase_end both sides are the VLA's own
                  continuations (on good vs bad states); using them as
                  preference material only teaches state discrimination
                  (64% of chunks under the pre-v3 pairing — found 2026-09-13).
                  Pairs without a phase record are excluded."""
    t_star_by_tag = {c["tag"]: c.get("t_star") for c in repair_summary.get("cases", [])}
    chosen = tags_from_dir(chosen_dir, "True")
    rejected = tags_from_dir(rejected_dir, "False")
    pairs, stats = [], {
        "chosen_files": len(chosen), "rejected_files": len(rejected),
        "paired": 0, "no_chosen": [], "no_rejected": [], "unknown_tag": [],
        "t_star_fallback": 0, "no_phase_record": [],
    }
    for tag in sorted(set(chosen) | set(rejected)):
        meta = screening_meta.get(tag)
        if meta is None:
            stats["unknown_tag"].append(tag)
            continue
        if tag not in chosen:
            stats["no_chosen"].append(tag)
            continue
        if tag not in rejected:
            stats["no_rejected"].append(tag)
            continue
        t_star = t_star_by_tag.get(tag)
        if isinstance(t_star, int):
            t_star = max(1, int(t_star))
        else:
            t_star = 10
            stats["t_star_fallback"] += 1
        phase_end = None
        record_path = diagnosis_root / meta["task"] / meta["init"] / "phase_record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            candidate = record.get("candidate") or {}
            phase_index = candidate.get("phase_index")
            phase = next((x for x in record.get("phases", [])
                          if x.get("phase_index") == phase_index), None)
            if phase is not None:
                phase_end = phase.get("end_step")
                if phase_end is None:
                    phase_end = record.get("total_action_steps")
        if phase_end is None:
            stats["no_phase_record"].append(tag)
            continue
        pairs.append({
            "pair_id": tag,
            "task": meta.get("task"),
            "init": meta.get("init"),
            "chosen_success": True,
            "rejected_success": False,
            "chosen_path": chosen[tag],
            "rejected_path": rejected[tag],
            "t_star": t_star,
            "phase_end": int(phase_end),
        })
        stats["paired"] += 1
    return pairs, stats


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
    parser.add_argument("--diagnosis-root", default=str(REPO / "memory_system/pointcloud_action/results/tta_failure_screening_68"))
    parser.add_argument("--chosen-dir", default=str(REPO / "training/tta_sft_success_v2"))
    parser.add_argument("--rejected-dir", default=str(REPO / "experiments/tta/dpo_rejected_images"))
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

    pairs, stats = build_pairs(load_json(Path(args.screening_meta)),
                               Path(args.chosen_dir), Path(args.rejected_dir),
                               load_json(Path(args.repair_summary)),
                               Path(args.diagnosis_root))
    manifest = {
        "created": datetime.now().isoformat(),
        "stats": stats,
        "pairs": pairs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[build_manifest] {out}: {stats}")


if __name__ == "__main__":
    main()
