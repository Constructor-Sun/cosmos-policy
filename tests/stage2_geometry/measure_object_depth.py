"""Stage 2: per-pixel depth validation over target-object masks.

For each (task, demo, segment frame, object): replay the demo state, take
the object's mask from simulator segmentation (given target region), and for
every (eroded) mask pixel compare the RGB-D metric depth against the
simulator-geometry surface-depth Oracle (MuJoCo mj_ray):

    per-pixel error = |obs metric depth - oracle surface depth|

Supported skills and measured objects:
- Pick: measure `arguments.item`
- PlaceIn: measure `arguments.item` + resolvable `arguments.target`
- PlaceOn: measure `arguments.item` + resolvable `arguments.target`
- TurnOn: measure `arguments.target`
- Close: measure resolvable `arguments.target`

Primary metric: median / P90 of per-pixel errors (median < 5 mm, P90 < 2 cm).

Usage (cosmospolicy env, single-env-at-a-time):

    python tests/stage2_geometry/measure_object_depth.py --max-demos 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    TASKS,
    create_seg_env,
    demo_hdf5,
    instance_mask,
    load_manifest,
    patch_numpy2_segmentation,
    phase_frames,
    ray_surface_depth,
)

TOPM = 0.005  # median target tolerance (5 mm)
TOPP = 0.020  # P90 target tolerance (2 cm)

SUPPORTED_SKILLS = ("Pick", "PlaceIn", "PlaceOn", "TurnOn", "Close")


def _resolve_instance(env, name: str | None) -> str | None:
    """Resolve an argument to a segmentable simulator instance.

    Handles both direct instance names (plate_1, microwave_1) and region
    names that embed an instance name (basket_1_contain_region ->
    basket_1, white_cabinet_1_bottom_region -> white_cabinet_1).
    """
    if not name:
        return None
    if name in env.instance_to_id:
        return name
    for inst in env.instance_to_id:
        if name.startswith(inst):
            return inst
    return None


def objects_to_measure(env, skill: str, arguments: dict | None) -> list[tuple[str, str]]:
    """Return [(object_name, role), ...] for a segment."""
    args = arguments or {}
    out: list[tuple[str, str]] = []
    if skill in ("Pick", "PlaceIn", "PlaceOn"):
        item = args.get("item")
        if item:
            out.append((item, "item"))
    if skill in ("PlaceIn", "PlaceOn", "TurnOn", "Close"):
        target = _resolve_instance(env, args.get("target"))
        if target:
            out.append((target, "target"))
    # Deduplicate identical (object, role) pairs.
    return list(dict.fromkeys(out))


def collect(task: str, max_demos: int) -> list[dict]:
    manifest = load_manifest()
    records = [r for r in manifest["records"] if r.get("valid") and r["task_name"] == task][:max_demos]

    patch_numpy2_segmentation()
    import h5py

    env = create_seg_env(task)
    env.reset()
    h5 = h5py.File(demo_hdf5(task), "r")
    samples = []
    try:
        for record in records:
            demo_id = record["demo_id"]
            group = h5["data"][demo_id]
            states = group["states"]
            length = len(states)
            for segment in record["segments"]:
                if segment.get("status") == "already_satisfied":
                    continue

                skill = segment["skill"]
                if skill not in SUPPORTED_SKILLS:
                    continue
                for obj, role in objects_to_measure(env, skill, segment.get("arguments")):
                    for frame in phase_frames(segment, length):
                        obs = env.regenerate_obs_from_state(states[frame])
                        mask = instance_mask(env, obs, obj)
                        if mask is None or mask.sum() < 16:
                            samples.append({"ok": False, "reason": "mask_too_small",
                                            "task": task, "demo": demo_id, "skill": skill,
                                            "object": obj, "role": role, "frame": int(frame)})
                            continue
                        t0 = time.time()
                        obs_z, oracle, n_excluded = ray_surface_depth(env, obs, mask, obj)
                        if obs_z is None or len(obs_z) < 8:
                            samples.append({"ok": False, "reason": "no_valid_pixels",
                                            "task": task, "demo": demo_id, "skill": skill,
                                            "object": obj, "role": role, "frame": int(frame)})
                            continue
                        errs = np.abs(obs_z - oracle)
                        samples.append({
                            "ok": True, "task": task, "demo": demo_id, "skill": skill,
                            "object": obj, "role": role, "frame": int(frame),
                            "n_pixels": int(len(obs_z)),
                            "n_excluded": int(n_excluded),
                            "median_mm": float(np.median(errs) * 1e3),
                            "p90_mm": float(np.percentile(errs, 90) * 1e3),
                            "max_mm": float(np.max(errs) * 1e3),
                            "ms": int((time.time() - t0) * 1e3),
                        })
    finally:
        h5.close()
        env.close()
    return samples


def aggregate(samples: list[dict]) -> dict:
    ok = [s for s in samples if s.get("ok")]
    by_obj: dict[str, list[float]] = {}
    for s in ok:
        by_obj.setdefault(f"{s['task'].split('_SCENE')[0]}::{s['object']}", []).append(s["p90_mm"])
    items = {}
    for k, p90s in sorted(by_obj.items()):
        meds = [s["median_mm"] for s in ok if f"{s['task'].split('_SCENE')[0]}::{s['object']}" == k]
        items[k] = {
            "n": len(meds),
            "median_of_median_mm": float(np.median(meds)),
            "median_of_p90_mm": float(np.median(p90s)),
            "max_p90_mm": float(np.max(p90s)),
        }
    all_med = [s["median_mm"] for s in ok]
    all_p90 = [s["p90_mm"] for s in ok]
    return {
        "n_samples": len(samples),
        "n_ok": len(ok),
        "n_failed": len(samples) - len(ok),
        "failures": [s for s in samples if not s.get("ok")],
        "overall": {
            "median_of_median_mm": float(np.median(all_med)) if all_med else None,
            "median_of_p90_mm": float(np.median(all_p90)) if all_p90 else None,
            "max_p90_mm": float(np.max(all_p90)) if all_p90 else None,
        },
        "pass_median": bool(all_med and np.median(all_med) <= TOPM * 1e3),
        "pass_p90": bool(all_p90 and np.median(all_p90) <= TOPP * 1e3),
        "by_item": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--max-demos", type=int, default=2)
    parser.add_argument("--out", default="tests/stage2_geometry/results_object_depth.json")
    args = parser.parse_args()

    all_samples = []
    for task in args.tasks:
        print(f"[{task[:40]}...] collecting ...", flush=True)
        all_samples.extend(collect(task, args.max_demos))

    agg = aggregate(all_samples)
    print(json.dumps(agg["overall"], indent=2))
    print("by object (median of per-frame medians / median of per-frame P90):")
    for k, v in agg["by_item"].items():
        flag = "PASS" if v["median_of_p90_mm"] <= TOPP * 1e3 else "FAIL"
        print(f"  {flag} {k:42s} n={v['n']:3d} med={v['median_of_median_mm']:6.2f}mm "
              f"p90={v['median_of_p90_mm']:6.2f}mm max_p90={v['max_p90_mm']:6.2f}mm")
    print(f"failures ({len(agg['failures'])}):")
    for f in agg["failures"][:10]:
        print(f"  {f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"samples": samples_to_serializable(all_samples), "aggregate": agg}, fh, indent=2)
    print(f"saved {args.out}")


def samples_to_serializable(samples):
    return [dict(s) for s in samples]


if __name__ == "__main__":
    main()
