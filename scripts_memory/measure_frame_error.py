"""Measure the RGBD-only object-frame error against the simulator ground truth.

Question: if the simulator no longer provides ``body_xpos/body_xmat``, how far
is the object frame we can recover from an RGBD observation?  Recognition is
granted for free (the instance mask is the simulator's ground-truth
segmentation), so the measured quantity is exactly the partial-view
registration error: model cloud -> visible-cloud ICP.

Per instance and reset:
  observation: visible_point_cloud(env, obs, instance)   # depth back-projection
  model:       complete_point_cloud(env, instance) expressed in the body frame
               via the GT transform once (the offline "known CAD" step)
  estimate:    ICP (identity/PCA/yaw-ring initialisations, best fit kept)
  reference:   object_frame(env, instance)               # MuJoCo ground truth

Outputs rows (task, instance, t_err_cm, rot_deg, ready14_worst_cm, ...) to
memory/experiments/frame_error/results.json and prints a per-type table.

The rendered depth is noise-free and the mask is exact, so these numbers are a
lower bound: a real camera adds depth noise, hand-eye bias and mask errors on
top.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
for p in (str(ROOT), str(LIBERO_PLUS)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from scipy.spatial import cKDTree  # noqa: E402

from memory.build.pointcloud_action.offline.extraction import (  # noqa: E402
    complete_point_cloud,
    create_env,
    object_frame,
    visible_point_cloud,
)

DEFAULT_TASKS = (
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
)
READY_DISTANCE_M = 0.14  # memory's READY_DISTANCE_M; lever arm for the ready error
# Names whose visible geometry is close to rotationally symmetric; the yaw
# error for these is expected to be structurally large, not a registration bug.
ROT_SYMMETRIC_HINTS = ("bowl", "plate", "moka", "basket", "pan", "stove")


def _subsample(points: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= cap:
        return points
    return points[rng.choice(len(points), cap, replace=False)]


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimizing ||R @ src + t - dst|| (per-point, unweighted)."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, dst_c - R @ src_c


def _icp(
    model: np.ndarray,
    target: np.ndarray,
    R0: np.ndarray,
    t0: np.ndarray,
    iters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trimmed point-to-point ICP of ``model`` onto ``target`` from one init."""
    tree = cKDTree(target)
    R, t = R0, t0
    rmse = float("inf")
    for _ in range(iters):
        cur = model @ R.T + t
        dist, j = tree.query(cur, k=1)
        keep = dist < max(3.0 * float(np.median(dist)), 1e-4)
        if int(keep.sum()) < 10:
            break
        Rd, td = _kabsch(cur[keep], target[j[keep]])
        R, t = Rd @ R, Rd @ t + td
        rmse = float(np.sqrt(np.mean(dist[keep] ** 2)))
    return R, t, rmse


def _initial_guesses(model: np.ndarray, target: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Identity + PCA axis alignments + a yaw ring around the centroid init."""
    guesses: list[tuple[np.ndarray, np.ndarray]] = []
    cm, ct = model.mean(axis=0), target.mean(axis=0)
    guesses.append((np.eye(3), ct - cm))

    def axes(points: np.ndarray) -> np.ndarray:
        cov = np.cov((points - points.mean(axis=0)).T)
        _, vecs = np.linalg.eigh(cov)
        return vecs[:, ::-1]  # descending variance

    ev, em = axes(target), axes(model)
    for flips in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
        R = ev @ np.diag(flips) @ em.T
        if np.linalg.det(R) < 0:
            continue
        guesses.append((R, ct - R @ cm))

    for k in range(8):
        a = k * np.pi / 4
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        guesses.append((R, ct - R @ cm))
    return guesses


def estimate_frame(
    visible: np.ndarray,
    model_body: np.ndarray,
    *,
    cap: int,
    iters: int,
    rng: np.random.Generator,
) -> dict:
    """Register the body-frame model cloud onto the world-frame visible cloud."""
    vis = _subsample(visible, cap, rng)
    mod = _subsample(model_body, cap, rng)
    best = None
    for R0, t0 in _initial_guesses(mod, vis):
        R, t, rmse = _icp(mod, vis, R0, t0, iters)
        if best is None or rmse < best["rmse"]:
            best = {"R": R, "t": t, "rmse": rmse}
    return best


def rotation_deg(R_rel: np.ndarray) -> float:
    tr = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(tr, -1.0, 1.0))))


def object_type(name: str) -> str:
    import re

    return re.sub(r"_\d+$", "", str(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--resets", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--cap", type=int, default=600, help="ICP cloud subsample cap")
    parser.add_argument("--iters", type=int, default=40, help="ICP iterations per init")
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "memory/experiments/frame_error",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for task in args.tasks:
        t0 = time.time()
        env = create_env(task, args.resolution, suite=args.suite)
        try:
            for reset in range(args.resets):
                obs = env.reset()
                for instance in list(env.instance_to_id.keys()):
                    if "Panda" in instance:
                        continue
                    visible = visible_point_cloud(env, obs, instance, args.resolution)
                    if len(visible) < args.min_points:
                        rows.append({
                            "task": task, "reset": reset, "instance": instance,
                            "n_vis": int(len(visible)), "note": "skipped_small_vis",
                        })
                        continue
                    complete = complete_point_cloud(env, instance)
                    T_gt, t_gt, R_gt = object_frame(env, instance)
                    if len(complete) < args.min_points:
                        rows.append({
                            "task": task, "reset": reset, "instance": instance,
                            "n_vis": int(len(visible)), "note": "skipped_small_model",
                        })
                        continue
                    # The "known CAD" is the complete cloud expressed in the body
                    # frame; this is the one place GT is used for preparation.
                    model_body = (complete - t_gt) @ R_gt
                    est = estimate_frame(
                        visible, model_body, cap=args.cap, iters=args.iters, rng=rng
                    )
                    t_err = float(np.linalg.norm(est["t"] - t_gt))
                    rot = rotation_deg(est["R"] @ R_gt.T)
                    ready14 = t_err + 2.0 * np.sin(np.radians(rot) / 2.0) * READY_DISTANCE_M
                    rows.append({
                        "task": task,
                        "reset": reset,
                        "instance": instance,
                        "type": object_type(instance),
                        "sym_hint": any(h in instance for h in ROT_SYMMETRIC_HINTS),
                        "n_vis": int(len(visible)),
                        "n_model": int(len(complete)),
                        "rmse_mm": float(est["rmse"] * 1000.0),
                        "t_err_cm": float(t_err * 100.0),
                        "rot_deg": rot,
                        "ready14_worst_cm": float(ready14 * 100.0),
                    })
        finally:
            env.close()
        done = [r for r in rows if r.get("task") == task and "rot_deg" in r]
        print(f"[{time.time() - t0:6.1f}s] {task}: {len(done)} instances measured", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "results.json"
    config = {
        "tasks": list(args.tasks),
        "suite": args.suite,
        "resets": args.resets,
        "resolution": args.resolution,
        "cap": args.cap,
        "iters": args.iters,
        "min_points": args.min_points,
        "ready_distance_m": READY_DISTANCE_M,
    }
    out.write_text(json.dumps({"config": config, "rows": rows}, indent=1))

    measured = [r for r in rows if "rot_deg" in r]
    print(f"\n{len(measured)} measurements -> {out}\n")

    def stats(vals: list[float]) -> str:
        a = np.asarray(vals)
        return (
            f"n={len(a):3d}  median={np.median(a):7.2f}  mean={a.mean():7.2f}  "
            f"p90={np.percentile(a, 90):7.2f}  max={a.max():7.2f}"
        )

    by_type: dict[str, list[dict]] = {}
    for r in measured:
        by_type.setdefault(r["type"], []).append(r)
    print(f"{'type':34s} {'sym?':4s} | {'rot_deg (orientation error)':52s}")
    for name, rs in sorted(by_type.items()):
        sym = "Y" if rs[0]["sym_hint"] else "."
        print(f"{name:34s} {sym:4s} | {stats([r['rot_deg'] for r in rs])}")
    print(f"\n{'ALL':34s} {'':4s} | {stats([r['rot_deg'] for r in measured])}")
    asym = [r for r in measured if not r["sym_hint"]]
    sym = [r for r in measured if r["sym_hint"]]
    if asym:
        print(f"{'non-symmetric hint':34s} {'':4s} | {stats([r['rot_deg'] for r in asym])}")
    if sym:
        print(f"{'symmetric hint':34s} {'':4s} | {stats([r['rot_deg'] for r in sym])}")

    print(f"\n{'':34s} | {'t_err_cm (origin error)':52s}")
    print(f"{'ALL':34s} {'':4s} | {stats([r['t_err_cm'] for r in measured])}")
    if asym:
        print(f"{'non-symmetric hint':34s} {'':4s} | {stats([r['t_err_cm'] for r in asym])}")
    if sym:
        print(f"{'symmetric hint':34s} {'':4s} | {stats([r['t_err_cm'] for r in sym])}")

    print(f"\n{'':34s} | {'ready14_worst_cm (ready pose error @14cm)':52s}")
    print(f"{'ALL':34s} {'':4s} | {stats([r['ready14_worst_cm'] for r in measured])}")
    if asym:
        print(f"{'non-symmetric hint':34s} {'':4s} | {stats([r['ready14_worst_cm'] for r in asym])}")
    if sym:
        print(f"{'symmetric hint':34s} {'':4s} | {stats([r['ready14_worst_cm'] for r in sym])}")


if __name__ == "__main__":
    main()
