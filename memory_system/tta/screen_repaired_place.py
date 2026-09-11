"""Screen repaired episodes by action length and Place-phase tilt.

Input HDF5 contract (one episode, root datasets):
  actions (T, 7), wrist_depth (T, H, W), wrist_segmentation (T, H, W),
  wrist_c2w (T, 4, 4), camera_K (3, 3).
Root attrs provide place_start, release_frame, and target_id. Depth is metric
by default; set depth_metric=0 and provide depth_near/depth_far for normalized
MuJoCo depth.

The rule compares the target object's orientation at Place start and just
before release. Points are transformed to world coordinates before fitting a
normal, so wrist-camera motion does not look like object tilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def metric_depth(depth, attrs):
    if bool(attrs.get("depth_metric", True)):
        return np.asarray(depth, dtype=np.float64)
    near, far = float(attrs["depth_near"]), float(attrs["depth_far"])
    d = np.asarray(depth, dtype=np.float64)
    return near / (1.0 - d * (1.0 - near / far))


def object_points_world(depth, segmentation, c2w, K, target_id, attrs):
    depth = metric_depth(depth, attrs)
    mask = np.asarray(segmentation)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(bool) if target_id is None else mask == int(target_id)
    rows, cols = np.nonzero(mask)
    if len(rows) < 20:
        return np.empty((0, 3), dtype=np.float64)
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z > 0)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if len(z) < 20:
        return np.empty((0, 3), dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    cam = np.stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z), axis=1)
    T = np.asarray(c2w, dtype=np.float64)
    return cam @ T[:3, :3].T + T[:3, 3]


def cloud_normal(points):
    if len(points) < 20:
        return None
    centered = points - np.median(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    return None if norm == 0 or not np.isfinite(norm) else normal / norm


def tilt_change(points_start, points_release):
    n0, n1 = cloud_normal(points_start), cloud_normal(points_release)
    if n0 is None or n1 is None:
        return None
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n0, n1)), 0.0, 1.0))))


def screen_episode(actions, depth, segmentation, c2w, K, place_start,
                   release_frame, target_id=None, max_actions=0,
                   max_tilt_deg=10.0, attrs=None):
    attrs = {} if attrs is None else attrs
    actions = np.asarray(actions)
    total = len(actions)
    length_ok = max_actions <= 0 or total <= max_actions
    if not (0 <= place_start < total and 0 <= release_frame < total):
        return {"keep": False, "reason": "invalid_place_range",
                "num_actions": total}
    p0 = object_points_world(depth[place_start], segmentation[place_start],
                             c2w[place_start], K, target_id, attrs)
    p1 = object_points_world(depth[release_frame], segmentation[release_frame],
                             c2w[release_frame], K, target_id, attrs)
    angle = tilt_change(p0, p1)
    tilt_ok = angle is not None and angle <= max_tilt_deg
    reason = "ok" if length_ok and tilt_ok else (
        "too_long" if not length_ok else "tilt_exceeded_or_missing_cloud")
    return {"keep": bool(length_ok and tilt_ok), "reason": reason,
            "num_actions": total, "place_start": int(place_start),
            "release_frame": int(release_frame),
            "tilt_deg": None if angle is None else round(angle, 4),
            "length_ok": bool(length_ok), "tilt_ok": bool(tilt_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-actions", type=int, default=0)
    ap.add_argument("--max-tilt-deg", type=float, default=10.0)
    ap.add_argument("--place-start", type=int, default=None)
    ap.add_argument("--release-frame", type=int, default=None)
    ap.add_argument("--target-id", type=int, default=None)
    args = ap.parse_args()
    with h5py.File(args.episode, "r") as h5:
        attrs = dict(h5.attrs)
        result = screen_episode(
            h5["actions"][:], h5["wrist_depth"][:], h5["wrist_segmentation"][:],
            h5["wrist_c2w"][:], h5["camera_K"][:],
            int(attrs["place_start"] if args.place_start is None else args.place_start),
            int(attrs["release_frame"] if args.release_frame is None else args.release_frame),
            target_id=attrs.get("target_id", args.target_id)
            if args.target_id is None else args.target_id,
            max_actions=args.max_actions, max_tilt_deg=args.max_tilt_deg, attrs=attrs,
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
