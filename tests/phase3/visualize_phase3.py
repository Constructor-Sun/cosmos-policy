"""Visualize a single Phase3 Pick failure/example as an MP4.

Usage:
    python tests/phase3/visualize_phase3.py \
        --task "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
        --demo demo_0 \
        --step 3 \
        --out /tmp/phase3_failure.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    PHASE_TARGETS,
    SEGMENTS_MANIFEST,
    bbox_from_mask,
    create_env,
    demo_hdf5,
    instance_mask,
    load_manifest,
    make_template,
    patch_numpy2_segmentation,
    select_candidates,
    load_item_objects,
)
from memory_system.execute.phase import PhaseVerifier  # noqa: E402
from memory_system.geometry import (  # noqa: E402
    camera_params as build_camera_params,
    depth_to_metric,
    flip_depth,
    pixel_to_world,
)
from robosuite.utils.camera_utils import get_camera_transform_matrix  # noqa: E402

FRAME_STRIDE = 4
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
]


def sample_target(match, depth, cam, eef):
    if match is None:
        return None, None, 0.0
    mask = match["mask"]
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None, None, 0.0
    pts = pixel_to_world(np.stack([ys, xs], axis=-1), depth, cam)
    valid = np.isfinite(pts).all(axis=1)
    if int(valid.sum()) < 4:
        return None, None, float(valid.mean())
    target = np.median(pts[valid], axis=0)
    dist = float(np.linalg.norm(np.asarray(eef, dtype=np.float64) - target))
    return dist, target, float(valid.mean())


def draw_overlay(rgb, matches, dists, approaches, correct_item, t):
    img = rgb.copy()
    height, width = img.shape[:2]
    for idx, obj in enumerate(dists):
        color = COLORS[idx % len(COLORS)]
        match = matches.get(obj)
        if match is not None and match.get("bbox_xyxy") is not None:
            x0, y0, x1, y1 = match["bbox_xyxy"]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            label = obj
            if obj == correct_item:
                label += " [CORRECT]"
            cv2.putText(
                img, label, (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
        d = dists[obj][-1] if dists[obj] else None
        ap = approaches[obj][-1] if approaches[obj] else None
        text = f"{obj}: d={d:.3f}m ap={ap:+.3f}m" if d is not None else f"{obj}: N/A"
        cv2.putText(
            img, text, (10, 25 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    cv2.putText(
        img, f"t={t}", (width - 100, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE)
    args = parser.parse_args()

    manifest = load_manifest()
    item_objects = load_item_objects(manifest)
    record = next(
        r for r in manifest["records"]
        if r["task_name"] == args.task and r["demo_id"] == args.demo and r.get("valid")
    )
    segment = next(
        s for s in record["segments"]
        if s["skill"] == "Pick" and int(s["planner_step_id"]) == args.step
    )
    item = segment["arguments"]["item"]
    phase_start = int(segment["start"])
    ready_frame = int(segment["ready_frame"])
    success_start = int(segment["success_start"])

    patch_numpy2_segmentation()
    env = create_env(args.task)
    env.reset()
    try:
        h5 = h5py.File(demo_hdf5(args.task), "r")
        states = h5["data"][args.demo]["states"][:]
        length = len(states)
        search_end = min(max(success_start, phase_start + 1), length - 1)

        obs0 = env.regenerate_obs_from_state(states[0])
        rgb0 = np.flipud(obs0["agentview_image"])
        cam = build_camera_params(env.env.sim, "agentview", 256, 256)
        camera_transform = get_camera_transform_matrix(
            env.env.sim, "agentview", 256, 256
        )

        candidates = select_candidates(
            env, item, max_total=5, allowed_objects=item_objects
        )
        correct_tpl = make_template(
            rgb0, instance_mask(env, obs0, item), args.demo
        )
        if correct_tpl is None:
            raise RuntimeError(f"correct target template unavailable: {item}")

        matcher = PhaseVerifier(PHASE_TARGETS, min_demo_votes=1)
        prepared = {
            item: matcher._prepare_template(correct_tpl)
        }

        def match_correct_target(rgb):
            matcher.templates = [correct_tpl]
            matcher.prepared_templates = [(correct_tpl, prepared[item])]
            return matcher.match_current(rgb)

        sampled_frames = list(range(phase_start, search_end + 1, args.stride))
        dists = {obj: [] for obj in candidates}
        approaches = {obj: [] for obj in candidates}
        first_dist = {obj: None for obj in candidates}
        writer = imageio.get_writer(args.out, fps=10)

        for t in sampled_frames:
            obs = env.regenerate_obs_from_state(states[t])
            rgb = np.flipud(obs["agentview_image"])
            metric = depth_to_metric(obs["agentview_depth"], cam.near, cam.far)
            depth = flip_depth(metric)
            eef = obs["robot0_eef_pos"]

            matches = {}
            match_correct = match_correct_target(rgb)
            for obj in candidates:
                if obj == item:
                    match = match_correct
                else:
                    oracle_mask = instance_mask(env, obs, obj)
                    obb = bbox_from_mask(oracle_mask)
                    match = {"mask": oracle_mask, "bbox_xyxy": obb}
                matches[obj] = match
                d, _xyz, _ratio = sample_target(match, depth, cam, eef)
                dists[obj].append(d)
                if first_dist[obj] is None and d is not None:
                    first_dist[obj] = d
                ap = None if first_dist[obj] is None or d is None else first_dist[obj] - d
                approaches[obj].append(ap)

            frame = draw_overlay(rgb, matches, dists, approaches, item, t)
            writer.append_data(frame)

        writer.close()
        print(f"saved {args.out}")
        h5.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
