#!/usr/bin/env python3
"""Leave-one-demo-out evaluation for the training-free LIBERO phase verifier."""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from memory_system.artifacts import PhaseTargetMemory
from memory_system.execute.phase import PHASE_ERROR, PHASE_UNKNOWN, PhaseVerifier
from memory_system.types import VerifierObservation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-targets", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--shift-x", type=int, default=0)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument(
        "--synthetic-away", "--synthetic-reverse", dest="synthetic_away",
        action="store_true", help="test two consecutive gripper moves away from the target",
    )
    parser.add_argument("--vis-dir", type=Path, help="save offline diagnostic contact sheets")
    parser.add_argument("--vis-max", type=int, default=30, help="maximum visualized segments per task; 0 means all")
    parser.add_argument("--vis-filter", choices=("all", "error", "mismatch"), default="all")
    parser.add_argument("--show-matches", action="store_true", help="also save visual-similarity region views")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()

def decode_jpeg(value) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)


def perturb(image: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    value = image.astype(np.float32) * args.brightness
    if args.noise_std > 0:
        value += rng.normal(0.0, args.noise_std, value.shape)
    value = np.clip(value, 0, 255).astype(np.uint8)
    if args.shift_x or args.shift_y:
        value = np.roll(value, shift=(args.shift_y, args.shift_x), axis=(0, 1))
    return value


def shifted_xy(value, args) -> np.ndarray:
    point = np.asarray(value, dtype=np.float32).copy()
    point += np.asarray([args.shift_x, args.shift_y], dtype=np.float32)
    return point


def inside_bbox(point: tuple[float, float] | None, bbox, args) -> bool:
    if point is None:
        return False
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float32)
    x0, x1 = x0 + args.shift_x, x1 + args.shift_x
    y0, y1 = y0 + args.shift_y, y1 + args.shift_y
    return x0 <= point[0] < x1 and y0 <= point[1] < y1


def group_templates(templates):
    groups = defaultdict(list)
    for item in templates:
        key = (
            item["task_name"], item["demo_id"], int(item["planner_step_id"]),
            item["skill"], tuple(sorted(item.get("arguments", {}).items())),
        )
        groups[key].append(item)
    return groups


def run_group(verifier, key, samples, input_dir, args, rng):
    task_name, demo_id, step_id, skill, arguments_key = key
    verifier.reset(
        task_name, step_id, skill, dict(arguments_key), exclude_demo_ids=(demo_id,)
    )
    if not verifier.templates:
        return None
    path = input_dir / f"{task_name}_demo.hdf5"
    results, errors, frames = [], [], []
    with h5py.File(path, "r") as handle:
        images = handle["data"][demo_id]["obs"]["agentview_rgb_jpeg"]
        for sample in sorted(samples, key=lambda item: int(item["frame"])):
            image = perturb(decode_jpeg(images[int(sample["frame"])]), args, rng)
            gripper = shifted_xy(sample["gripper_xy"], args)
            result = verifier.update(
                VerifierObservation(third_view_rgb=image, gripper_xy=gripper)
            )
            results.append(result)
            frames.append({"image": image, "sample": sample, "gripper": gripper, "result": result})
            if result.target_xy is not None:
                expected = shifted_xy(sample["target_center_xy"], args)
                errors.append(float(np.linalg.norm(np.asarray(result.target_xy) - expected)))
    return {
        "results": results,
        "errors": errors,
        "frames": frames,
        "references": list(verifier.templates),
        "localized": [
            inside_bbox(result.target_xy, sample["bbox_xyxy"], args)
            for result, sample in zip(results, sorted(samples, key=lambda item: int(item["frame"])))
        ],
    }


def reference_rgb(reference, input_dir) -> np.ndarray:
    path = input_dir / f"{reference['task_name']}_demo.hdf5"
    with h5py.File(path, "r") as handle:
        value = handle["data"][reference["demo_id"]]["obs"]["agentview_rgb_jpeg"][
            int(reference["frame"])
        ]
        return decode_jpeg(value)


def _draw_point(image, point, color, radius=5):
    xy = tuple(np.rint(point).astype(int))
    cv2.circle(image, xy, radius, color, -1, cv2.LINE_AA)


def observation_panel(frame, previous_gripper, args, label) -> np.ndarray:
    image, sample, gripper, result = (
        frame["image"], frame["sample"], frame["gripper"], frame["result"]
    )
    height, width = image.shape[:2]
    panel = np.full((height + 44, width, 3), 20, dtype=np.uint8)
    panel[44:] = image
    offset = np.asarray([0, 44], dtype=np.float32)
    bbox = np.asarray(sample["bbox_xyxy"], dtype=np.float32)
    bbox += np.asarray([args.shift_x, args.shift_y] * 2, dtype=np.float32)
    cv2.rectangle(panel, tuple(bbox[:2].astype(int) + (0, 44)),
                  tuple(bbox[2:].astype(int) + (0, 44)), (40, 230, 70), 2)
    gt = shifted_xy(sample["target_center_xy"], args) + offset
    grip = np.asarray(gripper) + offset
    _draw_point(panel, gt, (40, 230, 70))
    _draw_point(panel, grip, (255, 255, 255), 4)
    if result.target_xy is not None:
        predicted = np.asarray(result.target_xy) + offset
        cv2.arrowedLine(panel, tuple(grip.astype(int)), tuple(predicted.astype(int)),
                        (30, 220, 255), 2, cv2.LINE_AA, tipLength=0.08)
        _draw_point(panel, predicted, (30, 220, 255))
    if previous_gripper is not None:
        previous = np.asarray(previous_gripper) + offset
        cv2.arrowedLine(panel, tuple(previous.astype(int)), tuple(grip.astype(int)),
                        (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.15)
    progress = "n/a" if result.progress_px is None else f"{result.progress_px:+.1f}px"
    wrong = result.details.get("wrong_way_count", 0)
    cv2.putText(panel, f"{label}  {result.status}", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, f"conf={result.confidence:.2f} inliers={result.match_count} "
                f"progress={progress} wrong={wrong}", (5, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def memory_panel(reference, input_dir, shape) -> np.ndarray:
    image = reference_rgb(reference, input_dir)
    height, width = shape
    source_h, source_w = image.shape[:2]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scale = np.asarray([width / source_w, height / source_h] * 2, dtype=np.float32)
    bbox = np.asarray(reference["bbox_xyxy"], dtype=np.float32) * scale
    panel = np.full((height + 44, width, 3), 20, dtype=np.uint8)
    panel[44:] = image
    cv2.rectangle(panel, tuple(bbox[:2].astype(int) + (0, 44)),
                  tuple(bbox[2:].astype(int) + (0, 44)), (40, 230, 70), 2)
    cv2.putText(panel, "MEMORY TARGET (offline)", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 230, 70), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{reference['demo_id']} frame={reference['frame']}", (5, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def save_match_view(reference, frame, path):
    bbox = frame["result"].details.get("matched_bbox_xyxy")
    if bbox is None:
        return
    crop = np.asarray(reference["crop_rgb"], dtype=np.uint8).copy()
    mask = np.asarray(reference["crop_mask"]) > 0
    crop[~mask] = 0
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    current = frame["image"].copy()
    x0, y0, x1, y1 = np.asarray(bbox, dtype=int)
    cv2.rectangle(current, (x0, y0), (x1, y1), (30, 220, 255), 2)
    height = max(crop.shape[0], current.shape[0])
    view = np.zeros((height, crop.shape[1] + current.shape[1], 3), dtype=np.uint8)
    view[:crop.shape[0], :crop.shape[1]] = crop
    view[:current.shape[0], crop.shape[1]:] = current
    Image.fromarray(view).save(path)


def save_visualization(index, key, evaluation, verifier, input_dir, args):
    task_name, demo_id, step_id, skill, _ = key
    frames, references = evaluation["frames"], evaluation["references"]
    matched = next((item for item in frames
                    if item["result"].details.get("matched_bbox_xyxy")), frames[0])
    mismatch = next((item for item, valid in zip(frames, evaluation["localized"])
                     if not valid), None) if args.vis_filter == "mismatch" else None
    debug_frame = mismatch or next((item for item in frames
                                    if item["result"].status == PHASE_ERROR), matched)
    matched_demo = debug_frame["result"].template_demo_id
    matched_frame = debug_frame["result"].details.get("template_frame")
    usable = [item for item in references if item.get("crop_rgb") is not None]
    reference = next((item for item in usable if item["demo_id"] == matched_demo
                      and int(item["frame"]) == matched_frame), usable[0])
    height, width = frames[0]["image"].shape[:2]
    panels = [memory_panel(reference, input_dir, (height, width))]
    previous = None
    labels = ("start", "middle", "ready") if len(frames) == 3 else None
    for frame_index, frame in enumerate(frames):
        label = labels[frame_index] if labels else f"frame={frame['sample']['frame']}"
        panels.append(observation_panel(frame, previous, args, label))
        previous = frame["gripper"]
    task_dir = args.vis_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:04d}_{demo_id}_step{step_id}_{skill}"
    Image.fromarray(np.concatenate(panels, axis=1)).save(task_dir / f"{stem}.jpg", quality=95)
    if args.show_matches:
        save_match_view(reference, debug_frame, task_dir / f"{stem}_matches.jpg")

def synthetic_away_detected(verifier, key, samples, input_dir, args, rng) -> bool | None:
    task_name, demo_id, step_id, skill, arguments_key = key
    ordered = sorted(samples, key=lambda item: int(item["frame"]))
    if len(ordered) < 3:
        return None
    verifier.reset(
        task_name, step_id, skill, dict(arguments_key), exclude_demo_ids=(demo_id,)
    )
    path = input_dir / f"{task_name}_demo.hdf5"
    detected = False
    gripper = shifted_xy(ordered[0]["gripper_xy"], args)
    with h5py.File(path, "r") as handle:
        images = handle["data"][demo_id]["obs"]["agentview_rgb_jpeg"]
        for index, sample in enumerate(ordered):
            if index:
                target = shifted_xy(sample["target_center_xy"], args)
                direction = target - gripper
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    gripper = gripper - 8.0 * direction / norm
            image = perturb(decode_jpeg(images[int(sample["frame"])]), args, rng)
            detected = detected or verifier.update(
                VerifierObservation(third_view_rgb=image, gripper_xy=gripper)
            ).status == PHASE_ERROR
    return detected


def main() -> int:
    args = parse_args()
    if args.brightness <= 0 or args.noise_std < 0:
        raise ValueError("brightness must be positive and noise-std must be non-negative")
    if args.vis_max < 0 or (args.show_matches and args.vis_dir is None):
        raise ValueError("vis-max must be non-negative and show-matches requires vis-dir")
    memory = PhaseTargetMemory(args.phase_targets)
    verifier = PhaseVerifier(memory)
    groups = list(group_templates(memory.templates).items())
    if args.max_segments > 0:
        groups = groups[: args.max_segments]
    rng = np.random.default_rng(0)
    counts, pixel_errors, localized, away = Counter(), [], [], []
    visualized = Counter()
    for key, samples in groups:
        evaluation = run_group(verifier, key, samples, args.input_dir, args, rng)
        if evaluation is None:
            counts["skipped_no_reference"] += 1
            continue
        counts["segments"] += 1
        counts.update(result.status for result in evaluation["results"])
        counts["observations"] += len(evaluation["results"])
        pixel_errors.extend(evaluation["errors"])
        localized.extend(evaluation["localized"])
        has_error = any(result.status == PHASE_ERROR for result in evaluation["results"])
        selected = args.vis_filter == "all" or (args.vis_filter == "error" and has_error)
        selected = selected or (args.vis_filter == "mismatch" and not all(evaluation["localized"]))
        if (args.vis_dir and selected
                and (args.vis_max == 0 or visualized[key[0]] < args.vis_max)):
            save_visualization(visualized[key[0]], key, evaluation, verifier, args.input_dir, args)
            visualized[key[0]] += 1
        if args.synthetic_away:
            detected = synthetic_away_detected(verifier, key, samples, args.input_dir, args, rng)
            if detected is not None:
                away.append(detected)
    observations = max(counts["observations"], 1)
    report = {
        "counts": dict(counts),
        "localization_accuracy": float(np.mean(localized)) if localized else None,
        "median_center_error_px": float(np.median(pixel_errors)) if pixel_errors else None,
        "phase_unknown_rate": counts[PHASE_UNKNOWN] / observations,
        "success_false_error_rate": counts[PHASE_ERROR] / observations,
        "synthetic_away_detection_rate": float(np.mean(away)) if away else None,
        "visualizations": {"total": sum(visualized.values()), "by_task": dict(visualized)},
        "settings": {
            "noise_std": args.noise_std,
            "brightness": args.brightness,
            "shift_x": args.shift_x,
            "shift_y": args.shift_y,
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
