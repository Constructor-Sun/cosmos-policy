#!/usr/bin/env python3
"""Baseline test: query Qwen3-VL-2B for recovery direction from HDF5 episode data.

Usage:
  python bin/test_vlm_recovery.py episode.hdf5 --t-star 200
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "/data1/liu/exp/counterfactual/checkpoints/Qwen3-VL-2B-Instruct"

SYSTEM_PROMPT = """\
You are a robot state observer. Your job is to describe the spatial \
relationship between the robot gripper and the task-relevant object, \
based on the third-view and wrist images.

Describe what you SEE — do NOT prescribe a correction. Just report the \
observable spatial features. The downstream controller will decide \
what to do with this information.

## Output Format
Output a JSON object only. No other text:
{
  "target_object": "...",
  "target_horizontal": "left|right|center",
  "target_vertical": "above|below|same",
  "target_depth": "near|far|same",
  "orientation_issue": "none|roll|pitch|yaw",
  "orientation_sign": "positive|negative|none",
  "gripper_state": "empty|holding",
  "confidence": 0.XX
}

### Field definitions
- target_object: what object is the gripper interacting with or aiming for?
- target_horizontal: looking at the third-view image, is the target to the \
LEFT, RIGHT, or CENTER of the gripper?
- target_vertical: is the target ABOVE, BELOW, or at the SAME height as the gripper?
- target_depth: is the target NEARER to the camera (in front of gripper), \
FARTHER from the camera (behind gripper), or at the SAME depth?
- orientation_issue: does the gripper's orientation look wrong for the task? \
If orientation looks natural, say "none". Otherwise pick ONE:
  * yaw: the gripper is facing the wrong horizontal direction — in the \
    third-view image it looks like the wrist/gripper should turn left or right \
    to face the target; in the wrist image the scene content shifts left/right.
  * pitch: the gripper is tilted too far up or too far down — in the \
    third-view image the wrist/gripper points too high or too low; in the \
    wrist image the horizon shifts up/down.
  * roll: the gripper is rotated around its own forward axis — in the \
    wrist image this is obvious: the entire camera view appears tilted \
    clockwise or counter-clockwise relative to the horizontal.
- orientation_sign: if orientation_issue is not "none", is the needed rotation \
positive or negative?
  * For yaw: positive = turn left (counter-clockwise from top-down view)
  * For pitch: positive = tilt up
  * For roll: positive = rotate clockwise in the wrist image
  If orientation_issue is "none", this must be "none".
- gripper_state: from the wrist image, is the gripper EMPTY or HOLDING an object?
- confidence: 0.7-1.0=confident, 0.5-0.7=moderate, <0.5=uncertain."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(f: h5py.File, jpeg_key: str, raw_key: str, t: int) -> np.ndarray:
    """Load image at timestep t from HDF5, decoding JPEG if necessary."""
    if jpeg_key in f:
        data = f[jpeg_key][t]

        # h5py variable-length bytes: may return a numpy byte array or bytes object
        if isinstance(data, np.ndarray):
            if data.dtype == object:
                data = data.item()  # unwrap scalar
            else:
                # Raw byte array (e.g. shape (N,), dtype=uint8) → decode as JPEG
                data = data.tobytes()
        if isinstance(data, bytes):
            return np.array(Image.open(io.BytesIO(data)))
        raise ValueError(f"Unexpected jpeg data type: {type(data)}")

    if raw_key in f:
        return f[raw_key][t][:]
    raise KeyError(f"Neither {jpeg_key} nor {raw_key} found in HDF5")


def _quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Quaternion (w,x,y,z) → roll, pitch, yaw (radians)."""
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _format_pose(proprio: np.ndarray) -> str:
    """Convert proprio array [g0, g1, x, y, z, qw, qx, qy, qz] to text."""
    g0, g1 = float(proprio[0]), float(proprio[1])
    x, y, z = float(proprio[2]), float(proprio[3]), float(proprio[4])
    qw, qx, qy, qz = float(proprio[5]), float(proprio[6]), float(proprio[7]), float(proprio[8])
    gripper_width = g0 - g1

    roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)

    # Semantic interpretation of pitch
    if pitch < -1.2:
        pitch_desc = "pointing downward (toward the table)"
    elif pitch > 0.8:
        pitch_desc = "pointing upward"
    else:
        pitch_desc = "pointing roughly forward/horizontal"

    # Semantic interpretation of gripper width
    if gripper_width < 0.015:
        grip_desc = "closed (holding or empty-closed)"
    elif gripper_width < 0.035:
        grip_desc = "partially open"
    else:
        grip_desc = "fully open"

    return (
        f"Position: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m\n"
        f"Orientation: roll={roll:.3f}rad, pitch={pitch:.3f}rad ({pitch_desc}), yaw={yaw:.3f}rad\n"
        f"Gripper: width={gripper_width:.4f}m ({grip_desc})"
    )


def _extract_json_blocks(text: str) -> list[str]:
    """Extract all brace-balanced {...} blocks from text."""
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1
    return candidates


def _parse_spatial_output(raw_text: str) -> dict:
    """Extract spatial-feature JSON from VLM output.

    Returns dict with keys: target_object, target_horizontal, target_vertical,
    target_depth, orientation_issue, orientation_sign, gripper_state, confidence.
    On parse failure, returns safe defaults (all "center"/"same"/"none").
    """
    for candidate in reversed(_extract_json_blocks(raw_text)):
        try:
            obj = json.loads(candidate)
            if "target_horizontal" in obj or "orientation_issue" in obj:
                return obj
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    print("[WARN] Could not parse spatial output from VLM", file=sys.stderr)
    return {
        "target_object": "unknown",
        "target_horizontal": "center",
        "target_vertical": "same",
        "target_depth": "same",
        "orientation_issue": "none",
        "orientation_sign": "none",
        "gripper_state": "empty",
        "confidence": 0.0,
    }


def _features_to_direction(
    features: dict,
) -> tuple[list[int], float, str]:
    """Map VLM spatial features to a single-axis 6D direction vector.

    Priority order (first non-"none"/"center"/"same" wins):
      1. orientation_issue  → VLM-detected rotation axis (roll / pitch / yaw)
      2. target_horizontal  → dy
      3. target_depth       → dx

    Returns (direction_6d, confidence, chosen_axis_label).
    """
    confidence = float(features.get("confidence", 0.5))

    # Priority 1: rotation correction (most important near grasp / insertion)
    orient = str(features.get("orientation_issue", "none")).strip().lower()
    orient_sign = str(features.get("orientation_sign", "none")).strip().lower()
    if orient in ("roll", "pitch", "yaw"):
        sign = 1 if orient_sign == "positive" else -1
        axis_map = {"roll": 3, "pitch": 4, "yaw": 5}
        idx = axis_map[orient]
        direction = [0, 0, 0, 0, 0, 0]
        direction[idx] = sign
        return direction, confidence, f"d{orient}{'+' if sign > 0 else '-'}"

    # Priority 2: horizontal misalignment → dy
    horiz = str(features.get("target_horizontal", "center")).strip().lower()
    if horiz == "left":
        return [0, 1, 0, 0, 0, 0], confidence, "dy+"
    elif horiz == "right":
        return [0, -1, 0, 0, 0, 0], confidence, "dy-"

    # Priority 3: depth misalignment → dx
    depth = str(features.get("target_depth", "same")).strip().lower()
    if depth == "near":
        return [1, 0, 0, 0, 0, 0], confidence, "dx+"
    elif depth == "far":
        return [-1, 0, 0, 0, 0, 0], confidence, "dx-"

    # No misalignment detected → do nothing
    return [0, 0, 0, 0, 0, 0], confidence, "none"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM recovery direction baseline test",
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to episode HDF5 file")
    parser.add_argument("--t-star", type=int, required=True, help="Intervention timestep")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT),
                       help="Path to Qwen3-VL-2B-Instruct checkpoint")
    parser.add_argument("--device", default="cuda", help="Device for model (cuda / cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--trigger-type", default="none",
                       choices=["none", "double_empty_grasp", "stagnation"],
                       help="Failure type detected by programmatic detector")
    args = parser.parse_args()

    # ---- Load VLM ----------------------------------------------------------
    print(f"Loading VLM from {args.checkpoint} ...", file=sys.stderr)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.checkpoint),
        dtype="auto",
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    print("VLM loaded.", file=sys.stderr)

    # ---- Load episode data at t* -------------------------------------------
    print(f"Loading episode data from {args.hdf5_path} ...", file=sys.stderr)
    with h5py.File(args.hdf5_path, "r") as f:
        if args.t_star < 0 or args.t_star >= f["proprio"].shape[0]:
            print(f"ERROR: t_star={args.t_star} out of range [0, {f['proprio'].shape[0]})",
                  file=sys.stderr)
            sys.exit(1)

        primary = _load_image(f, "primary_images_jpeg", "primary_images", args.t_star)
        wrist = _load_image(f, "wrist_images_jpeg", "wrist_images", args.t_star)
        proprio = f["proprio"][args.t_star][:]
        language = f.attrs.get("task_description", "")

    print(f"  primary_image: {primary.shape}, wrist_image: {wrist.shape}", file=sys.stderr)
    print(f"  proprio: {proprio}", file=sys.stderr)
    print(f"  language: {language}", file=sys.stderr)

    # ---- Build messages ----------------------------------------------------
    pose_text = _format_pose(proprio)

    # Inject failure-type context to guide the VLM's attention.
    _TRIGGER_CONTEXT: dict[str, str] = {
        "none": "",
        "double_empty_grasp": (
            "Context: the robot has just attempted two nearby grasps "
            "and both failed — the gripper closed but caught nothing. "
            "Focus on which spatial misalignment (horizontal, vertical, "
            "depth, or orientation) most plausibly explains why the grasp "
            "missed the target."
        ),
        "stagnation": (
            "Context: the robot is stuck in place — it has been commanding "
            "movement but barely displacing for several action chunks. "
            "Stagnation often means a pure translation correction is "
            "insufficient. Pay extra attention to the gripper orientation: "
            "is roll, pitch, or yaw misaligned for the current task stage?"
        ),
    }
    trigger_note = _TRIGGER_CONTEXT.get(args.trigger_type, "")
    user_text = f"Task: {language}\n\n{trigger_note}\n\nCurrent gripper pose:\n{pose_text}".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(primary)},
            {"type": "image", "image": Image.fromarray(wrist)},
            {"type": "text", "text": user_text},
        ]},
    ]

    # ---- Inference ---------------------------------------------------------
    print("Running VLM inference ...", file=sys.stderr)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    raw_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True,
    )[0]

    # ---- Parse + convert -------------------------------------------------
    features = _parse_spatial_output(raw_text)

    # ---- Detect stagnation from proprio data ------------------------------
    # Stagnation: the last N steps show negligible position change.
    _STAGNATION_WINDOW = 32  # two action chunks
    _STAGNATION_THRESHOLD = 0.005  # max displacement in meters
    with h5py.File(args.hdf5_path, "r") as f:
        T = f["proprio"].shape[0]
        start_t = max(0, args.t_star - _STAGNATION_WINDOW)
        end_t = min(T, args.t_star + 1)
        positions = f["proprio"][start_t:end_t, 2:5].astype(np.float64)
        displacement = float(np.max(np.linalg.norm(
            positions - positions[0], axis=1
        )))
        stagnant = displacement <= _STAGNATION_THRESHOLD

    direction, confidence, axis_label = _features_to_direction(features)

    print(f"\ntarget_object       = {features.get('target_object')}")
    print(f"target_horizontal   = {features.get('target_horizontal')}")
    print(f"target_vertical     = {features.get('target_vertical')}")
    print(f"target_depth        = {features.get('target_depth')}")
    print(f"orientation_issue   = {features.get('orientation_issue')}")
    print(f"orientation_sign    = {features.get('orientation_sign')}")
    print(f"gripper_state       = {features.get('gripper_state')}")
    print(f"stagnant (detected) = {stagnant}  (displacement={displacement:.4f}m over {_STAGNATION_WINDOW} steps)")
    print(f"trigger_type        = {args.trigger_type}")
    print(f"---")
    print(f"chosen_axis  = {axis_label}")
    print(f"direction_6d = {direction}")
    print(f"confidence   = {confidence:.3f}")
    print(f"\n{'─' * 60}\nRaw VLM output:\n{'─' * 60}\n{raw_text}")


if __name__ == "__main__":
    main()
