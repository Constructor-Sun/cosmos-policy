#!/usr/bin/env python3
"""Build visual crops and exact features for training-free LIBERO phase verifiers."""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS = ROOT.parent / "LIBERO-plus"
if LIBERO_PLUS.is_dir() and str(LIBERO_PLUS) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS))
os.environ.setdefault("MUJOCO_GL", "egl")


def patch_numpy2_segmentation() -> None:
    """Fix robosuite's uint8 segmentation decode under NumPy 2 without editing site-packages."""
    from robosuite.utils import binding_utils as binding

    original = binding.MjRenderContext.read_pixels
    if getattr(original, "_cosmos_numpy2_safe", False):
        return

    def read_pixels(self, width, height, depth=False, segmentation=False):
        if not segmentation:
            return original(self, width, height, depth=depth, segmentation=False)
        viewport = binding.mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        binding.mujoco.mjr_readPixels(rgb=rgb, depth=depth_img, viewport=viewport, con=self.con)
        rgb32 = rgb.astype(np.int32)
        encoded = rgb32[:, :, 0] + rgb32[:, :, 1] * 256 + rgb32[:, :, 2] * 65536
        encoded[encoded >= self.scn.ngeom + 1] = 0
        ids = np.full((self.scn.ngeom + 1, 2), -1, dtype=np.int32)
        for index in range(self.scn.ngeom):
            geom = self.scn.geoms[index]
            if geom.segid != -1:
                ids[geom.segid + 1] = (geom.objtype, geom.objid)
        result = ids[encoded]
        return (result, depth_img) if depth else result

    read_pixels._cosmos_numpy2_safe = True
    binding.MjRenderContext.read_pixels = read_pixels


def resolve_bddl(task_name: str) -> Path:
    from libero.libero import get_libero_path

    sibling = LIBERO_PLUS / "libero/libero/bddl_files/libero_10" / f"{task_name}.bddl"
    installed = Path(get_libero_path("bddl_files")) / "libero_10" / f"{task_name}.bddl"
    path = sibling if sibling.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def approach_argument(skill: str, arguments: dict[str, str]) -> tuple[str, str]:
    role = "item" if skill == "Pick" else "target"
    if role not in arguments:
        if "item" in arguments:
            role = "item"
        else:
            raise KeyError(f"No approach target for {skill}({arguments})")
    return role, str(arguments[role])


def resolve_instance(argument: str, parsed: dict[str, Any]) -> tuple[str, str]:
    region = parsed.get("regions", {}).get(argument)
    if region and region.get("target"):
        return str(region["target"]), "region_owner"
    return argument, "direct"


def match_instance(name: str, instance_to_id: dict[str, int]) -> str | None:
    if name in instance_to_id:
        return name
    normalized = name.lower().replace("_", "")
    matches = [key for key in instance_to_id if key.lower().replace("_", "") == normalized]
    if len(matches) == 1:
        return matches[0]
    matches = [key for key in instance_to_id if normalized in key.lower().replace("_", "")]
    return matches[0] if len(matches) == 1 else None


def match_region_anchor(argument: str, instance_to_id: dict[str, int]) -> str | None:
    """Find a visible anchor embedded in a relative region name, e.g. plate_1."""
    normalized = argument.lower().replace("_", "")
    matches = []
    for name in instance_to_id:
        stem, suffix = name.rsplit("_", 1) if "_" in name else (name, "")
        stem = stem if suffix.isdigit() else name
        token = stem.lower().replace("_", "")
        if len(token) >= 4 and token in normalized:
            matches.append((len(token), name))
    return max(matches)[1] if matches else None


def decode_jpeg(value: Any) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)


def bbox_from_mask(mask: np.ndarray, padding: int = 4) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1), min(height, int(ys.max()) + padding + 1),
    )


def make_template(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    crop, crop_mask = rgb[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=768)
    keypoints, descriptors = sift.detectAndCompute(
        cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), crop_mask
    )
    if descriptors is None or len(keypoints) < 2:
        descriptors, points = None, None
    else:
        descriptors = descriptors.astype(np.float32)
        points = np.asarray([point.pt for point in keypoints], dtype=np.float32)
    ys, xs = np.nonzero(mask)
    return {
        "crop_rgb": crop.copy(),
        "crop_mask": crop_mask.copy(),
        "bbox_xyxy": np.asarray(bbox, dtype=np.int16),
        "target_center_xy": np.asarray([xs.mean(), ys.mean()], dtype=np.float32),
        "crop_center_xy": np.asarray([(x1 - x0 - 1) / 2, (y1 - y0 - 1) / 2], dtype=np.float32),
        "keypoints_xy": points,
        "descriptors": descriptors,
        "visible_pixels": int(mask.sum()),
    }


def chosen_frames(segment: dict[str, Any], length: int) -> list[int]:
    start = min(max(int(segment["start"]), 0), length - 1)
    end = min(max(int(segment["end"]), start + 1), length)
    middle = min(start + max((end - start) // 2, 1), end - 1)
    ready = segment.get("ready_frame")
    ready = min(max(int(ready), start), end - 1) if ready is not None else end - 1
    return sorted(set((start, middle, ready)))


def build_task(args, task_name: str) -> dict[str, Any]:
    patch_numpy2_segmentation()
    from libero.libero.envs import SegmentationRenderEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from robosuite.utils.camera_utils import (
        get_camera_transform_matrix,
        project_points_from_world_to_camera,
    )

    manifest = json.loads(args.segments_manifest.read_text())
    records = [
        item for item in manifest["records"]
        if item.get("valid") and item["task_name"] == task_name
    ][: args.max_per_task or None]
    h5_path = args.input_dir / f"{task_name}_demo.hdf5"
    bddl = resolve_bddl(task_name)
    parsed = robosuite_parse_problem(str(bddl))
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.reset()
    templates, failures, warnings = [], [], []
    camera_matrix = get_camera_transform_matrix(
        env.sim, "agentview", args.resolution, args.resolution
    )
    with h5py.File(h5_path, "r") as handle:
        for record in records:
            group = handle["data"][record["demo_id"]]
            states, images, ee_pos = (
                group["states"], group["obs"]["agentview_rgb_jpeg"], group["obs"]["ee_pos"]
            )
            for segment in record["segments"]:
                if segment.get("status") == "already_satisfied":
                    continue
                try:
                    role, argument = approach_argument(segment["skill"], segment.get("arguments", {}))
                    owner, source = resolve_instance(argument, parsed)
                    instance = match_instance(owner, env.instance_to_id)
                    if instance is None and source == "region_owner":
                        instance = match_region_anchor(argument, env.instance_to_id)
                        source = "region_anchor" if instance else source
                    if instance is None:
                        raise KeyError(f"instance {owner!r} for {argument!r} not in "
                                       f"{sorted(env.instance_to_id)}")
                    instance_id = env.instance_to_id[instance]
                    frames, template_count = chosen_frames(segment, len(states)), len(templates)
                    for frame in frames:
                        observation = env.regenerate_obs_from_state(states[frame])
                        raw_mask = observation["agentview_segmentation_instance"][..., 0] == instance_id
                        rgb = decode_jpeg(images[frame])
                        mask = np.flipud(raw_mask) if args.flip_images else raw_mask
                        if mask.shape != rgb.shape[:2]:
                            mask = cv2.resize(
                                mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        template = make_template(rgb, mask)
                        if template is None:
                            warnings.append({
                                "demo_id": record["demo_id"],
                                "planner_step_id": segment.get("planner_step_id"),
                                "frame": frame, "warning": "target mask too small",
                            })
                            continue
                        row_col = project_points_from_world_to_camera(
                            np.asarray(ee_pos[frame], dtype=np.float64), camera_matrix,
                            args.resolution, args.resolution,
                        )
                        row, col = (int(row_col[0]), int(row_col[1]))
                        if args.flip_images:
                            row = args.resolution - 1 - row
                        template.update({
                            "task_name": task_name,
                            "demo_id": record["demo_id"],
                            "planner_step_id": int(segment["planner_step_id"]),
                            "skill": segment["skill"],
                            "arguments": dict(segment.get("arguments", {})),
                            "target_role": role,
                            "target_argument": argument,
                            "instance_name": instance,
                            "instance_source": source,
                            "frame": frame,
                            "gripper_xy": np.asarray([col, row], dtype=np.float32),
                        })
                        templates.append(template)
                    if len(templates) == template_count:
                        raise ValueError(f"no usable target mask in frames {frames}")
                except (KeyError, ValueError) as error:
                    failures.append({
                        "demo_id": record["demo_id"],
                        "planner_step_id": segment.get("planner_step_id"),
                        "error": str(error),
                    })
    env.close()
    return {"templates": templates, "failures": failures, "warnings": warnings}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_dir, args.segments_manifest = args.input_dir.resolve(), args.segments_manifest.resolve()
    args.output = args.output.resolve()
    if args.task:
        result = build_task(args, args.task)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "libero_phase_targets_v1", **result}, args.output)
        print(f"{args.task}: {len(result['templates'])} templates, "
              f"{len(result['failures'])} failures, {len(result['warnings'])} warnings")
        return 0
    manifest = json.loads(args.segments_manifest.read_text())
    tasks = sorted({item["task_name"] for item in manifest["records"] if item.get("valid")})
    templates, failures, warnings = [], [], []
    with tempfile.TemporaryDirectory(prefix="libero_phase_targets_") as temporary:
        for index, task in enumerate(tasks):
            part = Path(temporary) / f"part_{index}.pt"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--input-dir", str(args.input_dir),
                "--segments-manifest", str(args.segments_manifest),
                "--output", str(part), "--task", task,
                "--max-per-task", str(args.max_per_task),
                "--resolution", str(args.resolution),
                "--flip-images" if args.flip_images else "--no-flip-images",
            ]
            subprocess.run(command, check=True)
            payload = torch.load(part, map_location="cpu", weights_only=False)
            templates.extend(payload["templates"])
            failures.extend({"task_name": task, **item} for item in payload["failures"])
            warnings.extend({"task_name": task, **item} for item in payload["warnings"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "libero_phase_targets_v1",
        "suite": "libero_10",
        "templates": templates,
        "failures": failures,
        "warnings": warnings,
    }
    torch.save(payload, args.output)
    print(f"Wrote {len(templates)} templates to {args.output}; "
          f"failures={len(failures)}; warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
