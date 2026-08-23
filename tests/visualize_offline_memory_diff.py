#!/usr/bin/env python3
"""Visualize old vs new phase-target crops for the same task/demo/frame.

This helps confirm whether the visual differences between
``skill_memory/libero_10/phase_targets.pt`` and
``skill_memory_test/libero_10/phase_targets.pt`` are due to environment
rendering drift or migration logic.

Run:

    python tests/visualize_offline_memory_diff.py

Only the combined ``*_compare.png`` images are saved.

Optional filters:

    python tests/visualize_offline_memory_diff.py \
        --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
        --demo demo_0 \
        --limit 10

Use ``--limit 0`` to save all matched comparisons.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_PT = REPO_ROOT / "skill_memory/libero_10/phase_targets.pt"
NEW_PT = REPO_ROOT / "skill_memory_test/libero_10/phase_targets.pt"
OUT_DIR = REPO_ROOT / "outputs" / "offline_memory_visual_diff"


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)[:80]


def _to_image(array: np.ndarray, mode: str = "RGB") -> Image.Image:
    array = np.asarray(array)
    if mode == "L":
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        elif array.max() <= 1:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array, mode="RGB")


def _paste_centered(canvas: Image.Image, image: Image.Image) -> None:
    x = (canvas.width - image.width) // 2
    y = (canvas.height - image.height) // 2
    canvas.paste(image, (max(x, 0), max(y, 0)))


def _template_key(template: dict) -> tuple:
    return (
        str(template["task_name"]),
        str(template["demo_id"]),
        int(template["planner_step_id"]),
        str(template["skill"]),
        int(template["frame"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=OLD_PT)
    parser.add_argument("--new", type=Path, default=NEW_PT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--task", default=None)
    parser.add_argument("--demo", default=None)
    parser.add_argument("--planner-step", type=int, default=None)
    parser.add_argument("--skill", default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    old_payload = torch.load(args.old, map_location="cpu", weights_only=False)
    new_payload = torch.load(args.new, map_location="cpu", weights_only=False)
    old_templates = old_payload.get("templates", [])
    new_templates = new_payload.get("templates", [])

    new_by_key = {_template_key(item): item for item in new_templates}
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for old_tpl in old_templates:
        if args.task and old_tpl["task_name"] != args.task:
            continue
        if args.demo and old_tpl["demo_id"] != args.demo:
            continue
        if args.planner_step is not None and int(old_tpl["planner_step_id"]) != args.planner_step:
            continue
        if args.skill and old_tpl["skill"] != args.skill:
            continue
        if args.frame is not None and int(old_tpl["frame"]) != args.frame:
            continue

        key = _template_key(old_tpl)
        new_tpl = new_by_key.get(key)
        if new_tpl is None:
            continue

        old_crop = _to_image(old_tpl["crop_rgb"])
        new_crop = _to_image(new_tpl["crop_rgb"])
        old_mask = _to_image(old_tpl["crop_mask"], mode="L")
        new_mask = _to_image(new_tpl["crop_mask"], mode="L")

        # Save only the combined comparison sheet.
        stem = (
            f"{saved:04d}_{_sanitize(old_tpl['task_name'])}_{old_tpl['demo_id']}_"
            f"step{old_tpl['planner_step_id']}_{old_tpl['skill']}_"
            f"frame{old_tpl['frame']}"
        )

        # Build a combined sheet.
        height = max(old_crop.height, new_crop.height, old_mask.height, new_mask.height)
        width = old_crop.width + new_crop.width + old_mask.width + new_mask.width
        sheet = Image.new("RGB", (width, height + 20), (255, 255, 255))

        def paste_at(image: Image.Image, x_offset: int) -> None:
            y = 20 + (height - image.height) // 2
            sheet.paste(image, (x_offset, y))

        paste_at(old_crop, 0)
        paste_at(new_crop, old_crop.width)
        paste_at(old_mask.convert("RGB"), old_crop.width + new_crop.width)
        paste_at(new_mask.convert("RGB"), old_crop.width + new_crop.width + old_mask.width)
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 2), "old_crop", fill=(255, 0, 0))
        draw.text((old_crop.width + 4, 2), "new_crop", fill=(255, 0, 0))
        draw.text((old_crop.width + new_crop.width + 4, 2), "old_mask", fill=(255, 0, 0))
        draw.text((old_crop.width + new_crop.width + old_mask.width + 4, 2), "new_mask", fill=(255, 0, 0))
        sheet.save(out_dir / f"{stem}_compare.png")

        saved += 1
        if args.limit > 0 and saved >= args.limit:
            break

    print(f"Saved {saved} comparison images to: {out_dir}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
