#!/usr/bin/env python3
"""Render LIBERO demonstrations with skill-segment annotations.

The script reads JPEG observations directly from the regenerated HDF5 files and
uses a ``segments.json`` manifest produced by ``label_libero_skill_segments.py``.
It does not load a policy model and does not require a GPU.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
DEFAULT_MANIFEST = ROOT / "skill_memory/libero_10/segments.json"
DEFAULT_OUTPUT = ROOT / "skill_memory/libero_10/segment_videos"

VIEW_DATASETS = {
    "agentview": "agentview_rgb_jpeg",
    "wrist": "eye_in_hand_rgb_jpeg",
}
SKILL_COLORS = {
    "Pick": (52, 152, 219),
    "PlaceIn": (46, 204, 113),
    "PlaceOn": (39, 174, 96),
    "PushTo": (26, 188, 156),
    "Open": (243, 156, 18),
    "Close": (155, 89, 182),
    "TurnOn": (231, 76, 60),
    "TurnOff": (127, 140, 141),
}
BACKGROUND = (20, 23, 28)
PANEL = (31, 36, 43)
TEXT = (239, 242, 245)
MUTED = (168, 177, 186)
WARNING = (255, 95, 86)
READY_EVENT = (79, 209, 255)
READY_SHARED = (194, 126, 255)
READY_FALLBACK = (255, 184, 77)
PHASE_OK_COLOR = (110, 231, 183)
PHASE_ERROR_COLOR = WARNING
PHASE_UNKNOWN_COLOR = READY_FALLBACK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default="LIBERO-Cosmos-Policy/success_only/libero_10_regen")
    parser.add_argument("--segments-manifest", type=Path, default="skill_memory/libero_10/segments_ready.jso")
    parser.add_argument("--output-dir", type=Path, default="outputs/libero_10")
    parser.add_argument("--task", help="Render only one exact task name")
    parser.add_argument(
        "--demo",
        help="Render only one demo id, for example demo_0 or 0",
    )
    parser.add_argument(
        "--view",
        choices=("both", "agentview", "wrist"),
        default="both",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Integer rendering scale; 2 produces a 1024-pixel-wide video",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=0,
        help="Maximum demos per task after filtering; 0 means all manifest records",
    )
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned outputs without opening HDF5 files",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if args.max_demos < 0:
        parser.error("--max-demos cannot be negative")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if args.demo is not None and not args.demo.startswith("demo_"):
        args.demo = f"demo_{args.demo}"
    return args


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def decode_jpeg(encoded: Any) -> Image.Image:
    if hasattr(encoded, "tobytes"):
        encoded = encoded.tobytes()
    with Image.open(io.BytesIO(encoded)) as image:
        return image.convert("RGB")


def color_for(segment: dict[str, Any]) -> tuple[int, int, int]:
    return SKILL_COLORS.get(segment.get("skill", ""), (90, 110, 130))


def dim_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(24, round(channel * 0.38)) for channel in color)


def terminal_boundary(segment: dict[str, Any]) -> int | None:
    value = segment.get("terminal_start")
    if value is None:
        return None
    start, end, boundary = int(segment["start"]), int(segment["end"]), int(value)
    return boundary if start <= boundary <= end else None


def boundary_color(segment: dict[str, Any]) -> tuple[int, int, int]:
    method = str(segment.get("boundary_method", ""))
    if method.startswith("final_"):
        return READY_EVENT
    if method == "same_task_shared_suffix":
        return READY_SHARED
    return READY_FALLBACK


def format_segment(segment: dict[str, Any]) -> str:
    arguments = segment.get("arguments") or {}
    values = [str(value) for value in arguments.values()]
    return f"{segment.get('skill', 'Unknown')}({', '.join(values)})"


def fit_text(text: str, draw: ImageDraw.ImageDraw, text_font: ImageFont.ImageFont, width: int) -> str:
    """Truncate one line to a pixel width while preserving a useful prefix."""
    if draw.textlength(text, font=text_font) <= width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid] + suffix
        if draw.textlength(candidate, font=text_font) <= width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + suffix


def render_phase_verifier_overlay(
    image: np.ndarray,
    phase: dict[str, Any] | None,
) -> np.ndarray:
    """Overlay online phase-verifier state on one RGB rollout frame.

    This function only renders a copy. The policy observation and Cosmos future
    predictions remain untouched.
    """
    if phase is None:
        return image
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HxWx3 RGB image, got {image.shape}/{image.dtype}")

    height, width = image.shape[:2]
    scale = max(1, round(min(height, width) / 256))
    margin = 6 * scale
    padding = 7 * scale
    line_height = 15 * scale
    title_font = font(12 * scale, bold=True)
    body_font = font(9 * scale)
    event_font = font(10 * scale, bold=True)

    status = str(phase.get("status", "PHASE_UNKNOWN"))
    status_color = {
        "PHASE_OK": PHASE_OK_COLOR,
        "PHASE_ERROR": PHASE_ERROR_COLOR,
        "PHASE_UNKNOWN": PHASE_UNKNOWN_COLOR,
    }.get(status, MUTED)
    skill = str(phase.get("skill", "Unknown"))
    skill_color = SKILL_COLORS.get(skill, (90, 110, 130))
    progress = phase.get("progress_px")
    progress_text = "warmup" if progress is None else f"{float(progress):+.1f}px"
    confidence = float(phase.get("confidence", 0.0))
    votes = int(phase.get("votes", 0))
    next_step = phase.get("next_step_id")
    next_text = "end" if next_step is None else f"P{next_step}"
    evidence = int(phase.get("switch_evidence", 0))

    event = None
    event_color = MUTED
    if phase.get("switched"):
        event = f"SWITCH -> P{phase.get('step_id')}"
        event_color = PHASE_OK_COLOR
    elif phase.get("deviation_candidate"):
        event = "DEVIATION CANDIDATE"
        event_color = PHASE_ERROR_COLOR

    rows = 3 + int(event is not None)
    panel_height = padding * 2 + rows * line_height
    base = Image.fromarray(image, mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box = (margin, margin, width - margin, min(height - margin, margin + panel_height))
    border_color = event_color if event is not None else status_color
    draw.rounded_rectangle(
        box,
        radius=5 * scale,
        fill=(*BACKGROUND, 218),
        outline=(*border_color, 255),
        width=max(1, 2 * scale),
    )
    draw.rounded_rectangle(
        (margin, margin, margin + 5 * scale, box[3]),
        radius=2 * scale,
        fill=(*skill_color, 255),
    )

    left = margin + 11 * scale
    right = width - margin - padding
    available = max(1, right - left)
    y = margin + padding - scale
    status_width = draw.textlength(status, font=body_font)
    title = fit_text(
        f"P{phase.get('step_id')}  {skill}",
        draw,
        title_font,
        max(1, available - round(status_width) - 10 * scale),
    )
    draw.text((left, y), title, fill=TEXT, font=title_font)
    draw.text((right - status_width, y + 2 * scale), status, fill=status_color, font=body_font)

    y += line_height
    metrics = f"progress {progress_text}  |  conf {confidence:.2f}  |  votes {votes}"
    draw.text(
        (left, y),
        fit_text(metrics, draw, body_font, available),
        fill=TEXT,
        font=body_font,
    )
    y += line_height
    transition = f"next {next_text}  |  switch evidence {evidence}"
    draw.text(
        (left, y),
        fit_text(transition, draw, body_font, available),
        fill=MUTED,
        font=body_font,
    )
    if event is not None:
        y += line_height
        draw.text((left, y), event, fill=event_color, font=event_font)

    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"), dtype=np.uint8)


def active_segment(
    segments: list[dict[str, Any]], frame: int
) -> tuple[int, dict[str, Any]] | None:
    for index, segment in enumerate(segments):
        start, end = int(segment["start"]), int(segment["end"])
        if start <= frame < end:
            return index, segment
    return None


def successful_segments(
    segments: list[dict[str, Any]], frame: int
) -> list[dict[str, Any]]:
    return [
        segment
        for segment in segments
        if int(segment["success_start"]) <= frame < int(segment["success_end"])
    ]


def draw_timeline(
    draw: ImageDraw.ImageDraw,
    segments: list[dict[str, Any]],
    frame: int,
    total_frames: int,
    left: int,
    top: int,
    width: int,
    height: int,
    scale: int,
) -> None:
    bar_top = top + 20 * scale
    bar_height = 18 * scale
    draw.rounded_rectangle(
        (left, bar_top, left + width, bar_top + bar_height),
        radius=4 * scale,
        fill=(67, 74, 82),
    )

    def x_at(value: int) -> int:
        value = min(max(value, 0), total_frames)
        return left + round(value / max(total_frames, 1) * width)

    for segment in segments:
        start, end = int(segment["start"]), int(segment["end"])
        color = color_for(segment)
        boundary = terminal_boundary(segment)
        if end > start:
            x0, x1 = x_at(start), x_at(end)
            x1 = max(x1, x0 + 1)
            if boundary is None:
                draw.rectangle((x0, bar_top, x1, bar_top + bar_height), fill=color)
            else:
                boundary_x = x_at(boundary)
                if boundary_x > x0:
                    draw.rectangle(
                        (x0, bar_top, boundary_x, bar_top + bar_height),
                        fill=dim_color(color),
                    )
                if x1 > boundary_x:
                    draw.rectangle(
                        (boundary_x, bar_top, x1, bar_top + bar_height), fill=color
                    )
        else:
            x = x_at(start)
            points = [
                (x, bar_top - 7 * scale),
                (x - 5 * scale, bar_top - scale),
                (x + 5 * scale, bar_top - scale),
            ]
            draw.polygon(points, fill=color)

        success_start = int(segment["success_start"])
        success_end = int(segment["success_end"])
        sx0, sx1 = x_at(success_start), x_at(success_end)
        if sx0 < left + width:
            draw.rectangle(
                (
                    sx0,
                    bar_top + bar_height - 3 * scale,
                    max(sx1, sx0 + 2 * scale),
                    bar_top + bar_height,
                ),
                fill=(255, 255, 255),
            )

        if boundary is not None:
            boundary_x = x_at(boundary)
            marker_color = boundary_color(segment)
            draw.line(
                (
                    boundary_x,
                    bar_top - 2 * scale,
                    boundary_x,
                    bar_top + bar_height + 3 * scale,
                ),
                fill=marker_color,
                width=max(1, 2 * scale),
            )
            draw.polygon(
                [
                    (boundary_x, bar_top - 8 * scale),
                    (boundary_x - 5 * scale, bar_top - 3 * scale),
                    (boundary_x + 5 * scale, bar_top - 3 * scale),
                ],
                fill=marker_color,
            )

    marker_x = x_at(frame)
    draw.line(
        (marker_x, bar_top - 9 * scale, marker_x, bar_top + bar_height + 8 * scale),
        fill=(255, 255, 255),
        width=max(1, 2 * scale),
    )
    small = font(10 * scale)
    draw.text((left, top), "0", fill=MUTED, font=small)
    end_label = str(total_frames - 1)
    end_width = draw.textlength(end_label, font=small)
    draw.text((left + width - end_width, top), end_label, fill=MUTED, font=small)
    frame_label = f"frame {frame}/{total_frames - 1}"
    frame_width = draw.textlength(frame_label, font=small)
    draw.text(
        (left + (width - frame_width) / 2, top),
        frame_label,
        fill=TEXT,
        font=small,
    )


def render_frame(
    images: list[tuple[str, Image.Image]],
    record: dict[str, Any],
    frame: int,
    total_frames: int,
    scale: int,
) -> np.ndarray:
    segments = record.get("segments") or []
    image_width = 512 * scale // len(images)
    image_height = image_width
    width = image_width * len(images)
    header_height = 44 * scale
    info_height = 102 * scale
    timeline_height = 62 * scale
    height = header_height + image_height + info_height + timeline_height
    if height % 2:
        height += 1

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    header_font = font(15 * scale, bold=True)
    normal_font = font(13 * scale)
    small_font = font(11 * scale)

    current = active_segment(segments, frame)
    current_color = color_for(current[1]) if current else (91, 99, 110)
    draw.rectangle((0, 0, width, header_height), fill=current_color)
    demo_label = f"{record['demo_id']}  |  {frame:04d}/{total_frames - 1:04d}"
    draw.text((12 * scale, 11 * scale), demo_label, fill=(255, 255, 255), font=header_font)
    if current:
        skill_label = f"{current[0] + 1}/{len(segments)}  {format_segment(current[1])}"
    else:
        skill_label = "post-success tail" if segments else "unlabeled"
    skill_label = fit_text(skill_label, draw, header_font, width - 210 * scale)
    skill_width = draw.textlength(skill_label, font=header_font)
    draw.text(
        (width - skill_width - 12 * scale, 11 * scale),
        skill_label,
        fill=(255, 255, 255),
        font=header_font,
    )

    for index, (view_name, image) in enumerate(images):
        image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)
        x = index * image_width
        canvas.paste(image, (x, header_height))
        label_box = (x + 8 * scale, header_height + 8 * scale)
        label_width = draw.textlength(view_name, font=small_font) + 12 * scale
        draw.rounded_rectangle(
            (
                label_box[0] - 4 * scale,
                label_box[1] - 2 * scale,
                label_box[0] + label_width,
                label_box[1] + 15 * scale,
            ),
            radius=3 * scale,
            fill=(0, 0, 0),
        )
        draw.text(label_box, view_name, fill=(255, 255, 255), font=small_font)

    ready_segment = None
    if current:
        boundary = terminal_boundary(current[1])
        if boundary is not None and abs(frame - boundary) <= 2:
            ready_segment = current[1]
            draw.rectangle(
                (0, header_height, width - 1, header_height + image_height - 1),
                outline=boundary_color(ready_segment),
                width=max(2, 3 * scale),
            )

    info_top = header_height + image_height
    draw.rectangle((0, info_top, width, info_top + info_height), fill=PANEL)
    task_text = record["task_name"].replace("_", " ")
    task_text = fit_text(task_text, draw, normal_font, width - 24 * scale)
    draw.text((12 * scale, info_top + 8 * scale), task_text, fill=TEXT, font=normal_font)

    if current:
        segment = current[1]
        detail = (
            f"range [{segment['start']}, {segment['end']})  "
            f"success [{segment['success_start']}, {segment['success_end']})  "
            f"status={segment.get('status', 'unknown')}"
        )
    else:
        last_end = int(segments[-1]["end"]) if segments else 0
        detail = f"No active action segment; labeled actions end at frame {last_end}."
    draw.text((12 * scale, info_top + 35 * scale), detail, fill=MUTED, font=small_font)

    messages: list[tuple[str, tuple[int, int, int]]] = []
    if current and (boundary := terminal_boundary(current[1])) is not None:
        segment = current[1]
        method = segment.get("boundary_method", "unknown")
        score = float(segment.get("boundary_confidence", 0.0))
        prefix = "READY — " if ready_segment is not None else ""
        messages.append(
            (
                f"{prefix}τ={boundary}; terminal [{boundary}, {segment['end']}); "
                f"method={method}; score={score:.3f}",
                boundary_color(segment),
            )
        )
    successes = successful_segments(segments, frame)
    if successes:
        names = ", ".join(format_segment(segment) for segment in successes)
        messages.append((f"stable success window: {names}", (110, 231, 183)))
    if frame == 0:
        pre_satisfied = [
            format_segment(segment)
            for segment in segments
            if int(segment["start"]) == int(segment["end"]) == 0
        ]
        if pre_satisfied:
            messages.append(("already satisfied: " + ", ".join(pre_satisfied), (255, 190, 92)))
    terminal = [
        segment
        for segment in segments
        if int(segment["success_start"]) >= int(record["num_states"])
    ]
    if terminal and frame == total_frames - 1:
        names = ", ".join(format_segment(segment) for segment in terminal)
        messages.append((f"terminal replay success after this recorded frame: {names}", WARNING))
    if not record.get("valid", False):
        messages.append((f"INVALID LABEL: {record.get('error')}", WARNING))

    message_y = info_top + 60 * scale
    for message, color in messages[:2]:
        message = fit_text(message, draw, small_font, width - 24 * scale)
        draw.text((12 * scale, message_y), message, fill=color, font=small_font)
        message_y += 18 * scale

    timeline_top = info_top + info_height
    draw_timeline(
        draw,
        segments,
        frame,
        total_frames,
        left=12 * scale,
        top=timeline_top + 5 * scale,
        width=width - 24 * scale,
        height=timeline_height - 10 * scale,
        scale=scale,
    )
    return np.asarray(canvas, dtype=np.uint8)


def ffmpeg_command(
    ffmpeg: str,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def selected_views(view: str) -> tuple[str, ...]:
    if view == "both":
        return "agentview", "wrist"
    return (view,)


def render_video(
    h5_path: Path,
    record: dict[str, Any],
    output_path: Path,
    view: str,
    fps: float,
    scale: int,
    crf: int,
    preset: str,
    overwrite: bool,
    ffmpeg: str,
) -> str:
    if output_path.exists() and not overwrite:
        return "skipped"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    if temp_path.exists():
        temp_path.unlink()

    with h5py.File(h5_path, "r") as h5:
        data_path = f"data/{record['demo_id']}/obs"
        if data_path not in h5:
            raise KeyError(f"Missing {data_path} in {h5_path}")
        observations = h5[data_path]
        views = selected_views(view)
        datasets = [(name, observations[VIEW_DATASETS[name]]) for name in views]
        frame_counts = {name: len(dataset) for name, dataset in datasets}
        expected = int(record["num_states"])
        if any(count != expected for count in frame_counts.values()):
            print(
                f"  WARNING: {record['demo_id']} manifest states={expected}, "
                f"image frames={frame_counts}; rendering common prefix",
                file=sys.stderr,
            )
        total_frames = min(expected, *(len(dataset) for _, dataset in datasets))
        if total_frames <= 0:
            raise ValueError(f"No renderable frames for {record['task_name']}/{record['demo_id']}")

        first_images = [(name, decode_jpeg(dataset[0])) for name, dataset in datasets]
        first_frame = render_frame(first_images, record, 0, total_frames, scale)
        height, width = first_frame.shape[:2]
        command = ffmpeg_command(
            ffmpeg, temp_path, width, height, fps, crf, preset
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            process.stdin.write(first_frame.tobytes())
            for frame in range(1, total_frames):
                images = [
                    (name, decode_jpeg(dataset[frame]))
                    for name, dataset in datasets
                ]
                annotated = render_frame(images, record, frame, total_frames, scale)
                process.stdin.write(annotated.tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
        except (BrokenPipeError, KeyboardInterrupt):
            process.kill()
            process.wait()
            raise
        if return_code:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg exited with {return_code}: {stderr.strip()}")
    temp_path.replace(output_path)
    return "written"


def choose_records(
    records: Iterable[dict[str, Any]],
    task: str | None,
    demo: str | None,
    max_demos: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if task is not None and record["task_name"] != task:
            continue
        if demo is not None and record["demo_id"] != demo:
            continue
        grouped[record["task_name"]].append(record)
    chosen: list[dict[str, Any]] = []
    for task_name in sorted(grouped):
        task_records = sorted(
            grouped[task_name], key=lambda record: int(record["demo_id"].split("_")[1])
        )
        if max_demos:
            task_records = task_records[:max_demos]
        chosen.extend(task_records)
    return chosen


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    manifest_path = args.segments_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Segment manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    records = choose_records(
        manifest.get("records", []), args.task, args.demo, args.max_demos
    )
    if not records:
        raise ValueError("No manifest records matched the requested filters")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and not args.dry_run:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    print(
        f"Rendering {len(records)} videos from {manifest_path}\n"
        f"Input:  {input_dir}\nOutput: {output_dir}",
        flush=True,
    )
    written = skipped = failed = 0
    for index, record in enumerate(records, start=1):
        task_name, demo_id = record["task_name"], record["demo_id"]
        h5_path = input_dir / f"{task_name}_demo.hdf5"
        output_path = output_dir / task_name / f"{demo_id}_{args.view}_segments.mp4"
        print(f"[{index:03d}/{len(records):03d}] {task_name}/{demo_id}", flush=True)
        if args.dry_run:
            print(f"  -> {output_path}")
            continue
        if not h5_path.is_file():
            print(f"  ERROR: missing {h5_path}", file=sys.stderr)
            failed += 1
            continue
        try:
            result = render_video(
                h5_path=h5_path,
                record=record,
                output_path=output_path,
                view=args.view,
                fps=args.fps,
                scale=args.scale,
                crf=args.crf,
                preset=args.preset,
                overwrite=args.overwrite,
                ffmpeg=ffmpeg,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            failed += 1
            continue
        if result == "written":
            written += 1
            print(f"  wrote {output_path}")
        else:
            skipped += 1
            print(f"  exists, skipped (use --overwrite to replace): {output_path}")
    print(f"Done: written={written}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
