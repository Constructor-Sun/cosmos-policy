#!/usr/bin/env python3
"""Run Robometer-4B on one skill video.

This is an integration script rather than a pytest unit test because loading the
4B checkpoint requires a GPU and the optional upstream ``robometer`` package.

Video recommendation for the first experiment:
  * Use ONE fixed external/main RGB camera view.
  * The gripper, manipulated object, support surface, and destination should all
    remain visible. Start about 0.5-1 s before the skill and end about 1 s after
    the terminal state has settled.
  * Record one skill per clip and use a skill-specific instruction. Do not feed
    an entire pick-and-place episode with the instruction "pick".
  * A wrist-camera clip can be evaluated separately with a second invocation.
    Do not concatenate main and wrist images before validating each view alone.

Fill VIDEO_PATH and TASK_INSTRUCTION below, or pass --video and --task.

Example:
    conda activate cosmospolicy
    python tests/run_robometer_success_estimator.py \
        --video /path/to/main_camera_pick.mp4 \
        --task "Pick up the red mug and hold it securely above the table."

Use ``--mode prefix`` to simulate online/test-time inference: the script sends
only the frames available up to each query time and records the last-frame score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

MODEL_PATH = "/data1/liu/exp/counterfactual/checkpoints/Robometer-4B"
BASE_MODEL_PATH = "/data1/liu/exp/counterfactual/checkpoints/Qwen3-VL-4B-Instruct"
VIDEO_PATH = ""  # TODO: path to one RGB video containing one skill.
TASK_INSTRUCTION = ""  # TODO: a concrete, skill-level natural-language instruction.


def _uniform_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, num=max_frames).round().astype(np.int64)


def _load_video_rgb(video_path: Path, sample_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode a video and return RGB uint8 frames plus timestamps in seconds."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if sample_fps <= 0:
        raise ValueError("--fps must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = sample_fps
    stride = max(1, int(round(native_fps / sample_fps)))

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    source_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                timestamps.append(source_index / native_fps)
            source_index += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were decoded from: {video_path}")
    return np.stack(frames).astype(np.uint8, copy=False), np.asarray(timestamps, dtype=np.float32)


def _import_robometer() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        from robometer.models import RBM
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator
        from transformers import Qwen3VLConfig
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import Robometer or one of its dependencies from the active "
            f"Python environment. Original import error: {exc}"
        ) from exc

    # Upstream RBM supports Qwen3 per instance, but its class-level config_class
    # is still Qwen2.5. PreTrainedModel.from_pretrained() consults the class-level
    # value before constructing the instance, so make the checkpoint architecture
    # explicit for this Qwen3-based Robometer-4B checkpoint.
    RBM.config_class = Qwen3VLConfig
    return (
        ProgressSample,
        Trajectory,
        compute_batch_outputs,
        load_model_from_hf,
        setup_batch_collator,
    )


@contextmanager
def _checkpoint_with_local_base_model(
    checkpoint_path: Path,
    base_model_path: Path,
) -> Iterator[Path]:
    """Expose an unmodified checkpoint with a temporary local base-model config."""
    config_path = checkpoint_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Invalid Robometer config: {config_path}")
    config["model"]["base_model_id"] = str(base_model_path)
    # Unsloth is a training optimization and can spawn an Inductor worker pool
    # during local inference. The standard Transformers Qwen3 loader is simpler
    # and avoids that initialization path for this evaluation-only script.
    config["model"]["use_unsloth"] = False

    with tempfile.TemporaryDirectory(prefix="robometer-local-checkpoint-") as temp_dir:
        runtime_checkpoint = Path(temp_dir)
        for source in checkpoint_path.iterdir():
            # The monolithic file has tied-weight metadata but no `format=pt`.
            # Transformers otherwise prefers it over the valid indexed shards.
            if source.name in {"config.yaml", ".cache", "model.safetensors"}:
                continue
            os.symlink(
                source,
                runtime_checkpoint / source.name,
                target_is_directory=source.is_dir(),
            )
        (runtime_checkpoint / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        yield runtime_checkpoint


class RobometerSuccessEstimator:
    """Thin wrapper around the official Robometer local-inference API."""

    def __init__(
        self,
        model_path: Path,
        base_model_path: Path,
        device: torch.device,
    ) -> None:
        (
            self._progress_sample_cls,
            self._trajectory_cls,
            self._compute_batch_outputs,
            load_model_from_hf,
            setup_batch_collator,
        ) = _import_robometer()

        if not model_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")
        if not (model_path / "config.yaml").is_file():
            raise FileNotFoundError(f"Missing checkpoint config.yaml: {model_path}")
        if not base_model_path.is_dir():
            raise FileNotFoundError(f"Qwen base-model directory does not exist: {base_model_path}")
        required_base_files = (
            "config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer.json",
        )
        missing_base_files = [
            name for name in required_base_files if not (base_model_path / name).is_file()
        ]
        if missing_base_files:
            raise FileNotFoundError(
                f"Incomplete Qwen base-model directory {base_model_path}; missing {missing_base_files}"
            )

        self.device = device
        with _checkpoint_with_local_base_model(model_path, base_model_path) as runtime_checkpoint:
            self.exp_config, self.tokenizer, processor, self.model = load_model_from_hf(
                model_path=str(runtime_checkpoint),
                device=device,
            )
        self.model.eval()
        self.batch_collator = setup_batch_collator(
            processor,
            self.tokenizer,
            self.exp_config,
            is_eval=True,
        )

    @torch.inference_mode()
    def predict(self, frames: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Expected uint8 RGB frames shaped (T,H,W,3), got {frames.shape} {frames.dtype}"
            )

        trajectory = self._trajectory_cls(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": int(frames.shape[0])},
            video_embeddings=None,
        )
        sample = self._progress_sample_cls(trajectory=trajectory, sample_type="progress")
        batch = self.batch_collator([sample])
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)

        loss_config = getattr(self.exp_config, "loss", None)
        is_discrete = bool(
            loss_config and getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        )
        num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(self.exp_config.model, "progress_discrete_bins", 10)
        )
        outputs = self._compute_batch_outputs(
            self.model,
            self.tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )

        progress_pred = outputs.get("progress_pred", [])
        success_output = outputs.get("outputs_success", {}) or {}
        success_probs = success_output.get("success_probs", [])
        progress = (
            np.asarray(progress_pred[0], dtype=np.float32)
            if progress_pred
            else np.empty(0, np.float32)
        )
        success = (
            np.asarray(success_probs[0], dtype=np.float32)
            if success_probs
            else np.empty(0, np.float32)
        )
        return progress, success


def _last_or_none(values: np.ndarray) -> float | None:
    return float(values[-1]) if values.size else None


def _timed_predict(
    estimator: RobometerSuccessEstimator,
    frames: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if estimator.device.type == "cuda":
        torch.cuda.synchronize(estimator.device)
    started = time.perf_counter()
    progress, success = estimator.predict(frames, task)
    if estimator.device.type == "cuda":
        torch.cuda.synchronize(estimator.device)
    return progress, success, time.perf_counter() - started


def _run_full(
    estimator: RobometerSuccessEstimator,
    frames: np.ndarray,
    timestamps: np.ndarray,
    task: str,
    max_frames: int,
) -> dict[str, Any]:
    selected = _uniform_indices(len(frames), max_frames)
    progress, success, inference_seconds = _timed_predict(estimator, frames[selected], task)
    return {
        "mode": "full",
        "sampled_timestamps_s": timestamps[selected].round(4).tolist(),
        "progress": progress.round(6).tolist(),
        "success_probability": success.round(6).tolist(),
        "last_progress": _last_or_none(progress),
        "last_success_probability": _last_or_none(success),
        "inference_seconds": round(inference_seconds, 6),
    }


def _run_prefixes(
    estimator: RobometerSuccessEstimator,
    frames: np.ndarray,
    timestamps: np.ndarray,
    task: str,
    max_frames: int,
    prefix_step: int,
    min_prefix_frames: int,
) -> dict[str, Any]:
    """Simulate online inference without exposing any future frame to a query."""
    if prefix_step <= 0:
        raise ValueError("--prefix-step must be positive")
    if min_prefix_frames <= 0:
        raise ValueError("--min-prefix-frames must be positive")
    if len(frames) < min_prefix_frames:
        raise ValueError(
            f"Prefix mode needs at least {min_prefix_frames} decoded frames, got {len(frames)}"
        )

    prefix_ends = list(range(min_prefix_frames, len(frames) + 1, prefix_step))
    if prefix_ends[-1] != len(frames):
        prefix_ends.append(len(frames))

    queries: list[dict[str, Any]] = []
    for prefix_end in prefix_ends:
        selected = _uniform_indices(prefix_end, max_frames)
        progress, success, inference_seconds = _timed_predict(
            estimator,
            frames[selected],
            task,
        )
        queries.append(
            {
                "query_timestamp_s": round(float(timestamps[prefix_end - 1]), 4),
                "num_available_frames": prefix_end,
                "sampled_timestamps_s": timestamps[selected].round(4).tolist(),
                "last_progress": _last_or_none(progress),
                "last_success_probability": _last_or_none(success),
                "inference_seconds": round(inference_seconds, 6),
            }
        )
    inference_times = [query["inference_seconds"] for query in queries]
    return {
        "mode": "prefix",
        "queries": queries,
        "total_inference_seconds": round(sum(inference_times), 6),
        "mean_inference_seconds": round(float(np.mean(inference_times)), 6),
        "min_inference_seconds": round(min(inference_times), 6),
        "max_inference_seconds": round(max(inference_times), 6),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Robometer-4B success/progress inference on one skill video"
    )
    parser.add_argument("--model-path", default=MODEL_PATH, help="Local Robometer checkpoint directory")
    parser.add_argument(
        "--base-model-path",
        default=BASE_MODEL_PATH,
        help="Local Qwen3-VL-4B-Instruct directory",
    )
    parser.add_argument("--video", default=VIDEO_PATH, help="Path to one main- or wrist-camera RGB video")
    parser.add_argument("--task", default=TASK_INSTRUCTION, help="Concrete instruction for the skill in the video")
    parser.add_argument("--view", choices=("main", "wrist"), default="main", help="Metadata only; views are not fused")
    parser.add_argument("--fps", type=float, default=3.0, help="Decode/sample rate before selecting model frames")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum frames per model query; checkpoint used 8")
    parser.add_argument("--mode", choices=("full", "prefix"), default="full")
    parser.add_argument(
        "--prefix-step",
        type=int,
        default=3,
        help="In prefix mode, query after this many newly sampled frames",
    )
    parser.add_argument(
        "--min-prefix-frames",
        type=int,
        default=5,
        help="Delay the first prefix query; training data required at least 5 frames",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: tests/robometer_outputs)",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename stem (default: <video>_<view>_<mode>)",
    )
    return parser.parse_args()


def _curve_values(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return timestamp, progress, and success series for plotting."""
    if prediction["mode"] == "full":
        timestamps = np.asarray(prediction["sampled_timestamps_s"], dtype=np.float32)
        progress = np.asarray(prediction["progress"], dtype=np.float32)
        success = np.asarray(prediction["success_probability"], dtype=np.float32)
        return timestamps, progress, success

    queries = prediction["queries"]
    timestamps = np.asarray([query["query_timestamp_s"] for query in queries], dtype=np.float32)
    progress = np.asarray(
        [np.nan if query["last_progress"] is None else query["last_progress"] for query in queries],
        dtype=np.float32,
    )
    success = np.asarray(
        [
            np.nan
            if query["last_success_probability"] is None
            else query["last_success_probability"]
            for query in queries
        ],
        dtype=np.float32,
    )
    return timestamps, progress, success


def _plot_curve(prediction: dict[str, Any], task: str, output_path: Path) -> None:
    timestamps, progress, success = _curve_values(prediction)
    figure, axis = plt.subplots(figsize=(10, 5))

    progress_count = min(len(timestamps), len(progress))
    if progress_count:
        axis.plot(
            timestamps[:progress_count],
            progress[:progress_count],
            color="tab:blue",
            marker="o",
            linewidth=2,
            label="Progress",
        )

    success_count = min(len(timestamps), len(success))
    if success_count:
        axis.plot(
            timestamps[:success_count],
            success[:success_count],
            color="tab:orange",
            marker="o",
            linewidth=2,
            label="Success probability",
        )

    axis.set_xlabel("Video time (seconds)")
    axis.set_ylabel("Robometer score")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"Robometer {prediction['mode']} curve\n{task}")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    run_started = time.perf_counter()
    args = _parse_args()
    if not args.video:
        raise SystemExit("Set VIDEO_PATH in this file or pass --video /path/to/one_skill.mp4")
    if not args.task:
        raise SystemExit("Set TASK_INSTRUCTION in this file or pass --task 'skill instruction'")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this 4B integration test")

    video_path = Path(args.video).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    base_model_path = Path(args.base_model_path).expanduser().resolve()
    decode_started = time.perf_counter()
    frames, timestamps = _load_video_rgb(video_path, args.fps)
    decode_seconds = time.perf_counter() - decode_started
    device = torch.device("cuda")
    model_load_started = time.perf_counter()
    estimator = RobometerSuccessEstimator(model_path, base_model_path, device)
    model_load_seconds = time.perf_counter() - model_load_started

    if args.mode == "full":
        prediction = _run_full(estimator, frames, timestamps, args.task, args.max_frames)
    else:
        prediction = _run_prefixes(
            estimator,
            frames,
            timestamps,
            args.task,
            args.max_frames,
            args.prefix_step,
            args.min_prefix_frames,
        )

    result = {
        "model_path": str(model_path),
        "base_model_path": str(base_model_path),
        "video": str(video_path),
        "view": args.view,
        "task": args.task,
        "decoded_frames": int(len(frames)),
        "requested_sample_fps": float(args.fps),
        "timing_seconds": {
            "video_decode": round(decode_seconds, 6),
            "model_load": round(model_load_seconds, 6),
            "inference_total": (
                prediction["inference_seconds"]
                if prediction["mode"] == "full"
                else prediction["total_inference_seconds"]
            ),
        },
        "prediction": prediction,
    }

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parent / "robometer_outputs"
    )
    output_stem = args.output_stem or f"{video_path.stem}_{args.view}_{args.mode}"
    json_path = output_dir / f"{output_stem}.json"
    plot_path = output_dir / f"{output_stem}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_started = time.perf_counter()
    _plot_curve(prediction, args.task, plot_path)
    result["timing_seconds"]["plot"] = round(time.perf_counter() - plot_started, 6)
    result["timing_seconds"]["total_until_outputs"] = round(
        time.perf_counter() - run_started,
        6,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    json_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved raw scores: {json_path}", file=sys.stderr)
    print(f"Saved curve: {plot_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
