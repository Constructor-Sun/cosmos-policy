#!/usr/bin/env python3
"""Evaluate PCA-128 clean recovery with the existing online action evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import phase8_eval_latent_correction as evaluator


PACKAGE_DIR = Path(__file__).resolve().parents[3] / "checkpoints" / "mlp_pca128_rollout_package"
sys.path.insert(0, str(PACKAGE_DIR))
from inference import load_model, predict_delta_2048  # noqa: E402


class Pca128ActionCorrector(torch.nn.Module):
    """Broadcast a mean-pooled PCA prediction across the action latent's H,W axes."""

    def __init__(self, pipeline: dict, suite: str) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.suite = suite

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 5 or hidden.shape[-1] != 2048:
            raise ValueError(f"expected (B, slots, H, W, 2048), got {tuple(hidden.shape)}")
        pooled = hidden.float().mean(dim=(-3, -2))
        delta = predict_delta_2048(
            self.pipeline, pooled.detach().cpu().numpy().reshape(-1, 2048), suite=self.suite
        )
        delta = torch.as_tensor(delta, dtype=hidden.dtype, device=hidden.device)
        delta = delta.reshape(pooled.shape).unsqueeze(-2).unsqueeze(-2)
        return delta.expand_as(hidden)


def reconstruct_camera_pair(summary: dict, instruction: str, episode: int) -> dict:
    """Reconstruct one fixed-camera pair when the original episode JSON is absent."""

    conditions = {row["condition"]: row for row in summary["conditions"]}
    clean, camera = conditions["clean"], conditions["camera_viewpoints"]
    num_pairs = int(summary["num_pairs"])
    if not 0 <= episode < num_pairs:
        raise ValueError(f"--episode must be in [0, {num_pairs - 1}], got {episode}")
    if int(clean["successes"]) != int(clean["num_trials"]):
        raise ValueError("summary-only mode requires all clean episodes to have succeeded")

    status = next(row for row in summary["pair_status"] if row["condition"] == "camera_viewpoints")
    pert_success = episode not in set(status.get("flipped_episodes", []))
    common = {
        "suite": summary["suite"],
        "seed": int(summary["seed"]),
        "deterministic_reset": bool(summary["deterministic_reset"]),
        "deterministic_reset_seed": int(summary["deterministic_reset_seed"]),
        "base_task": summary["base_task"],
        "language": instruction,
        "episode": episode,
        "init_state_index": episode,
    }
    clean_ep = {**common, "condition": "clean", "task_name": clean["task_name"], "success": True}
    pert_ep = {
        **common,
        "condition": "camera_viewpoints",
        "task_name": camera["task_name"],
        "instruction_mode": "base",
        "success": pert_success,
    }
    return {
        "condition": "camera_viewpoints",
        "group": evaluator.phase2.pair_group(True, pert_success),
        "clean": clean_ep,
        "pert": pert_ep,
    }


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--summary", required=True)
    wrapper.add_argument("--suite", default="", help="Overrides the suite stored in --summary.")
    wrapper.add_argument("--cuda-device", type=int, default=None)
    wrapper.add_argument("--episode", type=int, default=None)
    wrapper.add_argument("--instruction", default="")
    args, forwarded = wrapper.parse_known_args()

    if args.cuda_device is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-device was provided, but CUDA is not available")
        torch.cuda.set_device(args.cuda_device)

    summary_path = Path(args.summary).expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(
            "--summary must name one full-rollout *summary.json file; "
            f"got {args.summary!r} (resolved to {summary_path.resolve()})"
        )
    args.summary = str(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suite = args.suite or str(summary.get("suite", ""))
    if not suite:
        raise ValueError("summary has no suite; pass --suite")
    pipeline = load_model(PACKAGE_DIR / "models" / "mlp_pca128_mixed_random.pkl")
    if pipeline["metrics"].get("suite_onehot", False) and suite not in pipeline["suite_to_index"]:
        raise ValueError(
            f"unsupported suite {suite!r}; expected one of {sorted(pipeline['suite_to_index'])}"
        )

    def load_pca_corrector(_path, _device):
        # The PCA pipeline returns deltas at their original scale, so target_rms is one.
        return Pca128ActionCorrector(pipeline, suite), {
            "target": {"layer": 27, "target": "action", "target_rms": 1.0}
        }

    evaluator.load_corrector = load_pca_corrector
    if args.episode is not None:
        if not args.instruction:
            raise ValueError("--instruction is required with --episode")
        pair = reconstruct_camera_pair(summary, args.instruction, args.episode)

        def discover_one(_summary_path, conditions, groups):
            if "camera_viewpoints" not in conditions or pair["group"] not in groups:
                return summary, []
            return summary, [pair]

        evaluator.phase2.discover_pairs = discover_one
    sys.argv = [
        sys.argv[0], "--checkpoint", str(PACKAGE_DIR), "--summary", args.summary, *forwarded
    ]
    evaluator.main()


if __name__ == "__main__":
    main()
