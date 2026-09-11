"""Build single-shot TTA repair requests from the diagnosis outputs and drive
run_libero_eval.py in repair mode WITH data collection, so the recovered
(chosen) trajectories land on disk as training-ready hdf5.

Per case:
  request = {
    "t_star":  <candidate.t_star from phase_record.json>,
    "actions": <episode.h5 actions>,
    "proprio": <episode.h5 proprio>,
    "phase":   <candidate.repair_phase dict>,
    "task":    <base task name>
  }
  env:   COSMOS_TTA_REPAIR=<request.json>  COSMOS_TTA_REPAIR_TAG=<tag>
         COSMOS_TTA_REPAIR_OUT=<result.json>  COSMOS_INIT_STATE_OFFSET=<abs_init>
  eval:  run_libero_eval.py --task-suite-name libero_10
         --task-filter <task substring> --num-trials-per-task 1 --data-collection True

Sequential, one case at a time; resumable (cases with an existing
success=True hdf5 matching the tag are skipped); a case whose repair result
reports failure is recorded and left out of the manifest (tools/build_manifest.py
also cross-checks). GPU budget/selection rules per the TTA contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from memory_system.tta.dpo_train import pick_gpu  # noqa: E402

EVAL_SCRIPT = REPO / "cosmos_policy" / "experiments" / "robot" / "libero" / "run_libero_eval.py"


def build_request(case_dir: Path, task: str) -> dict | None:
    record_path = case_dir / "phase_record.json"
    episode_path = case_dir / "episode.h5"
    if not record_path.exists() or not episode_path.exists():
        return None
    record = json.loads(record_path.read_text())
    candidate = record.get("candidate")
    if not candidate or not candidate.get("repairable", False):
        return None
    with h5py.File(episode_path, "r") as f:
        actions = f["actions"][:]
        proprio = f["proprio"][:]
    t_star = int(candidate["t_star"])
    if t_star <= 0 or t_star >= len(actions):
        return None
    return {
        "t_star": t_star,
        "actions": actions.astype(np.float32).tolist(),
        "proprio": proprio.astype(np.float32).tolist(),
        "phase": candidate["repair_phase"],
        "task": task,
    }


def find_existing_chosen(chosen_dir: Path, tag: str) -> Path | None:
    matches = sorted(chosen_dir.glob(f"*--{tag}--*success=True*.hdf5"))
    return matches[0] if matches else None


def check_episode(path: Path) -> list[str]:
    """Post-collection smoke check (absorbed into the manifest validation):
    required datasets present, all four streams the same length (obs[k] is the
    observation before action[k] — the capture appends them at one site),
    finite values, expected dims, success attr present."""
    problems = []
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if "primary_images_jpeg" in keys:
            n_primary = len(f["primary_images_jpeg"])
        elif "primary_images" in keys:
            n_primary = len(f["primary_images"])
        else:
            return ["missing primary images dataset"]
        if "wrist_images_jpeg" in keys:
            n_wrist = len(f["wrist_images_jpeg"])
        elif "wrist_images" in keys:
            n_wrist = len(f["wrist_images"])
        else:
            return ["missing wrist images dataset"]
        if "actions" not in f or "proprio" not in f:
            return ["missing actions/proprio datasets"]
        actions = f["actions"][:]
        proprio = f["proprio"][:]
        has_success = "success" in f.attrs
    lengths = {"primary": n_primary, "wrist": n_wrist, "actions": len(actions), "proprio": len(proprio)}
    if len(set(lengths.values())) != 1:
        problems.append(f"stream lengths differ: {lengths}")
    if actions.ndim != 2 or actions.shape[1] != 7:
        problems.append(f"actions shape {actions.shape} != (T, 7)")
    if proprio.ndim != 2 or proprio.shape[1] != 9:
        problems.append(f"proprio shape {proprio.shape} != (T, 9)")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(proprio)):
        problems.append("non-finite values in actions/proprio")
    if not has_success:
        problems.append("missing success attr")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect chosen (repair) episodes with data capture")
    parser.add_argument("--screening-meta", default=str(REPO / "memory_system/pointcloud_action/results/tta_screening_meta.json"))
    parser.add_argument("--repair-summary", default=str(REPO / "memory_system/pointcloud_action/results/tta_repair_68_summary.json"))
    parser.add_argument("--diagnosis-root", default=str(REPO / "experiments/tta/tta_phase_check_v3"))
    parser.add_argument("--request-dir", required=True, help="where request/result json files are written (new files)")
    parser.add_argument("--chosen-dir", required=True, help="eval rollout_data_dir (where hdf5 episodes land)")
    parser.add_argument("--tags", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget-gb", type=float, default=10.0)
    parser.add_argument("--log-dir", default=str(REPO / "experiments" / "tta_collect_logs"))
    args = parser.parse_args()

    gpu_index = pick_gpu(args.budget_gb)
    print(f"[collect] GPU {gpu_index} (least used), budget {args.budget_gb} GB")

    screening = json.load(open(args.screening_meta))
    summary = json.load(open(args.repair_summary))
    recovered = {c["tag"] for c in summary.get("cases", []) if c.get("task_success")}
    tags = sorted(t for t in screening if t in recovered)
    if args.tags:
        keep = set(args.tags.split(","))
        tags = [t for t in tags if t in keep]
    if args.limit:
        tags = tags[: args.limit]

    request_dir = Path(args.request_dir)
    chosen_dir = Path(args.chosen_dir)
    log_dir = Path(args.log_dir)
    for d in (request_dir, chosen_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"cases": [], "skipped_existing": 0, "no_request": []}

    for tag in tags:
        if find_existing_chosen(chosen_dir, tag):
            report["skipped_existing"] += 1
            continue
        meta = screening[tag]
        case_dir = Path(args.diagnosis_root) / meta["task"] / meta["init"]
        request = build_request(case_dir, meta["task"])
        if request is None:
            report["no_request"].append(tag)
            print(f"[collect] {tag}: no repairable candidate in diagnosis, skipped")
            continue

        request_path = request_dir / f"repair_request--{tag}--{stamp}.json"
        result_path = request_dir / f"repair_result--{tag}--{stamp}.json"
        request_path.write_text(json.dumps(request))

        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu_index),
            MUJOCO_GL="egl",
            HF_HUB_OFFLINE="1",
            HF_HUB_CACHE="/data1/liu/exp/counterfactual/checkpoints/huggingface-hub",
            WANDB_MODE="disabled",
            COSMOS_TTA_REPAIR=str(request_path),
            COSMOS_TTA_REPAIR_TAG=tag,
            COSMOS_TTA_REPAIR_OUT=str(result_path),
            COSMOS_INIT_STATE_OFFSET=str(int(meta["census_abs_init"])),
        )
        ckpt_dir = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            # canonical single-case invocation, underscore flags per draccus
            "--config", "cosmos_predict2_2b_480p_libero__inference_only",
            "--ckpt_path", f"{ckpt_dir}/Cosmos-Policy-LIBERO-Predict2-2B.pt",
            "--config_file", "cosmos_policy/config/config.py",
            "--dataset_stats_path", f"{ckpt_dir}/libero_dataset_statistics.json",
            "--t5_text_embeddings_path", f"{ckpt_dir}/libero_t5_embeddings.pkl",
            "--task_suite_name", "libero_10",
            "--task_filter", meta["task"],
            "--num_trials_per_task", "1",
            "--data_collection", "True",
            "--local_log_dir", str(chosen_dir.parent),
            "--run_id_note", f"tta_collect_{stamp}",
            "--seed", "7",
            "--deterministic", "True",
            "--chunk_size", "16",
            "--num_open_loop_steps", "16",
            "--use_wrist_image", "True",
            "--use_proprio", "True",
            "--normalize_proprio", "True",
            "--unnormalize_actions", "True",
            "--trained_with_image_aug", "True",
            "--use_jpeg_compression", "True",
            "--flip_images", "True",
            "--ar_future_prediction", "False",
            "--ar_value_prediction", "False",
            "--available_gpus", str(gpu_index),
        ]
        print(f"[collect] {tag}: running eval (t*={request['t_star']})")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO))
        log_path = log_dir / f"eval--{tag}--{stamp}.log"
        log_path.write_text(proc.stdout[-20000:] + "\n===== STDERR =====\n" + proc.stderr[-20000:])

        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "missing_result"}
        chosen = find_existing_chosen(chosen_dir, tag)
        smoke = check_episode(chosen) if chosen else ["no h5 produced"]
        entry = {"tag": tag, "status": result.get("status"), "task_success": result.get("task_success"),
                 "chosen_h5": str(chosen) if chosen else None, "smoke_problems": smoke,
                 "log": str(log_path)}
        report["cases"].append(entry)
        print(f"[collect] {tag}: status={entry['status']} task_success={entry['task_success']} "
              f"chosen={bool(chosen)} smoke={'PASS' if not smoke else smoke}")

    report_path = request_dir / f"collect_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=1))
    print(f"[collect] report -> {report_path}")


if __name__ == "__main__":
    main()
