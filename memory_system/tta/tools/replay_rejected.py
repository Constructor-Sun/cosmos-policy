"""One-time data prep: replay STORED baseline-failure actions in the simulator
and capture full episode data (primary/wrist images, proprio, success).

Per the agreed design (2026-09-09) this does NOT run the policy: the actions
come from the diagnosis episode.h5, so the replay is deterministic by
construction, needs no GPU policy inference, and cannot drift. The env is
stepped once per stored action; observation[k] is captured BEFORE action[k]
(the write order below is the timing guarantee — it is not checkable from the
h5 afterwards).

Output (new files only): <out>/tta_rejected--<tag>--success=<bool>.hdf5 with
primary_images_jpeg / wrist_images_jpeg / actions / proprio + attrs, exactly
what memory_system/tta/dataset.py consumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/data1/liu/exp/counterfactual/external/LIBERO-plus")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

MAX_STEPS = 520
NUM_WAIT = 10


def replay_case(env, init_states, init_idx, actions: np.ndarray, resize_size: int, flip: bool):
    """Step the stored actions; capture observations. No policy involved."""
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation

    env.reset()
    obs = env.set_init_state(init_states[init_idx])
    for _ in range(NUM_WAIT):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    primary, wrist, proprio_buf = [], [], []
    success = False
    for t in range(min(len(actions), MAX_STEPS)):
        observation = prepare_observation(obs, resize_size, flip)
        primary.append(observation["primary_image"].copy())
        wrist.append(observation["wrist_image"].copy())
        proprio_buf.append(observation["proprio"].astype(np.float32))
        obs, _, done, _ = env.step(np.asarray(actions[t], dtype=np.float64).tolist())
        if done:
            success = True
            break
    return success, primary, wrist, np.stack(proprio_buf), actions[: len(primary)]


def save_episode(path: Path, primary, wrist, proprio, actions, success, task_description, tag):
    from cosmos_policy.datasets.dataset_utils import apply_jpeg_compression_np

    with h5py.File(path, "w") as f:
        for key, frames in (("primary", primary), ("wrist", wrist)):
            jpeg = [apply_jpeg_compression_np(frame, quality=95) for frame in frames]
            f.create_dataset(f"{key}_images_jpeg", data=jpeg, dtype=h5py.vlen_dtype(np.dtype("uint8")))
        f.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
        f.create_dataset("proprio", data=np.asarray(proprio, dtype=np.float32))
        f.attrs["success"] = bool(success)
        f.attrs["task_description"] = task_description
        f.attrs["tag"] = tag
        f.attrs["recorded_at"] = datetime.now().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay stored failure actions with full data capture (no policy)")
    parser.add_argument("--screening-meta", default=str(REPO / "memory_system/pointcloud_action/results/tta_screening_meta.json"))
    parser.add_argument("--diagnosis-root", default=str(REPO / "experiments/tta/tta_phase_check_v3"),
                        help="directory with <task>/<init>/episode.h5 (stored actions source)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tags", default="", help="comma-separated tags; default all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    screening = json.load(open(args.screening_meta))
    tags = sorted(screening.keys())
    if args.tags:
        keep = set(args.tags.split(","))
        tags = [t for t in tags if t in keep]
    if args.limit:
        tags = tags[: args.limit]

    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env, get_image_resize_size

    resize_size = get_image_resize_size("cosmos")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[tuple[str, dict]]] = {}
    for tag in tags:
        by_task.setdefault(screening[tag]["task"], []).append((tag, screening[tag]))

    report = {"cases": [], "skipped_existing": 0}
    for task, cases in by_task.items():
        suite = benchmark.get_benchmark_dict()["libero_10"](category_value="Robot Initial States")
        match = None
        for i in range(suite.n_tasks):
            name = suite.get_task(i).name
            if name.startswith(f"{task}_") or name == task:
                match = (i, name)
                if name.startswith(f"{task}_view"):
                    break
        if match is None:
            report["cases"] += [{"tag": tag, "status": "task_not_found"} for tag, _ in cases]
            continue
        tid, _suite_task_name = match
        env, task_description = get_libero_env(suite.get_task(tid), "cosmos", resolution=256, camera_depths=[True, False])
        init_states = suite.get_task_init_states(tid)
        try:
            for tag, meta in cases:
                out_path = out_dir / f"tta_rejected--{tag}--success=False.hdf5"
                if out_path.exists():
                    report["skipped_existing"] += 1
                    continue
                source = Path(args.diagnosis_root) / task / meta["init"] / "episode.h5"
                if not source.exists():
                    report["cases"].append({"tag": tag, "status": "no_stored_actions"})
                    continue
                with h5py.File(source, "r") as f:
                    actions = f["actions"][:]
                init_idx = int(meta["census_abs_init"])
                success, primary, wrist, proprio, used = replay_case(
                    env, init_states, init_idx, actions, resize_size, flip=True
                )
                if success:
                    # A baseline failure that succeeds on replay is not a valid rejected side.
                    report["cases"].append({"tag": tag, "status": "rerun_succeeded_skip"})
                    print(f"[replay] {tag}: replay succeeded (expected failure), skipped")
                    continue
                save_episode(out_path, primary, wrist, proprio, used, success, task_description, tag)
                report["cases"].append({"tag": tag, "status": "saved", "path": str(out_path), "steps": len(used)})
                print(f"[replay] {tag}: saved ({len(used)} steps)")
        finally:
            try:
                env.close()
            except Exception:
                pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"replay_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=1))
    print(f"[replay] report -> {report_path}")


if __name__ == "__main__":
    main()
