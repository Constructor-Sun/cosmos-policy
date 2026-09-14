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


def _jpeg_bytes(frame: np.ndarray) -> np.ndarray:
    """Encode one uint8 frame as JPEG bytes. The *_images_jpeg vlen convention
    stores REAL jpeg byte strings (decode_single_jpeg_frame does PIL
    Image.open on each element) — apply_jpeg_compression_np returns a
    re-decoded image ARRAY and must not be used here (data-corruption bug
    found 2026-09-13: vlen-of-pixels wrote garbage). Returned as a 1-D uint8
    ndarray because h5py's vlen writer requires ndarray elements (raw bytes
    raise 'bytes object has no attribute dtype')."""
    import io

    from PIL import Image

    assert frame.dtype == np.uint8, f"expected uint8 frame, got {frame.dtype}"
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=95)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def episode_file_complete(path: Path) -> bool:
    """True iff the file opens and carries every dataset/attr the dataset
    loader needs — a crash mid-write leaves a partial file that must be
    re-captured (overwritten via 'w' mode), never trusted."""
    try:
        with h5py.File(path, "r") as f:
            needed = ("primary_images_jpeg", "wrist_images_jpeg", "actions", "proprio")
            return all(k in f for k in needed) and "success" in f.attrs
    except OSError:
        return False


def save_episode(path: Path, primary, wrist, proprio, actions, success, task_description, tag,
                 suite_task_name: str):
    with h5py.File(path, "w") as f:
        for key, frames in (("primary", primary), ("wrist", wrist)):
            jpeg = [_jpeg_bytes(frame) for frame in frames]
            f.create_dataset(f"{key}_images_jpeg", data=jpeg, dtype=h5py.vlen_dtype(np.dtype("uint8")))
        f.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
        f.create_dataset("proprio", data=np.asarray(proprio, dtype=np.float32))
        f.attrs["success"] = bool(success)
        f.attrs["task_description"] = task_description
        f.attrs["suite_task_name"] = suite_task_name
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
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    resize_size = get_image_resize_size("cosmos")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One env per EXACT suite task variant: the init-state list and the
    # language description (T5 lookup key) are per-variant. The old
    # task-prefix matcher always hit the FIRST *_view variant (e.g.
    # initstate_4 instead of the meta's initstate_274) and replayed every
    # case from the WRONG initial state (proprio off by ~0.2 from t=0,
    # found 2026-09-13) — group by suite_task_name and match by equality.
    by_task: dict[str, list[tuple[str, dict]]] = {}
    for tag in tags:
        meta_i = screening[tag]
        key = meta_i.get("suite_task_name") or meta_i["task"]
        by_task.setdefault(key, []).append((tag, meta_i))

    suite = benchmark.get_benchmark_dict()["libero_10"](category_value="Robot Initial States")
    suite_names = [suite.get_task(i).name for i in range(suite.n_tasks)]

    report = {"cases": [], "skipped_existing": 0}
    for suite_task_name, cases in by_task.items():
        if suite_task_name in suite_names:
            tid = suite_names.index(suite_task_name)
        else:
            # legacy fallback: prefix match on the base task name
            base_task = cases[0][1]["task"]
            tid = next((i for i, n in enumerate(suite_names)
                        if n.startswith(f"{base_task}_view") or n == base_task), None)
        if tid is None:
            report["cases"] += [{"tag": tag, "status": "task_not_found"} for tag, _ in cases]
            continue
        env, task_description = get_libero_env(suite.get_task(tid), "cosmos", resolution=256, camera_depths=[True, False])
        init_states = suite.get_task_init_states(tid)
        try:
            for tag, meta in cases:
                out_path = out_dir / f"tta_rejected--{tag}--success=False.hdf5"
                if out_path.exists() and episode_file_complete(out_path) \
                        and str(h5py.File(out_path, "r").attrs.get("suite_task_name", "")) == suite_task_name:
                    report["skipped_existing"] += 1
                    continue
                if out_path.exists():
                    print(f"[replay] {tag}: existing file stale/incomplete, re-capturing")
                source = Path(args.diagnosis_root) / meta["task"] / meta["init"] / "episode.h5"
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
                save_episode(out_path, primary, wrist, proprio, used, success, task_description, tag,
                             suite_task_name=suite_task_name)
                report["cases"].append({"tag": tag, "status": "saved", "path": str(out_path),
                                        "steps": len(used), "suite_task_name": suite_task_name})
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
