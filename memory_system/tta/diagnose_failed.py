#!/usr/bin/env python
"""Phase/point validation harness: one failed census init per task (TMP.MD).

For each libero_10 task with census failures, rerun ONE failed init with the
plain policy under the census config, record phases/events observation-only
(PhaseEventRecorder), pick the repair candidate and cut-in step t*, and write
a PNG whose left panel is the observation the repair would start from
(replay of ``actions[:t*]`` lands exactly on this frame) plus a timeline
panel showing phase spans, the command event and t*.

No repair is executed.  Output: <out>/<base_task>/init<abs>/
  - tstar.png            annotated observation @ t* + timeline
  - episode.h5           actions (T,7) and proprio (T,9), both per action step
  - phase_record.json    spans, events, classification, candidate, t*

Usage:
  python memory_system/tta/diagnose_failed.py                 # all tasks
  python memory_system/tta/diagnose_failed.py --tasks 273,274 # variant ids
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")
os.environ["HF_HUB_CACHE"] = "/data1/liu/exp/counterfactual/checkpoints/huggingface-hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = "/data1/liu/exp/counterfactual/external/cosmos-policy"
LIB = "/data1/liu/exp/counterfactual/external/LIBERO-plus"
sys.path[:0] = [REPO, LIB]

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from memory_system.tta.variant_spec import resolve_variant

MAX_STEPS = 520
NUM_WAIT = 10

CENSUS = Path(REPO) / "experiments" / "tta_census" / "robotinit_baseline_failures.json"
OUT_DIR = Path(REPO) / "experiments" / "tta_phase_check"

STATUS_COLORS = {
    "confirmed": "#2e7d32",
    "released_only": "#f9a825",
    "grasp_failed": "#c62828",
    "never_released": "#c62828",
    "timeout_gap": "#ef6c00",
    "pending": "#6a1b9a",
    "object_mismatch": "#b71c1c",
    "object_unknown": "#757575",
    "close_recorded": "#546e7a",
    "unlocatable": "#616161",
    "none": "#616161",
}


def load_policy():
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    policy_dir = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"
    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=f"{policy_dir}/Cosmos-Policy-LIBERO-Predict2-2B.pt",
        dataset_stats_path=f"{policy_dir}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{policy_dir}/libero_t5_embeddings.pkl",
        task_suite_name="libero_10",
        seed=7,
        randomize_seed=False,
        flip_images=True,
        deterministic=True,
        deterministic_reset=True,
        deterministic_reset_seed=0,
    )
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)
    resize_size = get_image_resize_size(cfg.model_family)
    return cfg, model, dataset_stats, resize_size


def run_one_init(env, task_description, init_states, init_idx, recorder,
                 cfg, model, dataset_stats, resize_size):
    """Run the plain policy once with observation-only recording.

    proprio[t] is the observation BEFORE action t, so replaying
    ``actions[:k]`` lands on state ``proprio[k]``; frames[i] is the
    agentview observation after i+1 actions (= proprio[i+1]).
    """
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    from memory_system.execute.skill_completion.shadow import PickTargetPointCloud

    env.reset()
    obs = env.set_init_state(init_states[init_idx])
    for _ in range(NUM_WAIT):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    frames_init = np.asarray(obs["agentview_image"]).copy()

    proprio_buf, action_buf, frames = [], [], []
    t = 0
    queue = []
    success = False

    def pick_points(phase, observation):
        if phase is None or phase.skill != "Pick":
            return None
        item = phase.arguments.get("item")
        if not item:
            return None
        try:
            resolution = observation["agentview_image"].shape[0]
            return PickTargetPointCloud(env, resolution).points(observation, item)
        except Exception:
            return None

    # The ten settle actions explicitly command open. This establishes the
    # cross-phase gripper state without fabricating a close event at step 0.
    recorder.begin(step=0, initial_gripper_state=0)
    while t < MAX_STEPS:
        if not queue:
            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            ard = get_action(
                cfg, model, dataset_stats, observation, task_description,
                seed=cfg.seed, randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            queue = list(ard["actions"])[: cfg.num_open_loop_steps]
        action = np.asarray(queue.pop(0), dtype=np.float64)

        proprio_buf.append(np.concatenate([
            obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"],
        ]).astype(np.float64))
        action_buf.append(action)

        obs, _, done, _ = env.step(action.tolist())
        frames.append(np.asarray(obs["agentview_image"]).copy())

        phase = recorder.active_phase
        decision = recorder.observe(
            step=t, action=action, obs=obs,
            target_points=pick_points(phase, obs),
        )
        t += 1

        if done:
            success = True
            break
        if decision is not None and decision.advance:
            continue  # next phase window already opened by the recorder
        if len(queue) == 0:
            recorder.finish_chunk(step=t - 1)

    proprio = np.stack(proprio_buf) if proprio_buf else np.zeros((0, 9))
    actions = np.stack(action_buf) if action_buf else np.zeros((0, 7))
    return success, frames_init, frames, proprio, actions


def make_png(path: Path, title: str, info_lines, img, record,
             candidate, t_star, event_step, proprio, actions):
    T = len(actions)
    fig = plt.figure(figsize=(15, 6.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5])

    # -- left: observation at t* (state the repair would start from) --------
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(np.flipud(img))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(
        f"observation at t* = {t_star}" if t_star is not None
        else "observation at diagnostic baseline (no valid t*)",
        fontsize=10,
    )
    status = candidate["status"] if candidate else "none"
    color = STATUS_COLORS.get(status, "black")
    ax_img.text(
        0.02, 0.02,
        "\n".join(info_lines),
        transform=ax_img.transAxes, fontsize=8.2, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, alpha=0.88),
    )

    # -- right: timeline (EE xyz, gripper width, phase spans, events) -------
    ax = fig.add_subplot(gs[0, 1])
    steps = np.arange(T)
    width = np.abs(proprio[:, 0] - proprio[:, 1]) * 1000.0  # mm
    for i, ph in enumerate(record["phases"]):
        start = ph["start_step"]
        end = (ph["end_step"] + 1) if ph["end_step"] is not None else T
        status = "pending" if ph["end_step"] is None else None
        if status is None:
            from memory_system.tta.phase_record import classify_phase
            status = classify_phase(ph)
        ax.axvspan(start, min(end, T), color=STATUS_COLORS[status], alpha=0.10)
        ax.text((start + min(end, T)) / 2, 1.01, f"{ph['skill']}#{ph['phase_index']}",
                transform=ax.transAxes, rotation=0, ha="center", va="bottom",
                fontsize=7, color=STATUS_COLORS[status])
    ax.plot(steps, proprio[:, 2] * 100, label="EE x (cm)", lw=1.0)
    ax.plot(steps, proprio[:, 3] * 100, label="EE y (cm)", lw=1.0)
    ax.plot(steps, proprio[:, 4] * 100, label="EE z (cm)", lw=1.4)
    ax.plot(steps, width, label="gripper width (mm)", lw=1.4, ls="--", color="k")
    if event_step is not None:
        ax.axvline(event_step, color="green", ls="--", lw=1.4,
                   label=f"gripper flip @ {event_step}")
    if t_star is not None:
        ax.axvline(t_star, color="red", lw=1.8, label=f"t* = {t_star}")
    ax.set_xlabel("policy action step")
    ax.set_ylabel("cm / mm")
    ax.set_ylim(bottom=min(-5, np.min(proprio[:, 2:5]) * 100 - 2))
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.set_title("phase spans / events / cut-in", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def diagnose_task(task_entry, variant, cfg, model, dataset_stats, resize_size, out_root,
                  video=False):
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from memory_system.tta.phase_record import (
        PhaseEventRecorder,
        compute_t_star,
        load_task_sequence,
        save_record,
    )
    from memory_system.tta.object_query import SceneObjectQuery

    base_task = task_entry["task"]
    pert_name = task_entry["task_name_perturbed"]
    abs_inits = task_entry["fail_init_indices_abs"]

    variant_spec = resolve_variant(pert_name, suite="libero_10")
    suite = benchmark.get_benchmark_dict()[variant_spec.suite](
        category_value=variant_spec.category
    )
    exact_matches = [
        i for i in range(suite.n_tasks) if suite.get_task(i).name == pert_name
    ]
    if len(exact_matches) != 1:
        raise RuntimeError(
            f"Expected one exact task match for {pert_name!r} in "
            f"category {variant_spec.category!r}, found {len(exact_matches)}"
        )
    tid = exact_matches[0]
    suite_task = suite.get_task(tid)
    suite_task_name = suite_task.name
    env, task_description = get_libero_env(
        suite_task, "cosmos", resolution=256, camera_depths=[True, False]
    )
    init_states = suite.get_task_init_states(tid)
    object_query = SceneObjectQuery(env, resolution=256)

    base_resolved, demo_id, phases = load_task_sequence(suite_task_name)
    print(
        f"[{variant}] category={variant_spec.category!r} "
        f"task={suite_task_name} bddl={suite_task.bddl_file} "
        f"phases={[p.skill for p in phases]} demo={demo_id}",
        flush=True,
    )

    result = {
        "task": base_task,
        "variant": variant,
        "suite_task": suite_task_name,
        "perturbation_category": variant_spec.category,
        "perturbation_condition": variant_spec.condition,
        "bddl_file": suite_task.bddl_file,
    }
    init_indices = abs_inits
    for init_idx in init_indices:
        recorder = PhaseEventRecorder(
            phases, task_name=base_resolved, demo_id=demo_id,
            episode_id=f"{variant}-init{init_idx}",
            object_query=object_query,
        )
        try:
            success, frames_init, frames, proprio, actions = run_one_init(
                env, task_description, init_states, init_idx,
                recorder, cfg, model, dataset_stats, resize_size,
            )
        except Exception:
            result.update({"status": "error", "init": init_idx,
                           "error": traceback.format_exc()[-800:]})
            print(f"[{variant}] init {init_idx}: ERROR\n{traceback.format_exc()[-800:]}",
                  flush=True)
            continue

        record = recorder.finalize(total_steps=len(actions), success=success)
        record["census_abs_init"] = init_idx
        record["suite_task_name"] = suite_task_name
        record["task_name_perturbed"] = pert_name
        record["perturbation_category"] = variant_spec.category
        record["perturbation_condition"] = variant_spec.condition
        record["bddl_file"] = suite_task.bddl_file
        candidate = None if success else record.get("candidate")
        t_star = event_step = None
        if candidate is not None:
            t_star, event_step = compute_t_star(candidate)
            candidate["t_star"] = t_star
            candidate["event_step"] = event_step

        case_dir = out_root / base_task / f"init{init_idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(case_dir / "episode.h5", "w") as h5:
            h5.create_dataset("actions", data=actions)
            h5.create_dataset("proprio", data=proprio)
        save_record(record, case_dir / "phase_record.json")

        cand = record["candidate"]
        info_lines = [
            f"task: {base_task[:52]}",
            f"category: {variant_spec.category}",
            f"actual: {suite_task_name[-62:]}",
            f"variant {variant} / init {init_idx}  success={success}",
            "candidate: none" if cand is None else
            (f"candidate: {cand['span']['skill']}#{cand['span']['phase_index']}"
             f"  status={cand['status']}"),
        ]
        if cand is not None:
            info_lines += [
                f"first interaction: {cand['event_step']}",
                f"t* = {t_star}",
                f"total action steps: {len(actions)}",
            ]
        title = (f"{base_task[:60]}  |  {variant} init {init_idx}  |  "
                 f"success={success}  t*={t_star}")
        image_step = 0 if t_star is None else int(t_star)
        img = frames_init if (image_step == 0 or not frames) else \
            frames[min(image_step - 1, len(frames) - 1)]
        png = case_dir / "tstar.png"
        make_png(png, title, info_lines, img, record,
                 candidate or {"status": "none"},
                 image_step, event_step, proprio, actions)

        if video:
            import imageio
            mp4 = case_dir / "episode.mp4"
            with imageio.get_writer(mp4, fps=20) as w:
                for fr in [frames_init, *frames]:
                    w.append_data(np.flipud(np.asarray(fr)))

        result.update({
            "status": "ok" if not success else "census_mismatch",
            "init": init_idx,
            "success": success,
            "candidate": record["candidate"],
            "png": str(png),
        })
        print(f"[{variant}] init {init_idx}: success={success} "
              f"candidate={record['candidate']} -> {png.name}", flush=True)
    try:
        env.close()
    except Exception:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="",
                        help="comma-separated census variant ids to include (default: all)")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--census", default=str(CENSUS),
                        help="failure census JSON (default: repository census path)")
    parser.add_argument("--video", action="store_true",
                        help="also write episode.mp4 (default: off, images only)")
    args = parser.parse_args()

    census = json.loads(Path(args.census).read_text())
    wanted = {s.strip() for s in args.tasks.split(",") if s.strip()}
    entries = [
        (t, str(t["variant_state"]))
        for t in census["tasks"]
        if t["n_fail"] > 0 and (not wanted or str(t["variant_state"]) in wanted)
    ]

    print("loading policy...", flush=True)
    cfg, model, dataset_stats, resize_size = load_policy()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for entry, variant in entries:
        summary.append(diagnose_task(entry, variant, cfg, model, dataset_stats,
                                     resize_size, out_root, video=args.video))

    print("\n=== SUMMARY ===", flush=True)
    for r in summary:
        cand = r.get("candidate")
        cand_txt = (
            f"{cand['span']['skill']}#{cand['span']['phase_index']} {cand['status']} "
            f"t*={cand['t_star']} (event={cand['event_step']})"
            if cand else "-"
        )
        print(f"{r.get('status', '?'):>16}  {r['task'][:56]:56s}  init={r.get('init')}  {cand_txt}",
              flush=True)
    ok = sum(1 for r in summary if r.get("status") == "ok")
    mismatch = sum(1 for r in summary if r.get("status") == "census_mismatch")
    print(f"\ndone: {ok} failed-reproduced, {mismatch} census mismatches, "
          f"{len(summary)} tasks", flush=True)


if __name__ == "__main__":
    main()
