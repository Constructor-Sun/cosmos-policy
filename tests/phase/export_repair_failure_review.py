#!/usr/bin/env python
"""Review artifacts for post-repair failures (no policy inference).

For every case whose paired repair run still failed, replay the saved baseline
action stream, then export per case:
  - tstar.png   same layout as diagnose_failed.py (observation at the repair
                cut-in + phase/event/t* timeline), built from the replay
  - failure_phase.mp4   the full repair-target phase span with HUD markers
  - a summary JSON of old-config vs new-config candidates
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "LIBERO-plus")]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")

REPAIR_SETS = {
    "repro68": {
        "results": ROOT / "experiments/tta/tta_repair_work_repro_68/results",
        "requests": ROOT / "experiments/tta/tta_repair_work_repro_68/requests",
        "source_root": ROOT / "experiments/tta/tta_phase_check_repro_68",
        "replay_root": ROOT / "experiments/tta/phase_rules_pick9_placeon5/tta_phase_check_repro_68",
    },
    "bg5": {
        "results": ROOT / "experiments/tta/tta_repair_bg5_tstar_correct/results",
        "requests": ROOT / "experiments/tta/tta_repair_bg5_tstar_correct/requests",
        "source_root": ROOT / "experiments/tta/tta_phase_check_bg5_correct",
        "replay_root": ROOT / "experiments/tta/phase_rules_pick9_placeon5/tta_phase_check_bg5_correct",
    },
}
OUT_ROOT = ROOT / "experiments/tta/phase_check_videos/repair_failures"


def find_failures():
    """[(set_name, tag, task, repair_phase, repair_t_star)] still failing."""
    out = []
    for set_name, cfg in REPAIR_SETS.items():
        for summary in sorted(cfg["results"].glob("*/*1pair_summary.json")):
            payload = json.loads(summary.read_text())
            if payload["conditions"][0]["success_rate"] >= 1.0:
                continue
            tag = summary.parent.name
            request = json.loads(
                (cfg["requests"] / f"repair_{tag}.json").read_text())
            out.append((set_name, tag, request["task"], request["phase"],
                        request["t_star"]))
    return out


def cand_text(record):
    cand = (record or {}).get("candidate")
    if cand is None:
        return "none"
    span = cand["span"]
    return (f"{span['skill']}#{span['phase_index']} {cand['status']} "
            f"t*={cand.get('t_star')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from memory_system.execute.skill_completion.shadow import PickTargetPointCloud
    from memory_system.tta.diagnose_failed import make_png
    from memory_system.tta.phase_record import PhaseEventRecorder, load_task_sequence
    from replay_phase_rules import make_replay_context
    from export_case_videos import annotate

    failures = find_failures()
    if args.limit:
        failures = failures[:args.limit]
    print(f"{len(failures)} post-repair failures to review", flush=True)

    # The repair request stores the base task name; the env must be built from
    # the exact variant recorded in the source phase_record.json.
    # The repair request stores a task name that may be the variant (bg5) or
    # the base task (repro_68); dataset dirs are named by the base task, which
    # is always a prefix of the variant name.
    def find_task_dir(source_root: Path, task: str) -> Path:
        exact = source_root / task
        if exact.is_dir():
            return exact
        matches = [d for d in source_root.iterdir() if d.is_dir()
                   and task.startswith(d.name + "_")]
        if len(matches) != 1:
            raise FileNotFoundError(f"no unique task dir for {task!r}: {matches}")
        return matches[0]

    grouped = {}
    for set_name, tag, base_task, phase, t_star in failures:
        init = "init" + tag.split("-init")[1]
        task_dir = find_task_dir(REPAIR_SETS[set_name]["source_root"], base_task)
        source_record = json.loads(
            (task_dir / init / "phase_record.json").read_text())
        grouped.setdefault(
            (set_name, task_dir.name, source_record["suite_task_name"]), []).append(
            (tag, phase, t_star))

    summary = []
    for (set_name, base_task, variant), cases in grouped.items():
        cfg = REPAIR_SETS[set_name]
        task_dir = find_task_dir(cfg["source_root"], base_task)
        targets = {"init" + tag.split("-init")[1]: (tag, repair_phase, t_star)
                   for tag, repair_phase, t_star in cases}
        # Replay the task's whole sorted case sequence: MuJoCo episode history
        # (qacc_warmstart) leaks across episodes, and only this order makes the
        # target replays bitwise-identical to the saved episodes.
        all_cases = sorted(p.parent for p in task_dir.rglob("phase_record.json"))
        env, init_states, phases, object_query = make_replay_context(variant)
        base, demo_id, _ = load_task_sequence(variant)
        point_cloud = PickTargetPointCloud(env, 256)

        def pick_points(phase, observation):
            if phase is None or phase.skill != "Pick":
                return None
            item = phase.arguments.get("item")
            if not item:
                return None
            try:
                return point_cloud.points(observation, item)
            except Exception:
                return None

        try:
            for case_dir in all_cases:
                init = case_dir.name
                source_dir = cfg["source_root"] / base_task / init
                with h5py.File(source_dir / "episode.h5") as h5:
                    actions = np.asarray(h5["actions"][:], dtype=np.float64)
                    source_proprio = h5["proprio"][:]
                source_record = json.loads(
                    (source_dir / "phase_record.json").read_text())
                if init not in targets:
                    # filler case: advance the env's episode history only
                    env.reset()
                    obs = env.set_init_state(
                        init_states[source_record["census_abs_init"]])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    for action in actions:
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            break
                    continue
                tag, repair_phase, repair_t_star = targets[init]
                replay_path = cfg["replay_root"] / base_task / init / "phase_record.json"
                new_record = json.loads(replay_path.read_text()) \
                    if replay_path.exists() else None

                recorder = PhaseEventRecorder(
                    phases, task_name=base, demo_id=demo_id,
                    object_query=object_query)
                env.reset()
                obs = env.set_init_state(init_states[source_record["census_abs_init"]])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                recorder.begin(step=0, initial_gripper_state=0)
                proprio_buf, frames, success, executed = [], [], False, 0
                for step, action in enumerate(actions):
                    phase = recorder.active_phase
                    proprio_buf.append(np.concatenate([
                        obs["robot0_gripper_qpos"], obs["robot0_eef_pos"],
                        obs["robot0_eef_quat"]]).astype(np.float64))
                    obs, _, done, _ = env.step(action.tolist())
                    frames.append(np.flipud(np.asarray(obs["agentview_image"])))
                    decision = recorder.observe(
                        step=step, action=action, obs=obs,
                        target_points=pick_points(phase, obs))
                    executed = step + 1
                    if done:
                        success = True
                        break
                    advanced = decision is not None and decision.advance
                    if not advanced and (step + 1) % 16 == 0:
                        recorder.finish_chunk(step=step)
                proprio = np.stack(proprio_buf)
                record = recorder.finalize(total_steps=executed, success=success)

                # failing-phase span: prefer the repair-target phase in the new
                # record, fall back to the old record's span with that index.
                span = next((p for p in record["phases"]
                             if p["phase_index"] == repair_phase["phase_index"]),
                            None) or next(
                    (p for p in source_record["phases"]
                     if p["phase_index"] == repair_phase["phase_index"]), None)
                clip_lo = span["start_step"] if span else 0
                clip_hi = (span["end_step"] + 1 if span and span["end_step"]
                           is not None else executed)
                event = None
                new_cand = record.get("candidate")
                if new_cand is not None and new_cand.get("event_step") is not None:
                    event = new_cand["event_step"]

                out_dir = args.output / set_name / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                t_star = repair_t_star
                img = frames[0] if (t_star is None or t_star == 0 or not frames) \
                    else frames[min(t_star - 1, len(frames) - 1)]
                info = [
                    f"tag: {tag}  ({'success' if success else 'failed'} baseline replay)",
                    f"repair used: {repair_phase['skill']}#"
                    f"{repair_phase['phase_index']} t*={repair_t_star}",
                    f"old candidate: {cand_text(source_record)}",
                    f"new(9/5) candidate: {cand_text(record)}",
                    f"actions: {executed}",
                ]
                make_png(out_dir / "tstar.png",
                         f"{tag}  {base_task[:52]}",
                         info, img, record, new_cand or {"status": "none"},
                         t_star, event, proprio, actions)

                label = (f"{tag} {repair_phase['skill']}#"
                         f"{repair_phase['phase_index']} t*={repair_t_star}")
                import imageio
                with imageio.get_writer(out_dir / "failure_phase.mp4", fps=20) as w:
                    for step in range(clip_lo, min(clip_hi, len(frames))):
                        note = ""
                        if step == clip_lo:
                            note = f" phase start"
                        if step + 1 == min(clip_hi, len(frames)):
                            note = " phase end"
                        w.append_data(annotate(
                            frames[step], [label, f"step {step + 1}{note}"],
                            event, step, t_star))
                # full baseline replay, from step 0
                with imageio.get_writer(out_dir / "baseline_full.mp4", fps=20) as w:
                    for step in range(len(frames)):
                        w.append_data(annotate(
                            frames[step], [label + " baseline",
                                           f"step {step + 1}/{len(frames)}"],
                            event, step, t_star))
                # true post-repair rollout, from the frames recorded at repair
                # time (no replay involved)
                repaired_success = None
                repaired_path = None
                rollout_dirs = list((cfg["results"] / tag / "logs" /
                                     "rollout_data").glob("episode_data--*.hdf5"))
                if rollout_dirs:
                    repaired_path = rollout_dirs[0]
                    repaired_success = "--success=True--" in repaired_path.name
                    import io
                    from PIL import Image as PILImage
                    with h5py.File(repaired_path) as h5:
                        jpegs = h5["primary_images_jpeg"]
                        n = len(jpegs)
                        with imageio.get_writer(out_dir / "repaired_full.mp4",
                                                fps=20) as w:
                            for step in range(n):
                                frame = np.asarray(
                                    PILImage.open(io.BytesIO(jpegs[step])))
                                w.append_data(annotate(
                                    frame,
                                    [f"{tag} REPAIRED rollout "
                                     f"success={repaired_success}",
                                     f"step {step + 1}/{n}"],
                                    None, step, t_star))
                summary.append({
                    "set": set_name, "tag": tag, "task": base_task,
                    "repair_phase": repair_phase, "repair_t_star": repair_t_star,
                    "replay_success": success, "executed_steps": executed,
                    "proprio_max_abs_diff": float(np.abs(
                        proprio - source_proprio[:len(proprio)]).max()),
                    "old_candidate": cand_text(source_record),
                    "new_candidate": cand_text(record),
                    "clip": [int(clip_lo), int(clip_hi)],
                    "repaired_rollout_success": repaired_success,
                    "artifacts": [str(out_dir / "tstar.png"),
                                  str(out_dir / "failure_phase.mp4"),
                                  str(out_dir / "baseline_full.mp4")] +
                                 ([] if repaired_path is None else
                                  [str(out_dir / "repaired_full.mp4")]),
                })
                print(f"done {set_name}/{tag}: clip={clip_lo}..{clip_hi} "
                      f"new={cand_text(record)}", flush=True)
        finally:
            env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(summary)} case reviews to {args.output}", flush=True)


if __name__ == "__main__":
    main()
