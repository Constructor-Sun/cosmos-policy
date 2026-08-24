"""Phase 3 replay test: RGB-D Pick approach discrimination.

Replays LIBERO-10 training demos from global t=0.  For each active Pick phase,
it evaluates the correct target and nearby distractors through the same
template-match -> mask -> RGB-D -> 3D representative-point path, reports
approach ranking, premature gripper changes, and whether the Phase reaches the
Feasible stage via the existing 2D FeasibleVerifier.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    PHASE_TARGETS,
    SEGMENTS_MANIFEST,
    TASKS,
    create_env,
    demo_hdf5,
    instance_mask,
    load_item_objects,
    load_manifest,
    load_target_arguments,
    make_template,
    patch_numpy2_segmentation,
    select_candidates,
    select_target_candidates,
    target_underlying_objects,
    underlying_instance,
)
from memory_system.artifacts import PhaseTargetMemory  # noqa: E402
from memory_system.execute.feasible import FEASIBLE, FeasibleVerifier  # noqa: E402
from memory_system.execute.phase import PhaseVerifier  # noqa: E402
from memory_system.geometry import (  # noqa: E402
    camera_params as build_camera_params,
    depth_to_metric,
    flip_depth,
    pixel_to_world,
)
from memory_system.types import VerifierObservation  # noqa: E402
from robosuite.utils.camera_utils import (  # noqa: E402
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

GRIPPER_CLOSE_ACTION = 0.0
EVIDENCE_UPDATES = 2
SUPPORTED_SKILLS = ("Pick", "PlaceIn", "PlaceOn", "Close", "TurnOn")
CANDIDATE_TOLERANCES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def simulate_phase3(
    distances: list[float | None],
    tolerance_m: float,
    evidence_updates: int = EVIDENCE_UPDATES,
) -> tuple[bool, str | None, int | None]:
    """Simulate Phase 3D away/stall evidence on a distance sequence.

    Returns (error, error_type, index_in_distances).  Frames with None
    distance are skipped; progress is measured between consecutive valid
    distances.
    """
    prev = None
    away = 0
    stall = 0
    abnormal = 0
    for i, d in enumerate(distances):
        if d is None:
            continue
        if prev is None:
            prev = d
            continue
        progress = prev - d
        if progress > tolerance_m:
            away = 0
            stall = 0
            abnormal = 0
        elif progress < -tolerance_m:
            away += 1
            stall = 0
            abnormal += 1
            if abnormal >= evidence_updates:
                return True, "away", i
        else:
            stall += 1
            away = 0
            abnormal += 1
            if abnormal >= evidence_updates:
                return True, "stall", i
        prev = d
    return False, None, None


def gripper_xy(obs, camera_transform) -> np.ndarray:
    """Project EEF to pixel using the same raw-space convention as eval."""
    row_col = project_points_from_world_to_camera(
        np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
        camera_transform,
        256,
        256,
    )
    return np.asarray([row_col[1], row_col[0]], dtype=np.float32)


def sample_target(match, depth, cam, eef):
    """Return (distance_m, target_xyz, valid_ratio) for a matched mask."""
    if match is None:
        return None, None, None
    mask = match["mask"]
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None, None, 0.0
    pts = pixel_to_world(np.stack([ys, xs], axis=-1), depth, cam)
    valid = np.isfinite(pts).all(axis=1)
    if int(valid.sum()) < 4:
        return None, None, float(valid.mean())
    target = np.median(pts[valid], axis=0)
    distance = float(np.linalg.norm(np.asarray(eef, dtype=np.float64) - target))
    return distance, target, float(valid.mean())


def process_demo(
    task: str,
    demo_id: str,
    max_candidates: int = 5,
    frame_stride: int = 4,
) -> dict:
    manifest = load_manifest()
    item_objects = load_item_objects(manifest)
    record = next(
        r for r in manifest["records"]
        if r["task_name"] == task and r["demo_id"] == demo_id and r.get("valid")
    )

    patch_numpy2_segmentation()
    env = create_env(task)
    env.reset()
    try:
        h5 = h5py.File(demo_hdf5(task), "r")
        group = h5["data"][demo_id]
        states = group["states"][:]
        actions = group["actions"][:]
        length = len(states)

        obs0 = env.regenerate_obs_from_state(states[0])
        rgb0 = np.flipud(obs0["agentview_image"])
        cam = build_camera_params(env.env.sim, "agentview", 256, 256)
        camera_transform = get_camera_transform_matrix(
            env.env.sim, "agentview", 256, 256
        )

        demo_result = {"task": task, "demo": demo_id, "phases": []}

        phase_memory = PhaseTargetMemory(PHASE_TARGETS)
        target_objects = target_underlying_objects(env, manifest)

        for segment in record["segments"]:
            skill = str(segment["skill"])
            if skill not in SUPPORTED_SKILLS or segment.get("status") == "already_satisfied":
                continue

            step_id = int(segment["planner_step_id"])
            arguments = segment.get("arguments", {})
            phase_start = int(segment["start"])
            ready_frame = int(segment["ready_frame"])
            success_start = int(segment["success_start"])
            search_end = min(max(success_start, phase_start + 1), length - 1)

            if skill == "Pick":
                correct_ref = str(arguments["item"])
                correct_label = correct_ref
                candidates = select_candidates(
                    env,
                    correct_ref,
                    max_total=max_candidates,
                    allowed_objects=item_objects,
                )
                correct_tpl = make_template(
                    rgb0, instance_mask(env, obs0, correct_ref), demo_id
                )
                if correct_tpl is None:
                    demo_result["phases"].append({
                        "skill": skill,
                        "planner_step_id": step_id,
                        "correct_item": correct_ref,
                        "error": "correct_target_template_unavailable",
                    })
                    continue
                memory_templates = [correct_tpl]
            else:
                correct_ref = str(arguments["target"])
                correct_label = underlying_instance(env, correct_ref) or correct_ref
                candidates = select_target_candidates(
                    env,
                    correct_label,
                    target_objects,
                    max_total=max_candidates,
                )
                memory_templates = phase_memory.select(
                    task, step_id, skill, arguments
                )
                if not memory_templates:
                    demo_result["phases"].append({
                        "skill": skill,
                        "planner_step_id": step_id,
                        "correct_item": correct_ref,
                        "error": "correct_target_template_unavailable",
                    })
                    continue

            matcher = PhaseVerifier(PHASE_TARGETS, min_demo_votes=1)
            prepared_list = [
                (tpl, matcher._prepare_template(tpl)) for tpl in memory_templates
            ]

            def match_correct_target(rgb: np.ndarray):
                matcher.templates = memory_templates
                matcher.prepared_templates = prepared_list
                return matcher.match_current(rgb)

            feasible = FeasibleVerifier(PHASE_TARGETS, SEGMENTS_MANIFEST)
            feasible.reset(task, step_id, skill, arguments)

            dists = {obj: [] for obj in candidates}
            targets = {obj: [] for obj in candidates}
            valid_ratios = {obj: [] for obj in candidates}
            feasible_entry_frame = None
            premature_frame = None
            prev_closed = None
            sampled_frames = list(range(phase_start, search_end + 1, frame_stride))

            for t in sampled_frames:
                obs = env.regenerate_obs_from_state(states[t])
                rgb = np.flipud(obs["agentview_image"])
                metric = depth_to_metric(obs["agentview_depth"], cam.near, cam.far)
                depth = flip_depth(metric)
                eef = obs["robot0_eef_pos"]
                closed = bool(float(actions[t][6]) > GRIPPER_CLOSE_ACTION)

                match_correct = match_correct_target(rgb)
                if feasible_entry_frame is None and match_correct is not None:
                    vobs = VerifierObservation(
                        third_view_rgb=rgb,
                        gripper_xy=gripper_xy(obs, camera_transform),
                        gripper_closed=closed,
                        timestep=t,
                    )
                    fresult = feasible.update(
                        vobs,
                        match_correct["target_xy"],
                        match_correct["bbox_xyxy"],
                        confidence=match_correct["confidence"],
                    )
                    if fresult.status == FEASIBLE:
                        feasible_entry_frame = t

                if t == phase_start:
                    prev_closed = closed
                elif feasible_entry_frame is None and closed != prev_closed:
                    if premature_frame is None:
                        premature_frame = t
                    prev_closed = closed
                else:
                    prev_closed = closed

                for obj in candidates:
                    if obj == correct_label:
                        match = match_correct
                    else:
                        oracle_mask = instance_mask(env, obs, obj)
                        match = {"mask": oracle_mask}
                    d, xyz, ratio = sample_target(match, depth, cam, eef)
                    dists[obj].append(d)
                    valid_ratios[obj].append(ratio)
                    if obj == correct_label and xyz is not None:
                        targets[obj].append(xyz)

            phase = {
                "skill": skill,
                "planner_step_id": step_id,
                "correct_item": correct_ref,
                "correct_label": correct_label,
                "phase_start": phase_start,
                "ready_frame": ready_frame,
                "success_start": success_start,
                "feasible_entry_frame": feasible_entry_frame,
                "premature_gripper_frame": premature_frame,
                "phase_normal_end": feasible_entry_frame is not None,
                "phase_normal_end_strict": (
                    feasible_entry_frame is not None and premature_frame is None
                ),
                "candidates": [],
            }

            end_frame = ready_frame if phase_start <= ready_frame <= search_end else search_end
            end_idx = max(
                (i for i, f in enumerate(sampled_frames) if f <= end_frame),
                default=0,
            )
            for obj in candidates:
                ds = dists[obj]
                if not ds or end_idx < 0 or end_idx >= len(ds):
                    phase["candidates"].append({
                        "object": obj,
                        "distance_start": None,
                        "distance_end": None,
                        "approach": None,
                    })
                    continue
                d0, d1 = ds[0], ds[end_idx]
                approach = None if d0 is None or d1 is None else d0 - d1
                phase["candidates"].append({
                    "object": obj,
                    "distance_start": d0,
                    "distance_end": d1,
                    "approach": approach,
                })

            valid_cands = [c for c in phase["candidates"] if c["approach"] is not None]
            valid_cands.sort(key=lambda c: c["approach"], reverse=True)
            for rank, cand in enumerate(valid_cands, start=1):
                cand["rank"] = rank
                cand["is_correct"] = cand["object"] == correct_label
            correct_rank = next(
                (c["rank"] for c in valid_cands if c["is_correct"]), None
            )
            phase["correct_rank"] = correct_rank

            correct_cand = next(
                (c for c in phase["candidates"] if c.get("is_correct")), None
            )
            distractor_cands = [
                c for c in phase["candidates"]
                if not c.get("is_correct") and c.get("approach") is not None
            ]
            best_distractor = max(
                distractor_cands,
                key=lambda c: c["approach"],
                default=None,
            )
            phase["correct_approach_positive"] = bool(
                correct_cand is not None
                and correct_cand.get("approach") is not None
                and correct_cand["approach"] > 0
            )
            phase["best_distractor_object"] = (
                best_distractor["object"] if best_distractor is not None else None
            )
            phase["best_distractor_approach"] = (
                best_distractor["approach"] if best_distractor is not None else None
            )
            if (
                correct_cand is not None
                and correct_cand.get("approach") is not None
                and best_distractor is not None
            ):
                phase["approach_margin_to_best_distractor"] = (
                    correct_cand["approach"] - best_distractor["approach"]
                )
            else:
                phase["approach_margin_to_best_distractor"] = None

            if (
                feasible_entry_frame is not None
                and correct_label in dists
                and feasible_entry_frame in sampled_frames
                and sampled_frames.index(feasible_entry_frame) < len(dists[correct_label])
            ):
                phase["feasible_entry_distance_to_item_m"] = dists[correct_label][
                    sampled_frames.index(feasible_entry_frame)
                ]
            else:
                phase["feasible_entry_distance_to_item_m"] = None

            sim_end_frame = (
                feasible_entry_frame - 1
                if feasible_entry_frame is not None
                else end_frame
            )
            sim_end_idx = max(
                (i for i, f in enumerate(sampled_frames) if f <= sim_end_frame),
                default=-1,
            )
            sim = {
                "evidence_updates": EVIDENCE_UPDATES,
                "tolerances": list(CANDIDATE_TOLERANCES),
                "per_candidate": {},
            }
            for obj in candidates:
                prefix = dists[obj][: sim_end_idx + 1] if sim_end_idx >= 0 else []
                sim["per_candidate"][obj] = []
                for tol in CANDIDATE_TOLERANCES:
                    error, error_type, error_idx = simulate_phase3(prefix, tol)
                    sim["per_candidate"][obj].append({
                        "tolerance_m": tol,
                        "error": error,
                        "error_type": error_type,
                        "error_frame": (
                            phase_start + error_idx
                            if error and error_idx is not None
                            else None
                        ),
                    })
            phase["phase3_simulation"] = sim

            if correct_label in valid_ratios:
                ratios = [r for r in valid_ratios[correct_label] if r is not None]
                phase["mask_stability"] = {
                    "valid_ratio_mean": float(np.mean(ratios)) if ratios else None,
                    "valid_ratio_min": float(np.min(ratios)) if ratios else None,
                }
            if correct_label in targets and len(targets[correct_label]) >= 2:
                jumps = [
                    float(np.linalg.norm(targets[correct_label][i + 1] - targets[correct_label][i]))
                    for i in range(len(targets[correct_label]) - 1)
                ]
                phase["target_stability"] = {
                    "jump_mean_m": float(np.mean(jumps)) if jumps else None,
                    "jump_max_m": float(np.max(jumps)) if jumps else None,
                }

            demo_result["phases"].append(phase)

        h5.close()
        return demo_result
    finally:
        env.close()


def summarize(results: list[dict]) -> dict:
    phases = [p for r in results for p in r["phases"] if "correct_rank" in p]
    rank1 = sum(1 for p in phases if p["correct_rank"] == 1)
    rank2 = sum(1 for p in phases if p["correct_rank"] is not None and p["correct_rank"] <= 2)
    rank3 = sum(1 for p in phases if p["correct_rank"] is not None and p["correct_rank"] <= 3)
    normal_end = sum(1 for p in phases if p.get("phase_normal_end"))
    normal_end_strict = sum(1 for p in phases if p.get("phase_normal_end_strict"))
    premature = sum(1 for p in phases if p.get("premature_gripper_frame") is not None)
    approach_positive = sum(1 for p in phases if p.get("correct_approach_positive"))
    margins = [
        p["approach_margin_to_best_distractor"]
        for p in phases
        if p.get("approach_margin_to_best_distractor") is not None
    ]
    margin_stats = None
    if margins:
        margin_stats = {
            "mean_m": float(np.mean(margins)),
            "median_m": float(np.median(margins)),
            "min_m": float(np.min(margins)),
            "max_m": float(np.max(margins)),
        }

    tolerance_summary = []
    suggested_tolerance = None
    for tol in CANDIDATE_TOLERANCES:
        correct_ok = 0
        any_distractor_error = 0
        distractor_error_candidates = 0
        distractor_candidates = 0
        for p in phases:
            sim = p.get("phase3_simulation")
            if not sim:
                continue
            per_candidate = sim.get("per_candidate", {})
            correct = per_candidate.get(p.get("correct_item"))
            if correct is not None:
                entry = next((x for x in correct if x["tolerance_m"] == tol), None)
                if entry is not None and not entry["error"]:
                    correct_ok += 1
            for obj, entries in per_candidate.items():
                if obj == p.get("correct_item"):
                    continue
                entry = next((x for x in entries if x["tolerance_m"] == tol), None)
                if entry is None:
                    continue
                distractor_candidates += 1
                if entry["error"]:
                    distractor_error_candidates += 1
                    any_distractor_error += 1
        n_phases = len(phases)
        correct_no_error_rate = correct_ok / n_phases if n_phases else None
        any_distractor_error_rate = (
            any_distractor_error / n_phases if n_phases else None
        )
        distractor_error_rate = (
            distractor_error_candidates / distractor_candidates
            if distractor_candidates
            else None
        )
        tolerance_summary.append({
            "tolerance_m": tol,
            "correct_no_error": correct_ok,
            "correct_no_error_rate": correct_no_error_rate,
            "any_distractor_error": any_distractor_error,
            "any_distractor_error_rate": any_distractor_error_rate,
            "distractor_error_candidates": distractor_error_candidates,
            "distractor_candidates": distractor_candidates,
            "distractor_error_rate": distractor_error_rate,
        })
        if (
            suggested_tolerance is None
            and correct_no_error_rate == 1.0
            and any_distractor_error_rate is not None
            and any_distractor_error_rate > 0.0
        ):
            suggested_tolerance = tol

    return {
        "phases": len(phases),
        "correct_rank_1": rank1,
        "rank1_rate": rank1 / len(phases) if phases else None,
        "correct_rank_top2": rank2,
        "rank_top2_rate": rank2 / len(phases) if phases else None,
        "correct_rank_top3": rank3,
        "rank_top3_rate": rank3 / len(phases) if phases else None,
        "correct_approach_positive": approach_positive,
        "correct_approach_positive_rate": (
            approach_positive / len(phases) if phases else None
        ),
        "approach_margin_to_best_distractor": margin_stats,
        "phase_normal_end": normal_end,
        "phase_normal_end_rate": normal_end / len(phases) if phases else None,
        "phase_normal_end_strict": normal_end_strict,
        "phase_normal_end_strict_rate": (
            normal_end_strict / len(phases) if phases else None
        ),
        "premature_gripper_change": premature,
        "premature_rate": premature / len(phases) if phases else None,
        "phase3_tolerance_sweep": tolerance_summary,
        "suggested_progress_tolerance_m": suggested_tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--max-demos", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--out", default="tests/phase3/results.json")
    args = parser.parse_args()

    manifest = load_manifest()
    all_results = []
    t0 = time.time()
    for task in args.tasks:
        records = [
            r for r in manifest["records"]
            if r.get("valid") and r["task_name"] == task
        ][: args.max_demos]
        for record in records:
            print(f"[{task[:40]}...] {record['demo_id']}", flush=True)
            all_results.append(
                process_demo(
                    task,
                    record["demo_id"],
                    args.max_candidates,
                    args.frame_stride,
                )
            )

    summary = summarize(all_results)
    output = {"summary": summary, "results": all_results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=float))
    print(json.dumps(summary, indent=2))
    print(f"saved {out_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
