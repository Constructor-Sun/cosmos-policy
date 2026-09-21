#!/usr/bin/env python
"""Batch TTA repair driver: rebuild requests from saved episodes and run
the repair mode of run_libero_eval (COSMOS_TTA_REPAIR) across GPUs.

One process per GPU runs its assigned cases sequentially.  Requests are
rebuilt from the recorded episodes, so nothing outside the repo is needed.
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from memory_system.tta.variant_spec import resolve_variant
DEFAULT_RESULT_DIR = REPO / "memory_system/pointcloud_action/results/tta_failure_screening_68"
DEFAULT_META = REPO / "memory_system/pointcloud_action/results/tta_screening_meta.json"
DEFAULT_SUMMARY = REPO / "memory_system/pointcloud_action/results/tta_repair_68_summary.json"
PRO_LIBERO_ROOT = Path("/data1/liu/exp/counterfactual/external/LIBERO-PRO")
LANGS = {
    "KSCENE3": "turn on the stove and put the moka pot on it",
    "KSCENE4": "put the black bowl in the bottom drawer of the cabinet and close it",
    "KSCENE6": "put the yellow and white mug in the microwave and close it",
    "KSCENE8": "put both moka pots on the stove",
    "LRSCENE2": "put both the alphabet soup and the tomato sauce in the basket",
    "LRSCENE5": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "LRSCENE6": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
}
ROBOTS = {"KSCENE3": 273, "KSCENE4": 274, "KSCENE6": 270,
          "KSCENE8": 269, "LRSCENE2": 271, "LRSCENE5": 265,
          "LRSCENE6": 267}


def _case_tag(task, init):
    """Keep the historical compact tags without assuming a perturbation type."""
    prefixes = {
        "KITCHEN": "KSCENE",
        "LIVING_ROOM": "LRSCENE",
        "STUDY": "SSCENE",
    }
    match = re.match(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE(\d+)_", task)
    short = (
        f"{prefixes[match.group(1)]}{match.group(2)}"
        if match else re.sub(r"[^A-Za-z0-9]+", "_", task).strip("_")
    )
    return f"{short}-{init}"


def _clean_language(task):
    instruction = re.sub(
        r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", task
    )
    return instruction.replace("_", " ")


def prepare_inputs_from_diagnosis(diagnosis_root, out_dir):
    """Build batch inputs directly from generic per-case phase records.

    This is perturbation-agnostic: the exact variant task/category remains in
    phase_record.json and is resolved later by variant_spec.  It preserves the
    RobotInit convention that a missing t* falls back to step 10.
    """
    diagnosis_root = Path(diagnosis_root)
    meta, cases = {}, []
    numeric = fallback = 0
    for record_path in sorted(diagnosis_root.glob("*/init*/phase_record.json")):
        record = json.loads(record_path.read_text())
        task = record_path.parent.parent.name
        init = record_path.parent.name
        tag = _case_tag(task, init)
        if tag in meta:
            # Same scene hosting two base tasks (LIBERO-PRO swap suites):
            # extend the compact tag with the task's distinguishing prefix.
            distinct = re.sub(r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", task)
            tag = f"{tag}-{re.sub(r'[^A-Za-z0-9]+', '', distinct)[:12].lower()}"
        if tag in meta:
            raise ValueError(f"duplicate generated tag {tag!r} under {diagnosis_root}")
        variant_task = (
            record.get("task_name_perturbed")
            or record.get("suite_task_name")
            or task
        )
        candidate = record.get("candidate") or {}
        t_star = candidate.get("t_star")
        if isinstance(t_star, int):
            numeric += 1
        else:
            t_star = None
            fallback += 1
        meta[tag] = {
            "task": variant_task,
            "base_task": task,
            "init": init,
            "census_abs_init": int(record["census_abs_init"]),
            "suite_task_name": variant_task,
            "clean_language": _clean_language(task),
            # Case identity as recorded by diagnosis; the repair env must be
            # rebuilt from these, never re-derived from the task name.
            "suite": record.get("suite"),
            "bddl_file": record.get("bddl_file"),
            "flavor": record.get("flavor", "plus"),
            # Only diagnosis-root inputs opt into phase-aware requests.  The
            # legacy RobotInit meta remains byte-for-byte behavior compatible.
            "use_diagnosed_phase": True,
        }
        cases.append({
            "tag": tag,
            "t_star": t_star,
            "task_success": bool(record.get("task_success", False)),
        })
    if not meta:
        raise FileNotFoundError(
            f"no */init*/phase_record.json found under {diagnosis_root}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "diagnosis_meta.json"
    summary_path = out_dir / "diagnosis_summary.json"
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    summary_path.write_text(json.dumps({
        "version": "diagnosis_root_v1",
        "total": len(cases),
        "numeric_t_star": numeric,
        "fallback_t_star_10": fallback,
        "cases": cases,
    }, indent=1, ensure_ascii=False))
    print(
        f"[prepare] {len(cases)} cases: {numeric} numeric t*, "
        f"{fallback} fallback t*=10",
        flush=True,
    )
    return meta_path, summary_path


def _plan_from_record(record):
    """Plan recorded at diagnosis time (BDDL-derived, current-task arguments).

    ``memory_phases`` entries already carry planner_step_id/skill/arguments in
    plan order, so no task-name re-derivation happens here.
    """
    if not record:
        return None
    specs = [
        {
            "phase_index": index,
            "planner_step_id": int(entry["planner_step_id"]),
            "skill": str(entry["skill"]),
            "arguments": {
                str(key): str(value)
                for key, value in (entry.get("arguments") or {}).items()
            },
        }
        for index, entry in enumerate(record.get("memory_phases") or [])
        if entry.get("skill")
    ]
    return specs or None


def build_request(tag, out_dir, *, result_dir=DEFAULT_RESULT_DIR,
                  meta_path=DEFAULT_META, summary_path=DEFAULT_SUMMARY):
    import h5py, numpy as np
    from memory_system.tta.phase_record import load_task_sequence
    meta = json.loads(Path(meta_path).read_text())[tag]
    task, init = meta["task"], meta["init"]
    record_task = meta.get("base_task", task)
    t_star = {c["tag"]: c["t_star"]
              for c in json.loads(Path(summary_path).read_text())["cases"]}[tag]
    if t_star is None:
        # No interaction event in the recording: documented fallback restarts
        # the phase at the first-phase floor.
        t_star = 10
    record_path = Path(result_dir) / record_task / init / "phase_record.json"
    record = (
        json.loads(record_path.read_text()) if record_path.exists() else None
    )
    phases = _plan_from_record(record)
    if phases is None:
        # Legacy records predating the BDDL-derived plan.
        phases = [
            {
                "phase_index": index,
                "planner_step_id": int(spec.planner_step_id),
                "skill": spec.skill,
                "arguments": dict(spec.arguments),
            }
            for index, spec in enumerate(load_task_sequence(task)[2])
        ]
    request_phase = {
        "phase_index": 0,
        "planner_step_id": int(phases[0]["planner_step_id"]),
        "skill": phases[0]["skill"],
        "arguments": dict(phases[0]["arguments"]),
    }
    if meta.get("use_diagnosed_phase") and record is not None:
        candidate = record.get("candidate") or {}
        # A numeric event-based t* belongs to the diagnosed phase.  When the
        # event is missing and t*=10 is used, retain the proven RobotInit
        # behavior: repair the first phase from the beginning.
        if isinstance(candidate.get("t_star"), int):
            raw_phase = (
                candidate.get("repair_phase")
                or candidate.get("runtime_phase")
                or candidate.get("span")
                or {}
            )
            if all(k in raw_phase for k in ("planner_step_id", "skill")):
                request_phase = {
                    "phase_index": int(raw_phase.get("phase_index", 0)),
                    "planner_step_id": int(raw_phase["planner_step_id"]),
                    "skill": raw_phase["skill"],
                    "arguments": dict(raw_phase.get("arguments") or {}),
                }
    with h5py.File(Path(result_dir) / record_task / init / "episode.h5", "r") as h5:
        actions = np.asarray(h5["actions"], dtype=np.float32)
    req = {"task": task, "tag": tag,
           "phase": request_phase,
           # Full BDDL-derived phase plan, so the per-phase loop
           # (COSMOS_TTA_PHASE_LOOP) reads the same order the diagnosis used
           # instead of re-deriving it from the task name.
           "plan": phases,
           "t_star": int(t_star), "actions": actions.tolist()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"repair_{tag}.json"
    path.write_text(json.dumps(req))
    return path, meta


def run_case(gpu, tag, req_path, out_path, *, meta_path=DEFAULT_META, args_work_dir=None):
    meta = json.loads(Path(meta_path).read_text())[tag]
    short = tag.split("-init")[0]
    base_task = meta.get("base_task", meta["task"])
    target_task = (
        meta.get("pert_task")
        or meta.get("suite_task_name")
        or meta["task"]
    )
    clean_language = (
        meta.get("clean_language")
        or meta.get("language")
        or LANGS.get(short)
        or base_task.replace("_", " ")
    )
    env = dict(os.environ)
    variant_spec = None
    if meta.get("flavor") == "pro":
        # PRO suites are their own perturbation: the suite selects it and the
        # task keeps the base name, so resolve_variant (plus classification)
        # does not apply.  The smoke test runs strict-instruction mode for the
        # empty perturbation category, i.e. the real BDDL instruction.
        pair_suite = meta["suite"]
        pert_category = ""
        pert_task = target_task
        condition = "pro_variant"
        env.setdefault("COSMOS_LIBERO_ROOT", str(PRO_LIBERO_ROOT))
        # PRO 修复 selector：align（只对齐）/timegrip/rawactions（对齐+回放记忆段）。
        env["COSMOS_SKILL_READY_MEMORY"] = os.environ.get(
            "COSMOS_SKILL_READY_MEMORY", "align")
        env.setdefault(
            "LIBERO_CONFIG_PATH", str(PRO_LIBERO_ROOT / "configs" / "libero_pro")
        )
        # PRO init-state files only load with the legacy torch.load default
        # (same workaround as scripts/libero_pro/run_swap.sh).
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    else:
        variant_spec = resolve_variant(target_task, suite="libero_10")
        pair_suite = variant_spec.suite
        pert_category = variant_spec.category
        pert_task = variant_spec.task_name
        condition = variant_spec.condition
    env.update({
        "GPU_ID": str(gpu),
        "SMOKE_ONLY_CONDITION": "perturb",
        "SMOKE_PAIR_SUITE": pair_suite,
        "SMOKE_PAIR_BASE_TASK": base_task,
        "SMOKE_PAIR_CLEAN_LANGUAGE": clean_language,
        "SMOKE_PAIR_PERT_NAME": condition,
        "SMOKE_PAIR_PERT_CATEGORY": pert_category,
        "SMOKE_PAIR_PERT_TASK": pert_task,
        "SMOKE_NUM_PAIRS": "1",
        "SMOKE_SEED": str(meta.get("seed", 7)),
        "SMOKE_RESULTS_DIR": str(Path(args_work_dir) / "results" / tag),
        "SMOKE_RUN_ID": f"tta_repair_{tag}",
        "COSMOS_INIT_STATE_OFFSET": str(meta["census_abs_init"]),
        "COSMOS_DATA_COLLECTION": "1",
        # run_libero_smoke_test.sh re-exports COSMOS_DATA_COLLECTION from
        # SMOKE_DATA_COLLECTION, so this is the one that must be set.
        "SMOKE_DATA_COLLECTION": "1",
        # The repair alignment selector is only built when this is on.
        "COSMOS_INITIAL_ALIGNMENT": "1",
        "COSMOS_TTA_RGBD": "1",
        # Instance segmentation (wrist_segmentation) is only rendered when
        # this flag builds a SegmentationRenderEnv; item left empty keeps the
        # completion telemetry off.
        "COSMOS_SKILL_COMPLETION_SHADOW": "1",
        "COSMOS_TTA_REPAIR": str(req_path), "COSMOS_TTA_REPAIR_OUT": str(out_path),
        "COSMOS_TTA_REPAIR_TAG": tag,
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
    })
    print(
        f"[gpu{gpu}] {tag}: flavor={meta.get('flavor', 'plus')} suite={pair_suite} "
        f"category={pert_category!r} task={pert_task} bddl={meta.get('bddl_file') or '-'}",
        flush=True,
    )
    subprocess.run(["sh", "run_libero_smoke_test.sh"], env=env,
                   cwd=str(REPO / "scripts"), check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3", help="GPUs to use, one chain each")
    ap.add_argument("--tags", default="", help="comma-separated case tags (default: all 68)")
    ap.add_argument("--work-dir", default=str(REPO / "scripts/experiments/tta_repair_work"),
                    help="work directory; use an ABSOLUTE path -- the smoke child "
                         "process runs with cwd=scripts/ and would resolve relative "
                         "paths differently from this driver")
    ap.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR),
                    help="directory holding <task>/<initNNN>/episode.h5 recordings")
    ap.add_argument("--meta", default=str(DEFAULT_META),
                    help="tag -> {task, init, census_abs_init, ...} mapping")
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY),
                    help="summary JSON with per-case t_star")
    ap.add_argument(
        "--diagnosis-root", default="",
        help=("build meta/summary automatically from */init*/phase_record.json; "
              "missing t* uses the RobotInit fallback t*=10"),
    )
    args = ap.parse_args()
    work = Path(args.work_dir)
    if args.diagnosis_root:
        meta_path, summary_path = prepare_inputs_from_diagnosis(
            args.diagnosis_root, work / "inputs"
        )
        args.meta, args.summary = str(meta_path), str(summary_path)
    tags = ([t.strip() for t in args.tags.split(",") if t.strip()]
            or sorted(json.loads(Path(args.meta).read_text()).keys()))
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    chains = {g: [t for i, t in enumerate(tags) if i % len(gpus) == j]
              for j, g in enumerate(gpus)}
    if len(gpus) > 1 and args.tags == "":
        # One child process per GPU; the parent only waits.
        children = []
        for g, chain in chains.items():
            cmd = [sys.executable, __file__, "--gpus", g, "--tags", ",".join(chain),
                   "--work-dir", str(work), "--result-dir", args.result_dir,
                   "--meta", args.meta, "--summary", args.summary]
            children.append(subprocess.Popen(cmd))
        for c in children:
            c.wait()
        return
    gpu = gpus[0]
    (work / "outputs").mkdir(parents=True, exist_ok=True)
    for tag in chains[gpu]:
        req, _ = build_request(tag, work / "requests", result_dir=args.result_dir,
                               meta_path=args.meta, summary_path=args.summary)
        run_case(int(gpu), tag, req, work / "outputs" / f"{tag}.json",
                     args_work_dir=work,
                 meta_path=args.meta)
        print(f"[gpu{gpu}] {tag} done", flush=True)


if __name__ == "__main__":
    main()
