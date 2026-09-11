#!/usr/bin/env python
"""Per-case (task + init state + seed) evaluation of an SFT checkpoint over the
68 screened failure cases.

Same precision machinery as scripts/tta_repair_batch.py (one deterministic
rollout per case), but WITHOUT the online TTA repair injection — the SFT
checkpoint is evaluated standalone. Pass the checkpoint via
--ckpt (a .pt file or a DCP checkpoint directory containing .metadata).
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
DEFAULT_META = REPO / "memory_system/pointcloud_action/results/tta_screening_meta.json"
LANGS = {
    "KSCENE3": "turn on the stove and put the moka pot on it",
    "KSCENE4": "put the black bowl in the bottom drawer of the cabinet and close it",
    "KSCENE6": "put the yellow and white mug in the microwave and close it",
    "KSCENE8": "put both moka pots on the stove",
    "LRSCENE2": "put both the alphabet soup and the tomato sauce in the basket",
    "LRSCENE5": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "LRSCENE6": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
}


def run_case(gpu, tag, meta, ckpt, work_dir):
    short = tag.split("-init")[0]
    base_task = meta.get("base_task", meta["task"])
    env = dict(os.environ)
    env.update({
        "GPU_ID": str(gpu),
        "SMOKE_ONLY_CONDITION": "perturb",
        "SMOKE_PAIR_SUITE": "libero_10",
        "SMOKE_PAIR_BASE_TASK": base_task,
        "SMOKE_PAIR_CLEAN_LANGUAGE": meta.get("language") or LANGS[short],
        "SMOKE_PAIR_PERT_NAME": meta.get("pert_name", "robot_initial_states" if meta["task"] == base_task else "background_textures"),
        "SMOKE_PAIR_PERT_CATEGORY": meta.get("pert_category", "Robot Initial States" if meta["task"] == base_task else "Background Textures"),
        "SMOKE_PAIR_PERT_TASK": meta.get("pert_task") or (
            meta.get("suite_task_name", meta["task"]) if meta["task"] == base_task
            else meta["task"]),
        "SMOKE_NUM_PAIRS": "1",
        "SMOKE_SEED": str(meta.get("seed", 7)),
        "SMOKE_RESULTS_DIR": str(Path(work_dir) / "results" / tag),
        "SMOKE_RUN_ID": f"tta_sft_eval_{tag}",
        "SMOKE_POLICY_CKPT_PATH": str(Path(ckpt).resolve()),
        "COSMOS_INIT_STATE_OFFSET": str(meta["census_abs_init"]),
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
    })
    subprocess.run(["sh", "run_libero_smoke_test.sh"], env=env,
                   cwd=str(REPO / "scripts"), check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3", help="GPUs to use, one chain each")
    ap.add_argument("--tags", default="", help="comma-separated case tags (default: all 68)")
    ap.add_argument("--ckpt", required=True, help="SFT checkpoint (.pt or DCP dir)")
    ap.add_argument("--work-dir", default=str(REPO / "scripts/experiments/tta_sft_eval"))
    ap.add_argument("--meta", default=str(DEFAULT_META),
                    help="tag -> {task, init, census_abs_init, seed, ...} mapping")
    args = ap.parse_args()
    tags = ([t.strip() for t in args.tags.split(",") if t.strip()]
            or sorted(json.loads(Path(args.meta).read_text()).keys()))
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    work = Path(args.work_dir)
    chains = {g: [t for i, t in enumerate(tags) if i % len(gpus) == j]
              for j, g in enumerate(gpus)}
    if len(gpus) > 1:
        # One child process per GPU; the parent only waits.
        children = []
        for g, chain in chains.items():
            if not chain:
                continue
            cmd = [sys.executable, __file__, "--gpus", g, "--tags", ",".join(chain),
                   "--ckpt", args.ckpt, "--work-dir", args.work_dir, "--meta", args.meta]
            children.append(subprocess.Popen(cmd))
        for c in children:
            c.wait()
        return
    gpu = gpus[0]
    meta_all = json.loads(Path(args.meta).read_text())
    for tag in chains[gpu]:
        run_case(int(gpu), tag, meta_all[tag], args.ckpt, work)
        print(f"[gpu{gpu}] {tag} done", flush=True)


if __name__ == "__main__":
    main()
