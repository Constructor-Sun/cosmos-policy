"""Run a LIBERO-10 Pick sweep against the LIBERO-90 pointcloud_action memory."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
MANIFEST = REPO / "skill_memory_test/libero_10/segments_ready_fixed16.json"
DEMO_DIR = REPO / "LIBERO-Cosmos-Policy/success_only/libero_10_regen"
PYTHON = "/data1/liu/miniconda3/envs/cosmospolicy/bin/python"


def collect_cases(manifest_path: Path, max_per_task: int):
    manifest = json.loads(Path(manifest_path).read_text())
    per_task = {}
    cases = []
    for rec in manifest.get("records", []):
        if not rec.get("valid"):
            continue
        task = rec["task_name"]
        demo = rec["demo_id"]
        for seg in rec.get("segments", []):
            if seg.get("skill") != "Pick":
                continue
            key = (task, demo, str(seg.get("arguments", {}).get("item")))
            if key in per_task:
                continue
            per_task[key] = True
            cases.append(
                {
                    "task": task,
                    "demo": demo,
                    "item": seg.get("arguments", {}).get("item"),
                    "frame": int(seg.get("start", 0)),
                }
            )
    # Limit per task, preserving first occurrences.
    limited = []
    counts = {}
    for case in cases:
        counts[case["task"]] = counts.get(case["task"], 0) + 1
        if counts[case["task"]] <= max_per_task:
            limited.append(case)
    return limited


def parse_result(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return ast.literal_eval(line)
            except Exception:
                return {"raw": line}
    return {"raw": text[-500:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", type=Path, default=REPO / "rollouts/libero10_pick_sweep_results.jsonl")
    parser.add_argument("--save-video-dir", type=Path, default=None,
                        help="If set, save one MP4 per evaluated case.")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Zero-based shard id when running parallel shards.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards.")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must be in [0, --num-shards)")

    output = args.output
    video_dir = args.save_video_dir
    if args.num_shards > 1:
        output = output.with_name(
            f"{output.stem}.shard{args.shard_id}{output.suffix}"
        )
        if video_dir is not None:
            video_dir = video_dir / f"shard_{args.shard_id}"
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(MANIFEST, args.max_per_task)
    cases = [
        case for idx, case in enumerate(cases)
        if idx % args.num_shards == args.shard_id
    ]
    print(f"Shard {args.shard_id}/{args.num_shards}: running {len(cases)} cases", flush=True)
    results = []
    for idx, case in enumerate(cases, 1):
        cmd = [
            PYTHON,
            "-m",
            "memory_system.pointcloud_action.eval.eval_pointcloud_pick",
            "--memory", str(MEMORY),
            "--task", case["task"],
            "--demo", case["demo"],
            "--frame", str(case["frame"]),
            "--item", str(case["item"]),
            "--suite", "libero_10",
            "--demo-dir", str(DEMO_DIR),
            "--resolution", "256",
            "--max-steps", str(args.max_steps),
        ]
        if video_dir is not None:
            # Pass the directory (not a fixed .mp4 path) so eval_pointcloud_pick
            # appends success/frame info to the generated video filename.
            cmd += ["--save-video", str(video_dir)]
        print(f"[{idx}/{len(cases)}] {case['task']} {case['demo']} item={case['item']} frame={case['frame']}", flush=True)
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
        result = parse_result(proc.stdout)
        row = {**case, "success": result.get("success"), "memory_id": result.get("memory_id"),
               "distance": result.get("distance"), "controller_status": result.get("controller_status"),
               "error": result.get("error"), "returncode": proc.returncode}
        results.append(row)
        print(f"  -> success={row['success']} mem={row['memory_id']} dist={row['distance']}", flush=True)
        with open(output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"Done. Wrote {len(results)} results to {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
