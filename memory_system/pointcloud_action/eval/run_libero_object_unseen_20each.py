"""Run 20 cases each for bbq_sauce and salad_dressing (unseen targets), shardable."""
from __future__ import annotations

import argparse
import h5py
import json
from pathlib import Path

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
DEMO_DIR = REPO / "LIBERO-Cosmos-Policy/success_only/libero_object_regen"
MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
DEFAULT_OUTPUT = REPO / "rollouts/libero_object_unseen_20each_results.jsonl"

# Must be set before importing eval_pointcloud_pick.
from memory_system.pointcloud_action import config as pc_config
pc_config.POINT_CLOUD_SOURCE = "complete"

from memory_system.pointcloud_action.eval.eval_pointcloud_pick import evaluate  # noqa: E402

TASKS = {
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": "bbq_sauce_1",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket": "salad_dressing_1",
}
NUM_CASES_PER_TASK = 20


def first_n_demos(task: str, n: int):
    with h5py.File(DEMO_DIR / f"{task}_demo.hdf5", "r") as handle:
        demos = list(handle["data"].keys())
    demos.sort(key=lambda x: int(x.split("_")[1]))
    return demos[:n]


def build_cases():
    cases = []
    for task, item in TASKS.items():
        for demo in first_n_demos(task, NUM_CASES_PER_TASK):
            cases.append({"task": task, "demo": demo, "item": item})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        parser.error("invalid shard settings")

    output = args.output
    if args.num_shards > 1:
        output = output.with_name(f"{output.stem}.shard{args.shard_id}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    cases = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard_id]
    print(f"Shard {args.shard_id}/{args.num_shards}: running {len(cases)} cases", flush=True)

    results = []
    for case in cases:
        print(f"[run] {case['task']} {case['demo']}", flush=True)
        try:
            res = evaluate(
                str(MEMORY),
                case["task"],
                case["demo"],
                0,
                case["item"],
                resolution=256,
                top_k=1,
                max_steps=200,
                suite="libero_object",
                demo_dir=str(DEMO_DIR),
                move_to_ready=True,
            )
        except Exception as exc:
            res = {"success": False, "error": f"exception: {exc}"}

        row = {
            "task": case["task"],
            "demo": case["demo"],
            "item": case["item"],
            "success": res.get("success"),
            "memory_id": res.get("memory_id"),
            "distance": res.get("distance"),
            "controller_status": res.get("controller_status"),
            "error": res.get("error"),
        }
        results.append(row)
        print(
            f"  -> success={row['success']} mem={row['memory_id']} "
            f"dist={row['distance']}",
            flush=True,
        )
        with open(output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print(f"Done. Wrote {len(results)} results to {output}", flush=True)


if __name__ == "__main__":
    main()
