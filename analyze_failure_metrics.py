"""Run detailed diagnostics for retrieval-success but Pick/stable-failure cases.

For each failed case from the merged object-coverage results, this script:
  - generates the same single-object BDDL from the original task scene,
  - runs the full local Pick pipeline,
  - records ready-pose error/reached, lift after replay, grasping after replay,
    and final lift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
BDDL_ROOT = Path(
    "/data1/liu/exp/counterfactual/external/LIBERO-plus/libero/libero/bddl_files"
)
RESULTS_PATHS = [
    REPO / "rollouts/object_coverage_1per_results.jsonl",
    REPO / "rollouts/object_coverage_fixed4_results.jsonl",
]
OUTPUT = REPO / "rollouts/object_coverage_failure_metrics.jsonl"


def load_merged_rows():
    rows = []
    for path in RESULTS_PATHS:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["object"], row["scene"], row["seed"])
            rows = [r for r in rows if (r["object"], r["scene"], r["seed"]) != key]
            rows.append(row)
    return rows


def main() -> int:
    sys.path.insert(0, str(REPO))
    from memory_system.pointcloud_action import config as pc_config

    pc_config.POINT_CLOUD_SOURCE = "complete"

    from memory_system.pointcloud_action.eval.local_pick_core import run_local_pick
    from memory_system.pointcloud_action.execute.pointcloud_selector import (
        PointCloudSelector,
    )
    from memory_system.pointcloud_action.offline.extraction import create_env
    from memory_system.pointcloud_action.offline.object_inventory import (
        discover_pick_objects,
    )
    from memory_system.pointcloud_action.offline.single_object_scene import (
        generate_single_object_bddl,
    )

    objs = discover_pick_objects()
    by_suite_type = {(o.suite, o.object_type): o for o in objs}

    merged = load_merged_rows()
    failures = [
        r for r in merged
        if not r.get("stable_success") and r.get("distance") is not None
    ]
    print(f"Analyzing {len(failures)} retrieval-success but action-failure cases", flush=True)

    selector = PointCloudSelector(
        REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt",
        cloud_key="complete_points_object",
    )

    out_rows = []
    for row in failures:
        scene = row["scene"]
        suite, _, task_name = scene.partition("/")
        obj = by_suite_type.get((suite, row["object"]))
        if obj is None:
            print(f"skip missing object {scene}", flush=True)
            continue

        out_dir = REPO / "memory_system/pointcloud_action/generated_failure_metrics_bddl"
        out_dir.mkdir(parents=True, exist_ok=True)
        bddl_path = out_dir / f"{suite}__{row['object']}.bddl"
        generate_single_object_bddl(
            obj.bddl_path,
            obj.object_name,
            obj.object_type,
            output_path=bddl_path,
        )

        print(f"[run] {scene} / {row['object']}", flush=True)
        try:
            np.random.seed(row["seed"])
            env = create_env(
                "failure_metrics",
                256,
                suite=suite,
                bddl_file_name=bddl_path,
            )
            try:
                obs = env.reset()
                result = run_local_pick(
                    env,
                    obs,
                    obj.object_name,
                    selector,
                    resolution=256,
                    top_k=1,
                    max_steps=200,
                    stable_hold_steps=20,
                    move_to_ready=True,
                    save_video=None,
                    task=f"{suite}__{row['object']}",
                    demo=f"seed{row['seed']}",
                    frame=0,
                )
                detail = {
                    "object": row["object"],
                    "scene": scene,
                    "seed": row["seed"],
                    "distance": result.get("distance"),
                    "ready_reached": result.get("ready_reached"),
                    "ready_pos_error": result.get("ready_pos_error"),
                    "grasping_after_replay": result.get("grasping_after_replay"),
                    "lift_after_replay": result.get("lift_after_replay"),
                    "lift_after_hold": result.get("lift"),
                    "stable_success": bool(result.get("success", False)),
                    "error": result.get("error"),
                }
                out_rows.append(detail)
                print(
                    f"  ready_reached={detail['ready_reached']} "
                    f"ready_err={detail['ready_pos_error']} "
                    f"grasp_after={detail['grasping_after_replay']} "
                    f"lift_after={detail['lift_after_replay']} "
                    f"lift_hold={detail['lift_after_hold']}",
                    flush=True,
                )
            finally:
                env.close()
        except Exception as exc:  # noqa: BLE001
            out_rows.append(
                {
                    "object": row["object"],
                    "scene": scene,
                    "seed": row["seed"],
                    "distance": row.get("distance"),
                    "ready_reached": False,
                    "ready_pos_error": None,
                    "grasping_after_replay": None,
                    "lift_after_replay": None,
                    "lift_after_hold": None,
                    "stable_success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  exception: {exc}", flush=True)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

    print(f"Done. Wrote {len(out_rows)} detailed rows to {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
