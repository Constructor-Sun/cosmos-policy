"""Check which memory object is retrieved for each action-failure case."""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
RESULTS_PATHS = [
    REPO / "rollouts/object_coverage_1per_results.jsonl",
    REPO / "rollouts/object_coverage_fixed4_results.jsonl",
]
OUTPUT = REPO / "rollouts/object_coverage_failure_retrieval_names.jsonl"


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
    from memory_system.pointcloud_action.execute.pointcloud_selector import PointCloudSelector
    from memory_system.pointcloud_action.offline.extraction import create_env, complete_point_cloud
    from memory_system.pointcloud_action.offline.object_inventory import discover_pick_objects
    from memory_system.pointcloud_action.offline.single_object_scene import generate_single_object_bddl
    from memory_system.offline.build_ready3d import resolve_instance
    from memory_system.pointcloud_action.offline.extraction import object_frame

    objs = discover_pick_objects()
    by_suite_type = {(o.suite, o.object_type): o for o in objs}
    failures = [r for r in load_merged_rows() if not r["stable_success"] and r["distance"] is not None]
    selector = PointCloudSelector(REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt", cloud_key="complete_points_object")
    out = []
    for row in failures:
        scene = row["scene"]
        suite, _, _ = scene.partition("/")
        obj = by_suite_type.get((suite, row["object"]))
        if obj is None:
            continue
        bddl_dir = REPO / "memory_system/pointcloud_action/generated_retrieval_check_bddl"
        bddl_dir.mkdir(parents=True, exist_ok=True)
        bddl_path = bddl_dir / f"{suite}__{row['object']}.bddl"
        generate_single_object_bddl(obj.bddl_path, obj.object_name, obj.object_type, output_path=bddl_path)
        print("[check]", scene, row["object"], flush=True)
        try:
            np.random.seed(row["seed"])
            env = create_env("check", 256, suite=suite, bddl_file_name=bddl_path)
            try:
                obs = env.reset()
                inst = resolve_instance(env, {"item": obj.object_name}, "Pick")
                pts = complete_point_cloud(env, inst)
                T, _, _ = object_frame(env, inst)
                hits = selector.select(pts, T, skill="Pick", top_k=1)
                hit = hits[0] if hits else None
                rec = hit["record"] if hit else None
                item_name = str(rec.get("arguments", {}).get("item")) if rec else None
                source_task = str(rec.get("source_task")) if rec else None
                out.append({
                    "object": row["object"],
                    "scene": scene,
                    "seed": row["seed"],
                    "retrieved_item": item_name,
                    "retrieved_type": (item_name.rsplit("_", 1)[0] if item_name else None),
                    "source_task": source_task,
                    "distance": hit["distance"] if hit else None,
                })
                print("  retrieved_item", item_name, "source", source_task, "dist", hit["distance"] if hit else None, flush=True)
            finally:
                env.close()
        except Exception as e:
            out.append({"object": row["object"], "scene": scene, "seed": row["seed"], "error": f"{type(e).__name__}: {e}"})
            print("  error", e, flush=True)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    print("Done", len(out), OUTPUT, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
