"""Render point clouds for retrieval-success but action-failure cases as PNG."""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
RESULTS_PATHS = [
    REPO / "rollouts/object_coverage_1per_results.jsonl",
    REPO / "rollouts/object_coverage_fixed4_results.jsonl",
]
OUTPUT_DIR = REPO / "rollouts/failure_pointcloud_pngs_fixed"


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


def plot_pointcloud(points, path):
    fig = plt.figure(figsize=(12, 4))
    titles = ["3D view", "X-Y plane (top)", "X-Z plane (side)", "Y-Z plane (side)"]
    views = [
        (30, -60),
        (90, -90),
        (0, -90),
        (0, 0),
    ]
    for idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d" if idx == 0 else None)
        if idx == 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap="viridis", alpha=0.8)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
        elif idx == 1:
            ax.scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap="viridis", alpha=0.8)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
        elif idx == 2:
            ax.scatter(points[:, 0], points[:, 2], s=1, c=points[:, 2], cmap="viridis", alpha=0.8)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_aspect("equal")
        else:
            ax.scatter(points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap="viridis", alpha=0.8)
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_aspect("equal")
        ax.set_title(titles[idx])
    fig.suptitle(f"{path.stem}  N={len(points)}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    sys.path.insert(0, str(REPO))
    from memory_system.pointcloud_action import config as pc_config
    pc_config.POINT_CLOUD_SOURCE = "complete"
    from memory_system.pointcloud_action.offline.extraction import create_env, complete_point_cloud
    from memory_system.pointcloud_action.offline.object_inventory import discover_pick_objects
    from memory_system.pointcloud_action.offline.single_object_scene import generate_single_object_bddl
    from memory_system.offline.build_ready3d import resolve_instance

    objs = discover_pick_objects()
    by_suite_type = {(o.suite, o.object_type): o for o in objs}
    failures = [r for r in load_merged_rows() if not r["stable_success"] and r["distance"] is not None]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(failures)} failure point clouds to {OUTPUT_DIR}", flush=True)

    for row in failures:
        scene = row["scene"]
        suite, _, _ = scene.partition("/")
        obj = by_suite_type.get((suite, row["object"]))
        if obj is None:
            continue
        bddl_dir = REPO / "memory_system/pointcloud_action/generated_failure_pc_bddl"
        bddl_dir.mkdir(parents=True, exist_ok=True)
        bddl_path = bddl_dir / f"{suite}__{row['object']}.bddl"
        generate_single_object_bddl(obj.bddl_path, obj.object_name, obj.object_type, output_path=bddl_path)
        png_path = OUTPUT_DIR / f"{suite}__{row['object']}__seed{row['seed']}.png"
        print("[render]", scene, row["object"], flush=True)
        try:
            np.random.seed(row["seed"])
            env = create_env("pc", 256, suite=suite, bddl_file_name=bddl_path)
            try:
                obs = env.reset()
                inst = resolve_instance(env, {"item": obj.object_name}, "Pick")
                pts = complete_point_cloud(env, inst)
                print("  points", len(pts), "bbox min", pts.min(axis=0), "max", pts.max(axis=0), flush=True)
                if len(pts) >= 4:
                    plot_pointcloud(pts, png_path)
                else:
                    print("  too few points, skip png", flush=True)
            finally:
                env.close()
        except Exception as exc:
            print("  error", type(exc).__name__, exc, flush=True)
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
