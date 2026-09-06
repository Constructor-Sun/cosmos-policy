"""Run object-interaction coverage sweeps for single-object local Pick.

For every Pick-target object discovered in the four LIBERO suites, generate a
single-object BDDL from its original task scene and evaluate the pointcloud
action memory with several random placements.

Output only:
    object, scene, seed, distance, stable_success
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
BDDL_ROOT = Path(
    "/data1/liu/exp/counterfactual/external/LIBERO-plus/libero/libero/bddl_files"
)
MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
DEFAULT_OUTPUT = REPO / "rollouts/object_interaction_coverage_results.jsonl"
DEFAULT_GENERATED_DIR = (
    REPO / "memory_system/pointcloud_action/generated_object_coverage_bddl"
)

ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_suites(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ALL_SUITES
    suites = []
    for value in values:
        if value not in ALL_SUITES:
            raise argparse.ArgumentTypeError(
                f"unknown suite {value!r}; choose from {ALL_SUITES}"
            )
        suites.append(value)
    return tuple(suites)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-object Pick interaction coverage across LIBERO suites."
    )
    parser.add_argument("--memory", type=Path, default=MEMORY)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=list(ALL_SUITES),
        help="Suite(s) to scan. If omitted, scan all four suites.",
    )
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--stable-hold-steps", type=int, default=20)
    parser.add_argument(
        "--point-cloud-source",
        choices=["complete", "visible"],
        default="complete",
    )
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-video-dir", type=Path, default=None)
    parser.add_argument(
        "--settle-steps", type=int, default=0,
        help="Zero-displacement/gripper-open actions after env.reset(), before "
             "anchoring (V1-style settle-first; 15 steps covers the measured "
             "12-15 step convergence confirmation of coverage objects).",
    )
    parser.add_argument(
        "--no-reanchor", action="store_true",
        help="Disable REANCHOR_AFTER_READY (V1 semantics; default config uses re-anchor).",
    )
    parser.add_argument(
        "--retry-wptime", action="store_true",
        help="Hybrid max-success strategy: attempt 1 = V1 semantics (settle-first, "
             "open_loop, no re-anchor); on failure reset the scene and retry once "
             "with V2 semantics (REANCHOR_AFTER_READY + wp_time).  Success is "
             "judged on final object state; the producing attempt is recorded.",
    )
    args = parser.parse_args()

    # Set config before importing modules that read config values at import time.
    from memory_system.pointcloud_action import config as pc_config

    pc_config.POINT_CLOUD_SOURCE = args.point_cloud_source
    if args.no_reanchor:
        pc_config.REANCHOR_AFTER_READY = False

    from memory_system.pointcloud_action.eval.local_pick_core import run_local_pick
    from memory_system.offline.label_segments import object_position
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

    suites = parse_suites(args.suite)
    objects = discover_pick_objects(bddl_root=BDDL_ROOT, suites=suites)
    if not objects:
        print(f"No Pick objects discovered in suites: {suites}", file=sys.stderr)
        return 1

    print(f"Discovered {len(objects)} Pick object types from {suites}", flush=True)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video_dir is not None:
        args.save_video_dir.mkdir(parents=True, exist_ok=True)

    selector = PointCloudSelector(
        args.memory,
        cloud_key=(
            "complete_points_object"
            if args.point_cloud_source == "complete"
            else "target_points_object"
        ),
    )

    results = []
    for obj in objects:
        object_type = obj.object_type
        object_name = obj.object_name
        # Use the original task BDDL as the scene template.
        safe_name = f"{obj.suite}__{object_type}".replace("/", "_")
        bddl_path = args.generated_dir / f"{safe_name}.bddl"
        generate_single_object_bddl(
            obj.bddl_path,
            object_name,
            object_type,
            output_path=bddl_path,
        )

        for seed in range(args.num_cases):
            print(
                f"[run] object={object_type} suite={obj.suite} seed={seed}",
                flush=True,
            )
            row = {
                "object": object_type,
                "scene": f"{obj.suite}/{obj.task_name}",
                "seed": seed,
                "distance": None,
                "stable_success": False,
            }
            try:
                np.random.seed(seed)
                env = create_env(
                    "object_interaction_pick",
                    args.resolution,
                    suite=obj.suite,
                    bddl_file_name=bddl_path,
                )
                try:
                    obs = env.reset()
                    row["object_pos_reset"] = [
                        round(float(v), 4)
                        for v in object_position(env.env, object_name)
                    ]
                    if args.settle_steps > 0:
                        settle_action = np.zeros(7, dtype=np.float32)
                        settle_action[-1] = -1.0
                        for _ in range(args.settle_steps):
                            step_result = env.step(settle_action.tolist())
                            obs = (
                                step_result[0]
                                if isinstance(step_result, tuple)
                                else step_result
                            )
                        row["object_pos_anchor"] = [
                            round(float(v), 4)
                            for v in object_position(env.env, object_name)
                        ]
                    first_mode = "open_loop" if args.retry_wptime else None
                    if args.retry_wptime:
                        pc_config.REANCHOR_AFTER_READY = False
                    result = run_local_pick(
                        env,
                        obs,
                        object_name,
                        selector,
                        resolution=args.resolution,
                        top_k=args.top_k,
                        max_steps=args.max_steps,
                        stable_hold_steps=args.stable_hold_steps,
                        move_to_ready=True,
                        save_video=(
                            str(args.save_video_dir) if args.save_video_dir else None
                        ),
                        task=safe_name,
                        demo=f"seed{seed}",
                        frame=0,
                        replay_mode=first_mode,
                    )
                    row["attempt1_success"] = bool(result.get("success", False))
                    row["attempts"] = 1
                    if args.retry_wptime:
                        row["strategy"] = "settle_first_openloop"
                    if args.retry_wptime and not result.get("success", False):
                        # Clean-scene retry with V2 semantics.
                        row["attempts"] = 2
                        row["strategy"] = "settle_first_openloop_then_wptime_retry"
                        pc_config.REANCHOR_AFTER_READY = True
                        np.random.seed(seed)
                        obs = env.reset()
                        row["object_pos_retry"] = [
                            round(float(v), 4)
                            for v in object_position(env.env, object_name)
                        ]
                        result = run_local_pick(
                            env,
                            obs,
                            object_name,
                            selector,
                            resolution=args.resolution,
                            top_k=args.top_k,
                            max_steps=args.max_steps,
                            stable_hold_steps=args.stable_hold_steps,
                            move_to_ready=True,
                            save_video=(
                                str(args.save_video_dir)
                                if args.save_video_dir
                                else None
                            ),
                            task=safe_name,
                            demo=f"seed{seed}_v2retry",
                            frame=0,
                            replay_mode="wp_time",
                        )
                    row["distance"] = result.get("distance")
                    row["stable_success"] = bool(result.get("success", False))
                finally:
                    env.close()
            except Exception as exc:
                # Keep the row slim but record why it failed for later triage.
                row["distance"] = None
                row["stable_success"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"

            results.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(
                f"  -> distance={row['distance']} stable_success={row['stable_success']}",
                flush=True,
            )

    print(f"Done. Wrote {len(results)} results to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
