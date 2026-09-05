"""Run PointCloud Action Memory Pick on LIBERO-Plus Objects Layout variants.

This evaluates only the local Pick subtask:
  - create a LIBERO-Plus objects_layout variant of a LIBERO-10 task
  - start from that variant initial state
  - retrieve from the LIBERO-90 pointcloud_action memory
  - move to ready and replay the Pick action
  - report whether the target object is grasped/lifted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
LIBERO_PLUS = Path("/data1/liu/exp/counterfactual/external/LIBERO-plus")
if str(LIBERO_PLUS) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS))

from libero.libero import benchmark  # noqa: E402

# The target Pick item(s) for each LIBERO-10 base task used by pointcloud_action.
BASE_TASK_ITEMS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": ["moka_pot_1"],
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": ["akita_black_bowl_1"],
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": ["white_yellow_mug_1"],
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": ["moka_pot_1", "moka_pot_2"],
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": ["alphabet_soup_1", "cream_cheese_1"],
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": ["alphabet_soup_1", "tomato_sauce_1"],
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": ["butter_1", "cream_cheese_1"],
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": ["porcelain_mug_1", "white_yellow_mug_1"],
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": ["chocolate_pudding_1", "porcelain_mug_1"],
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": ["black_book_1"],
}


def _as_1d_state(state):
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state)
    if state.ndim == 2:
        state = state[0]
    return np.asarray(state, dtype=np.float64).reshape(-1)


def _load_variants(base_task: str):
    suite = benchmark.get_benchmark_dict()["libero_10"](category_value="Objects Layout")
    variants = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        if task.name.startswith(base_task):
            variants.append((task_id, task))
    return suite, variants


def evaluate_variant(
    memory_path: str,
    variant_task_name: str,
    state: np.ndarray,
    item: str,
    resolution: int,
    top_k: int,
    max_steps: int,
    save_video_dir: str | None,
    move_to_ready: bool = True,
):
    # Must set this before importing eval_pointcloud_pick because that module
    # reads POINT_CLOUD_SOURCE at import time.
    from memory_system.pointcloud_action import config as pc_config
    from memory_system.pointcloud_action.execute.pointcloud_controller import (
        PointCloudPickController,
    )
    from memory_system.pointcloud_action.execute.pointcloud_selector import (
        PointCloudSelector,
    )
    from memory_system.pointcloud_action.eval.eval_pointcloud_pick import (
        _execute_controller,
        _main_image,
        _move_to_ready,
        _pick_succeeded,
        _replay_actions,
        _set_state,
        _sync_controller,
        _write_video,
    )
    from memory_system.pointcloud_action.offline.extraction import (
        complete_point_cloud,
        create_env,
        object_frame,
        visible_point_cloud,
    )
    from memory_system.offline.build_ready3d import resolve_instance
    from memory_system.offline.label_segments import object_position

    cloud_key = (
        "complete_points_object"
        if pc_config.POINT_CLOUD_SOURCE == "complete"
        else "target_points_object"
    )
    selector = PointCloudSelector(memory_path, cloud_key=cloud_key)

    env = create_env(variant_task_name, resolution, suite="libero_10")
    try:
        env.reset()
        _set_state(env, state)
        obs = env.regenerate_obs_from_state(state)
        _sync_controller(env)
        frames = [_main_image(obs)]

        instance = resolve_instance(env, {"item": item}, "Pick")
        if instance is None:
            return {"success": False, "error": f"instance not found: {item}"}

        if pc_config.POINT_CLOUD_SOURCE == "complete":
            points = complete_point_cloud(env, instance)
        else:
            points = visible_point_cloud(env, obs, instance, resolution)
        if len(points) < 4:
            return {"success": False, "error": "empty point cloud"}

        T_world_object, _, _ = object_frame(env, instance)
        candidates = selector.select(points, T_world_object, skill="Pick", top_k=top_k)
        if not candidates:
            return {"success": False, "error": "no retrieved memory"}

        best = candidates[0]
        if move_to_ready:
            obs = _move_to_ready(
                env,
                obs,
                best["ready_ee_states"],
                resolution,
                max_steps,
                frames=frames,
            )

        start_pos = object_position(env.env, item)
        _, _, R_cur = object_frame(env, instance)
        replay_actions = _replay_actions(best["record"], R_cur)
        pick_controller = PointCloudPickController(actions=replay_actions)
        obs = _execute_controller(
            env, pick_controller, obs, max_steps, frames=frames
        )
        success = _pick_succeeded(env, item, start_pos)

        video_path = None
        if save_video_dir is not None:
            video_path = _write_video(
                frames, save_video_dir, variant_task_name, "init", 0, success
            )

        return {
            "success": success,
            "memory_id": best["record"]["memory_id"],
            "distance": best["distance"],
            "controller_status": pick_controller.status,
            "video_path": video_path,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=str, required=True)
    parser.add_argument("--max-cases-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--point-cloud-source", choices=["visible", "complete"], default="visible")
    parser.add_argument("--output", type=Path, default=REPO / "rollouts/libero10_objects_layout_pick_sweep_results.jsonl")
    parser.add_argument("--save-video-dir", type=Path, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    # This must happen before importing eval_pointcloud_pick.
    from memory_system.pointcloud_action import config as pc_config
    pc_config.POINT_CLOUD_SOURCE = args.point_cloud_source

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        parser.error("invalid shard settings")
    output = args.output
    if args.num_shards > 1:
        output = output.with_name(f"{output.stem}.shard{args.shard_id}{output.suffix}")
    if args.save_video_dir is not None:
        args.save_video_dir.mkdir(parents=True, exist_ok=True)

    # Build at most args.max_cases_per_task cases for each LIBERO-10 base task.
    cases = []
    for base_task, items in BASE_TASK_ITEMS.items():
        suite, variants = _load_variants(base_task)
        task_cases = []
        for task_id, task in variants:
            state = _as_1d_state(suite.get_task_init_states(task_id))
            for item in items:
                task_cases.append({
                    "base_task": base_task,
                    "task": task.name,
                    "task_id": task_id,
                    "item": item,
                    "state": state,
                })
                if len(task_cases) >= args.max_cases_per_task:
                    break
            if len(task_cases) >= args.max_cases_per_task:
                break
        cases.extend(task_cases)

    cases = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard_id]
    print(f"Shard {args.shard_id}/{args.num_shards}: running {len(cases)} cases", flush=True)

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['task']} item={case['item']}", flush=True)
        try:
            res = evaluate_variant(
                args.memory,
                case["task"],
                case["state"],
                case["item"],
                resolution=args.resolution,
                top_k=args.top_k,
                max_steps=args.max_steps,
                save_video_dir=str(args.save_video_dir) if args.save_video_dir else None,
            )
        except Exception as exc:  # keep sweep alive
            res = {"success": False, "error": f"exception: {exc}"}
        row = {
            "base_task": case["base_task"],
            "task": case["task"],
            "task_id": case["task_id"],
            "item": case["item"],
            "success": res.get("success"),
            "memory_id": res.get("memory_id"),
            "distance": res.get("distance"),
            "controller_status": res.get("controller_status"),
            "error": res.get("error"),
        }
        results.append(row)
        print(f"  -> success={row['success']} mem={row['memory_id']} dist={row['distance']}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"Done. Wrote {len(results)} results to {output}", flush=True)


if __name__ == "__main__":
    main()
