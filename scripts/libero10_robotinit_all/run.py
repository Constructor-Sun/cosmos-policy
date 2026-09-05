#!/usr/bin/env python
"""Measure LIBERO-90 pointcloud memory retrieval on LIBERO-10 robotinit states.

For each LIBERO-10 Pick task, load its correct robot_initial_states variant,
take N initial states, extract the first Pick item's visible point cloud, and
query the LIBERO-90 pointcloud_action memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memory_system.pointcloud_action.execute.pointcloud_selector import PointCloudSelector
from memory_system.pointcloud_action.offline.extraction import (
    create_env,
    object_frame,
    visible_point_cloud,
)
from memory_system.offline.build_ready3d import resolve_instance
from memory_system.offline.build_targets import patch_numpy2_segmentation

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt"
MANIFEST = REPO / "skill_memory_test/libero_10/segments_ready_fixed16.json"

# task -> (robot_initial_states variant number, first Pick item)
TASKS = [
    ("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", 273, "moka_pot_1"),
    ("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", 274, "akita_black_bowl_1"),
    ("KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", 270, "white_yellow_mug_1"),
    ("KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", 269, "moka_pot_2"),
    ("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 268, "alphabet_soup_1"),
    ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", 271, "alphabet_soup_1"),
    ("LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket", 282, "cream_cheese_1"),
    ("LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", 265, "porcelain_mug_1"),
    ("LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", 267, "porcelain_mug_1"),
    ("STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", 276, "black_book_1"),
]


def load_robot_init_states(task: str, state: int):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"](
        category_value="Robot Initial States"
    )
    variant = f"{task}_view_0_0_100_0_0_initstate_{state}"
    for task_id in range(suite.n_tasks):
        if suite.get_task(task_id).name == variant:
            return suite.get_task_init_states(task_id)
    raise RuntimeError(f"variant not found: {variant}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "rollouts/libero10_robotinit_memory_retrieval.jsonl",
    )
    args = parser.parse_args()

    patch_numpy2_segmentation()
    selector = PointCloudSelector(MEMORY)
    results = []
    with open(args.output, "w") as out:
        for task, state, item in TASKS:
            print(f"Task {task} initstate_{state}", flush=True)
            init_states = load_robot_init_states(task, state)
            env = create_env(task, args.resolution, suite="libero_10")
            try:
                for trial in range(min(args.num_trials, len(init_states))):
                    env.reset()
                    init_state = np.asarray(init_states[trial])
                    obs = env.set_init_state(init_state)
                    instance = resolve_instance(env, {"item": item}, "Pick")
                    if instance is None:
                        row = {"task": task, "state": state, "trial": trial, "item": item,
                               "points": 0, "default_hits": 0, "error": "no instance"}
                        results.append(row)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        continue
                    points = visible_point_cloud(env, obs, instance, args.resolution)
                    T_world_object, _, _ = object_frame(env, instance)
                    hits = selector.memory.retrieve(points, skill="Pick", top_k=5)
                    top = hits[0] if hits else None
                    row = {
                        "task": task,
                        "state": state,
                        "trial": trial,
                        "item": item,
                        "points": int(len(points)),
                        "default_hits": len(hits),
                        "top_task": top["record"]["source_task"] if top else None,
                        "top_demo": top["record"]["source_demo"] if top else None,
                        "top_item": top["record"].get("arguments", {}).get("item") if top else None,
                        "top_distance": round(float(top["distance"]), 6) if top else None,
                    }
                    results.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"  trial={trial} points={len(points)} hits={len(hits)} top={row['top_item']}", flush=True)
            finally:
                env.close()
    print(f"Wrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
