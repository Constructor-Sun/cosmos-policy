"""Counterfactual Pick evaluation with LIBERO-Plus-only physical objects.

Selects object classes registered by LIBERO-Plus custom_objects.py, substitutes
each into a simple LIBERO-10 Pick BDDL, and evaluates random reset states with
the frozen pointcloud-action memory. Generation and evaluation are separate;
--list-only never creates an environment or executes a rollout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
LIBERO_PLUS = REPO.parent / "LIBERO-plus"
BDDL_DIR = LIBERO_PLUS / "libero/libero/bddl_files/libero_10"
TEMPLATE = BDDL_DIR / "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"
DEFAULT_GENERATED_DIR = BDDL_DIR
DEFAULT_OUTPUT = REPO / "rollouts/plus_only_object_replacement.jsonl"
sys.path.insert(0, str(LIBERO_PLUS))

REPLACEMENT_OBJECTS = [
    ("can_of_sardines", "can"),
    ("can_of_soda__1", "can"),
    ("canned_food__1", "can"),
    ("bottle_of_alfredo_sauce", "bottle"),
    ("bottle_of_antihistamines", "bottle"),
    ("bottle_of_aspirin", "bottle"),
    ("bottle_of_baby_oil", "bottle"),
    ("bottle_of_barbecue_sauce__2", "bottle"),
    ("box_of_yogurt__1", "box"),
    ("box_of_vegetable_juice__1", "box"),
]


def plus_only_objects(limit: int | None = None) -> list[dict[str, str]]:
    """Return curated Plus-only container objects and verify registration."""
    import libero.libero.envs.objects  # noqa: F401
    from libero.libero.envs.base_object import OBJECTS_DICT

    selected = []
    for key, family in REPLACEMENT_OBJECTS[:limit]:
        cls = OBJECTS_DICT.get(key)
        if cls is None or not cls.__module__.endswith(".custom_objects"):
            raise RuntimeError(f"not a registered Plus custom object: {key}")
        selected.append({"type": key, "family": family, "asset": "assets/new_objects/" + family})
    return selected


def generate_bddl(object_type: str, output_dir: Path) -> tuple[str, str]:
    """Create one BDDL by replacing only the original Pick target object."""
    text = TEMPLATE.read_text()
    item = f"{object_type}_1"
    old_decl = "alphabet_soup_1 - alphabet_soup"
    if old_decl not in text:
        raise ValueError(f"template target declaration not found: {old_decl}")
    text = text.replace(old_decl, f"{item} - {object_type}")
    text = text.replace("alphabet_soup_1", item)
    text = text.replace(
        "both the alphabet soup and the tomato sauce",
        f"the {object_type.replace('_', ' ')} and the tomato sauce",
    )
    task_name = f"counterfactual_plusonly_{object_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{task_name}.bddl").write_text(text)
    return task_name, item


def evaluate_random_reset(memory_path: str, task_name: str, item: str, seed: int,
                          resolution: int = 256, top_k: int = 1,
                          max_steps: int = 200,
                          save_video_dir: str | None = None) -> dict:
    """Evaluate one random-reset Pick case using the existing pipeline."""
    from memory_system.pointcloud_action import config as pc_config
    pc_config.POINT_CLOUD_SOURCE = "complete"
    from memory_system.pointcloud_action.execute.pointcloud_controller import PointCloudPickController
    from memory_system.pointcloud_action.execute.pointcloud_selector import PointCloudSelector
    from memory_system.pointcloud_action.eval.eval_pointcloud_pick import (
        _execute_controller, _main_image, _move_to_ready, _pick_succeeded,
        _replay_actions, _sync_controller,
    )
    from memory_system.pointcloud_action.offline.extraction import complete_point_cloud, create_env, object_frame
    from memory_system.offline.build_ready3d import resolve_instance
    from memory_system.offline.label_segments import object_position

    np.random.seed(seed)
    selector = PointCloudSelector(memory_path, cloud_key="complete_points_object")
    env = create_env(task_name, resolution, suite="libero_10")
    try:
        obs = env.reset()
        _sync_controller(env)
        frames = [_main_image(obs)]
        instance = resolve_instance(env, {"item": item}, "Pick")
        if instance is None:
            return {"success": False, "error": f"instance not found: {item}"}
        points = complete_point_cloud(env, instance)
        if len(points) < 4:
            return {"success": False, "error": "empty point cloud"}
        T_world_object, _, _ = object_frame(env, instance)
        candidates = selector.select(points, T_world_object, skill="Pick", top_k=top_k)
        if not candidates:
            return {"success": False, "error": "no retrieved memory"}
        best = candidates[0]
        start_pos = object_position(env.env, item)
        obs = _move_to_ready(env, obs, best["ready_ee_states"], resolution, max_steps, frames=frames)
        _, _, current_rotation = object_frame(env, instance)
        controller = PointCloudPickController(actions=_replay_actions(best["record"], current_rotation))
        _execute_controller(env, controller, obs, max_steps, frames=frames)
        success = _pick_succeeded(env, item, start_pos)
        video_path = None
        if save_video_dir is not None:
            from memory_system.pointcloud_action.eval.eval_pointcloud_pick import _write_video
            video_path = _write_video(frames, save_video_dir, task_name, item, seed, success)
        return {
            "success": success,
            "memory_id": best["record"]["memory_id"],
            "distance": best["distance"],
            "controller_status": controller.status,
            "video_path": video_path,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt")
    parser.add_argument("--num-objects", type=int, default=10)
    parser.add_argument("--cases-per-object", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-video-dir", type=Path, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("invalid --shard-id/--num-shards")

    objects = plus_only_objects(args.num_objects)
    if len(objects) < args.num_objects:
        raise RuntimeError(f"requested {args.num_objects} distinct Plus-only assets, found {len(objects)}")
    print(json.dumps(objects, indent=2), flush=True)
    if args.list_only:
        return

    objects = objects[args.shard_id::args.num_shards]
    output = args.output
    if args.num_shards > 1:
        output = output.with_name(f"{output.stem}.shard{args.shard_id}{output.suffix}")
        video_dir = args.save_video_dir / f"shard_{args.shard_id}" if args.save_video_dir else None
    else:
        video_dir = args.save_video_dir
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(objects)} objects", flush=True)

    results = []
    for obj in objects:
        task_name, item = generate_bddl(obj["type"], args.generated_dir)
        for seed in range(args.cases_per_object):
            print(f"[run] object={obj['type']} seed={seed}", flush=True)
            try:
                result = evaluate_random_reset(str(args.memory), task_name, item, seed, max_steps=args.max_steps, save_video_dir=str(video_dir) if video_dir else None)
            except Exception as exc:
                result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append({"object_type": obj["type"], "family": obj["family"], "asset": obj["asset"], "task": task_name, "seed": seed, **result})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(json.dumps(x) for x in results) + "\n")


if __name__ == "__main__":
    main()
