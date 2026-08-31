"""Smoke verification for held-object mask matching on a real LIBERO frame.

Run in the cosmospolicy conda env:

    conda activate cosmospolicy
    cd <repo>
    python scripts/verify_held_object_smoke.py
"""
from __future__ import annotations

import os

os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")

import numpy as np
from libero.libero import benchmark

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.artifacts import PhaseTargetMemory
from memory_system.execute.planner.held_object.geometry_builder import (
    HeldObjectGeometryBuilder,
)
from memory_system.execute.planner.held_object.mask_matcher import MemoryTemplateMatcher
from memory_system.geometry import camera_params as build_camera_params

BASE_TASK = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
PHASE_TARGETS = "skill_memory_test/libero_10/phase_targets.pt"


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    matches = [
        (index, suite.get_task(index))
        for index in range(suite.n_tasks)
        if suite.get_task(index).name == BASE_TASK
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one task, found {len(matches)}")
    _, task = matches[0]
    env, _ = get_libero_env(task, "cosmos", resolution=256, camera_depths=[True, False])
    try:
        env.reset()
        obs = env.env._get_observations()
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        memory = PhaseTargetMemory(PHASE_TARGETS)
        templates = [
            t
            for t in memory.templates
            if t["task_name"] == BASE_TASK
            and t["skill"] == "Pick"
            and t["arguments"].get("item") == "white_yellow_mug_1"
        ]
        if not templates:
            raise RuntimeError("no Pick template found for white_yellow_mug_1")
        template = templates[0]

        # Memory templates and metric depth are stored in canonical (flipped) space.
        rgb = np.flipud(np.asarray(obs["agentview_image"]))
        height, width = rgb.shape[:2]
        camera = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, camera, flip_images=True)

        matcher = MemoryTemplateMatcher()
        match = matcher.match(rgb, template)
        print("TEMPLATE", template["demo_id"], template["frame"], template["crop_rgb"].shape)
        if match is None:
            print("MATCH", "failed")
            return
        ys, xs = np.nonzero(match.mask)
        print("MATCH", f"conf={match.confidence:.3f}", f"pixels={len(ys)}", f"bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")

        instance_name = "white_yellow_mug_1"
        instance_id = getattr(env, "instance_to_id", {}).get(instance_name)
        if instance_id is not None and "agentview_segmentation_instance" in obs:
            seg = np.asarray(obs["agentview_segmentation_instance"])[..., 0]
            gt = np.flipud(seg == instance_id)
            intersection = float(np.logical_and(match.mask, gt).sum())
            union = float(np.logical_or(match.mask, gt).sum())
            iou = intersection / union if union > 0 else 0.0
            print(f"IOU {iou:.3f}")
        else:
            print("IOU unavailable")

        builder = HeldObjectGeometryBuilder(matcher, "white_yellow_mug_1", voxel_size=0.005)
        hand_to_world = np.eye(4)
        ok = builder.update(rgb, depth, camera, template, hand_to_world)
        obs_out = builder.build()
        if ok and obs_out is not None:
            print("GEOMETRY", f"frames={builder.matched_frames}", f"points={len(obs_out.points_hand)}")
        else:
            print("GEOMETRY", "failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
