"""Verify HeldObjectPlanner in a real LIBERO PlaceIn scenario.

This script uses simulator instance segmentation only to build the held-object
observation (test oracle). Robot removal is handled by HeldObjectPlanner's
URDF depth filter.
"""
from __future__ import annotations

import os

os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")

import h5py
import numpy as np
from libero.libero import benchmark
from scipy.spatial.transform import Rotation
import torch

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
from memory_system.execute.planner.held_object.types import (
    HeldObjectObservation,
    HeldObjectPlannerInput,
)
from memory_system.geometry import camera_params as build_camera_params, pixel_to_world

BASE_TASK = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
DEMO_ID = "demo_1"
START_FRAME = 110
HDF5 = "LIBERO-Cosmos-Policy/success_only/libero_10_regen/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"
FEASIBLE = "skill_memory_test/libero_10/feasible_recovery_targets.pt"


def _current_ee(obs):
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
        ]
    ).astype(np.float32)


def _hand_to_world(planner, obs, robot_base_pose):
    from curobo._src.state.state_joint import JointState

    curobo = planner._ensure_planner()
    state = JointState.from_position(
        torch.as_tensor(
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(1, -1),
            device=planner.device,
            dtype=torch.float32,
        ),
        joint_names=curobo.joint_names,
    )
    kin = curobo.compute_kinematics(state)
    sk = kin.tool_poses.get_link_pose(curobo.tool_frames[0])
    p0 = sk.position.detach().cpu().numpy().reshape(3)
    q0 = sk.quaternion.detach().cpu().numpy().reshape(4)
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    base = np.asarray(robot_base_pose, dtype=np.float64).reshape(7)
    rb = Rotation.from_quat(base[3:]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rb @ r0
    mat[:3, 3] = rb @ p0 + base[:3]
    return mat


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    matches = [
        (i, suite.get_task(i))
        for i in range(suite.n_tasks)
        if suite.get_task(i).name == BASE_TASK
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one task, found {len(matches)}")
    _, task = matches[0]
    env, _ = get_libero_env(task, "cosmos", resolution=256, camera_depths=[True, False])
    try:
        with h5py.File(HDF5, "r") as handle:
            states = handle["data"][DEMO_ID]["states"][:]
        env.reset()
        obs = env.set_init_state(states[START_FRAME])
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        height, width = obs["agentview_image"].shape[:2]
        camera = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(obs, camera, flip_images=True)
        robot_base_pose = np.concatenate([env.robots[0].base_pos, env.robots[0].base_ori])

        planner = HeldObjectPlanner()
        hand_to_world = _hand_to_world(planner, obs, robot_base_pose)
        seg_inst = np.flipud(np.asarray(obs["agentview_segmentation_instance"])[..., 0])
        instance_id = env.instance_to_id["white_yellow_mug_1"]
        mask = seg_inst == instance_id
        pixels = np.stack(np.nonzero(mask), axis=-1)
        world = pixel_to_world(pixels, depth, camera)
        rotation = hand_to_world[:3, :3]
        translation = hand_to_world[:3, 3]
        points_hand = (rotation.T @ (world - translation).T).T
        held_object = HeldObjectObservation(
            item="white_yellow_mug_1",
            points_hand=points_hand.astype(np.float32),
            source="simulator_oracle_for_test",
        )
        print(
            "HELD_OBJECT",
            f"points={len(points_hand)}",
            f"hand_bbox_min={points_hand.min(0).tolist()}",
            f"hand_bbox_max={points_hand.max(0).tolist()}",
        )

        ready_pose = _current_ee(obs)
        ready_pose[2] += 0.02

        inp = HeldObjectPlannerInput(
            joint_positions=obs["robot0_joint_pos"],
            ee_states=_current_ee(obs),
            depth=depth,
            camera_params=camera,
            held_object=held_object,
            ready_pose=ready_pose,
            gripper_joint_positions=obs.get("robot0_gripper_qpos"),
            robot_base_pose=robot_base_pose,
        )
        result = planner.plan(inp)
        if result is None:
            print("PLAN", "failed")
            return
        print("PLAN", "ok", f"waypoints={len(result.waypoints)}", f"target={result.target_ee_states.tolist()}")

        controller = result.controller
        contacts_before = env.sim.data.ncon
        for step in range(controller.max_steps):
            current = _current_ee(obs)
            action6 = controller.step(current)
            action = np.zeros(7, dtype=np.float32)
            action[:6] = action6
            action[6] = -1.0
            obs, _, _, _ = env.step(action.tolist())
            if controller.finished:
                break
        print("EXEC", f"status={controller.status}", f"steps={controller.step_count}", f"contacts_before={contacts_before}", f"contacts_after={env.sim.data.ncon}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
