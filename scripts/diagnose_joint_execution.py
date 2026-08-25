import numpy as np
from libero.libero import benchmark
from scipy.spatial.transform import Rotation

from cosmos_policy.experiments.robot.libero.libero_joint_control import (
    LiberoJointTrajectoryController,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from memory_system.execute.curobo_planner import CuroboPlanner
from memory_system.geometry import camera_params as build_camera_params


TASK_NAME = (
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_"
    "view_0_0_100_0_0_initstate_270"
)
TARGET = np.array(
    [
        -0.01693102903664112,
        -0.022844258695840836,
        1.005768060684204,
        2.7328341007232666,
        -2.389669179916382,
        -0.574185848236084,
    ],
    dtype=np.float32,
)


suite = benchmark.get_benchmark_dict()["libero_10"](
    category_value="Robot Initial States"
)
matches = [
    (index, suite.get_task(index))
    for index in range(suite.n_tasks)
    if suite.get_task(index).name == TASK_NAME
]
if len(matches) != 1:
    raise RuntimeError(f"expected one task, found {len(matches)}")
task_index, task = matches[0]
initial_state = suite.get_task_init_states(task_index)[0]
env, _ = get_libero_env(
    task,
    "cosmos",
    resolution=256,
    camera_depths=[True, False],
)
controller = None
try:
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
    height, width = obs["agentview_image"].shape[:2]
    camera = build_camera_params(env.sim, "agentview", height, width)
    depth = _make_main_depth(obs, camera, flip_images=True)
    current_ee = np.concatenate(
        [
            obs["robot0_eef_pos"],
            Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
        ]
    ).astype(np.float32)
    print("ROBOT_BASE", env.robots[0].base_pos, env.robots[0].base_ori)
    planner = CuroboPlanner(joint_execution=True)
    raw_points = planner._points_from_depth(depth, camera)
    result = planner.plan(
        current_ee_states=current_ee,
        target_ee_states=TARGET,
        depth=depth,
        camera_params=camera,
        joint_positions=obs["robot0_joint_pos"],
        robot_base_pose=np.concatenate([env.robots[0].base_pos, env.robots[0].base_ori]),
    )
    if result is None or result.joint_trajectory is None:
        raise RuntimeError("cuRobo did not return a joint trajectory")
    plan = result.joint_trajectory
    selected_target = np.asarray(result.target_ee_states)
    print(
        "PLAN",
        f"points={len(plan.position)}",
        f"dt={plan.dt:.6f}",
        f"motion_time={plan.motion_time:.6f}",
        f"required_steps={plan.required_steps(0.05)}",
    )
    robot = env.robots[0]
    q_indexes = robot._ref_joint_pos_indexes
    saved_q = env.sim.data.qpos[q_indexes].copy()
    env.sim.data.qpos[q_indexes] = plan.position[-1]
    env.sim.forward()
    robot_geoms = set(robot.robot_model.contact_geoms)
    robot_geoms.update(robot.gripper.contact_geoms)
    theoretical_contacts = []
    for contact_index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_index]
        first = env.sim.model.geom_id2name(contact.geom1) or ""
        second = env.sim.model.geom_id2name(contact.geom2) or ""
        if (first in robot_geoms) != (second in robot_geoms):
            contact_position = np.asarray(contact.pos).copy()
            nearest_depth = float(
                np.min(np.linalg.norm(raw_points - contact_position, axis=1))
            )
            theoretical_contacts.append(
                (first, second, float(contact.dist), contact_position, nearest_depth)
            )
    env._update_observables(force=True)
    final_obs = env.env._get_observations()
    final_ee = np.concatenate(
        [
            final_obs["robot0_eef_pos"],
            Rotation.from_quat(final_obs["robot0_eef_quat"]).as_rotvec(),
        ]
    )
    theoretical_pos = np.linalg.norm(selected_target[:3] - final_ee[:3])
    theoretical_original_pos = np.linalg.norm(TARGET[:3] - final_ee[:3])
    theoretical_rot = (
        Rotation.from_rotvec(selected_target[3:])
        * Rotation.from_rotvec(final_ee[3:]).inv()
    ).magnitude()
    print(
        "THEORETICAL", f"pos={theoretical_pos:.6f}",
        f"original_pos={theoretical_original_pos:.6f}",
        f"rot={theoretical_rot:.6f}", f"contacts={theoretical_contacts}",
    )
    planned_contacts = {}
    for path_index, q_ref in enumerate(plan.position):
        env.sim.data.qpos[q_indexes] = q_ref
        env.sim.forward()
        for contact_index in range(env.sim.data.ncon):
            contact = env.sim.data.contact[contact_index]
            first = env.sim.model.geom_id2name(contact.geom1) or ""
            second = env.sim.model.geom_id2name(contact.geom2) or ""
            if (first in robot_geoms) != (second in robot_geoms):
                pair = (first, second)
                event = planned_contacts.setdefault(
                    pair,
                    {"first": path_index, "last": path_index, "min_dist": 0.0},
                )
                event["last"] = path_index
                if float(contact.dist) < event["min_dist"]:
                    event["min_dist"] = float(contact.dist)
                    event["min_index"] = path_index
                    event["position"] = np.asarray(contact.pos).copy()
    print("PLANNED_CONTACTS", planned_contacts)
    env.sim.data.qpos[q_indexes] = saved_q
    env.sim.forward()
    env._update_observables(force=True)
    obs = env.env._get_observations()
    controller = LiberoJointTrajectoryController(
        env,
        plan,
        selected_target,
        gripper_command=-1.0,
    )
    print("ROBOT_GEOMS", sorted(robot_geoms))
    robot_contacts = {}
    for step in range(controller.max_steps):
        action = controller.step(obs)
        obs, _, _, _ = env.step(action.tolist())
        controller.observe(obs)
        for contact_index in range(env.sim.data.ncon):
            contact = env.sim.data.contact[contact_index]
            first = env.sim.model.geom_id2name(contact.geom1) or ""
            second = env.sim.model.geom_id2name(contact.geom2) or ""
            first_robot = first in robot_geoms
            second_robot = second in robot_geoms
            if first_robot != second_robot:
                pair = (first, second)
                distance = float(contact.dist)
                event = robot_contacts.setdefault(
                    pair,
                    {"first": step + 1, "last": step + 1, "min_dist": distance},
                )
                event["last"] = step + 1
                event["min_dist"] = min(event["min_dist"], distance)
        if step % 5 == 0 or controller.finished:
            print(
                "STEP",
                step + 1,
                controller.status,
                f"pos={controller.position_error:.6f}",
                f"rot={controller.rotation_error:.6f}",
            )
        if controller.finished:
            break
    controller.close()
    next_index = min(controller.reference_index + 1, len(controller.references) - 1)
    q_actual = np.asarray(obs["robot0_joint_pos"])
    print(
        "FINAL",
        controller.status,
        f"steps={controller.step_count}",
        f"pos={controller.position_error:.6f}",
        f"rot={controller.rotation_error:.6f}",
        f"tracking={controller.tracking_error:.6f}",
        f"next={np.max(np.abs(controller.references[next_index] - q_actual)):.6f}",
        f"contacts={env.sim.data.ncon}",
        f"robot_contacts={robot_contacts}",
        f"restored={env.robots[0].controller.name}",
        f"action_dim={env.env.action_dim}",
    )
finally:
    if controller is not None:
        controller.close()
    env.close()
