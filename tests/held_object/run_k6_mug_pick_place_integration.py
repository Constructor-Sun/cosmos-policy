#!/usr/bin/env python3
"""K6 single-case integration runner for held-object Pick->Place.

This runner is intentionally task-specific and keeps K6 constants out of the
generic ``run_libero_eval.py`` path.  It uses the new no-oracle extractor and
feeds the result into ``HeldObjectPlanner`` through the optional
``phase_transition_hook`` added to ``run_episode``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]

TASK = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
ITEM = "white_yellow_mug_1"
FEASIBLE_RECOVERY_TARGETS = (
    _REPO_ROOT / "skill_memory_test" / "libero_10" / "feasible_recovery_targets.pt"
)
OUTPUT_DIR = _REPO_ROOT / "tests" / "held_object" / "k6_pick_place_integration"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument("--config_file", default="cosmos_policy/config/config.py")
    parser.add_argument("--dataset_stats_path", default="/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
    parser.add_argument("--t5_text_embeddings_path", default="/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
    parser.add_argument("--local_log_dir", default=str(OUTPUT_DIR / "logs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    return parser.parse_args()


def _build_cfg(args: argparse.Namespace):
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

    return PolicyEvalConfig(
        suite="libero",
        model_family="cosmos",
        config=args.config,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        flip_images=True,
        use_variance_scale=False,
        use_jpeg_compression=True,
        ar_future_prediction=False,
        ar_value_prediction=False,
        ar_qvalue_prediction=False,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        unnormalize_actions=True,
        normalize_proprio=True,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        trained_with_image_aug=True,
        chunk_size=16,
        num_open_loop_steps=16,
        deterministic=True,
        deterministic_reset=True,
        deterministic_reset_seed=0,
        num_queries_best_of_n=1,
        use_parallel_inference=False,
        task_suite_name="libero_10",
        num_trials_per_task=args.num_trials,
        task_filter=TASK,
        env_img_res=256,
        local_log_dir=args.local_log_dir,
        run_id_note="k6-held-object",
        use_wandb=False,
        seed=args.seed,
        randomize_seed=False,
        data_collection=False,
        enable_initial_alignment=True,
        enable_collision_aware_initial_alignment=True,
        enable_curobo_joint_execution=False,
        enable_urdf_robot_filter=True,
    )


def _make_phase_transition_hook():
    from memory_system.artifacts import FeasibleRecoveryMemory
    from memory_system.execute.planner.held_object.connected_component import (
        HeldObjectConnectedComponentExtractor,
    )
    from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
    from memory_system.execute.planner.held_object.types import HeldObjectPlannerInput
    from memory_system.geometry import camera_params as build_camera_params

    from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth

    extractor = HeldObjectConnectedComponentExtractor()
    planner = HeldObjectPlanner()
    memory = FeasibleRecoveryMemory(str(FEASIBLE_RECOVERY_TARGETS))

    # Test-only: use the GPU Warp URDF filter from the diagnostic script so the
    # K6 runner does not stall on CPU ray casting.  This is not promoted into
    # the production extractor.
    import importlib.util
    import sys

    import curobo
    from memory_system.execute.urdf_depth_filter import UrdfDepthFilterConfig

    _base_path = Path(__file__).resolve().parents[2] / "tests" / "held_object" / "visualize_held_object_after_robot_removal.py"
    _spec = importlib.util.spec_from_file_location("held_object_gpu_urdf_base", _base_path)
    _base = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
    _urdf_path = str(
        Path(curobo.__file__).resolve().parent
        / "content" / "assets" / "robot" / "franka_description" / "franka_panda.urdf"
    )
    extractor.planner._urdf_filter = _base.WarpUrdfDepthFilter(
        UrdfDepthFilterConfig(urdf_path=_urdf_path), device="cuda:0"
    )

    def hook(
        *,
        observation,
        completed_phase,
        next_phase,
        task_name,
        demo_id,
        env,
        cfg,
        log_file,
    ):
        item = completed_phase.arguments.get("item") or ITEM
        height, width = observation["agentview_image"].shape[:2]
        camera = build_camera_params(env.sim, "agentview", height, width)
        depth = _make_main_depth(observation, camera, cfg.flip_images)
        robot_base_pose = np.concatenate(
            [env.robots[0].base_pos, env.robots[0].base_ori]
        ).astype(np.float64)

        held = extractor.extract(
            depth=depth,
            camera_params=camera,
            joint_positions=observation["robot0_joint_pos"],
            gripper_joint_positions=observation.get("robot0_gripper_qpos"),
            robot_base_pose=robot_base_pose,
            eef_pos=observation["robot0_eef_pos"],
            item=item,
        )
        if held is None:
            from cosmos_policy.experiments.robot.robot_utils import log_message
            log_message(f"[K6] held-object extraction failed for {item}", log_file)
            return None

        candidates = memory.select(
            task_name,
            next_phase.planner_step_id,
            next_phase.skill,
            next_phase.arguments,
        )
        exact = [c for c in candidates if str(c.get("demo_id")) == str(demo_id)]
        if not exact:
            from cosmos_policy.experiments.robot.robot_utils import log_message
            log_message(f"[K6] no exact ready pose for demo={demo_id}", log_file)
            return None

        ready_pose = np.asarray(exact[0]["ee_states"], dtype=np.float64).reshape(6)
        current_ee = np.concatenate(
            [
                observation["robot0_eef_pos"],
                Rotation.from_quat(observation["robot0_eef_quat"]).as_rotvec(),
            ]
        ).astype(np.float32)

        inp = HeldObjectPlannerInput(
            joint_positions=observation["robot0_joint_pos"],
            ee_states=current_ee,
            depth=depth,
            camera_params=camera,
            held_object=held,
            ready_pose=ready_pose,
            gripper_joint_positions=observation.get("robot0_gripper_qpos"),
            robot_base_pose=robot_base_pose,
        )
        result = planner.plan(inp)
        from cosmos_policy.experiments.robot.robot_utils import log_message
        if result is None:
            log_message("[K6] HeldObjectPlanner failed", log_file)
        else:
            log_message(
                f"[K6] HeldObjectPlanner ok waypoints={len(result.waypoints)}",
                log_file,
            )
        return result

    return hook


def main() -> None:
    args = _parse_args()
    os.environ["COSMOS_SKILL_COMPLETION_ACTIVE"] = "1"
    os.environ.setdefault("COSMOS_SKILL_COMPLETION_ITEM", ITEM)

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        _create_initial_alignment_selector,
        run_task,
        validate_config,
    )
    from cosmos_policy.experiments.robot.robot_utils import (
        get_image_resize_size,
        setup_logging,
    )
    from cosmos_policy.utils.utils import set_seed_everywhere

    cfg = _build_cfg(args)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)
    planning_model = None
    resize_size = get_image_resize_size(cfg.model_family)

    log_file, _, _ = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_suite_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task_id = next(
        i for i in range(task_suite.n_tasks) if task_suite.get_task(i).name == TASK
    )
    initial_alignment_selector = _create_initial_alignment_selector(cfg)

    total_episodes, total_successes = run_task(
        cfg,
        task_suite,
        task_id,
        model,
        planning_model,
        dataset_stats,
        None,
        resize_size,
        total_episodes=0,
        total_successes=0,
        log_file=log_file,
        initial_alignment_selector=initial_alignment_selector,
        phase_transition_hook=_make_phase_transition_hook(),
    )
    print("K6_RESULT", f"episodes={total_episodes}", f"successes={total_successes}")
    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
