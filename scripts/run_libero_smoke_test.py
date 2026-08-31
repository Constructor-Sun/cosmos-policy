"""Run Cosmos Policy LIBERO smoke and paired evaluations."""

import builtins
import gc
import json
import logging
import os
import pathlib
import pickle
import re
import time
import types

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Task


mode = os.environ["COSMOS_SMOKE_MODE"]
flip_images = os.environ["COSMOS_SMOKE_FLIP_IMAGES"].lower() in {"1", "true", "yes"}
policy_dir = os.environ["COSMOS_POLICY_MODEL_DIR"]
policy_ckpt_path = os.environ.get(
    "COSMOS_POLICY_CKPT_PATH",
    os.path.join(policy_dir, "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
)

benchmark_dict = benchmark.get_benchmark_dict()

# config_v2 imports every experiment module. The unrelated ALOHA module resolves
# its checkpoint eagerly, so give that import-only URI an existing local path.
from cosmos_policy._src.imaginaire.utils import checkpoint_db


original_get_checkpoint_path = checkpoint_db.get_checkpoint_path
aloha_checkpoint_uri = (
    "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/"
    "Cosmos-Policy-ALOHA-Predict2-2B.pt"
)


def get_checkpoint_path_offline(checkpoint_uri):
    if checkpoint_uri.rstrip("/") == aloha_checkpoint_uri:
        return os.path.join(policy_dir, "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    return original_get_checkpoint_path(checkpoint_uri)


checkpoint_db.get_checkpoint_path = get_checkpoint_path_offline

from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    PolicyEvalConfig,
    eval_libero,
)
import cosmos_policy.experiments.robot.libero.run_libero_eval as run_libero_eval_mod
from cosmos_policy.experiments.robot import cosmos_utils


prompt_override = os.environ.get("COSMOS_SMOKE_PROMPT_OVERRIDE", "").strip()
if prompt_override:
    original_get_libero_env = run_libero_eval_mod.get_libero_env

    def get_libero_env_with_prompt(*args, **kwargs):
        env, _task_description = original_get_libero_env(*args, **kwargs)
        return env, prompt_override

    run_libero_eval_mod.get_libero_env = get_libero_env_with_prompt


quiet_step_logs = os.environ.get("COSMOS_SMOKE_QUIET_STEP_LOGS", "true").lower() in {
    "1",
    "true",
    "yes",
}


def is_step_log(message):
    return message.startswith("Query ") or (
        message.startswith("t=")
        and ("Current base seed" in message or "Selected seed" in message)
    )


if quiet_step_logs:
    logging.getLogger("cosmos_policy.experiments.robot.robot_utils").setLevel(logging.WARNING)
    logging.getLogger("cosmos_policy.experiments.robot.libero.run_libero_eval").setLevel(logging.WARNING)
    try:
        from cosmos_policy._src.imaginaire.utils import log as imaginaire_log

        imaginaire_log.LEVEL = "WARNING"
        imaginaire_log.init_loguru_stdout()
    except Exception:
        pass

    def log_message_quiet(message, log_file=None, console=True):
        if console and not is_step_log(message):
            print(message)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    run_libero_eval_mod.log_message = log_message_quiet


latent_adapter_path = os.environ.get("COSMOS_SMOKE_LATENT_ADAPTER", "").strip()
if latent_adapter_path:
    from counterfactual_experiments.direct_sft.adapter import load_model as load_latent_adapter

    original_get_model = cosmos_utils.get_model

    def get_model_with_latent_adapter(cfg):
        model, config = original_get_model(cfg)
        adapter = load_latent_adapter(latent_adapter_path, device=next(model.parameters()).device)
        adapter.eval()
        adapter.requires_grad_(False)
        original_get_data_and_condition = model.get_data_and_condition
        adapter_diagnostic_printed = False

        def corrected_get_data_and_condition(self, data_batch):
            nonlocal adapter_diagnostic_printed
            raw_state, latent_state, condition = original_get_data_and_condition(data_batch)
            batch = torch.arange(latent_state.shape[0], device=latent_state.device)
            wrist_idx = data_batch["current_wrist_image_latent_idx"].to(latent_state.device)
            image_idx = data_batch["current_image_latent_idx"].to(latent_state.device)
            video = torch.stack(
                (latent_state[batch, :, wrist_idx], latent_state[batch, :, image_idx]), dim=2
            ).float()
            corrected = adapter(video)
            if not adapter_diagnostic_printed:
                mean_abs_shift = (corrected.float() - video).abs().mean().item()
                print(f"Latent adapter mean absolute shift: {mean_abs_shift:.8f}")
                adapter_diagnostic_printed = True
            corrected_latent_state = latent_state.clone()
            corrected_latent_state[batch, :, wrist_idx] = corrected[:, :, 0].to(latent_state.dtype)
            corrected_latent_state[batch, :, image_idx] = corrected[:, :, 1].to(latent_state.dtype)
            condition.gt_frames[batch, :, wrist_idx] = corrected[:, :, 0].to(condition.gt_frames.dtype)
            condition.gt_frames[batch, :, image_idx] = corrected[:, :, 1].to(condition.gt_frames.dtype)
            return raw_state, corrected_latent_state, condition

        model.get_data_and_condition = types.MethodType(corrected_get_data_and_condition, model)
        model.latent_adapter = adapter
        print(f"Loaded video-latent adapter: {latent_adapter_path}")
        return model, config

    cosmos_utils.get_model = get_model_with_latent_adapter
    run_libero_eval_mod.get_model = get_model_with_latent_adapter
    eval_libero.__wrapped__.__globals__["get_model"] = get_model_with_latent_adapter


original_libero_eval_print = getattr(run_libero_eval_mod, "print", builtins.print)


def print_without_action_spam(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("t: ") and "\t action:" in args[0]:
        return
    return original_libero_eval_print(*args, **kwargs)


run_libero_eval_mod.print = print_without_action_spam


original_init_t5_text_embeddings_cache = cosmos_utils.init_t5_text_embeddings_cache
original_get_t5_embedding = cosmos_utils.get_t5_embedding_from_cache


def init_t5_text_embeddings_cache_with_extra(t5_text_embeddings_path, *args, **kwargs):
    result = original_init_t5_text_embeddings_cache(t5_text_embeddings_path, *args, **kwargs)
    extra_path = os.environ.get("COSMOS_SMOKE_T5_EXTRA_EMBEDDINGS", "").strip()
    if extra_path:
        with open(extra_path, "rb") as file:
            extra_embeddings = pickle.load(file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key, value in extra_embeddings.items():
            if isinstance(value, torch.Tensor):
                extra_embeddings[key] = value.to(device)
        cosmos_utils.t5_text_embeddings_cache.update(extra_embeddings)
        print(f"Loaded {len(extra_embeddings)} extra T5 embeddings from {extra_path} onto device {device}")
    return result


eval_libero.__wrapped__.__globals__["init_t5_text_embeddings_cache"] = init_t5_text_embeddings_cache_with_extra


def strip_libero_plus_metadata(task_label):
    cache_label = task_label

    # Camera / init-state / sensor-noise variants.
    cache_label = re.sub(r" view .+ initstate \d+(?: noise \d+)?$", "", cache_label)

    # Other LIBERO-plus environment variants.
    for pattern in (
        r" table \d+$",
        r" tb \d+$",
        r" light \d+$",
        r" add \d+$",
        r" level\d+ sample\d+$",
        r" noise \d+$",
    ):
        cache_label = re.sub(pattern, "", cache_label)

    return cache_label


def get_t5_embedding_for_libero_plus(task_label):
    if prompt_override and task_label == prompt_override:
        return original_get_t5_embedding(task_label)
    instruction_mode = os.environ.get("COSMOS_SMOKE_INSTRUCTION_MODE", "task")
    if instruction_mode == "base":
        cache_label = strip_libero_plus_metadata(task_label)
        if (
            cache_label not in cosmos_utils.t5_text_embeddings_cache
            and os.environ["COSMOS_SMOKE_T5_FALLBACK_TO_BASE"].lower() in {"1", "true", "yes"}
        ):
            cache_label = os.environ["COSMOS_SMOKE_PAIR_CLEAN_LANGUAGE"]
        return original_get_t5_embedding(cache_label)

    if instruction_mode == "strict":
        if task_label not in cosmos_utils.t5_text_embeddings_cache:
            if os.environ["COSMOS_SMOKE_T5_ALLOW_STRICT_COMPUTE"].lower() not in {"1", "true", "yes"}:
                raise RuntimeError(
                    "Strict language perturbation T5 embedding is missing and exact on-demand computation is disabled. "
                    "Provide SMOKE_T5_EXTRA_EMBEDDINGS or set SMOKE_T5_ALLOW_STRICT_COMPUTE=true."
                )
            print(
                "Strict language perturbation T5 embedding is missing from cache; "
                f"computing exact instruction without base fallback: {task_label!r}"
            )
        return original_get_t5_embedding(task_label)

    cache_label = task_label
    return original_get_t5_embedding(cache_label)


cosmos_utils.get_t5_embedding_from_cache = get_t5_embedding_for_libero_plus


# --- Held-object Pick->Place pipeline ---
if os.environ.get("COSMOS_HELD_OBJECT", "").lower() in {"1", "true", "yes"}:
    import numpy as np
    from scipy.spatial.transform import Rotation

    from memory_system.artifacts import FeasibleRecoveryMemory
    from memory_system.geometry import camera_params as build_camera_params

    _repo_root = pathlib.Path(__file__).resolve().parents[1]
    _feasible_path = _repo_root / "skill_memory_test" / "libero_10" / "feasible_recovery_targets.pt"
    _held_memory = FeasibleRecoveryMemory(str(_feasible_path))
    _place_mode = os.environ.get("COSMOS_PLACE_MODE", "curobo").lower()

    if _place_mode == "simple":
        from memory_system.execute.planner.place_fine_aligner import PlaceFineAligner

        def _held_object_hook(
            *,
            observation,
            completed_phase,
            next_phase,
            task_name,
            demo_id,
            episode_id,
            env,
            cfg,
            log_file,
        ):
            del observation, completed_phase, env, cfg
            run_libero_eval_mod._PLACE_ALIGNER = None
            candidates = _held_memory.select(
                task_name,
                next_phase.planner_step_id,
                next_phase.skill,
                next_phase.arguments,
            )
            exact = [c for c in candidates if str(c.get("demo_id")) == str(demo_id)]
            if not exact:
                msg = f"[PLACE_ALIGN] no exact ready pose for demo={demo_id}"
                print(msg)
                run_libero_eval_mod.log_message(msg, log_file)
                return None
            ready_pose = np.asarray(exact[0]["ee_states"], dtype=np.float64).reshape(6)
            aligner = PlaceFineAligner(ready_pose)
            run_libero_eval_mod._PLACE_ALIGNER = aligner
            msg = (
                f"[PLACE_ALIGN] simple mode armed for demo={demo_id} "
                f"ready_pose={ready_pose.tolist()}"
            )
            print(msg)
            run_libero_eval_mod.log_message(msg, log_file)
            return None

        run_libero_eval_mod._PHASE_TRANSITION_HOOK = _held_object_hook
        print("[PLACE_ALIGN] simple Place fine-align mode enabled")
    else:
        from memory_system.execute.planner.held_object.connected_component import (
            HeldObjectConnectedComponentExtractor,
        )
        from memory_system.execute.planner.held_object.planner import HeldObjectPlanner
        from memory_system.execute.planner.held_object.types import HeldObjectPlannerInput

        _held_extractor = HeldObjectConnectedComponentExtractor()
        # Test-only: use GPU Warp URDF filter from the diagnostic script so the
        # held-object hook does not stall on CPU ray casting.
        import importlib.util
        import sys

        import curobo
        from memory_system.execute.urdf_depth_filter import UrdfDepthFilterConfig

        _base_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "tests" / "held_object" / "visualize_held_object_after_robot_removal.py"
        )
        _spec = importlib.util.spec_from_file_location("held_object_gpu_urdf_base", _base_path)
        _base = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _base
        _spec.loader.exec_module(_base)
        _urdf_path = str(
            pathlib.Path(curobo.__file__).resolve().parent
            / "content" / "assets" / "robot" / "franka_description" / "franka_panda.urdf"
        )
        _held_extractor.planner._urdf_filter = _base.WarpUrdfDepthFilter(
            UrdfDepthFilterConfig(urdf_path=_urdf_path), device="cuda:0"
        )
        _held_planner = HeldObjectPlanner()
        # Use the same GPU Warp URDF filter in HeldObjectPlanner so planning does not
        # fall back to slow CPU ray casting.
        _held_planner._urdf_filter = _held_extractor.planner._urdf_filter
        _failure_dump_dir = _repo_root / "tests" / "held_object" / "k6_pick_place_integration"
        _debug_enabled = os.environ.get("COSMOS_HELD_OBJECT_DEBUG", "1").lower() in {"1", "true", "yes"}

        def _dump_held_object_image(observation, env, cfg, debug, tag, task_name, episode_id):
            from memory_system.geometry import world_to_pixel

            _failure_dump_dir.mkdir(parents=True, exist_ok=True)
            height, width = observation["agentview_image"].shape[:2]
            camera = build_camera_params(env.sim, "agentview", height, width)

            r_world_base = np.asarray(debug["r_world_base"], dtype=np.float64).reshape(3, 3)
            t_world_base = np.asarray(debug["t_world_base"], dtype=np.float64).reshape(3)
            p0 = np.asarray(debug["p0"], dtype=np.float64).reshape(3)
            r0 = np.asarray(debug["r0"], dtype=np.float64).reshape(3, 3)
            points_hand = np.asarray(debug["points_hand"], dtype=np.float64).reshape(-1, 3)
            surface_base = np.asarray(debug["surface_points"], dtype=np.float64).reshape(-1, 3)

            held_world = (
                r_world_base @ (r0 @ points_hand.T + p0[:, None]) + t_world_base[:, None]
            ).T
            surface_world = (
                r_world_base @ surface_base.T + t_world_base[:, None]
            ).T

            robot_residual = np.zeros(len(surface_world), dtype=bool)
            if len(surface_world):
                px = world_to_pixel(surface_world, camera).astype(np.int64)
                valid = (
                    (px[:, 0] >= 0) & (px[:, 0] < height)
                    & (px[:, 1] >= 0) & (px[:, 1] < width)
                )
                if valid.any():
                    try:
                        seg_render, _ = env.sim.render(
                            height, width, camera_name="agentview", depth=True, segmentation=True
                        )
                        seg_canon = np.flipud(seg_render)
                        for i in np.flatnonzero(valid):
                            objtype, objid = seg_canon[px[i, 0], px[i, 1]]
                            objtype = int(objtype)
                            objid = int(objid)
                            if objtype == 5:
                                name = env.sim.model.geom_id2name(objid)
                                if name and (
                                    "robot" in name.lower()
                                    or "gripper" in name.lower()
                                    or name.lower().startswith("panda")
                                    or "link" in name.lower()
                                ):
                                    robot_residual[i] = True
                    except Exception:
                        pass

            other_world = surface_world[~robot_residual]
            robot_residual_world = surface_world[robot_residual]
            target = np.asarray(debug["target"], dtype=np.float64).reshape(6)

            conflict_point_world = None
            conflict_sphere_world = None
            last_conflict_info = debug.get("last_conflict_info")
            if last_conflict_info is not None:
                conflict_point_world = (
                    r_world_base @ np.asarray(last_conflict_info["point"], dtype=np.float64).reshape(3)
                    + t_world_base
                )
                conflict_sphere_world = (
                    r_world_base @ np.asarray(last_conflict_info["sphere_center"], dtype=np.float64).reshape(3)
                    + t_world_base
                )

            nearest_target_point_world = None
            nearest_target_info = debug.get("nearest_obstacle_to_target")
            if nearest_target_info is not None:
                nearest_target_point_world = (
                    r_world_base @ np.asarray(nearest_target_info["point"], dtype=np.float64).reshape(3)
                    + t_world_base
                )

            # Crop to the region around the EEF / main camera, matching the
            # existing held-object diagnostic visualizations.
            eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).reshape(3)
            crop_lo = eef + np.array([-0.55, -0.55, -0.40], dtype=np.float64)
            crop_hi = eef + np.array([0.55, 0.55, 0.35], dtype=np.float64)

            def _crop(points):
                if len(points) == 0:
                    return points
                return points[np.all((points >= crop_lo) & (points <= crop_hi), axis=1)]

            held_world = _crop(held_world)
            robot_residual_world = _crop(robot_residual_world)
            other_world = _crop(other_world)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            views = [
                ("front-left", 25, -60),
                ("front-right", 25, 30),
                ("top-ish", 60, -60),
                ("side", 10, -120),
            ]
            colors = {
                "robot_residual": "#d62728",
                "held_object": "#ff7f0e",
                "other": "#7f7f7f",
                "target": "#2ca02c",
            }
            rng = np.random.default_rng(0)

            def draw(ax, show_legend=False, title=None):
                for points, color, label, size in [
                    (robot_residual_world, colors["robot_residual"], f"robot_residual ({len(robot_residual_world)})", 0.8),
                    (held_world, colors["held_object"], f"held_object ({len(held_world)})", 0.8),
                    (other_world, colors["other"], f"other ({len(other_world)})", 0.6),
                ]:
                    if len(points) == 0:
                        continue
                    idx = np.arange(len(points))
                    if len(idx) > 20000:
                        idx = rng.choice(idx, 20000, replace=False)
                    pts = points[idx]
                    ax.scatter(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        s=size, c=color, alpha=0.8, label=label,
                    )
                ax.scatter(
                    [target[0]], [target[1]], [target[2]],
                    s=120, c=colors["target"], marker="*", label="ready_pose",
                )
                if conflict_point_world is not None:
                    ax.scatter(
                        [conflict_point_world[0]], [conflict_point_world[1]], [conflict_point_world[2]],
                        s=180, c="black", marker="X", label="closest_obstacle_point",
                    )
                    ax.scatter(
                        [conflict_sphere_world[0]], [conflict_sphere_world[1]], [conflict_sphere_world[2]],
                        s=140, c="blue", marker="o", label="robot_sphere_center",
                    )
                    ax.plot(
                        [conflict_point_world[0], conflict_sphere_world[0]],
                        [conflict_point_world[1], conflict_sphere_world[1]],
                        [conflict_point_world[2], conflict_sphere_world[2]],
                        c="black", linestyle="--", linewidth=1.0,
                    )
                if nearest_target_point_world is not None:
                    ax.scatter(
                        [nearest_target_point_world[0]], [nearest_target_point_world[1]], [nearest_target_point_world[2]],
                        s=180, c="purple", marker="^", label="nearest_obstacle_to_target",
                    )
                if title:
                    ax.set_title(title, fontsize=10)
                if show_legend:
                    ax.legend(loc="upper right", fontsize=8)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_zlabel("z (m)")

            fig = plt.figure(figsize=(20, 16))
            for idx, (name, elev, azim) in enumerate(views, start=1):
                ax = fig.add_subplot(2, 2, idx, projection="3d")
                draw(
                    ax,
                    show_legend=(idx == 1),
                    title=f"HeldObjectPlanner {tag} - {name}",
                )
                ax.view_init(elev=elev, azim=azim)
            fig.suptitle(f"HeldObjectPlanner {tag} point cloud", fontsize=14)
            safe_task = str(task_name).replace("/", "_").replace(" ", "_")
            out_png = _failure_dump_dir / f"i_planner_{tag}_{safe_task}_ep{episode_id}_multiview.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[HELD_OBJECT] saved {tag} point cloud image to {out_png}")

        def _held_object_hook(
            *,
            observation,
            completed_phase,
            next_phase,
            task_name,
            demo_id,
            episode_id,
            env,
            cfg,
            log_file,
        ):
            item = completed_phase.arguments.get("item") or "white_yellow_mug_1"
            height, width = observation["agentview_image"].shape[:2]
            camera = build_camera_params(env.sim, "agentview", height, width)
            depth = run_libero_eval_mod._make_main_depth(observation, camera, cfg.flip_images)
            robot_base_pose = np.concatenate(
                [env.robots[0].base_pos, env.robots[0].base_ori]
            ).astype(np.float64)

            held = _held_extractor.extract(
                depth=depth,
                camera_params=camera,
                joint_positions=observation["robot0_joint_pos"],
                gripper_joint_positions=observation.get("robot0_gripper_qpos"),
                robot_base_pose=robot_base_pose,
                eef_pos=observation["robot0_eef_pos"],
                item=item,
            )
            if held is None:
                msg = f"[HELD_OBJECT] extract failed for {item}"
                print(msg)
                run_libero_eval_mod.log_message(msg, log_file)
                return None

            candidates = _held_memory.select(
                task_name,
                next_phase.planner_step_id,
                next_phase.skill,
                next_phase.arguments,
            )
            exact = [c for c in candidates if str(c.get("demo_id")) == str(demo_id)]
            if not exact:
                msg = f"[HELD_OBJECT] no exact ready pose for demo={demo_id}"
                print(msg)
                run_libero_eval_mod.log_message(msg, log_file)
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
            result = _held_planner.plan(inp)
            if result is None:
                msg = "[HELD_OBJECT] HeldObjectPlanner failed"
                if getattr(_held_planner, "last_failure", None):
                    msg += f": {_held_planner.last_failure}"
                print(msg)
                run_libero_eval_mod.log_message(msg, log_file)
                if _debug_enabled and getattr(_held_planner, "last_debug", None):
                    try:
                        _dump_held_object_image(observation, env, cfg, _held_planner.last_debug, "failure", task_name, episode_id)
                    except Exception as exc:
                        print(f"[HELD_OBJECT] failed to dump failure point cloud: {exc}")
            else:
                msg = f"[HELD_OBJECT] plan ok waypoints={len(result.waypoints)}"
                print(msg)
                run_libero_eval_mod.log_message(msg, log_file)
                if _debug_enabled and getattr(_held_planner, "last_debug", None):
                    try:
                        _dump_held_object_image(observation, env, cfg, _held_planner.last_debug, "success", task_name, episode_id)
                    except Exception as exc:
                        print(f"[HELD_OBJECT] failed to dump success point cloud: {exc}")
            return result

        run_libero_eval_mod._PHASE_TRANSITION_HOOK = _held_object_hook
        print("[HELD_OBJECT] held-object Pick->Place pipeline enabled")


def make_cfg(suite_name, num_trials, run_id_note, local_log_dir="./experiments/logs"):
    unnorm_key = "libero_10" if suite_name == "libero_mix" else suite_name
    deterministic_reset = os.environ["COSMOS_SMOKE_DETERMINISTIC_RESET"].lower() in {"1", "true", "yes"}
    data_collection = os.environ.get("COSMOS_DATA_COLLECTION", "").lower() in {"1", "true", "yes"}
    save_vector_db = os.environ.get("COSMOS_VECTOR_DB", "").lower() in {"1", "true", "yes"}
    vector_db_output_dir = os.environ.get("COSMOS_VECTOR_DB_DIR", "")
    enable_initial_alignment = os.environ.get("COSMOS_INITIAL_ALIGNMENT", "").lower() in {"1", "true", "yes"}
    enable_curobo_joint_execution = os.environ.get(
        "COSMOS_CUROBO_JOINT_EXECUTION", ""
    ).lower() in {"1", "true", "yes"}
    enable_urdf_robot_filter = os.environ.get(
        "COSMOS_URDF_ROBOT_FILTER", ""
    ).lower() in {"1", "true", "yes"}
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=policy_ckpt_path,
        dataset_stats_path=os.path.join(policy_dir, "libero_dataset_statistics.json"),
        t5_text_embeddings_path=os.path.join(policy_dir, "libero_t5_embeddings.pkl"),
        task_suite_name=suite_name,
        unnorm_key=unnorm_key,
        num_trials_per_task=num_trials,
        flip_images=flip_images,
        local_log_dir=local_log_dir,
        run_id_note=run_id_note,
        seed=int(os.environ["COSMOS_SMOKE_SEED"]),
        randomize_seed=False,
        deterministic=True,
        deterministic_reset=deterministic_reset,
        deterministic_reset_seed=int(os.environ["COSMOS_SMOKE_DETERMINISTIC_RESET_SEED"]),
        data_collection=data_collection,
        save_vector_db=save_vector_db,
        vector_db_output_dir=vector_db_output_dir,
        enable_initial_alignment=enable_initial_alignment,
        enable_curobo_joint_execution=enable_curobo_joint_execution,
        enable_urdf_robot_filter=enable_urdf_robot_filter,
    )


def run_eval_with_task_filter(
    *,
    suite_name,
    task_name=None,
    task_id=None,
    category_value=None,
    synthetic_task=None,
    synthetic_init_states_path=None,
    num_trials=1,
    run_id_note,
    local_log_dir,
):
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite: {suite_name}")
    suite_class = benchmark_dict[suite_name]
    original_init = suite_class.__init__

    def init_one_task(self, *args, **kwargs):
        if category_value is not None:
            kwargs["category_value"] = category_value
        original_init(self, *args, **kwargs)

        if synthetic_task is not None:
            self.tasks = [synthetic_task]
            self.n_tasks = 1

            if synthetic_init_states_path is not None:
                def get_synthetic_init_states(_task_id):
                    return torch.load(synthetic_init_states_path, weights_only=False)

                self.get_task_init_states = get_synthetic_init_states
            return

        original_get_task_init_states = self.get_task_init_states
        if task_name is not None:
            matches = [(idx, task) for idx, task in enumerate(self.tasks) if task.name == task_name]
            if not matches:
                examples = "\n".join(f"  [{idx}] {task.name}" for idx, task in enumerate(self.tasks[:20]))
                raise RuntimeError(
                    f"Task not found in suite={suite_name}, category={category_value!r}: {task_name}\n"
                    f"First available tasks:\n{examples}"
                )
            selected_id, selected_task = matches[0]
        else:
            selected_id = int(task_id)
            if not 0 <= selected_id < self.n_tasks:
                raise IndexError(
                    f"Task ID {selected_id} is outside {suite_name}'s range [0, {self.n_tasks - 1}]"
                )
            selected_task = self.tasks[selected_id]

        selected_init_states = original_get_task_init_states(selected_id)
        self.tasks = [selected_task]
        self.n_tasks = 1

        def get_repeated_init_states(_task_id):
            states = selected_init_states
            _offset = int(os.environ.get("COSMOS_INIT_STATE_OFFSET", "0"))
            print(f"[init_state] offset={_offset} num_trials={num_trials} len(states)={len(states)} "
                  f"→ using indices [{_offset}:{_offset + num_trials}]", flush=True)
            if len(states) >= num_trials + _offset:
                return states[_offset:_offset + num_trials]
            if isinstance(states, torch.Tensor):
                return torch.stack([states[(_offset + idx) % len(states)] for idx in range(num_trials)], dim=0)
            return [states[(_offset + idx) % len(states)] for idx in range(num_trials)]

        self.get_task_init_states = get_repeated_init_states

    suite_class.__init__ = init_one_task
    try:
        cfg = make_cfg(
            suite_name,
            num_trials=num_trials,
            run_id_note=run_id_note,
            local_log_dir=local_log_dir,
        )
        return eval_libero.__wrapped__(cfg)
    finally:
        suite_class.__init__ = original_init
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_episode_successes(log_dir, run_id_note):
    log_paths = sorted(
        pathlib.Path(log_dir).glob(f"*--{run_id_note}.txt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not log_paths:
        return [], None

    log_path = log_paths[-1]
    episodes = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Success: "):
            episodes.append(
                {
                    "episode": len(episodes),
                    "success": line.split("Success: ", 1)[1].strip() == "True",
                }
            )
    return episodes, str(log_path)


def get_matching_variants(suite_name, category_value, base_task):
    task_suite = benchmark_dict[suite_name](category_value=category_value)
    variants = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if task.name.startswith(base_task):
            variants.append((task_id, task))
    return sorted(variants, key=lambda item: item[1].name)


def get_task_by_name(suite_name, category_value, task_name):
    task_suite = benchmark_dict[suite_name](category_value=category_value)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if task.name == task_name:
            return task_id, task
    raise RuntimeError(f"Task not found in suite={suite_name}, category={category_value}: {task_name}")


def load_available_t5_keys(policy_dir):
    keys = set()
    with open(os.path.join(policy_dir, "libero_t5_embeddings.pkl"), "rb") as file:
        keys.update(pickle.load(file).keys())

    extra_path = os.environ.get("COSMOS_SMOKE_T5_EXTRA_EMBEDDINGS", "").strip()
    if extra_path:
        with open(extra_path, "rb") as file:
            keys.update(pickle.load(file).keys())
    return keys


def annotate_episodes(
    episodes,
    *,
    run_id,
    seed,
    deterministic_reset,
    deterministic_reset_seed,
    suite,
    base_task,
    condition,
    category,
    task_id,
    task_name,
    language,
    log_path,
    instruction_mode,
):
    annotated = []
    for episode in episodes:
        episode_idx = int(episode["episode"])
        annotated_episode = {
            "run_id": run_id,
            "suite": suite,
            "seed": seed,
            "deterministic_reset": deterministic_reset,
            "deterministic_reset_seed": deterministic_reset_seed,
            "base_task": base_task,
            "condition": condition,
            "category": category,
            "task_id": task_id,
            "task_name": task_name,
            "language": language,
            "episode": episode_idx,
            "init_state_index": episode.get("init_state_index", episode_idx),
            "success": bool(episode["success"]),
            "log_path": log_path,
            "instruction_mode": instruction_mode,
        }
        for key, value in episode.items():
            if key not in annotated_episode:
                annotated_episode[key] = value
        annotated.append(annotated_episode)
    return annotated


if mode == "paired":
    pair_suite = os.environ["COSMOS_SMOKE_PAIR_SUITE"]
    base_task = os.environ["COSMOS_SMOKE_PAIR_BASE_TASK"]
    clean_language = os.environ["COSMOS_SMOKE_PAIR_CLEAN_LANGUAGE"]
    pert_name = os.environ["COSMOS_SMOKE_PAIR_PERT_NAME"]
    pert_category = os.environ["COSMOS_SMOKE_PAIR_PERT_CATEGORY"]
    pert_task_name = os.environ["COSMOS_SMOKE_PAIR_PERT_TASK"]
    num_pairs = int(os.environ["COSMOS_SMOKE_NUM_PAIRS"])
    seed = int(os.environ["COSMOS_SMOKE_SEED"])
    deterministic_reset = os.environ["COSMOS_SMOKE_DETERMINISTIC_RESET"].lower() in {"1", "true", "yes"}
    deterministic_reset_seed = int(os.environ["COSMOS_SMOKE_DETERMINISTIC_RESET_SEED"])
    run_id = os.environ.get("COSMOS_SMOKE_RUN_ID", "").strip() or time.strftime("%Y%m%d_%H%M%S")
    only_condition = os.environ.get("COSMOS_SMOKE_ONLY_CONDITION", "both").strip().lower()
    if only_condition not in {"both", "clean", "perturb", "background"}:
        raise ValueError(
            "SMOKE_ONLY_CONDITION must be one of: both, clean, perturb, background"
        )
    results_dir = pathlib.Path(os.environ["COSMOS_SMOKE_RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    clean_task = Task(
        name=base_task,
        language=clean_language,
        problem="Libero",
        problem_folder=pair_suite,
        bddl_file=f"{base_task}.bddl",
        init_states_file=f"{base_task}.pruned_init",
    )
    clean_init_states_path = (
        pathlib.Path(get_libero_path("init_states"))
        / pair_suite
        / f"{base_task}.pruned_init"
    )

    default_perturbations = [
        {
            "name": "camera_viewpoints",
            "category": "Camera Viewpoints",
            "task_name": f"{base_task}_view_50_0_100_0_0_initstate_0",
            "instruction_mode": "base",
        },
        {
            "name": "background_textures",
            "category": "Background Textures",
            "task_name": f"{base_task}_table_5",
            "instruction_mode": "base",
        },
        {
            "name": "light_conditions",
            "category": "Light Conditions",
            "task_name": f"{base_task}_light_5",
            "instruction_mode": "base",
        },
        {
            "name": "objects_layout",
            "category": "Objects Layout",
            "task_name": f"{base_task}_add_*",
            "instruction_mode": "base",
            "variant_mode": "all_variants",
        },
        {
            "name": "robot_initial_states",
            "category": "Robot Initial States",
            "task_name": f"{base_task}_view_0_0_100_0_0_initstate_274",
            "instruction_mode": "base",
        },
        {
            "name": "sensor_noise",
            "category": "Sensor Noise",
            "task_name": f"{base_task}_view_0_0_100_0_0_initstate_0_noise_5",
            "instruction_mode": "base",
        },
        {
            "name": "language_instructions",
            "category": "Language Instructions",
            "task_name": f"{base_task}_language_5_view_0_0_100_0_0_initstate_0",
            "instruction_mode": "strict",
        },
    ]
    by_name = {item["name"]: item for item in default_perturbations}
    if pert_name == "all":
        perturbations = default_perturbations
    elif "," in pert_name:
        perturbations = []
        for name in [item.strip() for item in pert_name.split(",") if item.strip()]:
            if name not in by_name:
                raise ValueError(f"Unknown perturbation name: {name}. Available: {sorted(by_name)}")
            perturbations.append(by_name[name])
    else:
        if pert_name in by_name and pert_task_name == os.environ["COSMOS_SMOKE_PAIR_BASE_TASK"] + "_view_50_0_100_0_0_initstate_0":
            perturbations = [by_name[pert_name]]
        else:
            perturbations = [
                {
                    "name": pert_name,
                    "category": pert_category,
                    "task_name": pert_task_name,
                    "instruction_mode": "strict" if pert_name == "language_instructions" else "base",
                }
            ]

    if only_condition == "clean":
        perturbations = []

    print("Running paired smoke test:")
    print(f"  suite: {pair_suite}")
    print(f"  num_pairs: {num_pairs}")
    print(f"  clean task: {base_task}")
    print("  perturbations:")
    for item in perturbations:
        print(
            f"    - {item['name']} ({item['category']}): {item['task_name']} "
            f"[instruction={item.get('instruction_mode', 'task')}]"
        )
    if os.environ["COSMOS_SMOKE_T5_FALLBACK_TO_BASE"].lower() in {"1", "true", "yes"}:
        print("  T5 fallback: non-language perturbation instructions may use the base instruction embedding")
    if os.environ.get("COSMOS_SMOKE_T5_EXTRA_EMBEDDINGS", "").strip():
        print(f"  extra T5 embeddings: {os.environ['COSMOS_SMOKE_T5_EXTRA_EMBEDDINGS']}")

    strict_missing = []
    available_t5_keys = load_available_t5_keys(policy_dir)
    for item in perturbations:
        if item.get("instruction_mode") != "strict":
            continue
        _, strict_task = get_task_by_name(pair_suite, item["category"], item["task_name"])
        item["language"] = strict_task.language
        if strict_task.language not in available_t5_keys:
            strict_missing.append((item["name"], item["task_name"], strict_task.language))

    if strict_missing:
        print("  strict language T5 embeddings missing:")
        for condition_name, task_name, language in strict_missing:
            print(f"    - {condition_name}: {task_name}")
            print(f"      language: {language}")
        if os.environ["COSMOS_SMOKE_T5_ALLOW_STRICT_COMPUTE"].lower() not in {"1", "true", "yes"}:
            raise RuntimeError(
                "Missing strict language perturbation T5 embeddings. "
                "Create an extra embeddings pkl and pass SMOKE_T5_EXTRA_EMBEDDINGS=/path/to/pkl, "
                "or set SMOKE_T5_ALLOW_STRICT_COMPUTE=true with a locally available T5-11B cache."
            )
        print("  strict language T5: missing embeddings will be computed exactly; no base fallback is allowed")

    clean_note = f"paired-clean-{num_pairs}pair"

    summary = {
        "mode": "paired",
        "run_id": run_id,
        "suite": pair_suite,
        "seed": seed,
        "deterministic_reset": deterministic_reset,
        "deterministic_reset_seed": deterministic_reset_seed,
        "num_pairs": num_pairs,
        "base_task": base_task,
        "conditions": [],
    }

    if only_condition not in {"perturb", "background"}:
        os.environ["COSMOS_SMOKE_INSTRUCTION_MODE"] = "task"
        clean_rate = run_eval_with_task_filter(
            suite_name=pair_suite,
            synthetic_task=clean_task,
            synthetic_init_states_path=str(clean_init_states_path),
            num_trials=num_pairs,
            run_id_note=clean_note,
            local_log_dir=str(log_dir),
        )
        clean_episodes, clean_log = parse_episode_successes(log_dir, clean_note)
        clean_episodes = annotate_episodes(
            clean_episodes,
            run_id=run_id,
            seed=seed,
            deterministic_reset=deterministic_reset,
            deterministic_reset_seed=deterministic_reset_seed,
            suite=pair_suite,
            base_task=base_task,
            condition="clean",
            category="clean",
            task_id=None,
            task_name=base_task,
            language=clean_language,
            log_path=clean_log,
            instruction_mode="task",
        )
        clean_dir = results_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        (clean_dir / "episodes.json").write_text(json.dumps(clean_episodes, indent=2), encoding="utf-8")
        summary["conditions"].append({
            "condition": "clean",
            "task_name": base_task,
            "category": "clean",
            "seed": seed,
            "success_rate": clean_rate,
            "successes": int(round(clean_rate * num_pairs)),
            "num_trials": num_pairs,
            "log_path": clean_log,
            "episodes_path": str(clean_dir / "episodes.json"),
        })

    for item in perturbations:
        os.environ["COSMOS_SMOKE_INSTRUCTION_MODE"] = item.get("instruction_mode", "task")
        pert_dir = results_dir / item["name"]
        pert_dir.mkdir(parents=True, exist_ok=True)

        if item.get("variant_mode") == "all_variants":
            variants = get_matching_variants(pair_suite, item["category"], base_task)
            if not variants:
                raise RuntimeError(
                    f"No variants found for suite={pair_suite}, category={item['category']}, base_task={base_task}"
                )
            if len(variants) < num_pairs:
                print(f"  Warning: only {len(variants)} {item['name']} variants available; requested {num_pairs}")

            pert_episodes = []
            pert_logs = []
            for episode_idx, (variant_id, variant_task) in enumerate(variants[:num_pairs]):
                pert_note = f"paired-{item['name']}-ep{episode_idx}-{num_pairs}pair"
                run_eval_with_task_filter(
                    suite_name=pair_suite,
                    task_name=variant_task.name,
                    category_value=item["category"],
                    num_trials=1,
                    run_id_note=pert_note,
                    local_log_dir=str(log_dir),
                )
                variant_episodes, variant_log = parse_episode_successes(log_dir, pert_note)
                if not variant_episodes:
                    raise RuntimeError(f"No episode result parsed for {variant_task.name}; log={variant_log}")
                pert_logs.append(variant_log)
                pert_episodes.append(
                    {
                        "episode": episode_idx,
                        "success": bool(variant_episodes[0]["success"]),
                        "seed": seed,
                        "deterministic_reset": deterministic_reset,
                        "deterministic_reset_seed": deterministic_reset_seed,
                        "base_task": base_task,
                        "condition": item["name"],
                        "category": item["category"],
                        "task_id": variant_id,
                        "task_name": variant_task.name,
                        "language": variant_task.language,
                        "init_state_index": 0,
                        "log_path": variant_log,
                        "instruction_mode": item.get("instruction_mode", "task"),
                        "run_id": run_id,
                        "suite": pair_suite,
                    }
                )
            pert_rate = (
                sum(1 for episode in pert_episodes if episode["success"]) / len(pert_episodes)
                if pert_episodes
                else 0.0
            )
            pert_log = pert_logs
            summary_task_name = item["task_name"]
        else:
            pert_task_id, pert_task = get_task_by_name(pair_suite, item["category"], item["task_name"])
            pert_note = f"paired-{item['name']}-{num_pairs}pair"
            pert_rate = run_eval_with_task_filter(
                suite_name=pair_suite,
                task_name=item["task_name"],
                category_value=item["category"],
                num_trials=num_pairs,
                run_id_note=pert_note,
                local_log_dir=str(log_dir),
            )
            pert_episodes, pert_log = parse_episode_successes(log_dir, pert_note)
            pert_episodes = annotate_episodes(
                pert_episodes,
                run_id=run_id,
                seed=seed,
                deterministic_reset=deterministic_reset,
                deterministic_reset_seed=deterministic_reset_seed,
                suite=pair_suite,
                base_task=base_task,
                condition=item["name"],
                category=item["category"],
                task_id=pert_task_id,
                task_name=pert_task.name,
                language=pert_task.language,
                log_path=pert_log,
                instruction_mode=item.get("instruction_mode", "task"),
            )
            summary_task_name = item["task_name"]

        (pert_dir / "episodes.json").write_text(json.dumps(pert_episodes, indent=2), encoding="utf-8")
        num_trials = len(pert_episodes)
        summary["conditions"].append(
            {
                "condition": item["name"],
                "task_name": summary_task_name,
                "category": item["category"],
                "seed": seed,
                "success_rate": pert_rate,
                "successes": sum(1 for episode in pert_episodes if episode["success"]),
                "num_trials": num_trials,
                "log_path": pert_log,
                "episodes_path": str(pert_dir / "episodes.json"),
                "instruction_mode": item.get("instruction_mode", "task"),
            }
        )

    summary_name = "clean" if only_condition == "clean" else (pert_name if "," not in pert_name else "selected")
    summary_path = results_dir / f"{base_task}__{summary_name}__{num_pairs}pair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Paired smoke summary saved to: {summary_path}")
else:
    suite_name = os.environ["COSMOS_SMOKE_SUITE"]
    task_id = int(os.environ["COSMOS_SMOKE_TASK_ID"])
    run_eval_with_task_filter(
        suite_name=suite_name,
        task_id=task_id,
        num_trials=1,
        run_id_note=f"smoke-task{task_id}",
        local_log_dir="./experiments/logs",
    )
