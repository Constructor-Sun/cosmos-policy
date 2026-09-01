# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.

Adapted from: https://github.com/user/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py

Parallel Inference:
    To enable parallel inference across multiple GPUs, use:
        --use_parallel_inference True
        --available_gpus "0,1,2,3"
        --num_queries_best_of_n 4

    This will run model queries in parallel across the specified GPUs using torch.multiprocessing, which can
    significantly speed up evaluation when using value functions that require multiple queries per action.

    Requirements:
    - Multiple GPUs must be available
    - CUDA must be properly configured
    - Sufficient GPU memory for multiple model copies

    Note: Uses torch.multiprocessing with 'spawn' start method for CUDA compatibility.

Usage examples:
    # *** Main checkpoint: 98.5% success rate ***
    #   Replace `task_suite_name` with one of {libero_spatial, libero_object, libero_goal, libero_10}
    #   Replace `seed` with one of {195, 196, 197}
    #   Replace `run_id_note` with a unique identifier for the run
    uv run -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1
    # Same as above, but with deterministic reset (seed=195/196/197, reset seed=0)
    # Also gets 98.5% success rate
    uv run -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note chkpt45000--5stepAct--seed195--deterministicand_deterministicResetSeed0 \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --deterministic_reset True \
        --deterministic_reset_seed 0

"""

import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import draccus
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import wandb
from libero.libero import benchmark

from cosmos_policy.experiments.robot.cosmos_utils import (
    WorkerPoolManager,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere
from memory_system.geometry import (
    camera_params as build_camera_params,
    depth_to_metric,
    flip_depth,
)
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Optional global phase-transition hook used by smoke-test held-object pipeline.
_PHASE_TRANSITION_HOOK = None

# Optional simple Place fine-aligner set by the smoke-test pipeline.
_PLACE_ALIGNER = None

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 3
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 5, 7


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_MIX = "libero_mix"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_MIX: 520,  # LIBERO-plus mixture; keep the conservative LIBERO-10 horizon
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "libero"                                                # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include primary (third-person) image in input
    num_third_person_images: int = 1                                     # Number of third-person images to include in input (LIBERO: 1 agentview image)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (LIBERO: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 16                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "first")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "gamma_weighted_average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL                      # Task suite (must be one of: LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_10, LIBERO_90, LIBERO_MIX)
    unnorm_key: str = ""                                                 # Optional action un-normalization key override (e.g., use libero_10 for LIBERO-plus)
    num_trials_per_task: int = 50                                        # Number of rollouts per task
    task_filter: str = ""                                                 # If non-empty, only run tasks whose name contains this substring
    initial_states_path: str = "DEFAULT"                                 # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                                               # Resolution for rendering environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 7                                                        # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save episodic data for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving
    save_vector_db: bool = False                                         # If True, save VAE latents + proprio at action chunk boundaries
    vector_db_output_dir: str = ""                                       # Output directory for vector DB .pt files
    enable_initial_alignment: bool = False                              # Enable one-shot initial ready-pose alignment before the first policy action
    enable_collision_aware_initial_alignment: bool = True              # Use cuRobo RGB-D collision-free planning for initial alignment
    enable_curobo_joint_execution: bool = False                         # Execute the timed cuRobo joint trajectory during initial alignment
    enable_urdf_robot_filter: bool = False                              # Remove robot depth with URDF before sphere filtering

    # fmt: on


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: PolicyEvalConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"

    if "image_aug" in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.enable_initial_alignment:
        if cfg.task_suite_name != TaskSuite.LIBERO_10:
            raise ValueError("Initial alignment memory currently supports only the libero_10 task suite")
        if cfg.env_img_res != 256:
            raise ValueError("Initial alignment memory was built at env_img_res=256")
        if not cfg.flip_images:
            raise ValueError("Initial alignment memory requires flip_images=True")



def _create_initial_alignment_selector(cfg: PolicyEvalConfig):
    """Build the one-shot Initial Alignment selector."""
    memory_dir = _REPO_ROOT / "skill_memory_test" / "libero_10"
    segments_manifest = memory_dir / "segments_ready_fixed16.json"
    feasible_recovery_targets = memory_dir / "feasible_recovery_targets.pt"
    ready3d_targets = memory_dir / "ready3d_targets.pt"
    missing = [
        path for path in (segments_manifest, feasible_recovery_targets)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing initial alignment inputs: {missing}")

    from memory_system.execute.curobo_planner import CuroboPlanner
    from memory_system.execute.initial_alignment import InitialAlignmentSelector
    planner = (
        CuroboPlanner(
            joint_execution=cfg.enable_curobo_joint_execution,
            enable_urdf_robot_filter=cfg.enable_urdf_robot_filter,
        )
        if cfg.enable_collision_aware_initial_alignment
        else None
    )
    return InitialAlignmentSelector(
        segments_manifest,
        feasible_recovery_targets,
        ready3d_targets=ready3d_targets if ready3d_targets.exists() else None,
        planner=planner,
    )


def _make_main_depth(obs, camera_params, flip_images: bool):
    """Return metric depth aligned with the canonical third-view RGB."""
    metric = depth_to_metric(
        obs["agentview_depth"], camera_params.near, camera_params.far
    )
    return flip_depth(metric) if flip_images else metric


def check_unnorm_key(cfg: PolicyEvalConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.unnorm_key or cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in Cosmos Policy `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def load_initial_states(cfg: PolicyEvalConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, flip_images: bool = False):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)

    # Prepare observations dict
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }

    return observation  # Return processed observation


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    initial_state=None,
    log_file=None,
    episode_index=0,
    alignment_task_name=None,
    initial_alignment_selector=None,
    phase_transition_hook=None,
    coordinator_factory=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Optional read-only skill-completion telemetry.  Its return value is
    # deliberately ignored: it never changes the action stream or episode end.
    skill_shadow = None
    if os.environ.get("COSMOS_SKILL_COMPLETION_SHADOW", "").lower() in {
        "1", "true", "yes"
    }:
        item = os.environ.get("COSMOS_SKILL_COMPLETION_ITEM", "").strip()
        if item:
            try:
                from memory_system.execute.skill_completion.shadow import (
                    PickCompletionShadow,
                )

                resolution = int(obs["agentview_image"].shape[0])
                skill_shadow = PickCompletionShadow(env, item, log_file, resolution)
                log_message(
                    f"[SKILL_COMPLETION] shadow enabled: item={item!r} "
                    f"instance_id={skill_shadow.target_instance_id}",
                    log_file,
                )
            except Exception as exc:
                log_message(f"[SKILL_COMPLETION] shadow init failed: {exc}", log_file)
        else:
            log_message(
                "[SKILL_COMPLETION] shadow requested but "
                "COSMOS_SKILL_COMPLETION_ITEM is empty",
                log_file,
            )

    def observe_skill_shadow(observation, action, frame):
        if skill_shadow is None:
            return
        try:
            # This is telemetry only; the boolean result is intentionally unused.
            skill_shadow.observe(observation, action, frame)
        except Exception as exc:
            log_message(f"[SKILL_COMPLETION] shadow frame failed: {exc}", log_file)

    # Active completion is deliberately kept separate from the read-only
    # shadow.  It is created only after Initial Alignment selects a concrete
    # memory demo, and it observes VLA actions only (never planner actions).
    skill_runtime = None
    pick_point_cloud = None
    session = None
    coordinator = None

    def _begin_vla_window(frame: int) -> None:
        if skill_runtime is None or skill_runtime.active or skill_runtime.exhausted:
            return
        try:
            initially_holding = (
                session.held_item is not None if session is not None else False
            )
            phase = skill_runtime.begin_vla(
                frame=frame,
                initially_holding=initially_holding,
            )
            log_message(
                f"[SKILL_COMPLETION] VLA start phase={skill_runtime.phase_index} "
                f"skill={phase.skill} args={phase.arguments} demo={skill_runtime.demo_id}",
                log_file,
            )
        except Exception as exc:
            log_message(f"[SKILL_COMPLETION] VLA start failed: {exc}", log_file)

    def _active_vla_points(observation, phase):
        nonlocal pick_point_cloud
        if phase is None or phase.skill != "Pick":
            return None
        item = phase.arguments.get("item")
        if not item:
            return None
        try:
            if pick_point_cloud is None:
                from memory_system.execute.skill_completion.shadow import (
                    PickTargetPointCloud,
                )

                resolution = int(observation["agentview_image"].shape[0])
                pick_point_cloud = PickTargetPointCloud(env, resolution)
            return pick_point_cloud.points(observation, item)
        except Exception as exc:
            log_message(f"[SKILL_COMPLETION] Pick point extraction failed: {exc}", log_file)
            return None

    def observe_active_vla(observation, action, frame: int):
        """Consume one VLA frame and clear the queue on phase advance."""
        if skill_runtime is None or not skill_runtime.active:
            return None
        phase = skill_runtime.active_phase
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        gripper_qpos = observation.get("robot0_gripper_qpos")
        if gripper_qpos is None:
            log_message(
                "[SKILL_COMPLETION] WARNING active gripper_qpos missing; "
                "Place completion will rely on timeout",
                log_file,
            )
        decision = skill_runtime.observe_vla_frame(
            target_points=_active_vla_points(observation, phase),
            eef_pos=observation.get("robot0_eef_pos"),
            eef_quat=observation.get("robot0_eef_quat"),
            gripper_closed=bool(action_array[-1] > 0.0) if len(action_array) else False,
            gripper_qpos=gripper_qpos,
            frame=frame,
        )
        if decision.advance:
            action_queue.clear()
            log_message(
                f"[SKILL_COMPLETION] advance phase={skill_runtime.phase_index - 1} "
                f"reason={decision.reason} semantic={decision.semantic_completed} "
                f"chunks={decision.action_chunks}; queue cleared",
                log_file,
            )
        return decision

    def finish_active_vla_chunk(frame: int):
        """Apply the timeout only when a VLA queue naturally exhausts."""
        if skill_runtime is None or not skill_runtime.active:
            return None
        decision = skill_runtime.finish_action_chunk(frame=frame)
        if decision.advance:
            action_queue.clear()
            log_message(
                f"[SKILL_COMPLETION] advance phase={skill_runtime.phase_index - 1} "
                f"reason={decision.reason} semantic={decision.semantic_completed} "
                f"chunks={decision.action_chunks}; queue cleared",
                log_file,
            )
        return decision

    alignment_camera_params = None
    if (
        cfg.enable_collision_aware_initial_alignment
        and initial_alignment_selector is not None
    ):
        height, width = obs["agentview_image"].shape[:2]
        alignment_camera_params = build_camera_params(
            env.sim, "agentview", height, width
        )

    # Initialize action queue
    if cfg.num_open_loop_steps != cfg.chunk_size:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match cfg.chunk_size "
            f"{cfg.chunk_size}! For best performance (in terms of both speed and success rate), we "
            "recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    alignment_gripper_action = None
    last_gripper_closed = None

    _alignment_controller = None
    _alignment_steps_remaining = 0
    _alignment_step_index = 0
    _captured_latent = None
    _policy_step_count = 0
    _initial_align_attempted = False
    from cosmos_policy.experiments.robot.libero.libero_joint_control import step_correction_controller

    def _close_alignment_controller(controller) -> None:
        if controller is not None and hasattr(controller, "close"):
            controller.close()

    def _maybe_start_initial_alignment(t_now, observation, obs, log_fh) -> None:
        nonlocal _initial_align_attempted, _policy_step_count, skill_runtime
        nonlocal _alignment_steps_remaining
        nonlocal _alignment_controller, _alignment_step_index
        nonlocal alignment_gripper_action, action_queue
        nonlocal session, coordinator
        if initial_alignment_selector is None or _initial_align_attempted:
            return
        # Trigger before the first policy action is executed.
        if _policy_step_count != 0:
            return
        _initial_align_attempted = True
        if not _captured_latent:
            log_message(
                f"[INIT ALIGN] t={t_now}: skip, no captured VAE latent",
                log_fh,
            )
            return
        current_vae_main = _captured_latent[0][0, :, 3:4, :, :]
        current_ee_states = np.concatenate([
            obs["robot0_eef_pos"],
            Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
        ]).astype(np.float32)
        main_depth = None
        joint_positions = None
        if alignment_camera_params is not None and "agentview_depth" in obs:
            main_depth = _make_main_depth(
                obs, alignment_camera_params, cfg.flip_images
            )
        if "robot0_joint_pos" in obs:
            joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        robot_base_pose = np.concatenate([env.robots[0].base_pos, env.robots[0].base_ori])
        try:
            alignment = initial_alignment_selector.select(
                alignment_task_name,
                current_vae_main,
                current_ee_states,
                main_depth=main_depth,
                camera_params=alignment_camera_params,
                joint_positions=joint_positions,
                gripper_joint_positions=obs.get("robot0_gripper_qpos"),
                robot_base_pose=robot_base_pose,
                observation=obs,
                env=env,
            )
            spatial_match = initial_alignment_selector.last_spatial_match
            if spatial_match is not None:
                xyz, ranked = spatial_match
                log_message(
                    f"[INIT ALIGN 3D] xyz={np.asarray(xyz).round(6).tolist()} "
                    f"top3={[(str(p['demo_id']), round(d, 4)) for p, d in ranked]}",
                    log_fh,
                )
        except Exception as exc:
            log_message(
                f"[INIT ALIGN] t={t_now}: error selecting target: {exc}",
                log_fh,
            )
            return
        if alignment is None:
            log_message(
                f"[INIT ALIGN] t={t_now}: no matching ready pose; skip",
                log_fh,
            )
            return
        # Bind active completion to the exact demo selected by memory retrieval.
        # If that demo has no valid ordered sequence, retain the baseline path.
        if (
            skill_runtime is None
            and os.environ.get("COSMOS_SKILL_COMPLETION_ACTIVE", "0").lower()
            not in {"0", "false", "no"}
        ):
            demo_id = alignment.demo_ids[0] if alignment.demo_ids else None
            sequence = None
            sequence_loader = getattr(
                initial_alignment_selector, "sequence_for_demo", None
            )
            if demo_id is not None and callable(sequence_loader):
                try:
                    sequence = sequence_loader(alignment_task_name, demo_id)
                except Exception as exc:
                    log_message(
                        f"[SKILL_COMPLETION] sequence load failed for demo={demo_id}: {exc}",
                        log_fh,
                    )
            if sequence:
                from memory_system.execute.vla_skill_runtime import VLASkillRuntime

                skill_runtime = VLASkillRuntime(
                    sequence,
                    task_name=alignment_task_name,
                    demo_id=demo_id,
                    episode_id=episode_index,
                )
                if coordinator_factory is not None:
                    coordinator = coordinator_factory(
                        alignment_task_name,
                        demo_id,
                    )
                    session = coordinator.session
                log_message(
                    f"[SKILL_COMPLETION] active sequence loaded: demo={demo_id} "
                    f"phases={len(sequence)}",
                    log_fh,
                )
            else:
                log_message(
                    f"[SKILL_COMPLETION] no valid sequence for demo={demo_id}; "
                    "active continuation disabled",
                    log_fh,
                )
        alignment_gripper_action = float(bool(last_gripper_closed))
        if alignment.joint_trajectory is not None:
            from cosmos_policy.experiments.robot.libero.libero_joint_control import (
                LiberoJointTrajectoryController,
            )
            _alignment_controller = LiberoJointTrajectoryController(
                env,
                alignment.joint_trajectory,
                alignment.target_ee_states,
                alignment_gripper_action,
            )
            _alignment_steps_remaining = _alignment_controller.max_steps
        else:
            _alignment_steps_remaining = alignment.correction_steps
            _alignment_controller = alignment.controller
        _alignment_step_index = 0
        action_queue.clear()
        if (
            os.environ.get("COSMOS_DEBUG_INIT_ALIGN", "").lower()
            in {"1", "true", "yes"}
        ):
            _full_plan_waypoints = getattr(_alignment_controller, "waypoints", None)
            if _full_plan_waypoints is not None:
                log_message(
                    "[INIT_ALIGN_FULL_PLAN] "
                    + json.dumps(
                        {
                            "episode": episode_index,
                            "waypoints": np.asarray(
                                _full_plan_waypoints, dtype=np.float32
                            ).tolist(),
                            "target": alignment.target_ee_states.tolist(),
                        }
                    ),
                    log_fh,
                    console=False,
                )
        log_message(
            f"[INIT ALIGN] t={t_now}: target={alignment.target_ee_states.tolist()} "
            f"sim={alignment.similarity:.3f} demos={alignment.demo_ids} "
            f"steps={_alignment_steps_remaining}",
            log_fh,
        )

    def _maybe_run_phase_transition_hook(decision, observation, frame):
        nonlocal _alignment_controller, _alignment_steps_remaining
        nonlocal _alignment_step_index, alignment_gripper_action
        if decision is None or not decision.advance:
            return
        if skill_runtime is None or skill_runtime.phase_index < 1:
            return
        completed_phase = skill_runtime.phases[skill_runtime.phase_index - 1]
        next_phase = skill_runtime.active_phase
        if completed_phase is None or next_phase is None:
            return

        if coordinator is not None:
            try:
                intervention = coordinator.on_phase_advance(
                    observation=observation,
                    completed_phase=completed_phase,
                    next_phase=next_phase,
                    env=env,
                    cfg=cfg,
                    log_file=log_file,
                    episode_id=episode_index,
                )
            except Exception as exc:
                log_message(f"[SKILL_TRANSITION] coordinator failed: {exc}", log_file)
                return
            if intervention is None:
                return
            if intervention.controller is not None:
                _alignment_controller = intervention.controller
                _alignment_steps_remaining = int(intervention.step_budget)
                _alignment_step_index = 0
                alignment_gripper_action = float(intervention.gripper_action)
                action_queue.clear()
                log_message(
                    f"[SKILL_TRANSITION] t={frame}: intervention armed "
                    f"kind={intervention.kind} steps={_alignment_steps_remaining}",
                    log_file,
                )
            return

        # Legacy hook fallback for existing held-object integration.
        hook = phase_transition_hook or _PHASE_TRANSITION_HOOK
        if hook is None:
            return
        if completed_phase.skill != "Pick" or next_phase.skill not in {"PlaceIn", "PlaceOn"}:
            return
        try:
            result = hook(
                observation=observation,
                completed_phase=completed_phase,
                next_phase=next_phase,
                task_name=skill_runtime.task_name,
                demo_id=skill_runtime.demo_id,
                episode_id=episode_index,
                env=env,
                cfg=cfg,
                log_file=log_file,
            )
        except Exception as exc:
            log_message(f"[HELD_OBJECT] phase transition hook failed: {exc}", log_file)
            return
        if result is None:
            return
        controller = getattr(result, "controller", None)
        if controller is None:
            return
        _alignment_controller = controller
        _alignment_steps_remaining = int(getattr(result, "correction_steps", 1))
        _alignment_step_index = 0
        alignment_gripper_action = 1.0
        action_queue.clear()
        log_message(
            f"[HELD_OBJECT] t={frame}: held-object controller armed steps={_alignment_steps_remaining}",
            log_file,
        )

    def _maybe_start_place_fine_alignment(obs, frame):
        nonlocal _alignment_controller, _alignment_steps_remaining
        nonlocal _alignment_step_index, alignment_gripper_action
        aligner = session.pending_place_aligner if session is not None else None
        if aligner is None or _alignment_controller is not None:
            return
        if skill_runtime is None or not skill_runtime.active:
            return
        phase = skill_runtime.active_phase
        if phase is None or phase.skill not in {"PlaceIn", "PlaceOn"}:
            return
        try:
            controller = aligner.maybe_controller(obs, frame, phase)
        except Exception as exc:
            log_message(f"[PLACE_ALIGN] simple check failed: {exc}", log_file)
            return
        if controller is None:
            return
        _alignment_controller = controller
        _alignment_steps_remaining = int(getattr(controller, "max_steps", 48))
        _alignment_step_index = 0
        alignment_gripper_action = 1.0
        action_queue.clear()
        log_message(
            f"[PLACE_ALIGN] t={frame}: simple controller armed "
            f"pos_err={getattr(aligner, 'last_pos_error', float('nan')):.4f} "
            f"rot_err={getattr(aligner, 'last_rot_error', float('nan')):.4f} "
            f"steps={_alignment_steps_remaining}",
            log_file,
        )

    # Setup
    t = 0
    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Best-of-N search variables
    base_seed = cfg.seed  # Used for seed switching (if applicable)

    # Data collection buffers
    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []
    vector_db_chunks: list = []  # always created; populated only when save_vector_db=True
    _dump_wrist_dir = os.environ.get("COSMOS_DUMP_WRIST_DIR")
    _wrist_dump: list = []
    if _dump_wrist_dir:
        os.makedirs(_dump_wrist_dir, exist_ok=True)

    # Run episode
    success = False
    try:
        NUM_STEPS_WAIT = 10
        while t < max_steps + NUM_STEPS_WAIT:
            # If the deterministic flag is set, reset the random state with the same seed in every step
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                seed = 0
                set_seed_everywhere(seed)

            # Do nothing for the first few timesteps to let objects stabilize
            if t < NUM_STEPS_WAIT:
                dummy_action = get_libero_dummy_action(cfg.model_family)
                obs, reward, done, info = env.step(dummy_action)
                observe_skill_shadow(obs, dummy_action, t)
                t += 1
                continue

            # Prepare observation
            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            if _dump_wrist_dir is not None:
                _wrist_dump.append((
                    t,
                    np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8).copy(),
                    np.asarray(obs["agentview_image"], dtype=np.uint8).copy(),
                    np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy(),
                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).copy(),
                    last_gripper_closed,
                ))
            replay_images.append(observation["primary_image"])
            if replay_wrist_images is not None:
                replay_wrist_images.append(observation["wrist_image"])

            if cfg.data_collection:
                primary_images_list.append(observation["primary_image"])
                wrist_images_list.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

            _alignment_active = (
                _alignment_steps_remaining > 0
                and _alignment_controller is not None
            )
            # Initial Alignment owns the action stream until it finishes, then
            # the policy is queried again because its queued chunk was cleared.
            if _alignment_active:
                if _alignment_controller is not None:
                    # Step-level closed loop: regenerate the action from the
                    # current measured EE pose every step.
                    _step_action = step_correction_controller(_alignment_controller, obs)
                if np.ndim(_step_action) == 1 and _step_action.shape[0] in (7, 8):
                    action = _step_action.astype(np.float32).copy()
                    # Preserve the gripper command captured before alignment.
                    if alignment_gripper_action is not None:
                        action[-1] = alignment_gripper_action
                else:
                    action = np.zeros(7, dtype=np.float32)
                    action[:6] = _step_action
                    action[6] = alignment_gripper_action
                print(f"t: {t}\t initial alignment action: {action}")

                _debug_init_align = (
                    os.environ.get("COSMOS_DEBUG_INIT_ALIGN", "").lower()
                    in {"1", "true", "yes"}
                )
                _debug_before_ee = None
                _debug_wp_index = None
                _debug_wp = None
                if _debug_init_align:
                    _debug_before_ee = np.concatenate([
                        obs["robot0_eef_pos"],
                        Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
                    ]).astype(np.float32)
                    _debug_wp_index = getattr(_alignment_controller, "index", None)
                    _debug_wps = getattr(_alignment_controller, "waypoints", None)
                    if _debug_wps is not None and _debug_wp_index is not None:
                        _debug_wp = np.asarray(
                            _debug_wps[min(_debug_wp_index, len(_debug_wps) - 1)],
                            dtype=np.float32,
                        )

                if cfg.data_collection:
                    actions_list.append(action.copy())

                last_gripper_closed = bool(float(action[-1]) > 0.0)
                obs, reward, done, info = env.step(action.tolist())
                observe_skill_shadow(obs, action, t)
                _alignment_steps_remaining -= 1
                _alignment_step_index += 1
                if _debug_init_align:
                    _debug_after_ee = np.concatenate([
                        obs["robot0_eef_pos"],
                        Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(),
                    ]).astype(np.float32)
                    log_message(
                        "[INIT_ALIGN_DEBUG] "
                        + json.dumps(
                            {
                                "episode": episode_index,
                                "t": t,
                                "correction_step_index": _alignment_step_index,
                                "waypoint_index": _debug_wp_index,
                                "waypoint": None if _debug_wp is None else _debug_wp.tolist(),
                                "eef_before": None if _debug_before_ee is None else _debug_before_ee.tolist(),
                                "action": np.asarray(action, dtype=np.float32).tolist(),
                                "eef_after": _debug_after_ee.tolist(),
                                "steps_remaining": _alignment_steps_remaining,
                            }
                        ),
                        log_file,
                        console=False,
                    )
                if hasattr(_alignment_controller, "observe"):
                    _alignment_controller.observe(obs)
                if getattr(_alignment_controller, "finished", False):
                    _alignment_steps_remaining = 0
                if _alignment_steps_remaining == 0:
                    _finished_controller = _alignment_controller
                    _close_alignment_controller(_finished_controller)
                    alignment_gripper_action = None
                    _alignment_controller = None
                    _alignment_step_index = 0
                    log_message(
                        f"[INIT ALIGN] t={t}: alignment finished; "
                        f"status={getattr(_finished_controller, 'status', 'completed')}; resuming policy "
                        f"eef_pos={np.asarray(obs['robot0_eef_pos'], dtype=np.float64).round(6).tolist()} "
                        f"eef_quat={np.asarray(obs['robot0_eef_quat'], dtype=np.float64).round(6).tolist()}",
                        log_file,
                    )
                    _begin_vla_window(t + 1)
                if done:
                    success = True
                    break
                t += 1
                continue

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                _begin_vla_window(t)
                best_actions = None
                best_future_predictions = None

                # Capture VAE latent during the first model forward pass in get_action()
                _capture_vae = (
                    cfg.save_vector_db
                    or (cfg.enable_initial_alignment and initial_alignment_selector is not None)
                )
                if _capture_vae:
                    _captured_latent = []
                    _orig_gdac = model.get_data_and_condition
                    def _hook_gdac(data_batch):
                        raw, latent, cond = _orig_gdac(data_batch)
                        lat = latent.detach().cpu()
                        if not _captured_latent:
                            _captured_latent.append(lat)
                        return raw, latent, cond
                    model.get_data_and_condition = _hook_gdac

                # Query model multiple times if value functions are available
                num_queries = cfg.num_queries_best_of_n

                # Use parallel inference if enabled and multiple queries are needed
                if cfg.use_parallel_inference and num_queries > 1 and worker_pool and worker_pool.initialized:
                    # Query model in parallel
                    start_time = time.time()
                    query_results = query_model_parallel(
                        cfg, observation, task_description, worker_pool, cfg.parallel_timeout
                    )
                    total_query_time = time.time() - start_time

                    log_message(
                        f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                    )

                else:
                    # Serial execution (original behavior)
                    query_results = []
                    for query_idx in range(num_queries):
                        actions_by_depth = []  # Action chunks across all depths of the search
                        future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                        value_predictions_by_depth = []  # Value predictions across all depths of the search
                        return_dict = {}
                        # Query model to get action
                        start_time = time.time()
                        action_return_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=not (
                                cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                            ),
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec", log_file
                        )
                        return_dict["actions"] = action_return_dict["actions"]
                        actions_by_depth.append(return_dict["actions"])

                        if cfg.ar_future_prediction:
                            # Autoregressively query model to get future state prediction
                            start_time = time.time()
                            future_state_return_dict = get_future_state_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                generated_latent_with_action=action_return_dict["generated_latent"],
                                orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                                future_proprio_latent_idx=action_return_dict["latent_indices"][
                                    "future_proprio_latent_idx"
                                ],
                                future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image_latent_idx"
                                ],
                                future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image2_latent_idx"
                                ],
                                future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                                future_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_image2_latent_idx"
                                ],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Future state prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["future_image_predictions"] = future_state_return_dict[
                                "future_image_predictions"
                            ]
                            future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                        else:
                            return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                        if cfg.ar_value_prediction:
                            # Autoregressively query model to get value prediction
                            start_time = time.time()
                            value_return_dict = get_value_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        elif cfg.ar_qvalue_prediction:
                            # Autoregressively query model to get Q-value prediction
                            start_time = time.time()
                            value_return_dict = get_qvalue_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                action_sample=action_return_dict["generated_latent"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        else:
                            return_dict["value_prediction"] = action_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])

                        return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                        return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                        return_dict["actions_by_depth"] = actions_by_depth
                        query_results.append(return_dict)

                # Print all value predictions
                log_message(f"t={t}: Current base seed: {base_seed}", log_file)
                for query_idx, return_dict in enumerate(query_results):
                    predicted_value = return_dict["value_prediction"]
                    log_message(
                        f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                        log_file,
                    )
                # Get dict: seed number -> (action chunk, future state, value)
                seed_to_return_dict = {
                    cfg.seed + query_idx: (
                        return_dict["actions"],
                        return_dict["future_image_predictions"],
                        return_dict["value_prediction"],
                    )
                    for query_idx, return_dict in enumerate(query_results)
                }
                # Get seed with highest value
                best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
                best_actions = best_return_dict[0]
                best_future_predictions = best_return_dict[1]
                best_value_predictions = best_return_dict[2]
                # Use the best actions, future predictions, and value predictions found
                action_queue.extend(best_actions)
                future_image_predictions_list.append(best_future_predictions)
                log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)

                if cfg.save_vector_db:
                    if _captured_latent:
                        vae_video = _captured_latent[0][0, :, [2, 3], :, :].half()
                        vector_db_chunks.append({
                            "vae_video": vae_video,
                            "proprio": torch.from_numpy(observation["proprio"].copy()).half(),
                            "action_chunk": torch.from_numpy(np.array(best_actions)).half(),
                            "step_index": t,
                        })

                _maybe_start_initial_alignment(t, observation, obs, log_file)

                if _capture_vae:
                    model.get_data_and_condition = _orig_gdac

            _maybe_start_place_fine_alignment(obs, t)

            # A newly selected Initial Alignment begins immediately on this timestep.
            _is_alignment_action = (
                _alignment_steps_remaining > 0
                and _alignment_controller is not None
            )
            if _is_alignment_action:
                if _alignment_controller is not None:
                    # Step-level closed loop: regenerate the action from the
                    # current measured EE pose every step.
                    _step_action = step_correction_controller(_alignment_controller, obs)
                if np.ndim(_step_action) == 1 and _step_action.shape[0] in (7, 8):
                    action = _step_action.astype(np.float32).copy()
                else:
                    action = np.zeros(7, dtype=np.float32)
                    action[:6] = _step_action
                    action[6] = alignment_gripper_action
                _alignment_steps_remaining -= 1
                _alignment_step_index += 1
            else:
                action = action_queue.popleft()
                _policy_step_count += 1

            # Process action
            print(f"t: {t}\t action: {action}")

            if cfg.data_collection:
                actions_list.append(action.copy())
            # Execute action in environment
            last_gripper_closed = bool(float(action[-1]) > 0.0)
            obs, reward, done, info = env.step(action.tolist())
            observe_skill_shadow(obs, action, t)
            if hasattr(_alignment_controller, "observe"):
                _alignment_controller.observe(obs)
            if getattr(_alignment_controller, "finished", False):
                _alignment_steps_remaining = 0
            if (
                _alignment_controller is not None
                and _alignment_steps_remaining == 0
            ):
                _finished_controller = _alignment_controller
                _close_alignment_controller(_finished_controller)
                alignment_gripper_action = None
                _alignment_controller = None
                _alignment_step_index = 0
                log_message(
                    f"[INIT ALIGN] t={t}: alignment finished; "
                    f"status={getattr(_finished_controller, 'status', 'completed')}; resuming policy "
                    f"eef_pos={np.asarray(obs['robot0_eef_pos'], dtype=np.float64).round(6).tolist()} "
                    f"eef_quat={np.asarray(obs['robot0_eef_quat'], dtype=np.float64).round(6).tolist()}",
                    log_file,
                )
                _begin_vla_window(t + 1)
            if not _is_alignment_action:
                decision = observe_active_vla(obs, action, t)
                if decision is not None and decision.advance:
                    _maybe_run_phase_transition_hook(decision, obs, t)
                else:
                    # A chunk boundary is the only place where timeout may be
                    # applied; planner actions never reach this hook.
                    if len(action_queue) == 0:
                        decision = finish_active_vla_chunk(t)
                        if decision is not None and decision.advance:
                            _maybe_run_phase_transition_hook(decision, obs, t)
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_msg = f"Episode error: {e}"
        traceback_str = traceback.format_exc()
        log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    _close_alignment_controller(_alignment_controller)
    if skill_runtime is not None:
        for summary in skill_runtime.finalize(success):
            log_message(
                "[SKILL_COMPLETION] summary " + json.dumps(summary),
                log_file,
            )
    if skill_shadow is not None:
        skill_shadow.close()

    # Fill data collection buffers
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C)
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
        )
        # Add future image predictions if available
        if len(future_image_predictions_list) > 0:
            if cfg.use_third_person_image:
                future_primary_images = [
                    x["future_image"] for x in future_image_predictions_list if x["future_image"] is not None
                ]
                if len(future_primary_images) > 0:
                    collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Wrist image predictions (may be None depending on config)
            if (
                cfg.use_wrist_image
                and "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None

    if cfg.save_vector_db and vector_db_chunks:
        out_dir = cfg.vector_db_output_dir or os.path.join(cfg.local_log_dir, "vector_db")
        os.makedirs(out_dir, exist_ok=True)
        init_offset = int(os.environ.get("COSMOS_INIT_STATE_OFFSET", "0"))
        state_idx = init_offset + episode_index
        ts = int(time.time() * 1_000_000)
        fname = f"chunks_state{state_idx}_ep{episode_index + 1}_{ts}.pt"
        for chunk in vector_db_chunks:
            chunk["init_state_index"] = state_idx
        torch.save(vector_db_chunks, os.path.join(out_dir, fname))

    if _wrist_dump:
        import numpy as _np
        _np.savez(
            os.path.join(_dump_wrist_dir, f"ep{episode_index + 1}.npz"),
            times=_np.asarray([r[0] for r in _wrist_dump]),
            wrist=_np.stack([r[1] for r in _wrist_dump]),
            agentview=_np.stack([r[2] for r in _wrist_dump]),
            eef=_np.stack([r[3] for r in _wrist_dump]),
            qpos=_np.stack([r[4] for r in _wrist_dump]),
            closed=_np.asarray([r[5] for r in _wrist_dump]),
        )

    return success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data


def run_task(
    cfg: PolicyEvalConfig,
    task_suite,
    task_id: int,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    initial_alignment_selector=None,
    phase_transition_hook=None,
    coordinator_factory=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)
    alignment_task_name = None
    if initial_alignment_selector is not None:
        alignment_task_name = initial_alignment_selector.resolve_task_name(task.name)
        if alignment_task_name is not None and alignment_task_name != task.name:
            log_message(
                f"[INIT ALIGN] mapped perturbation task {task.name!r} to {alignment_task_name!r}",
                log_file,
            )

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
        camera_depths=[True, False] if (
            cfg.enable_initial_alignment and cfg.enable_collision_aware_initial_alignment
        ) else None,
    )

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data = run_episode(
            cfg,
            env,
            task_description,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            initial_state,
            log_file,
            episode_index=episode_idx,
            alignment_task_name=alignment_task_name,
            initial_alignment_selector=initial_alignment_selector,
            phase_transition_hook=phase_transition_hook,
            coordinator_factory=coordinator_factory,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video (skip if COSMOS_SKIP_PLAIN_ROLLOUT is set)
        if os.environ.get("COSMOS_SKIP_PLAIN_ROLLOUT", "").lower() not in ("1", "true", "yes"):
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
            )

        # Save replay video with future image predictions included
        future_primary_image_predictions = None
        if cfg.use_third_person_image:
            future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
        future_wrist_image_predictions = None
        if cfg.use_wrist_image:
            future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
        save_rollout_video_with_future_image_predictions(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            chunk_size=cfg.chunk_size,
            num_open_loop_steps=cfg.num_open_loop_steps,
            rollout_wrist_images=replay_wrist_images,
            future_primary_image_predictions=future_primary_image_predictions,
            future_wrist_image_predictions=future_wrist_image_predictions,
            log_file=log_file,
            show_diff=False,
        )

        # Save episodic data (in data collection mode)
        if cfg.data_collection and collected_data is not None:

            def _save_episode_data():
                """Save collected episode data to HDF5 file."""
                ep_filename = f"episode_data--suite={cfg.task_suite_name}--{DATE_TIME}--task={task_id}--ep={total_episodes}--success={success}--{cfg.run_id_note}.hdf5"
                rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data")
                os.makedirs(rollout_data_dir, exist_ok=True)
                ep_filepath = os.path.join(rollout_data_dir, ep_filename)
                with h5py.File(ep_filepath, "w") as f:
                    for k, v in collected_data.items():
                        if isinstance(v, np.ndarray):
                            is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                            if is_image and cfg.jpeg_compress:
                                jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                dt = h5py.vlen_dtype(np.dtype("uint8"))
                                f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                            else:
                                f.create_dataset(k, data=v)
                        else:
                            f.attrs[k] = v
                    f.attrs["task_description"] = task_description

            _save_episode_data()

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/{task_description}": task_success_rate,
                f"num_episodes/{cfg.task_suite_name}/{task_description}": task_episodes,
                f"num_successes/{cfg.task_suite_name}/{task_description}": task_successes,
            },
        )

    return (
        total_episodes,
        total_successes,
    )


@draccus.wrap()
def eval_libero(
    cfg: PolicyEvalConfig,
    *,
    coordinator_factory=None,
) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""

    # Set DETERMINISTIC environment variable if on deterministic mode (makes some model operations deterministic)
    assert not (cfg.deterministic and cfg.randomize_seed), (
        "Cannot enable both deterministic mode and randomize seed mode!"
    )
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize T5 text embeddings cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Load Cosmos Policy dataset stats
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # If using parallel inference, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        model = None
        planning_model = None

    # If using serial inference, initialize model and Cosmos config
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None

        # Initialize model for world model and value function
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg.model_family)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_suite_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)

    # Log parallel inference configuration and start worker pool
    if cfg.use_parallel_inference and worker_pool:
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
        log_message(f"Multiprocessing start method: {mp.get_start_method()}", log_file)

        # Verify GPUs are available
        for gpu_id in available_gpus:
            if gpu_id >= torch.cuda.device_count():
                log_message(
                    f"Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)", log_file
                )

        # Start worker pool
        try:
            log_message("Starting worker pool...", log_file)
            worker_pool.start_workers()
            log_message("Worker pool started successfully", log_file)
        except Exception as e:
            error_msg = f"Failed to start worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)
            log_message("Disabling parallel inference for this run", log_file)
            worker_pool = None
    else:
        log_message("Using serial inference (parallel inference disabled)", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    initial_alignment_selector = None
    if cfg.enable_initial_alignment:
        initial_alignment_selector = _create_initial_alignment_selector(cfg)
        log_message(
            "Initial alignment enabled (standalone one-shot path)",
            log_file,
        )

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Number of tasks: {num_tasks}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        if cfg.task_filter and cfg.task_filter not in task.name:
            continue
        (
            total_episodes,
            total_successes,
        ) = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
            initial_alignment_selector=initial_alignment_selector,
            coordinator_factory=coordinator_factory,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/total": final_success_rate,
                f"num_episodes/{cfg.task_suite_name}/total": total_episodes,
                f"num_successes/{cfg.task_suite_name}/total": total_successes,
            },
        )
        wandb.save(local_log_filepath)

    # Cleanup worker pool
    if worker_pool:
        try:
            worker_pool.shutdown()
        except Exception as e:
            error_msg = f"Error shutting down worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
