#!/usr/bin/env python3
"""
collect_shift_embeddings.py — Multi-Layer Shift Embedding Collection

Collects VAE latent, DiT L0/L14/L27 hidden states (video + action slots),
and action output during full policy rollouts across LIBERO-Plus tasks
under all perturbation conditions.

For each (suite, base_task, condition), runs one full policy episode.
At every N=5 steps, captures a single denoising step at sigma=80 with
hooks on DiT blocks 0, 14, 27.  Saves per-episode HDF5 files.

Usage:
  # Pilot — single task, all 8 conditions (use libero_10 for testing)
  MUJOCO_GL=egl PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \\
  python bin/collect_shift_embeddings.py \\
    --suite libero_10 \\
    --task "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \\
    --save-interval 5 --seed 7

  # Full — 3 suites (spatial/object/goal), all tasks, all conditions
  python bin/collect_shift_embeddings.py --all-suites --save-interval 5 --seed 7
"""

import sys, os, pathlib, json, time, ctypes, re, argparse, csv, pickle
from types import SimpleNamespace
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# Native runtime setup (ImageMagick / wand for LIBERO)
# ═══════════════════════════════════════════════════════════════════

def _prepend_env_path(name, value):
    parts = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    if value not in parts:
        os.environ[name] = value + (os.pathsep + os.environ[name] if parts else "")

def _preload_first_existing(lib_dir, names):
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    last_error = None
    for name in names:
        path = lib_dir / name
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=mode)
            return True
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        print(f"[warn] failed to preload ImageMagick library from {lib_dir}: {last_error}")
    return False

def _configure_native_runtime():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("DETERMINISTIC", "True")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy_numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy_mplconfig")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    prefix = pathlib.Path(conda_prefix) if conda_prefix else pathlib.Path(sys.executable).resolve().parents[1]
    lib_dir = prefix / "lib"
    if not lib_dir.exists():
        return

    os.environ.setdefault("MAGICK_HOME", str(prefix))
    os.environ.setdefault("WAND_MAGICK_LIBRARY_SUFFIX", "-7.Q16HDRI")
    _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir))
    _preload_first_existing(lib_dir, ["libMagickCore-7.Q16HDRI.so.10", "libMagickCore-7.Q16HDRI.so"])
    _preload_first_existing(lib_dir, ["libMagickWand-7.Q16HDRI.so.10", "libMagickWand-7.Q16HDRI.so"])

_configure_native_runtime()

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import h5py

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, prepare_images_for_model, duplicate_array,
    init_t5_text_embeddings_cache, load_dataset_stats, get_t5_embedding_from_cache,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from libero.libero import benchmark as libero_benchmark_module
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import Task
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

CKPT_ROOT = pathlib.Path("/data3/liu/exp/counterfactual/checkpoints")
POLICY_DIR = CKPT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
BASE_MODEL_DIR = CKPT_ROOT / "Cosmos-Predict2-2B-Video2World"

T = 4  # temporal copies per video slot
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Video slot layout (9 slots):
#   0: first frame    3: current primary   6: future wrist
#   1: blank          4: action            7: future primary
#   2: current wrist  5: proprio           8: value
SLOT_WRIST = 2
SLOT_PRIMARY = 3
SLOT_ACTION = 4

CAPTURE_TEMPORAL_SLOTS = [2, 3, 4]  # wrist, primary, action
CAPTURE_H_GRID = 14
CAPTURE_W_GRID = 14
CAPTURE_DIM = 2048

SIGMA_VAL = 80.0

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

# Perturbation categories as defined in LIBERO-plus task_classification.json
PERTURBATION_CATEGORIES = {
    "camera_viewpoints":    "Camera Viewpoints",
    "background_textures":  "Background Textures",
    "light_conditions":     "Light Conditions",
    "objects_layout":       "Objects Layout",
    "robot_initial_states": "Robot Initial States",
    "sensor_noise":         "Sensor Noise",
    "language_instructions": "Language Instructions",
}

# Perturbation suffix patterns for stripping (order matters: _language_ before _view_)
PERT_SUFFIX_RE = re.compile(
    r"_(language|light|table|tb|view|noise)_"   # prefix-style suffixes
    r"|_add_\d+$"                                 # _add_N at end
    r"|_level\d_sample\d$"                        # _levelN_sampleM at end
)

# ═══════════════════════════════════════════════════════════════════
# Checkpoint path monkey-patch
# ═══════════════════════════════════════════════════════════════════

from cosmos_policy._src.imaginaire.utils import checkpoint_db

_orig_get_checkpoint_path = checkpoint_db.get_checkpoint_path

def _get_checkpoint_path(uri):
    s = str(uri)
    if "Cosmos-Predict2-2B-Video2World" in s:
        return str(BASE_MODEL_DIR / s.split("Cosmos-Predict2-2B-Video2World/")[-1])
    if "ALOHA" in s or "LIBERO" in s:
        return str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    return _orig_get_checkpoint_path(uri)

checkpoint_db.get_checkpoint_path = _get_checkpoint_path

# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_model_and_stats():
    """Load Cosmos Policy model and dataset statistics."""
    cosmos_utils.init_t5_text_embeddings_cache(str(POLICY_DIR / "libero_t5_embeddings.pkl"))

    cfg = SimpleNamespace(
        suite="libero", config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(POLICY_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1, use_proprio=True, flip_images=True,
        use_variance_scale=False, use_jpeg_compression=True,
        num_denoising_steps_action=5, unnormalize_actions=True, normalize_proprio=True,
        dataset_stats_path=str(POLICY_DIR / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(POLICY_DIR / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
        num_open_loop_steps=16, ar_future_prediction=False,
        ar_value_prediction=False, ar_qvalue_prediction=False,
    )

    set_seed_everywhere(7)
    print("Loading model ...")
    model, _ = cosmos_utils.get_model(cfg)
    model.eval()
    device = next(model.parameters()).device
    print(f"Loaded: {len(model.net.blocks)} DiT blocks, device={device}")

    dataset_stats = load_dataset_stats(str(POLICY_DIR / "libero_dataset_statistics.json"))
    return model, dataset_stats, device


# ═══════════════════════════════════════════════════════════════════
# Video construction (for hidden state capture)
# ═══════════════════════════════════════════════════════════════════

def build_video_tensor(primary, wrist, *, T=T, device="cpu"):
    """Build the multi-slot video tensor for the Cosmos model.

    Constructs a 9-slot video where image slots are replicated T times
    along the temporal axis.  Slot 0 is a single zero frame.
    """
    images = [primary, wrist]
    processed = prepare_images_for_model(images, SimpleNamespace(
        use_jpeg_compression=True, use_variance_scale=False,
        flip_images=True, trained_with_image_aug=True,
        randomize_seed=False, num_third_person_images=1,
    ))
    wi, pi = processed[0], processed[1]
    blank = np.zeros_like(pi)

    wd = duplicate_array(wi, T)
    pd = duplicate_array(pi, T)
    bd = duplicate_array(blank, T)

    seq = [
        np.expand_dims(np.zeros_like(blank), axis=0),  # slot 0: first frame
        bd,                                              # slot 1: blank
        wd,                                              # slot 2: current wrist
        pd,                                              # slot 3: current primary
        bd,                                              # slot 4: action
        bd,                                              # slot 5: proprio
        wd.copy(),                                       # slot 6: future wrist
        pd.copy(),                                       # slot 7: future primary
        bd,                                              # slot 8: value
    ]
    raw = np.concatenate(seq, axis=0)          # (1+8*T, H, W, C)
    raw = np.expand_dims(raw, 0)                # (1, 1+8*T, H, W, C)
    raw = np.transpose(raw, (0, 4, 1, 2, 3))   # (1, C, 1+8*T, H, W)
    return torch.from_numpy(raw).to(dtype=torch.uint8, device=device)


# ═══════════════════════════════════════════════════════════════════
# Hidden state capture (sigma=80 single denoising step)
# ═══════════════════════════════════════════════════════════════════

class HiddenStateCapture:
    """Run one denoising step at sigma=80, capture VAE latent and DiT hidden states.

    Captures:
      - VAE latent: video slots (temporal idx 2,3) → [16, 2, 28, 28]
      - L0/L14/L27: video+action slots (temporal idx 2,3,4) → [3, 14, 14, 2048]
      - Action: x0 prediction from action slot → [112]
    """

    def __init__(self, model, primary, wrist, *, sigma_val=80.0, text_embedding, device):
        self.model = model
        self.sigma_val = sigma_val

        raw_video = build_video_tensor(primary, wrist, device=device)

        B = 1
        data_batch = {
            "dataset_name": "video_data",
            "video": raw_video,
            "t5_text_embeddings": text_embedding,
            "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
            "padding_mask": torch.zeros(B, 1, 256, 256, dtype=torch.bfloat16, device=device),
            "num_conditional_frames": model.config.min_num_conditional_frames,
            "current_wrist_image_latent_idx":  torch.tensor([SLOT_WRIST],   dtype=torch.int64, device=device),
            "current_image_latent_idx":         torch.tensor([SLOT_PRIMARY], dtype=torch.int64, device=device),
            "action_latent_idx":                torch.tensor([SLOT_ACTION],  dtype=torch.int64, device=device),
            "future_image_latent_idx":          torch.tensor([7],            dtype=torch.int64, device=device),
            "future_wrist_image_latent_idx":    torch.tensor([6],            dtype=torch.int64, device=device),
            "future_proprio_latent_idx":        torch.tensor([5],            dtype=torch.int64, device=device),
            "value_latent_idx":                 torch.tensor([8],            dtype=torch.int64, device=device),
        }
        for k in ("current_wrist_image2_latent_idx", "current_image2_latent_idx",
                  "future_wrist_image2_latent_idx", "future_image2_latent_idx",
                  "current_proprio_latent_idx"):
            data_batch[k] = torch.tensor([-1], dtype=torch.int64, device=device)

        raw_state, latent_state, cond_dict = model.get_data_and_condition(data_batch)
        self.latent_state = latent_state
        self.xt = latent_state + torch.randn_like(latent_state) * sigma_val
        self.cond_dict = cond_dict

        # Outputs populated by run()
        self.vae_latent = None    # [1, 16, 9, 28, 28]
        self.h_L0 = None          # [1, 9, 14, 14, 2048]
        self.h_L14 = None
        self.h_L27 = None
        self.action = None        # [1, 112]
        self._handles = []

    def _make_hook(self, name):
        def hook(_m, _in, out):
            x = out[0] if isinstance(out, tuple) else out
            setattr(self, name, x.detach().clone())
        return hook

    def run(self):
        """Execute one denoising step and capture VAE latent + L0/L14/L27."""
        blocks = self.model.net.blocks
        self._handles.append(blocks[0].register_forward_hook(self._make_hook("h_L0")))
        self._handles.append(blocks[14].register_forward_hook(self._make_hook("h_L14")))
        self._handles.append(blocks[27].register_forward_hook(self._make_hook("h_L27")))
        try:
            with torch.no_grad():
                sigma_t = torch.full((1, 1), self.sigma_val, device=self.xt.device, dtype=torch.float32)
                out = self.model.denoise(self.xt, sigma_t.squeeze(-1), self.cond_dict)
                # VAE latent: full tensor [1, 16, 9, 28, 28]
                self.vae_latent = self.latent_state.detach().cpu().numpy()
                # Action from action slot (temporal idx 4)
                al = out.x0[:, :, [SLOT_ACTION], :, :]
                self.action = al.reshape(1, -1)[:, :112].cpu().numpy()
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
        return self

    def get_snapshot(self):
        """Extract saved tensors into a snapshot dict with float16 precision.

        Both VAE latent and DiT hidden states use temporal-first layout [B, T, C, H, W/D].
        We select video slots [2, 3] (wrist, primary) plus action slot [4].
        """
        # VAE: temporal-first [1, 9, 16, 28, 28] → video slots → [2, 16, 28, 28]
        vae = self.vae_latent[0, [2, 3], :, :, :].astype(np.float16)

        # DiT layers: temporal-first [1, 9, 14, 14, 2048] → video+action slots → [3, 14, 14, 2048]
        def extract_layer(hidden):
            return hidden[0, CAPTURE_TEMPORAL_SLOTS, :, :, :].float().cpu().numpy().astype(np.float16)

        return {
            "VAE": vae,                                       # [16, 2, 28, 28] fp16
            "L0": extract_layer(self.h_L0),                   # [3, 14, 14, 2048] fp16
            "L14": extract_layer(self.h_L14),                 # [3, 14, 14, 2048] fp16
            "L27": extract_layer(self.h_L27),                 # [3, 14, 14, 2048] fp16
            "action": self.action[0].astype(np.float16),      # [112] fp16
        }


# ═══════════════════════════════════════════════════════════════════
# Task discovery
# ═══════════════════════════════════════════════════════════════════

def strip_perturbation_suffix(task_name):
    """Remove perturbation suffix to recover the base task name."""
    # Handle _language_N_view_... → strip _language_N first, then _view_...
    # The order: _language_ before _view_ before others
    for sep in ["_language_", "_light_", "_table_", "_tb_"]:
        if sep in task_name:
            return task_name.split(sep)[0]
    if "_view_" in task_name:
        return task_name.split("_view_")[0]
    m = re.match(r"(.+)_add_\d+$", task_name)
    if m:
        return m.group(1)
    m = re.match(r"(.+)_noise_\d+$", task_name)
    if m:
        return m.group(1)
    m = re.match(r"(.+)_level\d_sample\d$", task_name)
    if m:
        return m.group(1)
    return task_name


def get_base_tasks(suite):
    """Extract unique base task names from libero_task_map for a suite."""
    all_tasks = libero_task_map[suite]
    seen = set()
    base_tasks = []
    for t in all_tasks:
        base = strip_perturbation_suffix(t)
        if base not in seen:
            seen.add(base)
            base_tasks.append(base)
    print(f"  {suite}: {len(base_tasks)} unique base tasks (from {len(all_tasks)} total)")
    return base_tasks


def resolve_condition_task(suite, base_task, condition, task_index_in_suite):
    """Resolve a (suite, base_task, condition) to a Task object.

    For 'clean': construct Task from base_task.
    For perturbations: use LIBERO-plus benchmark to find a matching variant.
    Different base tasks get different variant indices (via task_index_in_suite).
    """
    if condition == "clean":
        return Task(
            name=base_task,
            language="",  # will be resolved from BDDL
            problem="Libero",
            problem_folder=suite,
            bddl_file=f"{base_task}.bddl",
            init_states_file=f"{base_task}.pruned_init",
        )

    category = PERTURBATION_CATEGORIES[condition]
    benchmark_dict = libero_benchmark_module.get_benchmark_dict()
    suite_class = benchmark_dict[suite]

    # Get all perturbation tasks for this suite+category
    bench = suite_class(task_order_index=0, category_value=category)

    # Group tasks by base_task
    matching = []
    for i in range(bench.n_tasks):
        task = bench.tasks[i]
        task_base = strip_perturbation_suffix(task.name)
        if task_base == base_task:
            matching.append((i, task))

    if not matching:
        raise ValueError(f"No {condition} variant found for {base_task} in {suite}")

    # Select variant: different per base task
    variant_idx = task_index_in_suite % len(matching)
    selected_id, selected_task = matching[variant_idx]

    # If only one variant or first is selected, log explicitly
    if len(matching) > 1:
        print(f"    [{condition}] {base_task}: using variant "
              f"{variant_idx+1}/{len(matching)} → {selected_task.name}")
    else:
        print(f"    [{condition}] {base_task}: using {selected_task.name}")

    return selected_task


def get_clean_task_language(suite, base_task_name):
    """Get language instruction for a clean/base task from its BDDL filename."""
    from libero.libero.benchmark import grab_language_from_filename
    return grab_language_from_filename(suite, f"{base_task_name}.bddl")


def get_t5_embedding_safe(text, *, allow_compute=True, device="cuda"):
    """Get T5 embedding, computing on-the-fly if missing from cache."""
    try:
        return get_t5_embedding_from_cache(text)
    except KeyError:
        if not allow_compute:
            raise RuntimeError(
                f"T5 embedding not found in cache for language instruction:\n"
                f"  {text}\n"
                f"Use --extra-t5-embeddings PATH with a pre-computed .pkl file."
            )
        print(f"    Computing T5 embedding for new instruction: {text[:80]}...")
        from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder
        encoder = CosmosT5TextEncoder(
            model_name=str(CKPT_ROOT / "t5-11b"),
            device=device,
            local_files_only=True,
            use_safetensors=False,
        )
        emb = encoder.encode_prompts(text).to(dtype=torch.bfloat16).cpu()
        # Add to cache for future lookups
        cosmos_utils.t5_text_embeddings_cache[text] = emb
        return emb.clone()


# ═══════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════

def prepare_obs(env_obs):
    """Extract and flip images from env observation. Returns (primary, wrist, proprio)."""
    primary = np.flipud(env_obs["agentview_image"])
    wrist = np.flipud(env_obs["robot0_eye_in_hand_image"])
    proprio = np.concatenate((
        env_obs["robot0_gripper_qpos"],
        env_obs["robot0_eef_pos"],
        env_obs["robot0_eef_quat"],
    ))
    return primary, wrist, proprio


def run_episode(*, suite, bench_task, task_condition, t5_emb,
                model, dataset_stats, save_interval, seed,
                deterministic_reset_seed, device):
    """Run a full policy rollout and return collected snapshots + metadata.

    Returns:
      snapshots: list of dicts, each with keys VAE, L0, L14, L27, action
      success: bool
      total_steps: int
    """
    # Create environment
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / suite / bench_task.bddl_file
    if not bddl_path.exists():
        # Try stripping perturbation suffix (for env-level perturbations)
        stripped_name = strip_perturbation_suffix(bench_task.name)
        bddl_path = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{stripped_name}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)

    try:
        # Deterministic reset
        set_seed_everywhere(deterministic_reset_seed)
        env.seed(0)
        env.reset()

        # Set init state
        init_states_path = _resolve_init_states_path(suite, bench_task)
        states = torch.load(str(init_states_path), weights_only=False)
        is_newobj = "_add_" in bench_task.name or "_level" in bench_task.name
        if is_newobj:
            states = states.reshape(1, -1)
        obs = env.set_init_state(states[0])

        # Stabilize (10 dummy steps)
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        max_steps = TASK_MAX_STEPS.get(suite, 300)
        action_queue = []
        snapshots = []
        step_count = 0
        success = False

        for t in range(10, max_steps + 10):
            # Deterministic seed per step
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                set_seed_everywhere(0)

            primary, wrist, proprio = prepare_obs(obs)

            # Capture hidden states at every N steps
            if step_count % save_interval == 0:
                cap = HiddenStateCapture(model, primary, wrist,
                                         sigma_val=SIGMA_VAL,
                                         text_embedding=t5_emb,
                                         device=device)
                cap.run()
                snapshots.append(cap.get_snapshot())

            # Refresh action queue
            if len(action_queue) == 0:
                obs_dict = {
                    "primary_image": primary,
                    "wrist_image": wrist,
                    "proprio": proprio,
                }
                result = get_action(
                    cfg=SimpleNamespace(
                        suite="libero",
                        use_third_person_image=True, num_third_person_images=1,
                        use_wrist_image=True, num_wrist_images=1,
                        use_proprio=True, normalize_proprio=True,
                        flip_images=True, use_variance_scale=False,
                        use_jpeg_compression=True, trained_with_image_aug=True,
                        randomize_seed=False, chunk_size=16,
                        unnormalize_actions=True,
                        ar_future_prediction=False, ar_value_prediction=False,
                        ar_qvalue_prediction=False,
                    ),
                    model=model, dataset_stats=dataset_stats,
                    obs=obs_dict,
                    task_label_or_embedding=t5_emb,
                    seed=seed,
                    randomize_seed=False,
                    num_denoising_steps_action=5,
                    generate_future_state_and_value_in_parallel=False,
                )
                action_queue.extend(result["actions"])

            # Execute action
            action = action_queue.pop(0)
            obs, reward, done, info = env.step(action.tolist())
            step_count += 1
            if done:
                success = True
                break

        return snapshots, success, step_count

    finally:
        env.close()


def _resolve_init_states_path(suite, bench_task):
    """Resolve the init states file path following LIBERO-plus benchmark logic."""
    from libero.libero.benchmark import grab_language_from_filename
    init_file = bench_task.init_states_file
    base_init_dir = pathlib.Path(get_libero_path("init_states"))

    # objects_layout uses libero_newobj directory
    if "_add_" in init_file or "_level" in init_file:
        p = base_init_dir / "libero_newobj" / suite / init_file
        if p.exists():
            return p

    # Strip perturbation suffixes to get base init file
    for sep in ["_language_", "_view_", "_table_", "_tb_", "_light_"]:
        if sep in init_file:
            base_name = init_file.split(sep)[0] + "." + init_file.split(".")[-1]
            p = base_init_dir / suite / base_name
            if p.exists():
                return p

    p = base_init_dir / suite / init_file
    if p.exists():
        return p

    raise FileNotFoundError(f"Cannot find init states for {bench_task.name}: tried {p}")


# ═══════════════════════════════════════════════════════════════════
# HDF5 writing
# ═══════════════════════════════════════════════════════════════════

def write_hdf5(output_path, snapshots, metadata):
    """Write collected snapshots to an HDF5 file."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        # Write metadata as attributes
        for k, v in metadata.items():
            f.attrs[k] = v

        # Write snapshot data
        data_group = f.create_group("data")
        for i, snap in enumerate(snapshots):
            step_idx = i * metadata["save_interval"]
            step_name = f"step_{step_idx:04d}"
            step_group = data_group.create_group(step_name)
            for key, tensor in snap.items():
                step_group.create_dataset(key, data=tensor)

    print(f"  Wrote {output_path}  ({len(snapshots)} snapshots)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collect multi-layer shift embeddings during LIBERO-Plus policy rollouts"
    )
    parser.add_argument("--suite", default=None,
                        help="Suite name (libero_10, libero_spatial, libero_object, libero_goal)")
    parser.add_argument("--task", default=None,
                        help="Base task name (for pilot mode)")
    parser.add_argument("--all-suites", action="store_true",
                        help="Run all 3 main suites (30 tasks × 8 conditions = 240 episodes). "
                        "libero_10 is reserved for pilot testing.")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save hidden states every N policy steps (default: 5)")
    parser.add_argument("--sigma", type=float, default=80.0,
                        help="Sigma value for the capture denoising step (default: 80.0)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Policy sampling seed (default: 7)")
    parser.add_argument("--deterministic-reset-seed", type=int, default=0,
                        help="Seed for deterministic env reset (default: 0)")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/phase8_shift_embeddings"),
                        help="Output root directory")
    parser.add_argument("--extra-t5-embeddings", default=None,
                        help="Path to extra T5 embeddings .pkl for language perturbations")
    parser.add_argument("--max-steps-override", type=int, default=None,
                        help="Override max steps per episode (for debugging)")
    args = parser.parse_args()

    global SIGMA_VAL
    SIGMA_VAL = args.sigma

    if args.max_steps_override is not None:
        for k in TASK_MAX_STEPS:
            TASK_MAX_STEPS[k] = args.max_steps_override

    # Validate inputs
    if not args.all_suites and (not args.suite or not args.task):
        parser.error("Either --all-suites or both --suite and --task must be specified")

    if args.all_suites:
        suites = ["libero_spatial", "libero_object", "libero_goal"]
    else:
        suites = [args.suite]

    # Conditions to run
    conditions = ["clean"] + list(PERTURBATION_CATEGORIES.keys())

    # Output directories
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_root = out_dir / "h5"

    # Save config
    config = {
        "suites": suites,
        "conditions": conditions,
        "save_interval": args.save_interval,
        "sigma": args.sigma,
        "seed": args.seed,
        "deterministic_reset_seed": args.deterministic_reset_seed,
        "output_dir": str(out_dir),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Load model
    model, dataset_stats, device = load_model_and_stats()

    # Load extra T5 embeddings if provided
    if args.extra_t5_embeddings:
        with open(args.extra_t5_embeddings, "rb") as f:
            extra_emb = pickle.load(f)
        cosmos_utils.t5_text_embeddings_cache.update(extra_emb)
        print(f"Loaded {len(extra_emb)} extra T5 embeddings from {args.extra_t5_embeddings}")

    # Run episodes
    total_episodes = 0
    total_successes = 0
    errors = []
    t_start = time.time()
    all_metadata_rows = []

    for suite in suites:
        print(f"\n{'='*60}")
        print(f"Suite: {suite}")
        print(f"{'='*60}")

        base_tasks = get_base_tasks(suite)

        for task_idx, base_task in enumerate(base_tasks):
            if args.task and base_task != args.task:
                continue

            print(f"\n--- [{task_idx+1}/{len(base_tasks)}] {base_task} ---")

            for condition in conditions:
                total_episodes += 1
                ep_label = f"{suite}/{base_task}/{condition}"
                print(f"  [{total_episodes}] {condition} ...", end=" ", flush=True)

                try:
                    # Resolve task
                    bench_task = resolve_condition_task(suite, base_task, condition, task_idx)

                    # Get language instruction
                    # Perturbation tasks: benchmark already populated task.language
                    # Clean task: use grab_language_from_filename from base BDDL
                    if condition == "language_instructions":
                        language = bench_task.language
                    else:
                        language = get_clean_task_language(suite, base_task)

                    # Get T5 embedding (compute on-the-fly if missing, e.g. language_instructions)
                    text_emb = get_t5_embedding_safe(language, allow_compute=True, device=str(device))
                    text_emb = text_emb.to(device=device, dtype=torch.bfloat16)

                    # Run episode
                    snapshots, success, total_steps = run_episode(
                        suite=suite,
                        bench_task=bench_task,
                        task_condition=condition,
                        t5_emb=text_emb,
                        model=model,
                        dataset_stats=dataset_stats,
                        save_interval=args.save_interval,
                        seed=args.seed,
                        deterministic_reset_seed=args.deterministic_reset_seed,
                        device=device,
                    )

                    if success:
                        total_successes += 1

                    # Write HDF5
                    h5_path = h5_root / suite / base_task / f"{base_task}__{condition}.h5"
                    metadata = {
                        "suite": suite,
                        "task_name": base_task,
                        "variant_name": bench_task.name,
                        "condition": condition,
                        "language": language,
                        "seed": args.seed,
                        "deterministic_reset_seed": args.deterministic_reset_seed,
                        "init_state_index": 0,
                        "success": success,
                        "total_steps": total_steps,
                        "num_snapshots": len(snapshots),
                        "save_interval": args.save_interval,
                    }
                    write_hdf5(h5_path, snapshots, metadata)

                    all_metadata_rows.append({
                        "suite": suite,
                        "task_name": base_task,
                        "variant_name": bench_task.name,
                        "condition": condition,
                        "h5_path": str(h5_path.relative_to(out_dir)),
                        "seed": args.seed,
                        "success": success,
                        "total_steps": total_steps,
                        "num_snapshots": len(snapshots),
                        "save_interval": args.save_interval,
                    })

                    print(f"✓  steps={total_steps}  snapshots={len(snapshots)}  success={success}")

                except Exception as e:
                    errors.append({
                        "suite": suite, "base_task": base_task, "condition": condition, "error": str(e),
                    })
                    print(f"✗  ERROR: {e}")
                    if len(errors) <= 3:
                        import traceback
                        traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s  |  {total_successes}/{total_episodes} success  |  {len(errors)} errors")
    if errors:
        print("Errors:")
        for e in errors[:10]:
            print(f"  [{e['suite']}] {e['base_task']}  {e['condition']}: {e['error'][:120]}")

    # Write manifest and summary
    if all_metadata_rows:
        import pandas as pd
        df = pd.DataFrame(all_metadata_rows)
        df.to_parquet(out_dir / "manifest.parquet", index=False)
        print(f"Manifest: {len(df)} rows → {out_dir / 'manifest.parquet'}")

    summary = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0,
        "num_errors": len(errors),
        "elapsed_seconds": elapsed,
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
