#!/usr/bin/env python3
"""Estimate denoising-step Hessian spectra for Cosmos Policy pairs.

This script keeps the Phase-1 preserved/flipped sample identity fixed and
measures local curvature along the actual denoising trajectory for each branch:

  clean branch:     noise -> clean-conditioned sample
  perturbed branch: noise -> perturbed-conditioned sample

It does not interpolate clean -> perturb and does not do any intervention.

By default, the scalar loss is an EDM-style x0 reconstruction loss evaluated at
a captured denoising call, using the branch's own final generated latent sample
as the target. `--target-source clean_final_action` instead uses the successful
clean branch's final generated action as the shared action-space target.
`--target-source expert_action` uses a shared LeRobot demonstration action chunk
target:

    L = MSE(extract_action(pred_x0), expert_action_chunk)

The expert-action mode is an explicit approximation: Phase-1 eval episode ids
are mapped to the same-task demonstration order in the LeRobot zip unless a
stricter mapping is added later.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import pathlib
import site
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy-matplotlib")
os.environ.setdefault("DETERMINISTIC", "True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", str(ROOT.parent / "LIBERO-plus")))
for item in (str(LIBERO_PLUS), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)


def preload_wand_imagemagick() -> None:
    """Load Wand before PIL/torchvision can load incompatible bundled libjpeg."""
    lib_dir = pathlib.Path(sys.prefix) / "lib"
    if (lib_dir / "libMagickWand-7.Q16HDRI.so").exists():
        os.environ.setdefault("MAGICK_HOME", sys.prefix)
        os.environ.setdefault("WAND_MAGICK_LIBRARY_SUFFIX", "-7.Q16HDRI")
    try:
        from wand.api import library as _wand_library  # noqa: F401
    except ImportError:
        return


preload_wand_imagemagick()

import numpy as np
import torch

from hessian_shift_alignment import ritz_shift_alignment

from cosmos_policy._src.imaginaire.functional.multi_step import is_multi_step_fn_supported
from cosmos_policy._src.imaginaire.functional.runge_kutta import is_runge_kutta_fn_supported
from cosmos_policy._src.imaginaire.modules.res_sampler import (
    SamplerConfig,
    SolverConfig,
    SolverTimestampConfig,
    differential_equation_solver,
    get_rev_ts,
)
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE,
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    extract_action_chunk_from_latent_sequence,
    get_model,
    get_t5_embedding_from_cache,
    load_dataset_stats,
    prepare_images_for_model,
    rescale_proprio,
)
from cosmos_policy.utils.utils import duplicate_array, set_seed_everywhere

CHECKPOINT_ROOT = ROOT.parent.parent / "checkpoints"
POLICY_DIR = CHECKPOINT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
EPS = 1e-12


DETAIL_FIELDS = [
    "condition",
    "episode",
    "group",
    "branch",
    "branch_success",
    "denoise_call_index",
    "denoise_num_calls",
    "sigma",
    "site",
    "layer",
    "target_source",
    "loss_frames",
    "expert_action_scale",
    "expert_demo_episode",
    "expert_demo_ordinal",
    "expert_start_frame",
    "loss_value",
    "lambda_max",
    "lambda_min",
    "lambda_abs_max",
    "hvp_mode",
    "fd_eps",
    "lanczos_iters",
    "hvp_dim",
    "clean_task_name",
    "pert_task_name",
    "instruction_mode",
]

DELTA_FIELDS = [
    "condition",
    "episode",
    "group",
    "denoise_call_index",
    "denoise_num_calls",
    "sigma_clean",
    "sigma_perturbed",
    "site",
    "layer",
    "target_source",
    "loss_frames",
    "expert_action_scale",
    "hvp_mode",
    "fd_eps",
    "lambda_clean",
    "lambda_perturbed",
    "delta_lambda",
    "ratio_lambda",
    "loss_clean",
    "loss_perturbed",
    "delta_loss",
]

ALIGNMENT_FIELDS = [
    "condition",
    "episode",
    "group",
    "anchor_branch",
    "denoise_call_index",
    "denoise_num_calls",
    "sigma",
    "site",
    "layer",
    "target_source",
    "hvp_mode",
    "fd_eps",
    "lambda_max",
    "hvp_dim",
    "shift_l2",
    "shift_rms",
    "alignment_valid",
    "alignment_k",
    "abs_cos_top1",
    "max_abs_cos_topk",
    "alignment_energy_topk",
    "alignment_rms_topk",
    "random_energy_topk",
    "abs_cosines_topk",
    "action_error_from_final_latent",
    "pert_action_mse_to_clean_final",
]


@dataclass
class CapturedCall:
    call_index: int
    sigma: float
    x_in: torch.Tensor
    hidden: dict[int, torch.Tensor]


@dataclass
class BranchResult:
    final_sample: torch.Tensor
    condition_target: torch.Tensor
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    captures: dict[int, CapturedCall]
    data_batch: dict[str, Any]


@dataclass
class ExpertActionTarget:
    raw_action: np.ndarray
    normalized_action: np.ndarray
    task_text: str
    task_index: int
    demo_episode: int
    demo_ordinal: int
    start_frame: int
    episode_length: int


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase2_helpers():
    from run_phase2_angular_cosmos import (
        discover_pairs,
        first_observation,
        load_extra_t5,
        make_cfg,
        patch_checkpoint_db,
    )

    return discover_pairs, first_observation, load_extra_t5, make_cfg, patch_checkpoint_db


def parse_layers(spec: list[str], n_layers: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in spec:
        if item in {"vae", "input"}:
            continue
        if item in {"0", "block0", "layer0"}:
            out["layer0"] = 0
        elif item in {"mid", "middle"}:
            out["mid"] = n_layers // 2
        elif item in {"last", "final"}:
            out["last"] = n_layers - 1
        else:
            layer = int(item)
            out[f"layer{layer}"] = layer
    return {name: layer for name, layer in out.items() if 0 <= layer < n_layers}


def parse_sites(spec: list[str], layer_map: dict[str, int]) -> list[str]:
    if spec == ["all"]:
        return ["vae", *layer_map.keys()]
    sites = []
    for item in spec:
        if item in {"vae", "input"}:
            sites.append("vae")
        elif item in {"0", "block0", "layer0"}:
            sites.append("layer0")
        elif item in {"mid", "middle"}:
            sites.append("mid")
        elif item in {"last", "final"}:
            sites.append("last")
        elif item.startswith("layer"):
            sites.append(item)
        else:
            sites.append(f"layer{int(item)}")
    out = []
    for site in sites:
        if site == "vae" or site in layer_map:
            out.append(site)
    return sorted(set(out), key=lambda x: (-1 if x == "vae" else layer_map[x]))


def select_call_indices(num_calls: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted({idx for idx in explicit if 0 <= idx < num_calls})
    if num_calls <= 1:
        return [0]
    return sorted({int(math.floor((num_calls - 1) * q / 4.0 + 0.5)) for q in range(5)})


def build_inference_data_batch(
    cfg: SimpleNamespace,
    model: torch.nn.Module,
    dataset_stats: dict[str, Any],
    obs: dict[str, Any],
    task_label_or_embedding: Any,
    device: torch.device,
    batch_size: int = 1,
) -> dict[str, Any]:
    if cfg.suite != "libero":
        raise ValueError(f"This script currently implements LIBERO only, got suite={cfg.suite}")

    if isinstance(task_label_or_embedding, str):
        text_embedding = get_t5_embedding_from_cache(task_label_or_embedding)
    elif isinstance(task_label_or_embedding, np.ndarray):
        text_embedding = torch.tensor(task_label_or_embedding, dtype=torch.bfloat16, device=device)
    elif torch.is_tensor(task_label_or_embedding):
        text_embedding = task_label_or_embedding.to(device=device, dtype=torch.bfloat16)
    else:
        raise TypeError(f"Unsupported task label/embedding type: {type(task_label_or_embedding)!r}")

    all_camera_images = [obs["wrist_image"], obs["primary_image"]]
    wrist_image_idx, image_idx = 0, 1
    all_camera_images = prepare_images_for_model(all_camera_images, cfg)

    proprio = None
    if cfg.use_proprio:
        proprio = obs["proprio"]
        if cfg.normalize_proprio:
            proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0)

    primary_image = all_camera_images[image_idx]
    wrist_image = all_camera_images[wrist_image_idx]
    blank_image = np.zeros_like(primary_image)
    blank_image_duplicated = duplicate_array(blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    wrist_image_duplicated = duplicate_array(wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_image_duplicated = duplicate_array(primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)

    image_sequence = []
    current_sequence_idx = 0
    image_sequence.append(np.expand_dims(np.zeros_like(blank_image), axis=0))
    current_sequence_idx += 1

    image_sequence.append(blank_image_duplicated.copy())
    current_proprio_latent_idx = current_sequence_idx
    current_sequence_idx += 1

    image_sequence.append(wrist_image_duplicated.copy())
    current_wrist_image_latent_idx = current_sequence_idx
    current_sequence_idx += 1
    current_wrist_image2_latent_idx = -1

    image_sequence.append(primary_image_duplicated.copy())
    current_image_latent_idx = current_sequence_idx
    current_sequence_idx += 1
    current_image2_latent_idx = -1

    image_sequence.append(blank_image_duplicated.copy())
    action_latent_idx = current_sequence_idx
    current_sequence_idx += 1

    image_sequence.append(blank_image_duplicated.copy())
    future_proprio_latent_idx = current_sequence_idx
    current_sequence_idx += 1

    image_sequence.append(wrist_image_duplicated.copy())
    future_wrist_image_latent_idx = current_sequence_idx
    current_sequence_idx += 1
    future_wrist_image2_latent_idx = -1

    image_sequence.append(primary_image_duplicated.copy())
    future_image_latent_idx = current_sequence_idx
    current_sequence_idx += 1
    future_image2_latent_idx = -1

    image_sequence.append(blank_image_duplicated.copy())
    value_latent_idx = current_sequence_idx

    raw_image_sequence = np.concatenate(image_sequence, axis=0)
    raw_image_sequence = np.expand_dims(raw_image_sequence, axis=0)
    raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))
    raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))
    raw_image_sequence_t = torch.from_numpy(raw_image_sequence).to(device=device, dtype=torch.uint8)

    proprio_tensor = None
    if cfg.use_proprio:
        proprio_tensor = torch.from_numpy(proprio).reshape(batch_size, -1).to(device=device, dtype=torch.bfloat16)

    def idx_tensor(value: int) -> torch.Tensor:
        return torch.tensor([value] * batch_size, dtype=torch.int64, device=device)

    return {
        "dataset_name": "video_data",
        "video": raw_image_sequence_t,
        "t5_text_embeddings": text_embedding.repeat(batch_size, 1, 1).to(device=device, dtype=torch.bfloat16),
        "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros((batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16, device=device),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio_tensor,
        "current_proprio_latent_idx": idx_tensor(current_proprio_latent_idx),
        "current_wrist_image_latent_idx": idx_tensor(current_wrist_image_latent_idx),
        "current_wrist_image2_latent_idx": idx_tensor(current_wrist_image2_latent_idx),
        "current_image_latent_idx": idx_tensor(current_image_latent_idx),
        "current_image2_latent_idx": idx_tensor(current_image2_latent_idx),
        "action_latent_idx": idx_tensor(action_latent_idx),
        "future_proprio_latent_idx": idx_tensor(future_proprio_latent_idx),
        "future_wrist_image_latent_idx": idx_tensor(future_wrist_image_latent_idx),
        "future_wrist_image2_latent_idx": idx_tensor(future_wrist_image2_latent_idx),
        "future_image_latent_idx": idx_tensor(future_image_latent_idx),
        "future_image2_latent_idx": idx_tensor(future_image2_latent_idx),
        "value_latent_idx": idx_tensor(value_latent_idx),
    }


def state_shape_from_batch(model: torch.nn.Module, data_batch: dict[str, Any]) -> tuple[int, int, int, int]:
    _, _, frames, height, width = data_batch["video"].shape
    return (
        model.config.state_ch,
        model.tokenizer.get_latent_num_frames(frames),
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    )


def initial_noise(
    model: torch.nn.Module,
    data_batch: dict[str, Any],
    seed: int,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(data_batch["video"].shape[0])
    shape = (batch_size,) + state_shape_from_batch(model, data_batch)
    return misc.arch_invariant_rand(shape, torch.float32, device, seed) * sigma_max


def normalize_actions_np(actions: np.ndarray, dataset_stats: dict[str, Any]) -> np.ndarray:
    actions_min = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
    actions_max = np.asarray(dataset_stats["actions_max"], dtype=np.float32)
    return (2.0 * ((actions.astype(np.float32) - actions_min) / (actions_max - actions_min)) - 1.0).astype(np.float32)


def unnormalize_actions_np(actions: np.ndarray, dataset_stats: dict[str, Any]) -> np.ndarray:
    actions_min = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
    actions_max = np.asarray(dataset_stats["actions_max"], dtype=np.float32)
    return (0.5 * (actions.astype(np.float32) + 1.0) * (actions_max - actions_min) + actions_min).astype(np.float32)


def unnormalize_actions_torch(actions: torch.Tensor, dataset_stats: dict[str, Any]) -> torch.Tensor:
    actions_min = torch.as_tensor(dataset_stats["actions_min"], device=actions.device, dtype=actions.dtype)
    actions_max = torch.as_tensor(dataset_stats["actions_max"], device=actions.device, dtype=actions.dtype)
    return 0.5 * (actions + 1.0) * (actions_max - actions_min) + actions_min


def action_chunk_with_padding(actions: np.ndarray, start_frame: int, chunk_size: int) -> np.ndarray:
    if start_frame < 0:
        raise ValueError(f"expert_start_frame must be non-negative, got {start_frame}")
    if start_frame >= len(actions):
        raise ValueError(f"expert_start_frame={start_frame} is outside episode length {len(actions)}")
    remaining = len(actions) - start_frame
    if remaining >= chunk_size:
        return actions[start_frame : start_frame + chunk_size].astype(np.float32)
    available = actions[start_frame:].astype(np.float32)
    padding = np.tile(actions[-1].astype(np.float32), (chunk_size - remaining, 1))
    return np.concatenate([available, padding], axis=0)


class ExpertActionStore:
    """Read same-task LeRobot expert action chunks directly from a zip archive."""

    def __init__(self, zip_path: pathlib.Path):
        if not zip_path.exists():
            raise FileNotFoundError(f"Expert action zip not found: {zip_path}")
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(zip_path)
        self.prefix = self._find_prefix()
        self.info = json.loads(self.zip_file.read(self.prefix + "meta/info.json").decode("utf-8"))
        self.tasks = [json.loads(line) for line in self.zip_file.read(self.prefix + "meta/tasks.jsonl").decode("utf-8").splitlines()]
        self.episodes = [
            json.loads(line) for line in self.zip_file.read(self.prefix + "meta/episodes.jsonl").decode("utf-8").splitlines()
        ]
        self.task_by_text = {item["task"]: item for item in self.tasks}
        self.task_by_index = {int(item["task_index"]): item for item in self.tasks}
        self.episodes_by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in self.episodes:
            if not episode.get("tasks"):
                continue
            self.episodes_by_task.setdefault(episode["tasks"][0], []).append(episode)
        for episodes in self.episodes_by_task.values():
            episodes.sort(key=lambda item: int(item["episode_index"]))
        self._actions_cache: dict[int, np.ndarray] = {}
        self._target_cache: dict[tuple[str, int, int, int, int], ExpertActionTarget] = {}

    def _find_prefix(self) -> str:
        for name in self.zip_file.namelist():
            if name.endswith("meta/info.json"):
                return name[: -len("meta/info.json")]
        raise RuntimeError(f"Could not find meta/info.json inside {self.zip_path}")

    def _episode_parquet_path(self, episode_index: int) -> str:
        chunks_size = int(self.info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size
        rel = self.info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        return self.prefix + rel.format(episode_chunk=episode_chunk, episode_index=episode_index)

    def _read_actions(self, episode_index: int) -> np.ndarray:
        if episode_index in self._actions_cache:
            return self._actions_cache[episode_index]
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading LeRobot parquet expert actions requires pyarrow in the active Python env.") from exc

        path = self._episode_parquet_path(episode_index)
        table = pq.read_table(io.BytesIO(self.zip_file.read(path)), columns=["action"])
        action_col = table.column("action").combine_chunks()
        actions = action_col.values.to_numpy(zero_copy_only=False).reshape(len(action_col), ACTION_DIM).astype(np.float32)
        self._actions_cache[episode_index] = actions
        return actions

    def resolve_task(self, task_text: str, task_index: int | None = None) -> dict[str, Any]:
        if task_index is not None:
            if int(task_index) not in self.task_by_index:
                raise KeyError(f"Expert task_index {task_index} not found in {self.zip_path}")
            return self.task_by_index[int(task_index)]
        if task_text in self.task_by_text:
            return self.task_by_text[task_text]
        available = "\n  ".join(sorted(self.task_by_text))
        raise KeyError(f"Expert task text not found: {task_text!r}\nAvailable tasks:\n  {available}")

    def get_target(
        self,
        phase_episode: int,
        task_text: str,
        task_index: int | None,
        demo_offset: int,
        start_frame: int,
        chunk_size: int,
        dataset_stats: dict[str, Any],
    ) -> ExpertActionTarget:
        task = self.resolve_task(task_text, task_index)
        resolved_task_text = task["task"]
        cache_key = (resolved_task_text, int(phase_episode), int(demo_offset), int(start_frame), int(chunk_size))
        if cache_key in self._target_cache:
            return self._target_cache[cache_key]
        demos = self.episodes_by_task.get(resolved_task_text, [])
        if not demos:
            raise RuntimeError(f"No expert episodes found for task {resolved_task_text!r}")
        demo_ordinal = int(phase_episode) + int(demo_offset)
        if not (0 <= demo_ordinal < len(demos)):
            raise IndexError(
                f"Phase episode {phase_episode} with demo_offset {demo_offset} maps to demo ordinal {demo_ordinal}, "
                f"but task {resolved_task_text!r} only has {len(demos)} demos"
            )
        demo = demos[demo_ordinal]
        demo_episode = int(demo["episode_index"])
        actions = self._read_actions(demo_episode)
        raw_chunk = action_chunk_with_padding(actions, start_frame, chunk_size)
        target = ExpertActionTarget(
            raw_action=raw_chunk,
            normalized_action=normalize_actions_np(raw_chunk, dataset_stats),
            task_text=resolved_task_text,
            task_index=int(task["task_index"]),
            demo_episode=demo_episode,
            demo_ordinal=demo_ordinal,
            start_frame=int(start_frame),
            episode_length=int(len(actions)),
        )
        self._target_cache[cache_key] = target
        return target


class DenoiseCapture:
    def __init__(self, model: torch.nn.Module, selected_calls: set[int], layers: dict[str, int]):
        self.selected_calls = selected_calls
        self.layers = set(layers.values())
        self.current_call_index = -1
        self.current_sigma = float("nan")
        self.captures: dict[int, CapturedCall] = {}
        self.handles = []
        for layer in sorted(self.layers):
            self.handles.append(model.net.blocks[layer].register_forward_hook(self._hook(layer)))

    def start_call(self, call_index: int, sigma: torch.Tensor, x_in: torch.Tensor) -> None:
        self.current_call_index = call_index
        sigma_value = float(sigma.reshape(-1)[0].detach().cpu())
        self.current_sigma = sigma_value
        if call_index in self.selected_calls:
            self.captures[call_index] = CapturedCall(
                call_index=call_index,
                sigma=sigma_value,
                x_in=x_in.detach().float().cpu(),
                hidden={},
            )

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self.current_call_index not in self.selected_calls:
                return
            x = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(x):
                return
            self.captures[self.current_call_index].hidden[layer] = x.detach().float().cpu()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def make_solver_cfg(solver_option: str) -> SolverConfig:
    is_multistep = is_multi_step_fn_supported(solver_option)
    is_rk = is_runge_kutta_fn_supported(solver_option)
    if not (is_multistep or is_rk):
        raise ValueError(f"Unsupported solver option: {solver_option}")
    return SolverConfig(
        s_churn=0,
        s_t_max=float("inf"),
        s_t_min=0,
        s_noise=1,
        is_multi=is_multistep,
        rk=solver_option,
        multistep=solver_option,
    )


def freeze_model_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def run_branch_trajectory(
    model: torch.nn.Module,
    data_batch: dict[str, Any],
    x_sigma_max: torch.Tensor,
    selected_calls: set[int],
    layers: dict[str, int],
    args,
) -> BranchResult:
    x0_fn, condition_target = model.get_x0_fn_from_batch(
        data_batch,
        guidance=args.guidance,
        is_negative_prompt=False,
        return_orig_clean_latent_frames=True,
    )

    in_dtype = x_sigma_max.dtype
    solver_steps = args.num_denoising_steps - 1 if args.num_denoising_steps > 1 else 1
    sigmas_l = get_rev_ts(args.sigma_min, args.sigma_max, solver_steps, args.rho).to(x_sigma_max.device)
    solver_cfg = make_solver_cfg(args.solver_option)
    timestamps_cfg = SolverTimestampConfig(nfe=solver_steps, t_min=args.sigma_min, t_max=args.sigma_max, order=args.rho)
    sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True)

    capture = DenoiseCapture(model, selected_calls, layers)
    call_index = {"value": -1}

    def recorded_x0_fn(x_state: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
        call_index["value"] += 1
        capture.start_call(call_index["value"], sigma_b, x_state)
        return x0_fn(x_state.to(in_dtype), sigma_b.to(in_dtype)).to(torch.float64)

    try:
        with torch.no_grad():
            if args.num_denoising_steps > 1:
                denoised = differential_equation_solver(
                    recorded_x0_fn,
                    sigmas_l,
                    sampler_cfg.solver,
                    callback_fns=None,
                )(x_sigma_max.to(torch.float64))
                ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
                final_sample = recorded_x0_fn(denoised, sigmas_l[-1] * ones)
            else:
                ones = torch.ones(x_sigma_max.size(0), device=x_sigma_max.device, dtype=torch.float64)
                final_sample = recorded_x0_fn(x_sigma_max.to(torch.float64), sigmas_l[0] * ones)
    finally:
        capture.close()

    return BranchResult(
        final_sample=final_sample.detach().float().cpu(),
        condition_target=condition_target.detach().float().cpu(),
        x0_fn=x0_fn,
        captures=capture.captures,
        data_batch=data_batch,
    )


def loss_value(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    data_batch: dict[str, Any],
    loss_frames: str,
) -> torch.Tensor:
    target = target_x0.to(device=pred_x0.device, dtype=pred_x0.dtype)
    diff = pred_x0 - target
    if loss_frames == "action":
        idx = data_batch["action_latent_idx"].to(device=pred_x0.device)
        batch = torch.arange(pred_x0.shape[0], device=pred_x0.device)
        diff = diff[batch, :, idx, :, :]
    elif loss_frames == "non_conditioned":
        mask = torch.ones((pred_x0.shape[0], pred_x0.shape[2]), device=pred_x0.device, dtype=pred_x0.dtype)
        for key in (
            "current_proprio_latent_idx",
            "current_wrist_image_latent_idx",
            "current_wrist_image2_latent_idx",
            "current_image_latent_idx",
            "current_image2_latent_idx",
        ):
            idx = data_batch[key].to(device=pred_x0.device)
            valid = idx >= 0
            if bool(valid.any()):
                mask[torch.arange(pred_x0.shape[0], device=pred_x0.device)[valid], idx[valid]] = 0
        diff = diff * mask[:, None, :, None, None]
    elif loss_frames != "all":
        raise ValueError(f"Unsupported loss frame mode: {loss_frames}")
    return (diff.float() ** 2).mean()


@contextmanager
def replace_block_output(block: torch.nn.Module, replacement: torch.Tensor):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (replacement, *output[1:])
        return replacement

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def make_loss_fn(
    model: torch.nn.Module,
    branch: BranchResult,
    capture: CapturedCall,
    site: str,
    layer: int | None,
    target: torch.Tensor,
    dataset_stats: dict[str, Any],
    args,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    device = torch.device(args.device)
    sigma = torch.full((capture.x_in.shape[0],), capture.sigma, device=device, dtype=torch.float32)
    x_in = capture.x_in.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)

    def weighted_loss(pred: torch.Tensor) -> torch.Tensor:
        if args.target_source in {"expert_action", "clean_final_action"}:
            action_scale = "normalized" if args.target_source == "clean_final_action" else args.expert_action_scale
            value = expert_action_mse_value(
                pred,
                target,
                branch.data_batch,
                args.chunk_size,
                action_scale,
                dataset_stats,
            )
        else:
            value = loss_value(pred, target, branch.data_batch, args.loss_frames)
        if args.edm_loss_weights:
            sigma_bt = sigma.reshape(-1, 1)
            value = value * model.get_per_sigma_loss_weights(sigma=sigma_bt).float().mean()
        return value

    if site == "vae":
        x0 = x_in

        def fn(x_var: torch.Tensor) -> torch.Tensor:
            pred = branch.x0_fn(x_var, sigma).float()
            return weighted_loss(pred)

        return fn, x0

    if layer is None:
        raise ValueError(f"Layer site {site} has no layer")
    if layer not in capture.hidden:
        raise RuntimeError(f"Missing hidden capture for layer {layer} at call {capture.call_index}")
    h0 = capture.hidden[layer].to(device=device, dtype=torch.float32)

    def fn(h_var: torch.Tensor) -> torch.Tensor:
        replacement = h_var
        if args.hidden_replacement_dtype == "model":
            replacement = replacement.to(dtype=next(model.parameters()).dtype)
        with replace_block_output(model.net.blocks[layer], replacement):
            pred = branch.x0_fn(x_in.detach(), sigma).float()
        return weighted_loss(pred)

    return fn, h0


def flatten_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.linalg.vector_norm(x) + EPS)


def captured_site_tensor(capture: CapturedCall, site: str, layer: int | None) -> torch.Tensor:
    if site == "vae":
        return capture.x_in
    if layer is None or layer not in capture.hidden:
        raise RuntimeError(f"Missing hidden capture for site={site} layer={layer} call={capture.call_index}")
    return capture.hidden[layer]


def hvp(loss_fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    x = x0.detach().clone().requires_grad_(True)
    loss = loss_fn(x)
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    grad_dot_v = flatten_dot(grad, vector)
    hv = torch.autograd.grad(grad_dot_v, x, retain_graph=False)[0]
    hv_detached = hv.detach()
    loss_value = float(loss.detach().cpu())
    del x, loss, grad, grad_dot_v, hv
    return hv_detached, loss_value


def grad_at(loss_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, float]:
    x_var = x.detach().clone().requires_grad_(True)
    loss = loss_fn(x_var)
    grad = torch.autograd.grad(loss, x_var, create_graph=False, retain_graph=False)[0]
    grad_detached = grad.detach()
    loss_value = float(loss.detach().cpu())
    del x_var, loss, grad
    return grad_detached, loss_value


def finite_difference_hvp(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    vector: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, float]:
    g_pos, loss_pos = grad_at(loss_fn, x0 + eps * vector)
    g_neg, loss_neg = grad_at(loss_fn, x0 - eps * vector)
    hv = (g_pos - g_neg) / (2.0 * eps)
    loss_value = 0.5 * (loss_pos + loss_neg)
    del g_pos, g_neg
    return hv.detach(), loss_value


def lanczos_extreme_eigs(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    iters: int,
    seed: int,
    hvp_mode: str,
    fd_eps: float,
    shift: torch.Tensor | None = None,
    alignment_top_k: int = 5,
) -> dict[str, Any]:
    generator = torch.Generator(device=x0.device)
    generator.manual_seed(seed)
    q = torch.randn(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
    q = l2_normalize(q)
    q_prev = torch.zeros_like(q)
    beta_prev = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    basis: list[torch.Tensor] = []
    last_loss = float("nan")

    for idx in range(iters):
        if hvp_mode == "autograd":
            z, last_loss = hvp(loss_fn, x0, q)
        elif hvp_mode == "finite_diff":
            z, last_loss = finite_difference_hvp(loss_fn, x0, q, fd_eps)
        else:
            raise ValueError(f"Unsupported HVP mode: {hvp_mode}")
        if idx > 0:
            z = z - beta_prev * q_prev
        alpha = float(flatten_dot(q, z).detach().cpu())
        z = z - alpha * q
        for prev_q in basis:
            z = z - flatten_dot(prev_q, z) * prev_q
        beta = float(torch.linalg.vector_norm(z).detach().cpu())
        alphas.append(alpha)
        basis.append(q.detach())
        if idx < iters - 1:
            betas.append(beta)
        if beta <= 1e-10:
            del z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            break
        q_prev = q
        q = z / beta
        beta_prev = beta
        del z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    tri = torch.zeros((len(alphas), len(alphas)), dtype=torch.float64)
    for i, alpha in enumerate(alphas):
        tri[i, i] = alpha
    for i, beta in enumerate(betas[: max(0, len(alphas) - 1)]):
        tri[i, i + 1] = beta
        tri[i + 1, i] = beta
    eigs, tri_eigvecs = torch.linalg.eigh(tri)
    lambda_min = float(eigs[0])
    lambda_max = float(eigs[-1])
    lambda_abs = float(eigs[torch.argmax(torch.abs(eigs))])
    result = {
        "loss_value": last_loss,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "lambda_abs_max": lambda_abs,
        "hvp_mode": hvp_mode,
        "fd_eps": fd_eps if hvp_mode == "finite_diff" else "",
        "lanczos_iters": len(alphas),
        "hvp_dim": int(x0.numel()),
    }
    if shift is not None:
        result.update(ritz_shift_alignment(basis, tri_eigvecs, shift, alignment_top_k))
    return result


def action_from_latent(sample: torch.Tensor, data_batch: dict[str, Any], chunk_size: int) -> np.ndarray:
    action_idx = data_batch["action_latent_idx"].to(device=sample.device)
    action = extract_action_chunk_from_latent_sequence(
        sample,
        action_shape=(chunk_size, ACTION_DIM),
        action_indices=action_idx,
    )
    return action.detach().float().cpu().numpy()


def action_tensor_from_latent(sample: torch.Tensor, data_batch: dict[str, Any], chunk_size: int) -> torch.Tensor:
    action_idx = data_batch["action_latent_idx"].to(device=sample.device)
    return extract_action_chunk_from_latent_sequence(
        sample,
        action_shape=(chunk_size, ACTION_DIM),
        action_indices=action_idx,
    ).to(torch.float32)


def expert_target_array(target: ExpertActionTarget, scale: str) -> np.ndarray:
    if scale == "normalized":
        return target.normalized_action
    if scale == "raw":
        return target.raw_action
    raise ValueError(f"Unsupported expert action scale: {scale}")


def expert_action_mse_value(
    pred_x0: torch.Tensor,
    target_action: torch.Tensor,
    data_batch: dict[str, Any],
    chunk_size: int,
    expert_action_scale: str,
    dataset_stats: dict[str, Any],
) -> torch.Tensor:
    pred_action = action_tensor_from_latent(pred_x0, data_batch, chunk_size)
    if expert_action_scale == "raw":
        pred_action = unnormalize_actions_torch(pred_action, dataset_stats)
    elif expert_action_scale != "normalized":
        raise ValueError(f"Unsupported expert action scale: {expert_action_scale}")
    target = target_action.to(device=pred_action.device, dtype=pred_action.dtype)
    return ((pred_action - target) ** 2).mean()


def target_for_branch(branch: BranchResult, target_source: str) -> torch.Tensor:
    if target_source == "final_sample":
        return branch.final_sample
    if target_source == "condition_latent":
        return branch.condition_target
    raise ValueError(f"Unsupported target source: {target_source}")


def finite_ratio(num: float, den: float) -> float:
    if math.isnan(num) or math.isnan(den) or abs(den) <= EPS:
        return float("nan")
    return num / den


def write_csv(path: pathlib.Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_delta_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in detail_rows:
        key = (
            row["condition"],
            row["episode"],
            row["group"],
            row["denoise_call_index"],
            row["site"],
            row["layer"],
            row["target_source"],
            row["loss_frames"],
            row.get("expert_action_scale", ""),
            row.get("hvp_mode", ""),
            row.get("fd_eps", ""),
        )
        keyed.setdefault(key, {})[row["branch"]] = row

    out = []
    for key, branches in sorted(keyed.items(), key=lambda item: item[0]):
        clean = branches.get("clean")
        pert = branches.get("perturbed")
        if clean is None or pert is None:
            continue
        lambda_clean = float(clean["lambda_max"])
        lambda_pert = float(pert["lambda_max"])
        loss_clean = float(clean["loss_value"])
        loss_pert = float(pert["loss_value"])
        out.append({
            "condition": key[0],
            "episode": key[1],
            "group": key[2],
            "denoise_call_index": key[3],
            "denoise_num_calls": clean["denoise_num_calls"],
            "sigma_clean": clean["sigma"],
            "sigma_perturbed": pert["sigma"],
            "site": key[4],
            "layer": key[5],
            "target_source": key[6],
            "loss_frames": key[7],
            "expert_action_scale": key[8],
            "hvp_mode": key[9],
            "fd_eps": key[10],
            "lambda_clean": lambda_clean,
            "lambda_perturbed": lambda_pert,
            "delta_lambda": lambda_pert - lambda_clean,
            "ratio_lambda": finite_ratio(lambda_pert, lambda_clean),
            "loss_clean": loss_clean,
            "loss_perturbed": loss_pert,
            "delta_loss": loss_pert - loss_clean,
        })
    return out


def main() -> None:
    args = build_parser().parse_args()
    if args.target_source in {"expert_action", "clean_final_action"} and args.loss_frames != "action":
        raise ValueError(f"--target-source {args.target_source} requires --loss-frames action")
    if args.hvp_mode == "finite_diff" and args.fd_eps <= 0:
        raise ValueError("--fd-eps must be positive for --hvp-mode finite_diff")
    if args.alignment_top_k <= 0:
        raise ValueError("--alignment-top-k must be positive")
    discover_pairs, first_observation, load_extra_t5, make_cfg, patch_checkpoint_db = load_phase2_helpers()
    start = time.time()
    args.policy_dir = pathlib.Path(args.policy_dir)
    patch_checkpoint_db(args.policy_dir)

    summary_path = pathlib.Path(args.summary)
    summary, pairs = discover_pairs(summary_path, set(args.conditions or []), set(args.groups))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"Discovered {len(pairs)} Phase-1 pairs from {summary_path}", flush=True)
    if not pairs:
        return

    cfg = make_cfg(args)
    cfg.num_denoising_steps_action = args.num_denoising_steps
    args.chunk_size = cfg.chunk_size
    set_seed_everywhere(args.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    expert_store = None
    if args.target_source == "expert_action":
        expert_store = ExpertActionStore(pathlib.Path(args.expert_actions_zip))
        print(f"Loaded expert action store: {args.expert_actions_zip}", flush=True)
    model, _ = get_model(cfg)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    freeze_model_parameters(model)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters frozen for Hessian input/hidden analysis; trainable_params={trainable_params}", flush=True)

    layer_map = parse_layers(args.layers, len(model.net.blocks))
    sites = parse_sites(args.sites, layer_map)
    selected_calls = select_call_indices(args.num_denoising_steps, args.step_indices)
    print(f"Sites: {sites}", flush=True)
    print(f"Layer map: {layer_map}", flush=True)
    print(f"Selected denoise calls: {selected_calls} / num_calls={args.num_denoising_steps}", flush=True)

    out_root = pathlib.Path(args.output_dir)
    detail_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(pairs, 1):
        clean_ep = pair["clean"]
        pert_ep = pair["pert"]
        condition = pair["condition"]
        group = pair["group"]
        episode = int(pert_ep["episode"])
        print(f"[{pair_idx}/{len(pairs)}] {condition}/ep{episode:02d} {group}", flush=True)

        clean_obs = first_observation(clean_ep, args)
        pert_obs = clean_obs if condition == "language_instructions" else first_observation(pert_ep, args)
        clean_instr = clean_ep["language"]
        pert_instr = pert_ep["language"] if pert_ep.get("instruction_mode") == "strict" else clean_instr
        expert_target = None
        expert_target_tensor = None
        if expert_store is not None:
            expert_task_text = args.expert_task or clean_instr
            expert_target = expert_store.get_target(
                phase_episode=episode,
                task_text=expert_task_text,
                task_index=args.expert_task_index,
                demo_offset=args.expert_demo_offset,
                start_frame=args.expert_start_frame,
                chunk_size=cfg.chunk_size,
                dataset_stats=stats,
            )
            expert_target_tensor = torch.from_numpy(expert_target_array(expert_target, args.expert_action_scale)).unsqueeze(0)

        clean_batch = build_inference_data_batch(cfg, model, stats, clean_obs, clean_instr, device)
        pert_batch = build_inference_data_batch(cfg, model, stats, pert_obs, pert_instr, device)
        noise_seed = args.seed * 100000 + episode if args.shared_noise else args.seed * 100000 + pair_idx
        x_sigma_clean = initial_noise(model, clean_batch, noise_seed, args.sigma_max, device)
        x_sigma_pert = x_sigma_clean.clone() if args.shared_noise else initial_noise(model, pert_batch, noise_seed + 17, args.sigma_max, device)

        branch_results = {
            "clean": run_branch_trajectory(model, clean_batch, x_sigma_clean, set(selected_calls), layer_map, args),
            "perturbed": run_branch_trajectory(model, pert_batch, x_sigma_pert, set(selected_calls), layer_map, args),
        }

        clean_action = action_from_latent(branch_results["clean"].final_sample.to(device), clean_batch, cfg.chunk_size)
        pert_action = action_from_latent(branch_results["perturbed"].final_sample.to(device), pert_batch, cfg.chunk_size)
        clean_final_action_target_tensor = torch.from_numpy(clean_action.astype(np.float32))
        action_error = float(np.linalg.norm(pert_action.reshape(-1) - clean_action.reshape(-1)) / (np.linalg.norm(clean_action.reshape(-1)) + EPS))
        pair_record = {
            "condition": condition,
            "episode": episode,
            "group": group,
            "action_error_from_final_latent": action_error,
            "clean_success": bool(clean_ep["success"]),
            "pert_success": bool(pert_ep["success"]),
        }
        if expert_target is not None:
            target_np = expert_target_array(expert_target, args.expert_action_scale)[None, ...]
            clean_cmp = clean_action if args.expert_action_scale == "normalized" else unnormalize_actions_np(clean_action, stats)
            pert_cmp = pert_action if args.expert_action_scale == "normalized" else unnormalize_actions_np(pert_action, stats)
            pair_record.update({
                "expert_task_text": expert_target.task_text,
                "expert_task_index": expert_target.task_index,
                "expert_demo_episode": expert_target.demo_episode,
                "expert_demo_ordinal": expert_target.demo_ordinal,
                "expert_start_frame": expert_target.start_frame,
                "expert_episode_length": expert_target.episode_length,
                "expert_action_scale": args.expert_action_scale,
                "clean_action_mse_to_expert": float(np.mean((clean_cmp - target_np) ** 2)),
                "pert_action_mse_to_expert": float(np.mean((pert_cmp - target_np) ** 2)),
            })
        if args.target_source == "clean_final_action":
            pair_record.update({
                "clean_final_action_scale": "normalized",
                "clean_action_mse_to_clean_final": 0.0,
                "pert_action_mse_to_clean_final": float(np.mean((pert_action - clean_action) ** 2)),
            })
        pair_records.append(pair_record)

        for branch_name, branch in branch_results.items():
            if args.target_source == "expert_action":
                target = expert_target_tensor
            elif args.target_source == "clean_final_action":
                target = clean_final_action_target_tensor
            else:
                target = target_for_branch(branch, args.target_source)
            if target is None:
                raise RuntimeError(f"{args.target_source} target requested but no target was loaded")
            branch_ep = clean_ep if branch_name == "clean" else pert_ep
            for call_idx in selected_calls:
                capture = branch.captures.get(call_idx)
                if capture is None:
                    print(f"  missing capture branch={branch_name} call={call_idx}; skipping", flush=True)
                    continue
                for site in sites:
                    layer = None if site == "vae" else layer_map[site]
                    clean_capture = branch_results["clean"].captures.get(call_idx)
                    pert_capture = branch_results["perturbed"].captures.get(call_idx)
                    if clean_capture is None or pert_capture is None:
                        continue
                    clean_site = captured_site_tensor(clean_capture, site, layer)
                    pert_site = captured_site_tensor(pert_capture, site, layer)
                    if clean_site.shape != pert_site.shape:
                        raise RuntimeError(
                            f"Shift shape mismatch at call={call_idx} site={site}: "
                            f"clean={tuple(clean_site.shape)} perturbed={tuple(pert_site.shape)}"
                        )
                    shift = pert_site - clean_site
                    print(f"  Hessian branch={branch_name} call={call_idx} site={site}", flush=True)
                    loss_fn, x0 = make_loss_fn(model, branch, capture, site, layer, target, stats, args)
                    spec = lanczos_extreme_eigs(
                        loss_fn,
                        x0,
                        args.lanczos_iters,
                        args.seed + call_idx + (layer or 0),
                        args.hvp_mode,
                        args.fd_eps,
                        shift=shift,
                        alignment_top_k=args.alignment_top_k,
                    )
                    detail_rows.append({
                        "condition": condition,
                        "episode": episode,
                        "group": group,
                        "branch": branch_name,
                        "branch_success": bool(branch_ep["success"]),
                        "denoise_call_index": call_idx,
                        "denoise_num_calls": args.num_denoising_steps,
                        "sigma": capture.sigma,
                        "site": site,
                        "layer": "" if layer is None else layer,
                        "target_source": args.target_source,
                        "loss_frames": args.loss_frames,
                        "expert_action_scale": (
                            args.expert_action_scale
                            if expert_target is not None
                            else ("normalized" if args.target_source == "clean_final_action" else "")
                        ),
                        "expert_demo_episode": "" if expert_target is None else expert_target.demo_episode,
                        "expert_demo_ordinal": "" if expert_target is None else expert_target.demo_ordinal,
                        "expert_start_frame": "" if expert_target is None else expert_target.start_frame,
                        "clean_task_name": clean_ep["task_name"],
                        "pert_task_name": pert_ep["task_name"],
                        "instruction_mode": pert_ep.get("instruction_mode", "task"),
                        **spec,
                    })
                    alignment_rows.append({
                        "condition": condition,
                        "episode": episode,
                        "group": group,
                        "anchor_branch": branch_name,
                        "denoise_call_index": call_idx,
                        "denoise_num_calls": args.num_denoising_steps,
                        "sigma": capture.sigma,
                        "site": site,
                        "layer": "" if layer is None else layer,
                        "target_source": args.target_source,
                        "hvp_mode": args.hvp_mode,
                        "fd_eps": args.fd_eps if args.hvp_mode == "finite_diff" else "",
                        "lambda_max": spec["lambda_max"],
                        "hvp_dim": spec["hvp_dim"],
                        "action_error_from_final_latent": action_error,
                        "pert_action_mse_to_clean_final": (
                            float(np.mean((pert_action - clean_action) ** 2))
                            if args.target_source == "clean_final_action"
                            else ""
                        ),
                        **{field: spec[field] for field in ALIGNMENT_FIELDS if field in spec},
                    })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        write_csv(out_root / "denoise_hessian_detail.csv", DETAIL_FIELDS, detail_rows)
        write_csv(out_root / "denoise_hessian_delta.csv", DELTA_FIELDS, build_delta_rows(detail_rows))
        write_csv(out_root / "hessian_shift_alignment.csv", ALIGNMENT_FIELDS, alignment_rows)

    summary_payload = {
        "summary": str(summary_path),
        "suite": summary.get("suite"),
        "base_task": summary.get("base_task"),
        "num_pairs": len(pairs),
        "num_denoising_steps": args.num_denoising_steps,
        "selected_calls": selected_calls,
        "sites": sites,
        "layer_map": layer_map,
        "target_source": args.target_source,
        "loss_frames": args.loss_frames,
        "expert_actions_zip": args.expert_actions_zip if args.target_source == "expert_action" else "",
        "expert_task": args.expert_task or "",
        "expert_task_index": args.expert_task_index,
        "expert_episode_mapping": "demo_ordinal = phase_episode + expert_demo_offset",
        "expert_demo_offset": args.expert_demo_offset,
        "expert_start_frame": args.expert_start_frame,
        "expert_action_scale": args.expert_action_scale if args.target_source == "expert_action" else "",
        "clean_final_action_scale": "normalized" if args.target_source == "clean_final_action" else "",
        "expert_target_assumption": (
            "clean and perturbed branches share the same same-task LeRobot base demonstration action chunk"
            if args.target_source == "expert_action"
            else ""
        ),
        "clean_final_action_target_assumption": (
            "clean and perturbed branches share the successful clean branch's final generated action chunk"
            if args.target_source == "clean_final_action"
            else ""
        ),
        "edm_loss_weights": bool(args.edm_loss_weights),
        "lanczos_iters": args.lanczos_iters,
        "alignment_top_k": args.alignment_top_k,
        "hvp_mode": args.hvp_mode,
        "fd_eps": args.fd_eps if args.hvp_mode == "finite_diff" else "",
        "shared_noise": bool(args.shared_noise),
        "elapsed_s": time.time() - start,
        "pairs": pair_records,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_csv(out_root / "denoise_hessian_detail.csv", DETAIL_FIELDS, detail_rows)
    write_csv(out_root / "denoise_hessian_delta.csv", DELTA_FIELDS, build_delta_rows(detail_rows))
    write_csv(out_root / "hessian_shift_alignment.csv", ALIGNMENT_FIELDS, alignment_rows)
    print(f"Saved {out_root / 'denoise_hessian_detail.csv'}", flush=True)
    print(f"Saved {out_root / 'denoise_hessian_delta.csv'}", flush=True)
    print(f"Saved {out_root / 'hessian_shift_alignment.csv'}", flush=True)
    print(f"Saved {out_root / 'summary.json'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_summary = ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json"
    parser.add_argument("--summary", default=str(default_summary))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/phase3_denoise_hessian/kitchen_scene4_seed7"))
    parser.add_argument("--policy-dir", default=str(POLICY_DIR))
    parser.add_argument("--t5-extra-embeddings", default=str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=["preserved", "flipped"])
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-seed", type=int, default=0)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-denoising-steps", type=int, default=50)
    parser.add_argument("--step-indices", nargs="*", type=int, default=None)
    parser.add_argument("--layers", nargs="+", default=["0", "mid", "last"])
    parser.add_argument("--sites", nargs="+", default=["vae", "0", "mid", "last"])
    parser.add_argument("--lanczos-iters", type=int, default=8)
    parser.add_argument("--alignment-top-k", type=int, default=5)
    parser.add_argument("--hvp-mode", choices=["autograd", "finite_diff"], default="autograd")
    parser.add_argument("--fd-eps", type=float, default=1e-2)
    parser.add_argument(
        "--target-source",
        choices=["final_sample", "condition_latent", "expert_action", "clean_final_action"],
        default="final_sample",
    )
    parser.add_argument("--loss-frames", choices=["action", "non_conditioned", "all"], default="action")
    parser.add_argument("--expert-actions-zip", default=str(ROOT.parent.parent / "libero_plus_10.zip"))
    parser.add_argument("--expert-task", default="")
    parser.add_argument("--expert-task-index", type=int, default=None)
    parser.add_argument("--expert-demo-offset", type=int, default=0)
    parser.add_argument("--expert-start-frame", type=int, default=0)
    parser.add_argument("--expert-action-scale", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--edm-loss-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-replacement-dtype", choices=["model", "float32"], default="model")
    parser.add_argument("--shared-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--solver-option", default="2ab")
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    return parser


if __name__ == "__main__":
    main()
