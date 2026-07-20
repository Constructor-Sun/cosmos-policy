#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECKPOINT_ROOT=$(CDPATH= cd -- "$REPO_ROOT/../../checkpoints" && pwd)
BASE_MODEL_DIR="$CHECKPOINT_ROOT/Cosmos-Predict2-2B-Video2World"
POLICY_MODEL_DIR="$CHECKPOINT_ROOT/Cosmos-Policy-LIBERO-Predict2-2B"
HF_REPO_DIR="$CHECKPOINT_ROOT/huggingface-hub/models--nvidia--Cosmos-Predict2-2B-Video2World"
HF_REVISION="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"

EXPECTED_PREFIX=/data2/haoze/miniconda3/envs/cosmospolicy
PYTHON_BIN=${PYTHON_BIN:-$EXPECTED_PREFIX/bin/python}
GPU_ID=${GPU_ID:-0}
SMOKE_SUITE=${SMOKE_SUITE:-libero_spatial}
SMOKE_TASK_ID=${SMOKE_TASK_ID:-0}
FLIP_IMAGES=${FLIP_IMAGES:-true}
SMOKE_SEED=${SMOKE_SEED:-7}
SMOKE_DETERMINISTIC_RESET=${SMOKE_DETERMINISTIC_RESET:-true}
SMOKE_DETERMINISTIC_RESET_SEED=${SMOKE_DETERMINISTIC_RESET_SEED:-0}
SMOKE_MODE=${SMOKE_MODE:-paired}
SMOKE_NUM_PAIRS=${SMOKE_NUM_PAIRS:-20}
SMOKE_RESULTS_DIR=${SMOKE_RESULTS_DIR:-./experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_20case}
SMOKE_RUN_ID=${SMOKE_RUN_ID:-vla_jepa_kitchen_scene4_seed7_20case}

# VLA-JEPA LIBERO-plus task used for paired clean/perturb diagnostics.
SMOKE_PAIR_SUITE=${SMOKE_PAIR_SUITE:-libero_10}
SMOKE_PAIR_BASE_TASK=${SMOKE_PAIR_BASE_TASK:-KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it}
SMOKE_PAIR_CLEAN_LANGUAGE=${SMOKE_PAIR_CLEAN_LANGUAGE:-put the black bowl in the bottom drawer of the cabinet and close it}
SMOKE_PAIR_PERT_NAME=${SMOKE_PAIR_PERT_NAME:-camera_viewpoints,background_textures,light_conditions,objects_layout,robot_initial_states,sensor_noise}
SMOKE_PAIR_PERT_CATEGORY=${SMOKE_PAIR_PERT_CATEGORY:-Camera Viewpoints}
SMOKE_PAIR_PERT_TASK=${SMOKE_PAIR_PERT_TASK:-${SMOKE_PAIR_BASE_TASK}_view_50_0_100_0_0_initstate_0}
SMOKE_T5_FALLBACK_TO_BASE=${SMOKE_T5_FALLBACK_TO_BASE:-true}
SMOKE_T5_EXTRA_EMBEDDINGS=${SMOKE_T5_EXTRA_EMBEDDINGS:-}
SMOKE_T5_ALLOW_STRICT_COMPUTE=${SMOKE_T5_ALLOW_STRICT_COMPUTE:-false}
SMOKE_HF_HUB_OFFLINE=${SMOKE_HF_HUB_OFFLINE:-1}

if [ ! -f "$BASE_MODEL_DIR/model-480p-16fps.pt" ]; then
    echo "Missing base checkpoint: $BASE_MODEL_DIR/model-480p-16fps.pt" >&2
    exit 1
fi

if [ ! -f "$BASE_MODEL_DIR/tokenizer/tokenizer.pth" ]; then
    echo "Missing tokenizer checkpoint: $BASE_MODEL_DIR/tokenizer/tokenizer.pth" >&2
    exit 1
fi

if [ ! -f "$POLICY_MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" ]; then
    echo "Missing policy checkpoint: $POLICY_MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" >&2
    exit 1
fi

PYTHON_PREFIX=$($PYTHON_BIN -c 'import sys; print(sys.prefix)')
if [ "$PYTHON_PREFIX" != "$EXPECTED_PREFIX" ]; then
    echo "Expected cosmospolicy Python at $EXPECTED_PREFIX, got $PYTHON_PREFIX" >&2
    echo "Activate cosmospolicy first, or set PYTHON_BIN explicitly." >&2
    exit 1
fi

# Populate the Hugging Face cache layout expected by the hard-coded base-model URI.
mkdir -p "$HF_REPO_DIR/refs" "$HF_REPO_DIR/snapshots"
printf '%s' "$HF_REVISION" > "$HF_REPO_DIR/refs/main"
ln -sfnT "$BASE_MODEL_DIR" "$HF_REPO_DIR/snapshots/$HF_REVISION"

mkdir -p "${TMPDIR:-/tmp}/cosmospolicy-numba" "${TMPDIR:-/tmp}/cosmospolicy-matplotlib"

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="../LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/cosmospolicy-numba"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cosmospolicy-matplotlib"
export HF_HUB_CACHE="$CHECKPOINT_ROOT/huggingface-hub"
export HF_HUB_OFFLINE="$SMOKE_HF_HUB_OFFLINE"
export PYTHONHASHSEED="$SMOKE_SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false

export COSMOS_SMOKE_SUITE="$SMOKE_SUITE"
export COSMOS_SMOKE_TASK_ID="$SMOKE_TASK_ID"
export COSMOS_SMOKE_FLIP_IMAGES="$FLIP_IMAGES"
export COSMOS_POLICY_MODEL_DIR="$POLICY_MODEL_DIR"
export COSMOS_SMOKE_MODE="$SMOKE_MODE"
export COSMOS_SMOKE_NUM_PAIRS="$SMOKE_NUM_PAIRS"
export COSMOS_SMOKE_RESULTS_DIR="$SMOKE_RESULTS_DIR"
export COSMOS_SMOKE_RUN_ID="$SMOKE_RUN_ID"
export COSMOS_SMOKE_SEED="$SMOKE_SEED"
export COSMOS_SMOKE_DETERMINISTIC_RESET="$SMOKE_DETERMINISTIC_RESET"
export COSMOS_SMOKE_DETERMINISTIC_RESET_SEED="$SMOKE_DETERMINISTIC_RESET_SEED"
export COSMOS_SMOKE_PAIR_SUITE="$SMOKE_PAIR_SUITE"
export COSMOS_SMOKE_PAIR_BASE_TASK="$SMOKE_PAIR_BASE_TASK"
export COSMOS_SMOKE_PAIR_CLEAN_LANGUAGE="$SMOKE_PAIR_CLEAN_LANGUAGE"
export COSMOS_SMOKE_PAIR_PERT_NAME="$SMOKE_PAIR_PERT_NAME"
export COSMOS_SMOKE_PAIR_PERT_CATEGORY="$SMOKE_PAIR_PERT_CATEGORY"
export COSMOS_SMOKE_PAIR_PERT_TASK="$SMOKE_PAIR_PERT_TASK"
export COSMOS_SMOKE_T5_FALLBACK_TO_BASE="$SMOKE_T5_FALLBACK_TO_BASE"
export COSMOS_SMOKE_T5_EXTRA_EMBEDDINGS="$SMOKE_T5_EXTRA_EMBEDDINGS"
export COSMOS_SMOKE_T5_ALLOW_STRICT_COMPUTE="$SMOKE_T5_ALLOW_STRICT_COMPUTE"

"$PYTHON_BIN" - <<'PY'
import gc
import json
import os
import pathlib
import pickle
import re
import time
import builtins

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Task


mode = os.environ["COSMOS_SMOKE_MODE"]
flip_images = os.environ["COSMOS_SMOKE_FLIP_IMAGES"].lower() in {"1", "true", "yes"}
policy_dir = os.environ["COSMOS_POLICY_MODEL_DIR"]

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


def make_cfg(suite_name, num_trials, run_id_note, local_log_dir="./experiments/logs"):
    unnorm_key = "libero_10" if suite_name == "libero_mix" else suite_name
    deterministic_reset = os.environ["COSMOS_SMOKE_DETERMINISTIC_RESET"].lower() in {"1", "true", "yes"}
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=os.path.join(policy_dir, "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
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
            if len(states) >= num_trials:
                return states
            if isinstance(states, torch.Tensor):
                return torch.stack([states[idx % len(states)] for idx in range(num_trials)], dim=0)
            return [states[idx % len(states)] for idx in range(num_trials)]

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

    summary = {
        "mode": "paired",
        "run_id": run_id,
        "suite": pair_suite,
        "seed": seed,
        "deterministic_reset": deterministic_reset,
        "deterministic_reset_seed": deterministic_reset_seed,
        "num_pairs": num_pairs,
        "base_task": base_task,
        "conditions": [
            {
                "condition": "clean",
                "task_name": base_task,
                "category": "clean",
                "seed": seed,
                "success_rate": clean_rate,
                "successes": int(round(clean_rate * num_pairs)),
                "num_trials": num_pairs,
                "log_path": clean_log,
                "episodes_path": str(clean_dir / "episodes.json"),
            },
        ],
    }

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

    summary_name = pert_name if "," not in pert_name else "selected"
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

PY
