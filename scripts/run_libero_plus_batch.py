#!/usr/bin/env python3
"""Evaluate CosmosPolicy on the seven LIBERO-plus LIBERO-10 categories.

Run with the CosmosPolicy Python environment from the repository root:
    python scripts/run_libero_plus_batch.py --dry-run
    python scripts/run_libero_plus_batch.py --validate-only
    python scripts/run_libero_plus_batch.py --gpus 0 1 2 3 4 5 6 7

One persistent model per GPU; categories run in order, tasks within each category
are dispatched as GPUs become free. Videos and episode HDF5/actions are saved by
default. Use --resume for interrupted runs and --retry-errors to retry runtime
errors. Task failures are completed trials and are never automatically retried.
Existing evaluation source files are imported, not modified.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


CATEGORY_KEYS = (
    "camera_viewpoints", "robot_initial_states", "language_instructions",
    "light_conditions", "background_textures", "sensor_noise", "objects_layout",
)
SOURCE_FILES = (
    "scripts/run_libero_smoke_test.sh",
    "cosmos_policy/experiments/robot/libero/run_libero_eval.py",
    "cosmos_policy/experiments/robot/libero/libero_utils.py",
    "cosmos_policy/experiments/robot/cosmos_utils.py",
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result_path(config, row):
    return Path(config["output_root"]) / row["group"] / "episodes" / f"task{row['task_id']:04d}" / "result.json"


def load_plan(args):
    root = args.repo_root.resolve()
    manifest_dir = (args.manifest_dir or root / "experiments/libero_plus/task_lists/libero_10").resolve()
    index = read_json(manifest_dir / "index.json")
    groups, seen, names = [], set(), set()
    for item in index["groups"]:
        key = Path(item["file"]).stem
        if key not in CATEGORY_KEYS:
            raise ValueError(f"Unknown category file: {item['file']}")
        path = manifest_dir / item["file"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Task manifest checksum changed: {path}")
        data = read_json(path)
        if (data["suite"] != "libero_10" or data["category"] != item["category"]
                or data["num_trials_per_task"] != 1 or data["init_state_index"] != 0
                or len(data["tasks"]) != item["num_tasks"]):
            raise ValueError(f"Invalid task manifest: {path}")
        rows = []
        for raw in data["tasks"]:
            row = dict(raw, group=key)
            if row["task_id"] in seen or row["task_name"] in names or row["category"] != item["category"]:
                raise ValueError(f"Duplicate task or invalid category: {row}")
            seen.add(row["task_id"])
            names.add(row["task_name"])
            rows.append(row)
        if args.categories is None or key in args.categories:
            groups.append(dict(key=key, category=item["category"], tasks=rows[:args.limit_per_category] if args.limit_per_category else rows))
    if len(groups) != len(set(g["key"] for g in groups)) or len(seen) != index["total_tasks"] or seen != set(range(index["total_tasks"])):
        raise ValueError("Task manifests do not cover the suite exactly once")
    if len(index["groups"]) != 7 or index["total_tasks"] != 2519:
        raise ValueError("Expected the pinned seven-category, 2519-task LIBERO-10 release")
    task_ids_file = getattr(args, "task_ids_file", None)
    if task_ids_file is not None:
        requested = read_json(task_ids_file)
        requested_ids = requested["task_ids"] if isinstance(requested, dict) else requested
        if (not isinstance(requested_ids, list) or not requested_ids
                or any(type(task_id) is not int or task_id < 0 for task_id in requested_ids)
                or len(set(requested_ids)) != len(requested_ids)):
            raise ValueError("Task selection must contain distinct nonnegative integer task IDs")
        available = {row["task_id"] for group in groups for row in group["tasks"]}
        if not set(requested_ids).issubset(available):
            raise ValueError("Selected task IDs are outside the requested categories or task limits")
        requested_ids = set(requested_ids)
        groups = [dict(group, tasks=[row for row in group["tasks"] if row["task_id"] in requested_ids])
                  for group in groups]
        groups = [group for group in groups if group["tasks"]]
    model_dir = (args.policy_dir or root.parents[1] / "checkpoints/Cosmos-Policy-LIBERO-Predict2-2B").resolve()
    ckpt = (args.ckpt or model_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt").resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    config = dict(
        version=1, repo_root=str(root), manifest_dir=str(manifest_dir),
        manifest_index_sha256=digest(manifest_dir / "index.json"),
        reference_commit=index["reference_commit"], source=index["source"],
        source_sha256=index["source_sha256"],
        output_root=str((args.output_root or root / "experiments/libero_plus/cosmospolicy_full_7dims").resolve()),
        policy_dir=str(model_dir), ckpt_path=str(ckpt),
        checkpoint_size=ckpt.stat().st_size, checkpoint_mtime_ns=ckpt.stat().st_mtime_ns,
        extra_t5=str((args.extra_t5 or root / "experiments/cache/libero_plus_language_t5_libero10.pkl").resolve()),
        base_tasks=str(root / "configs/libero10_experiment_tasks.json"),
        action_seed=args.action_seed, environment_seed=args.environment_seed,
        settle_steps=args.settle_steps, max_steps=args.max_steps,
        rng_mode="openwam", num_trials_per_task=1, init_state_index=0,
        save_episode_data=args.save_episode_data, groups=groups,
        upstream_sha256={name: digest(root / name) for name in SOURCE_FILES},
    )
    if getattr(args, "cuda_memory_limit_gib", None) is not None:
        config["cuda_memory_limit_gib"] = args.cuda_memory_limit_gib
    for name in ("extra_t5", "base_tasks"):
        config[name + "_sha256"] = digest(config[name])
    for name in ("libero_dataset_statistics.json", "libero_t5_embeddings.pkl"):
        config[name + "_sha256"] = digest(model_dir / name)
    if task_ids_file is not None:
        config["task_ids_file"] = str(task_ids_file.resolve())
        config["task_ids_file_sha256"] = digest(task_ids_file)
    identity = dict(config)
    identity.pop("output_root")
    config["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return config


def instruction_key(task, row, base_tasks):
    if row["group"] == "language_instructions":
        return task.language
    matches = [base["language"] for base in base_tasks if task.name.startswith(base["name"] + "_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve base instruction for {task.name}")
    return matches[0]


def configure_sensor_dependencies():
    """Keep compatibility fixes local to this evaluation process."""
    import numpy as np
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not os.environ.get("MAGICK_HOME"):
        envs = Path(sys.executable).resolve().parents[2]
        for name in ("libero_base", "vlajepa"):
            candidate = envs / name
            if list((candidate / "lib").glob("libMagickWand*.so")):
                os.environ["MAGICK_HOME"] = str(candidate)
                break


def load_catalog(config, check_states=True):
    # CPU preflight and GPU workers resolve the very same LIBERO package.
    from libero.libero import benchmark, get_libero_path
    with contextlib.redirect_stdout(io.StringIO()):
        suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
    if suite.n_tasks != 2519:
        raise ValueError(f"Expected LIBERO-plus with 2519 tasks, got {suite.n_tasks}")
    by_name = {task.name: i for i, task in enumerate(suite.tasks)}
    if len(by_name) != suite.n_tasks:
        raise ValueError("Duplicate task names in the installed suite")
    source = Path(benchmark.__file__).parent / "task_classification.json"
    if digest(source) != config["source_sha256"]:
        raise ValueError(f"LIBERO-plus classification differs from the pinned task manifests: {source}")
    base_tasks = read_json(config["base_tasks"])["tasks"]
    with (Path(config["policy_dir"]) / "libero_t5_embeddings.pkl").open("rb") as handle:
        cache = pickle.load(handle)
    with Path(config["extra_t5"]).open("rb") as handle:
        cache.update(pickle.load(handle))
    checked = 0
    for group in config["groups"]:
        for row in group["tasks"]:
            if by_name.get(row["task_name"]) != row["task_id"]:
                raise ValueError(f"Task ID/name mismatch: {row['task_id']} {row['task_name']}")
            task = suite.get_task(row["task_id"])
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            # LIBERO-plus's environment wrapper decodes camera/init/noise
            # suffixes before opening the physical BDDL file.
            if "_view_" in str(bddl) and "_initstate_" in str(bddl):
                bddl = Path(str(bddl).split("_view_", 1)[0] + ".bddl")
            if not bddl.is_file():
                raise FileNotFoundError(bddl)
            if check_states:
                states = suite.get_task_init_states(row["task_id"])
                if len(states.shape) != 2 or states.shape[0] < 1:
                    raise ValueError(f"Invalid initial states for {task.name}: {states.shape}")
            key = instruction_key(task, row, base_tasks)
            if key not in cache:
                raise ValueError(f"Missing exact T5 embedding for {task.name}: {key!r}")
            checked += 1
    return suite, base_tasks, cache, dict(
        checked_tasks=checked, benchmark_module=str(benchmark.__file__),
        initial_states="init_states[0]", t5_coverage="complete", source_sha256=digest(source),
    )


def worker_env(config, gpu, mode, config_path):
    env = os.environ.copy()
    env.update(
        GPU_ID=str(gpu), SMOKE_PYTHON_SCRIPT=str(Path(config["repo_root"]) / "scripts/run_libero_plus_batch.py"),
        LIBERO_PLUS_BATCH_MODE=mode, LIBERO_PLUS_BATCH_CONFIG=str(config_path),
        SMOKE_POLICY_CKPT_PATH=config["ckpt_path"], SMOKE_ACTION_SEED=str(config["action_seed"]),
        SMOKE_ENVIRONMENT_SEED=str(config["environment_seed"]), SMOKE_RNG_MODE="openwam",
        COSMOS_LIBERO_ROOT=str(Path(config["repo_root"]).parent / "LIBERO-plus"),
        COSMOS_SKILL_COMPLETION_SHADOW="0", COSMOS_SKILL_COMPLETION_ACTIVE="0",
        SMOKE_DATA_COLLECTION="true" if config["save_episode_data"] else "false",
        COSMOS_TTA_REPAIR="", COSMOS_TTA_RGBD="0", COSMOS_TTA_PHASE_LOOP="0",
        COSMOS_INIT_STATE_OFFSET="0",
        COSMOS_INITIAL_ALIGNMENT="0", COSMOS_VECTOR_DB="0", COSMOS_HELD_OBJECT="0",
        COSMOS_SKIP_PLAIN_ROLLOUT="0", PYTHONUNBUFFERED="1",
    )
    env.pop("COSMOS_DUMP_WRIST_DIR", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def validate(config, config_path):
    wrapper = Path(config["repo_root"]) / "scripts/run_libero_smoke_test.sh"
    completed = subprocess.run(
        ["sh", str(wrapper)], cwd=config["repo_root"],
        env=worker_env(config, -1, "validate", config_path), check=False,
    )
    if completed.returncode:
        raise RuntimeError("CPU validation failed; no evaluation workers were started")


class EpisodeLog:
    """Also detect exceptions that the existing episode runner logs and catches."""

    def __init__(self, handle):
        self.handle = handle
        self.errors = []

    def write(self, message):
        if "Episode error:" in message:
            self.errors.append(message)
        return self.handle.write(message)

    def flush(self):
        self.handle.flush()


class RecordedEnv:
    """Add settling without changing run_episode's local 10-step wait constant."""

    def __init__(self, env, extra_settle, native_wait):
        self.env = env
        self.extra_settle = extra_settle
        self.native_wait = native_wait
        self.initial_state = None
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        result = self.env.step(action)
        self.actions.append(list(action))
        return result

    def set_init_state(self, state):
        import numpy as np
        self.initial_state = np.asarray(state).copy()
        obs = self.env.set_init_state(state)
        for _ in range(self.extra_settle):
            obs, _, _, _ = self.step([0, 0, 0, 0, 0, 0, -1])
        return obs


def native_wait_steps(run_episode):
    node = ast.parse(inspect.getsource(run_episode))
    values = [ast.literal_eval(item.value) for item in ast.walk(node)
              if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "NUM_STEPS_WAIT" for target in item.targets)]
    if len(values) != 1 or not isinstance(values[0], int):
        raise ValueError("Cannot determine the existing episode runner's settle count")
    return values[0]


def gpu_worker(config):
    import numpy as np
    import torch
    memory_limit_gib = config.get("cuda_memory_limit_gib")
    if memory_limit_gib is not None:
        torch.cuda.set_device(0)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        budget_bytes = int(memory_limit_gib * 1024**3)
        if budget_bytes >= total_bytes:
            raise ValueError("CUDA memory limit must be smaller than device memory")
        torch.cuda.set_per_process_memory_fraction(budget_bytes / total_bytes, 0)
        print(json.dumps(dict(cuda_allocator_limit_gib=memory_limit_gib, pid=os.getpid(),
                              physical_gpu=os.environ.get("GPU_ID"))), flush=True)
    import h5py
    from cosmos_policy._src.imaginaire.utils import checkpoint_db

    original_checkpoint = checkpoint_db.get_checkpoint_path
    def offline_checkpoint(uri):
        if uri.rstrip("/") == "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt":
            return config["ckpt_path"]
        return original_checkpoint(uri)
    checkpoint_db.get_checkpoint_path = offline_checkpoint
    from cosmos_policy.experiments.robot.libero import run_libero_eval as evaluator
    from cosmos_policy.experiments.robot import cosmos_utils

    torch.cuda.set_device(0)
    suite, base_tasks, cache, validation = load_catalog(config, check_states=False)
    cfg = evaluator.PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only", ckpt_path=config["ckpt_path"],
        dataset_stats_path=str(Path(config["policy_dir"]) / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(Path(config["policy_dir"]) / "libero_t5_embeddings.pkl"),
        task_suite_name="libero_10", unnorm_key="libero_10", num_trials_per_task=1,
        seed=config["action_seed"], environment_seed=config["environment_seed"], rng_mode="openwam",
        deterministic=True, randomize_seed=False, deterministic_reset=False,
        data_collection=config["save_episode_data"], jpeg_compress=True,
        save_vector_db=False, enable_initial_alignment=False,
    )
    evaluator.validate_config(cfg)
    os.environ["DETERMINISTIC"] = "True"
    evaluator.set_seed_everywhere(cfg.seed)
    # Keep cached embeddings on CPU; the existing model input path moves the
    # selected embedding to CUDA. No on-demand T5 model or base fallback.
    cosmos_utils.t5_text_embeddings_cache.update(cache)
    stats = evaluator.load_dataset_stats(cfg.dataset_stats_path)
    model, model_config = evaluator.get_model(cfg)
    if cfg.chunk_size != model_config.dataloader_train.dataset.chunk_size:
        raise ValueError("Model/evaluation chunk size mismatch")
    resize = evaluator.get_image_resize_size(cfg.model_family)
    native_wait = native_wait_steps(evaluator.run_episode)
    if config["settle_steps"] < native_wait:
        raise ValueError(f"Settle steps must be at least the native count ({native_wait})")
    evaluator.TASK_MAX_STEPS["libero_10"] = config["max_steps"]
    original_env = evaluator.get_libero_env
    original_video = evaluator.save_rollout_video
    # The plain MP4 is sufficient; avoid an extra future-prediction panel video.
    evaluator.save_rollout_video_with_future_image_predictions = lambda *a, **kw: None
    ready = Path(os.environ["LIBERO_PLUS_BATCH_READY"])
    atomic_json(ready, dict(gpu=os.environ["GPU_ID"], pid=os.getpid(), model_loads=1, validation=validation))
    allowed = {row["task_id"]: row for group in config["groups"] for row in group["tasks"]}

    for line in sys.stdin:
        command = json.loads(line)
        if command.get("stop"):
            break
        row = command["task"]
        if allowed.get(row["task_id"]) != row:
            raise ValueError("Worker received a task outside its validated manifest")
        directory = result_path(config, row).parent
        directory.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result = dict(row, fingerprint=config["fingerprint"], status="error", success=None,
                      action_seed=cfg.seed, environment_seed=cfg.environment_seed, rng_mode=cfg.rng_mode,
                      init_state_index=0, num_trials_per_task=1, settle_steps=config["settle_steps"],
                      max_steps=config["max_steps"], gpu=os.environ["GPU_ID"], videos=[])
        environments = []
        key = instruction_key(suite.get_task(row["task_id"]), row, base_tasks)
        result["instruction"] = key

        def make_env(*args, **kwargs):
            env, description = original_env(*args, **kwargs)
            wrapped = RecordedEnv(env, config["settle_steps"] - native_wait, native_wait)
            environments.append(wrapped)
            return wrapped, description

        def embedding(label):
            if label != suite.get_task(row["task_id"]).language:
                raise ValueError(f"Unexpected policy instruction {label!r}")
            return cosmos_utils.t5_text_embeddings_cache[key]

        def save_video(*args, **kwargs):
            path = Path(original_video(*args, **kwargs)).resolve()
            destination = directory / "video.mp4"
            path.replace(destination)
            result["videos"].append(str(destination))
            return str(destination)

        evaluator.get_libero_env = make_env
        evaluator.save_rollout_video = save_video
        cosmos_utils.get_t5_embedding_from_cache = embedding
        cfg.local_log_dir = str(directory)
        cfg.run_id_note = f"plus-task{row['task_id']:04d}"
        os.environ["COSMOS_ROLLOUT_SUBDIR"] = os.path.relpath(directory, Path.cwd() / "rollouts")
        os.environ["COSMOS_TTA_REPAIR_TAG"] = f"plus-task{row['task_id']:04d}"
        with (directory / "eval.log").open("w") as handle:
            log = EpisodeLog(handle)
            try:
                # Each variant starts a new environment RNG stream. Reset and
                # simulation steps continue it; the existing action RNG scope
                # isolates model sampling and VAE seeds from that stream.
                evaluator.set_seed_everywhere(cfg.environment_seed)
                episodes, successes = evaluator.run_task(
                    cfg, suite, row["task_id"], model, None, stats, None, resize, log_file=log,
                )
                if log.errors:
                    raise RuntimeError("\n".join(log.errors))
                if episodes != 1 or successes not in (0, 1):
                    raise RuntimeError(f"Expected one trial, got episodes={episodes}, successes={successes}")
                if not result["videos"]:
                    raise RuntimeError("Episode completed without a saved video")
                if config["save_episode_data"]:
                    files = sorted((directory / "rollout_data").glob("*.hdf5"))
                    if len(files) != 1 or len(environments) != 1:
                        raise RuntimeError("Missing or ambiguous episode HDF5/environment")
                    recording = environments[0]
                    executed = np.asarray(recording.actions)
                    with h5py.File(files[0], "a") as data:
                        for name in ("initial_state", "executed_actions", "executed_action_is_settle", "environment_step_indices"):
                            if name in data:
                                del data[name]
                        data.create_dataset("initial_state", data=recording.initial_state)
                        data.create_dataset("executed_actions", data=executed)
                        data.create_dataset("executed_action_is_settle", data=np.arange(len(executed)) < config["settle_steps"])
                        indices = data["frame_indices"][:] + config["settle_steps"] - native_wait
                        data.create_dataset("environment_step_indices", data=indices)
                        actions = data["actions"][:]
                        if not np.array_equal(actions, executed[config["settle_steps"]:]):
                            raise RuntimeError("Recorded actions differ from the actions executed in the environment")
                        if len(actions) != len(data["proprio"]):
                            raise RuntimeError("Action/observation lengths differ")
                        data.attrs["libero_plus_metadata"] = json.dumps(dict(result, status="completed", success=bool(successes), episodes=1), ensure_ascii=False)
                        data.attrs["run_fingerprint"] = config["fingerprint"]
                    np.save(directory / "actions.npy", actions)
                    result.update(episode_hdf5=str(files[0]), actions_path=str(directory / "actions.npy"), policy_steps=len(actions))
                result.update(status="completed", success=bool(successes), episodes=1)
                if memory_limit_gib is not None:
                    result.update(cuda_allocator_limit_gib=memory_limit_gib,
                                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception:
                result["error"] = traceback.format_exc()
                log.write(result["error"])
            finally:
                for recording in environments:
                    try:
                        recording.env.close()
                    except Exception:
                        log.write(traceback.format_exc())
        result["elapsed_seconds"] = round(time.time() - started, 3)
        atomic_json(result_path(config, row), result)


def summaries(config):
    total = dict(planned_tasks=0, completed_episodes=0, successes=0, failures=0, errors=0, pending=0)
    categories = []
    for group in config["groups"]:
        rows = []
        counts = dict(planned_tasks=len(group["tasks"]), completed_episodes=0, successes=0, failures=0, errors=0, pending=0)
        for task in group["tasks"]:
            path = result_path(config, task)
            if not path.exists():
                counts["pending"] += 1
                continue
            result = read_json(path)
            if result.get("fingerprint") != config["fingerprint"] or result.get("task_name") != task["task_name"]:
                raise ValueError(f"Result belongs to another evaluation: {path}")
            rows.append(result)
            if result["status"] == "completed":
                counts["completed_episodes"] += 1
                counts["successes" if result["success"] else "failures"] += 1
            else:
                counts["errors"] += 1
        summary = dict(key=group["key"], category=group["category"], **counts)
        summary["complete"] = counts["pending"] == 0 and counts["errors"] == 0
        summary["success_rate"] = counts["successes"] / counts["completed_episodes"] if counts["completed_episodes"] else None
        destination = Path(config["output_root"]) / group["key"]
        atomic_json(destination / "summary.json", summary)
        temporary = destination / "results.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temporary.replace(destination / "results.jsonl")
        categories.append(summary)
        for key in total:
            total[key] += counts[key]
    overall = dict(**total, categories=categories, fingerprint=config["fingerprint"])
    overall["complete"] = total["pending"] == 0 and total["errors"] == 0
    overall["success_rate"] = total["successes"] / total["completed_episodes"] if total["completed_episodes"] else None
    atomic_json(Path(config["output_root"]) / "summary.json", overall)
    return overall


def pending_tasks(config, group, retry_errors):
    tasks = []
    for row in group["tasks"]:
        path = result_path(config, row)
        if not path.exists():
            tasks.append(row)
        elif retry_errors and read_json(path)["status"] == "error":
            tasks.append(row)
    return tasks


def run_parent(args, config):
    import fcntl
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "run_config.json"
        if config_path.exists():
            if not args.resume:
                raise ValueError(f"Existing run found; use --resume or choose another --output-root: {output}")
            if read_json(config_path) != config:
                raise ValueError("Resume parameters or source files differ from the existing run")
        else:
            if any(output.iterdir()) and any(path.name != ".runner.lock" for path in output.iterdir()):
                raise ValueError("Output directory is not empty and has no matching run_config.json")
            atomic_json(config_path, config)
        validate(config, config_path)
        summaries(config)  # Verify any existing results before dispatch.
        if not any(pending_tasks(config, group, args.retry_errors) for group in config["groups"]):
            return summaries(config)
        session = output / "workers" / uuid.uuid4().hex[:12]
        session.mkdir(parents=True)
        workers = []
        try:
            for gpu in args.gpus:
                log = (session / f"gpu{gpu}.log").open("w")
                ready = session / f"gpu{gpu}.ready.json"
                env = worker_env(config, gpu, "worker", config_path)
                env["LIBERO_PLUS_BATCH_READY"] = str(ready)
                process = subprocess.Popen(
                    ["sh", str(Path(config["repo_root"]) / "scripts/run_libero_smoke_test.sh")],
                    cwd=config["repo_root"], env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True,
                )
                workers.append(dict(gpu=gpu, process=process, log=log, ready=ready, task=None))
            started, progress = time.monotonic(), 0.0
            while not all(worker["ready"].exists() for worker in workers):
                for worker in workers:
                    if worker["process"].poll() is not None:
                        raise RuntimeError(f"GPU {worker['gpu']} failed during startup; see {worker['log'].name}")
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError("Model startup timeout; see worker logs")
                if time.monotonic() - progress > 30:
                    print("Waiting for persistent GPU workers to load their models...", flush=True)
                    progress = time.monotonic()
                time.sleep(1)
            for group in config["groups"]:
                queue = iter(pending_tasks(config, group, args.retry_errors))
                exhausted = False
                print(f"Starting category: {group['key']}", flush=True)
                while True:
                    for worker in workers:
                        process = worker["process"]
                        if process.poll() is not None:
                            raise RuntimeError(f"GPU {worker['gpu']} worker exited; see {worker['log'].name}")
                        row = worker["task"]
                        if row is not None:
                            path = result_path(config, row)
                            if path.exists():
                                result = read_json(path)
                                print(f"[{group['key']}] task={row['task_id']} gpu={worker['gpu']} status={result['status']} success={result['success']}", flush=True)
                                worker["task"] = None
                            elif time.monotonic() - worker["started"] > args.task_timeout:
                                raise TimeoutError(f"Task {row['task_id']} timed out on GPU {worker['gpu']}")
                        if worker["task"] is None and not exhausted:
                            row = next(queue, None)
                            if row is None:
                                exhausted = True
                            else:
                                # On an explicit error retry, move the old result out
                                # of the polling path and preserve its full artifacts.
                                directory = result_path(config, row).parent
                                if directory.exists():
                                    history = directory.parent / "previous_attempts"
                                    history.mkdir(exist_ok=True)
                                    directory.rename(history / (directory.name + "_" + uuid.uuid4().hex[:8]))
                                process.stdin.write(json.dumps(dict(task=row)) + "\n")
                                process.stdin.flush()
                                worker.update(task=row, started=time.monotonic())
                    if exhausted and all(worker["task"] is None for worker in workers):
                        break
                    if time.monotonic() - progress > 30:
                        overall = summaries(config)
                        print(f"Completed={overall['completed_episodes']} success={overall['successes']} errors={overall['errors']} pending={overall['pending']}", flush=True)
                        progress = time.monotonic()
                    time.sleep(1)
                summaries(config)
            for worker in workers:
                worker["process"].stdin.write('{"stop": true}\n')
                worker["process"].stdin.flush()
            return summaries(config)
        finally:
            for worker in workers:
                process = worker["process"]
                if process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                worker["log"].close()
            summaries(config)


def main():
    mode = os.environ.get("LIBERO_PLUS_BATCH_MODE")
    if mode:
        config = read_json(os.environ["LIBERO_PLUS_BATCH_CONFIG"])
        configure_sensor_dependencies()
        if mode == "validate":
            _, _, _, report = load_catalog(config)
            print(json.dumps(report, indent=2))
            return 0
        if mode == "worker":
            gpu_worker(config)
            return 0
        raise ValueError(f"Unknown worker mode: {mode}")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--extra-t5", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--cuda-memory-limit-gib", type=float,
                        help="Optional per-worker PyTorch CUDA allocator limit in GiB")
    parser.add_argument("--categories", nargs="+", choices=CATEGORY_KEYS)
    parser.add_argument("--task-ids-file", type=Path, help="JSON list or object with task_ids selecting exact cases")
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--action-seed", type=int, default=42)
    parser.add_argument("--environment-seed", type=int, default=10000)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--save-episode-data", action=argparse.BooleanOptionalAction, default=True,
                        help="Save images/proprio/actions in HDF5 plus actions.npy (default: enabled)")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests and print plan without importing/loading the model")
    parser.add_argument("--validate-only", action="store_true", help="Check installed tasks, initial states and T5 coverage on CPU; no rollouts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, retry runtime errors; completed failures are retained")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--task-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.cuda_memory_limit_gib is not None and args.cuda_memory_limit_gib <= 0:
        parser.error("CUDA memory limit must be positive")
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPU IDs must be nonnegative and unique")
    if args.categories and len(set(args.categories)) != len(args.categories):
        parser.error("Category names must be unique")
    if (args.limit_per_category is not None and args.limit_per_category < 1) or args.settle_steps < 10 or args.max_steps < 1:
        parser.error("Require limit >= 1, settle steps >= 10 and max steps >= 1")
    if min(args.action_seed, args.environment_seed) < 0 or min(args.startup_timeout, args.task_timeout) < 1:
        parser.error("Seeds must be nonnegative and timeouts positive")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    config = load_plan(args)
    print(json.dumps(dict(
        tasks=sum(len(group["tasks"]) for group in config["groups"]), gpus=args.gpus,
        categories={group["key"]: len(group["tasks"]) for group in config["groups"]},
        action_seed=config["action_seed"], environment_seed=config["environment_seed"],
        settle_steps=config["settle_steps"], max_steps=config["max_steps"],
        num_trials_per_task=1, init_state_index=0, videos=True, save_episode_data=config["save_episode_data"],
        output_root=config["output_root"],
    ), indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.validate_only:
        with tempfile.TemporaryDirectory(prefix="libero-plus-validate-") as directory:
            config_path = Path(directory) / "config.json"
            atomic_json(config_path, config)
            validate(config, config_path)
        return 0
    result = run_parent(args, config)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Saved results can be continued with --resume.", file=sys.stderr)
        sys.exit(130)
