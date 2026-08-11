#!/usr/bin/env python3
"""Run Phase 8 LIBERO camera full rollouts with one Cosmos model load."""

from __future__ import annotations

import argparse
import builtins
import contextlib
import gc
import io
import json
import logging
import os
import pathlib
import pickle
import re
import site
import sys
import time
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", ROOT.parent / "LIBERO-plus"))
if str(LIBERO_PLUS) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS))
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Task

from phase8_correction_lib import DynamicCorrectionContext, load_corrector


CHECKPOINT_ROOT = (ROOT / "../../checkpoints").resolve()
POLICY_DIR = CHECKPOINT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"

T5_CONTEXT = {
    "instruction_mode": "task",
    "base_language": "",
    "fallback_to_base": True,
}


@dataclass(frozen=True)
class BaseTaskSpec:
    suite: str
    name: str
    language: str


class SingleTaskSuite:
    def __init__(self, task: Task, init_states: Any, num_trials: int):
        self.tasks = [task]
        self.n_tasks = 1
        self._init_states = repeat_init_states(init_states, num_trials)

    def get_task(self, task_id: int) -> Task:
        if task_id != 0:
            raise IndexError(f"single-task suite only has task_id=0, got {task_id}")
        return self.tasks[0]

    def get_task_init_states(self, task_id: int) -> Any:
        if task_id != 0:
            raise IndexError(f"single-task suite only has task_id=0, got {task_id}")
        return self._init_states


class VariantLookup:
    def __init__(self):
        self._cache: dict[tuple[str, str], Any] = {}
        self._benchmark_dict = benchmark.get_benchmark_dict()

    def suite(self, suite_name: str, category: str):
        key = (suite_name, category)
        if key not in self._cache:
            if suite_name not in self._benchmark_dict:
                raise ValueError(f"Unknown LIBERO suite: {suite_name}")
            suite_cls = self._benchmark_dict[suite_name]
            # Some LIBERO-plus versions print thousands of task indices while
            # constructing a suite; that output obscures rollout progress.
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    self._cache[key] = suite_cls(category_value=category)
                except TypeError:
                    # Versions that put all variants directly in the suite
                    # task map do not expose category_value.
                    self._cache[key] = suite_cls()
        return self._cache[key]

    def task_by_name(self, suite_name: str, category: str, task_name: str) -> tuple[int, Task]:
        task_suite = self.suite(suite_name, category)
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.name == task_name:
                return task_id, task
        examples = "\n".join(f"  [{idx}] {task_suite.get_task(idx).name}" for idx in range(min(20, task_suite.n_tasks)))
        raise RuntimeError(
            f"Task not found in suite={suite_name}, category={category!r}: {task_name}\n"
            f"First available tasks:\n{examples}"
        )

    def camera_variant_for_base(
        self,
        suite_name: str,
        category: str,
        base_task: str,
        preferred_task_name: str,
        allow_fallback: bool,
    ) -> tuple[int, Task, bool]:
        try:
            task_id, task = self.task_by_name(suite_name, category, preferred_task_name)
            return task_id, task, False
        except RuntimeError:
            if not allow_fallback:
                raise

        task_suite = self.suite(suite_name, category)
        prefix = f"{base_task}_view_"
        candidates = [
            (task_id, task)
            for task_id in range(task_suite.n_tasks)
            for task in [task_suite.get_task(task_id)]
            if task.name.startswith(prefix) and "_noise_" not in task.name
        ]
        if not candidates:
            raise RuntimeError(
                f"No camera variants found for suite={suite_name}, category={category!r}, base_task={base_task}"
            )
        task_id, task = min(candidates, key=lambda item: camera_variant_score(item[1].name))
        return task_id, task, True


class EvalRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.correction_enabled = False
        self.corrector = None
        self.corrector_config: dict[str, Any] | None = None
        patch_checkpoint_db(args.policy_dir)

        from cosmos_policy.experiments.robot import cosmos_utils
        import cosmos_policy.experiments.robot.libero.run_libero_eval as run_libero_eval_mod

        self.cosmos_utils = cosmos_utils
        self.mod = run_libero_eval_mod
        patch_print_spam(run_libero_eval_mod)
        configure_terminal_logging(run_libero_eval_mod, args.quiet_step_logs)
        patch_t5(cosmos_utils, args.t5_extra_embeddings, args.t5_fallback_to_base)
        if not args.save_videos:
            patch_video_writes(run_libero_eval_mod)

        cfg = make_cfg(args, args.suites[0], args.num_pairs, run_id_note="phase8-fast-model-load")
        assert not (cfg.deterministic and cfg.randomize_seed), (
            "Cannot enable both deterministic mode and randomize seed mode."
        )
        if cfg.deterministic:
            os.environ["DETERMINISTIC"] = "True"

        run_libero_eval_mod.validate_config(cfg)
        run_libero_eval_mod.set_seed_everywhere(cfg.seed)
        cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        self.dataset_stats = run_libero_eval_mod.load_dataset_stats(cfg.dataset_stats_path)

        self.model, cosmos_config = run_libero_eval_mod.get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            "Mismatch found between train and test chunk sizes. "
            f"Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        if cfg.planning_model_ckpt_path:
            self.planning_model, _ = run_libero_eval_mod.get_planning_model(cfg)
        else:
            self.planning_model = None
        self.resize_size = run_libero_eval_mod.get_image_resize_size(cfg.model_family)

        if args.corrector_checkpoint is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.corrector, checkpoint = load_corrector(args.corrector_checkpoint, device)
            target = checkpoint["target"]
            self.corrector_config = {
                "checkpoint": str(args.corrector_checkpoint),
                "layer": int(target["layer"]),
                "target": str(target["target"]),
                "target_rms": float(target["target_rms"]),
                "alpha": float(args.correction_alpha),
            }
            original_get_action = run_libero_eval_mod.get_action

            def get_action_with_optional_correction(cfg, model, *action_args, **action_kwargs):
                if not self.correction_enabled:
                    return original_get_action(cfg, model, *action_args, **action_kwargs)
                with DynamicCorrectionContext(
                    model,
                    self.corrector,
                    layer=self.corrector_config["layer"],
                    target=self.corrector_config["target"],
                    target_rms=self.corrector_config["target_rms"],
                    alpha=self.corrector_config["alpha"],
                    condition_pass_only=True,
                ):
                    return original_get_action(cfg, model, *action_args, **action_kwargs)

            run_libero_eval_mod.get_action = get_action_with_optional_correction
            print(
                "Loaded rollout corrector: "
                f"layer={self.corrector_config['layer']} "
                f"target={self.corrector_config['target']} "
                f"alpha={self.corrector_config['alpha']}",
                flush=True,
            )

    def run_single_task(
        self,
        *,
        suite_name: str,
        task: Task,
        init_states: Any,
        num_trials: int,
        run_id_note: str,
        log_dir: pathlib.Path,
        instruction_mode: str,
        base_language: str,
        apply_correction: bool = False,
    ) -> tuple[float, list[dict[str, Any]], str]:
        if apply_correction and self.corrector is None:
            raise RuntimeError("corrected rollout requested without --corrector-checkpoint")
        cfg = make_cfg(self.args, suite_name, num_trials, run_id_note=run_id_note, local_log_dir=str(log_dir))
        self.mod.set_seed_everywhere(cfg.seed)
        T5_CONTEXT["instruction_mode"] = instruction_mode
        T5_CONTEXT["base_language"] = base_language
        task_suite = SingleTaskSuite(task, init_states, num_trials)

        log_file, local_log_filepath, _ = self.mod.setup_logging(
            cfg=cfg,
            task_identifier=cfg.task_suite_name,
            log_dir=cfg.local_log_dir,
            run_id_note=cfg.run_id_note,
            use_wandb=cfg.use_wandb,
            wandb_entity=cfg.wandb_entity,
            wandb_project=cfg.wandb_project,
        )
        self.correction_enabled = apply_correction
        try:
            self.mod.log_message(f"Eval config: {cfg}", log_file)
            total_episodes, total_successes = self.mod.run_task(
                cfg,
                task_suite,
                0,
                self.model,
                self.planning_model,
                self.dataset_stats,
                None,
                self.resize_size,
                0,
                0,
                log_file,
            )
            success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
            self.mod.log_message("Final results:", log_file)
            self.mod.log_message(f"Total episodes: {total_episodes}", log_file)
            self.mod.log_message(f"Total successes: {total_successes}", log_file)
            self.mod.log_message(
                f"Overall success rate: {success_rate:.4f} ({success_rate * 100:.1f}%)",
                log_file,
            )
        finally:
            self.correction_enabled = False
            log_file.close()
            T5_CONTEXT["instruction_mode"] = "task"
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        episodes = parse_episode_successes(pathlib.Path(local_log_filepath))
        return success_rate, episodes, local_log_filepath


def patch_checkpoint_db(policy_dir: pathlib.Path) -> None:
    from cosmos_policy._src.imaginaire.utils import checkpoint_db

    original_get_checkpoint_path = checkpoint_db.get_checkpoint_path
    aloha_checkpoint_uri = (
        "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/"
        "Cosmos-Policy-ALOHA-Predict2-2B.pt"
    )

    def get_checkpoint_path_offline(checkpoint_uri):
        if checkpoint_uri.rstrip("/") == aloha_checkpoint_uri:
            return str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        return original_get_checkpoint_path(checkpoint_uri)

    checkpoint_db.get_checkpoint_path = get_checkpoint_path_offline


def patch_print_spam(run_libero_eval_mod) -> None:
    original_print = getattr(run_libero_eval_mod, "print", builtins.print)

    def print_without_action_spam(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("t: ") and "\t action:" in args[0]:
            return
        return original_print(*args, **kwargs)

    run_libero_eval_mod.print = print_without_action_spam


def configure_terminal_logging(run_libero_eval_mod, quiet_step_logs: bool) -> None:
    if quiet_step_logs:
        logging.getLogger("cosmos_policy.experiments.robot.robot_utils").setLevel(logging.WARNING)
        logging.getLogger("cosmos_policy.experiments.robot.libero.run_libero_eval").setLevel(logging.WARNING)
        try:
            from cosmos_policy._src.imaginaire.utils import log as imaginaire_log

            imaginaire_log.LEVEL = "WARNING"
            imaginaire_log.init_loguru_stdout()
        except Exception:
            pass
    patch_robot_log_message(run_libero_eval_mod, quiet_step_logs)


def patch_robot_log_message(run_libero_eval_mod, quiet_step_logs: bool) -> None:
    def log_message_quiet(message: str, log_file=None):
        if not quiet_step_logs or not is_step_log(message):
            print(message)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    run_libero_eval_mod.log_message = log_message_quiet


def is_step_log(message: str) -> bool:
    if message.startswith("Query "):
        return True
    if message.startswith("t=") and (
        "Current base seed" in message
        or "Selected seed" in message
    ):
        return True
    if message.startswith("Task: "):
        return True
    if message.startswith("Starting episode "):
        return True
    if message.startswith("Success: "):
        return True
    if message.startswith("# episodes completed so far:"):
        return True
    if message.startswith("# successes:"):
        return True
    if message.startswith("Current task success rate:"):
        return True
    if message.startswith("Current total success rate:"):
        return True
    return message == "Using default initial states"


def patch_video_writes(run_libero_eval_mod) -> None:
    def skip_rollout_video(*_args, **_kwargs):
        return None

    run_libero_eval_mod.save_rollout_video = skip_rollout_video
    run_libero_eval_mod.save_rollout_video_with_future_image_predictions = skip_rollout_video


def patch_t5(cosmos_utils, extra_path: pathlib.Path | None, fallback_to_base: bool) -> None:
    original_init = cosmos_utils.init_t5_text_embeddings_cache
    original_get = cosmos_utils.get_t5_embedding_from_cache
    T5_CONTEXT["fallback_to_base"] = fallback_to_base

    def init_t5_text_embeddings_cache_with_extra(t5_text_embeddings_path, *args, **kwargs):
        result = original_init(t5_text_embeddings_path, *args, **kwargs)
        if extra_path is not None:
            with open(extra_path, "rb") as file:
                extra_embeddings = pickle.load(file)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for key, value in extra_embeddings.items():
                if isinstance(value, torch.Tensor):
                    extra_embeddings[key] = value.to(device)
            cosmos_utils.t5_text_embeddings_cache.update(extra_embeddings)
            print(f"Loaded {len(extra_embeddings)} extra T5 embeddings from {extra_path} onto device {device}")
        return result

    def get_t5_embedding_for_libero_plus(task_label: str):
        instruction_mode = T5_CONTEXT["instruction_mode"]
        if instruction_mode == "base":
            cache_label = strip_libero_plus_metadata(task_label)
            if (
                cache_label not in cosmos_utils.t5_text_embeddings_cache
                and T5_CONTEXT["fallback_to_base"]
                and T5_CONTEXT["base_language"]
            ):
                cache_label = T5_CONTEXT["base_language"]
            return original_get(cache_label)
        if instruction_mode == "strict":
            return original_get(task_label)
        return original_get(task_label)

    cosmos_utils.init_t5_text_embeddings_cache = init_t5_text_embeddings_cache_with_extra
    cosmos_utils.get_t5_embedding_from_cache = get_t5_embedding_for_libero_plus


def strip_libero_plus_metadata(task_label: str) -> str:
    cache_label = task_label
    cache_label = re.sub(r" view .+ initstate \d+(?: noise \d+)?$", "", cache_label)
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


def camera_variant_score(task_name: str) -> tuple[int, int, str]:
    match = re.search(r"_view_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)$", task_name)
    if not match:
        return (1_000_000, 1_000_000, task_name)
    azimuth, elevation, radius, x_offset, y_offset, init_state = [int(value) for value in match.groups()]
    distance = (
        abs(azimuth - 50)
        + abs(elevation - 0) * 10
        + abs(radius - 100)
        + abs(x_offset - 0) * 10
        + abs(y_offset - 0) * 10
        + abs(init_state - 0) * 1000
    )
    exact_init_state = 0 if init_state == 0 else 1
    return (exact_init_state, distance, task_name)


def make_cfg(
    args: argparse.Namespace,
    suite_name: str,
    num_trials: int,
    *,
    run_id_note: str,
    local_log_dir: str = "./experiments/logs",
):
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

    unnorm_key = "libero_10" if suite_name == "libero_mix" else suite_name
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(args.policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        dataset_stats_path=str(args.policy_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(args.policy_dir / "libero_t5_embeddings.pkl"),
        task_suite_name=suite_name,
        unnorm_key=unnorm_key,
        num_trials_per_task=num_trials,
        flip_images=args.flip_images,
        local_log_dir=local_log_dir,
        run_id_note=run_id_note,
        seed=args.seed,
        randomize_seed=False,
        deterministic=True,
        deterministic_reset=args.deterministic_reset,
        deterministic_reset_seed=args.deterministic_reset_seed,
    )


def repeat_init_states(states: Any, num_trials: int) -> Any:
    if len(states) >= num_trials:
        return states
    if isinstance(states, torch.Tensor):
        return torch.stack([states[idx % len(states)] for idx in range(num_trials)], dim=0)
    return [states[idx % len(states)] for idx in range(num_trials)]


def parse_episode_successes(log_path: pathlib.Path) -> list[dict[str, Any]]:
    episodes = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Success: "):
            episodes.append(
                {
                    "episode": len(episodes),
                    "success": line.split("Success: ", 1)[1].strip() == "True",
                }
            )
    return episodes


def annotate_episodes(
    episodes: list[dict[str, Any]],
    *,
    run_id: str,
    seed: int,
    deterministic_reset: bool,
    deterministic_reset_seed: int,
    suite: str,
    base_task: str,
    condition: str,
    category: str,
    task_id: int | None,
    task_name: str,
    language: str,
    log_path: str,
    instruction_mode: str,
) -> list[dict[str, Any]]:
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


def rollout_recovery_metrics(
    pert_episodes: list[dict[str, Any]],
    corrected_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    pert = {int(row["episode"]): bool(row["success"]) for row in pert_episodes}
    corrected = {int(row["episode"]): bool(row["success"]) for row in corrected_episodes}
    common = sorted(set(pert) & set(corrected))
    failed = [ep for ep in common if not pert[ep]]
    recovered = [ep for ep in failed if corrected[ep]]
    preserved = [ep for ep in common if pert[ep]]
    retained = [ep for ep in preserved if corrected[ep]]
    harmed = [ep for ep in preserved if not corrected[ep]]
    return {
        "num_paired_episodes": len(common),
        "num_pert_failures": len(failed),
        "num_recovered": len(recovered),
        "failure_recovery_rate": len(recovered) / len(failed) if failed else None,
        "recovered_episodes": recovered,
        "num_pert_successes": len(preserved),
        "num_preserved_after_correction": len(retained),
        "preservation_rate": len(retained) / len(preserved) if preserved else None,
        "num_harmed": len(harmed),
        "harm_rate": len(harmed) / len(preserved) if preserved else None,
        "harmed_episodes": harmed,
    }


def read_base_tasks(suites: list[str], task_limit: int, only_tasks: set[tuple[str, str]]) -> list[BaseTaskSpec]:
    specs = []
    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    for suite in suites:
        tasks_info = bddl_root / suite / "tasks_info.txt"
        if not tasks_info.exists():
            raise FileNotFoundError(f"Missing tasks_info.txt for {suite}: {tasks_info}")
        suite_specs = []
        for line in tasks_info.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            name = pathlib.Path(line).stem
            if only_tasks and (suite, name) not in only_tasks:
                continue
            filename = f"{name}.bddl"
            try:
                language = benchmark.grab_language_from_filename(suite, filename)
            except TypeError:
                # Compatibility with upstream LIBERO, whose helper does not
                # take a suite argument.
                language = benchmark.grab_language_from_filename(filename)
            suite_specs.append(BaseTaskSpec(suite=suite, name=name, language=language))
        if task_limit > 0:
            suite_specs = suite_specs[:task_limit]
        specs.extend(suite_specs)
    return specs


def parse_only_tasks(values: list[str]) -> set[tuple[str, str]]:
    parsed = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"--only-task must be formatted as suite:task_name, got {value!r}")
        suite, task_name = value.split(":", 1)
        parsed.add((suite, task_name))
    return parsed


def load_base_init_states(suite: str, base_task: str) -> Any:
    init_path = pathlib.Path(get_libero_path("init_states")) / suite / f"{base_task}.pruned_init"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing init states: {init_path}")
    return torch.load(init_path, weights_only=False)


def run_pair(
    runtime: EvalRuntime,
    lookup: VariantLookup,
    spec: BaseTaskSpec,
    args: argparse.Namespace,
) -> pathlib.Path:
    results_dir = args.output_root / spec.suite / spec.name
    summary_path = results_dir / f"{spec.name}__{args.condition}__{args.num_pairs}pair_summary.json"
    if args.resume and summary_path.exists():
        print(f"skip existing: {summary_path}", flush=True)
        return summary_path

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.run_id_prefix}_{spec.suite}_{spec.name}_camera_seed{args.seed}_{args.num_pairs}pair"
    init_states = load_base_init_states(spec.suite, spec.name)
    clean_task = Task(
        name=spec.name,
        language=spec.language,
        problem="Libero",
        problem_folder=spec.suite,
        bddl_file=f"{spec.name}.bddl",
        init_states_file=f"{spec.name}.pruned_init",
    )

    preferred_pert_task_name = f"{spec.name}{args.pert_suffix}"
    pert_task_id, pert_task, used_variant_fallback = lookup.camera_variant_for_base(
        spec.suite,
        args.category,
        spec.name,
        preferred_pert_task_name,
        args.allow_variant_fallback,
    )

    print(f"\n[{spec.suite}] {spec.name}", flush=True)
    if runtime.corrector is None:
        print(f"  clean: {spec.language}", flush=True)
    print(f"  pert:  {pert_task.name}", flush=True)
    if used_variant_fallback:
        print(f"  note: preferred variant missing, selected nearest camera variant to {preferred_pert_task_name}", flush=True)

    clean_rate = None
    clean_episodes = None
    clean_log = None
    clean_episodes_path = None
    if runtime.corrector is None:
        clean_rate, clean_episodes_raw, clean_log = runtime.run_single_task(
            suite_name=spec.suite,
            task=clean_task,
            init_states=init_states,
            num_trials=args.num_pairs,
            run_id_note=f"paired-clean-{args.num_pairs}pair",
            log_dir=log_dir,
            instruction_mode="task",
            base_language=spec.language,
        )
        clean_episodes = annotate_episodes(
            clean_episodes_raw,
            run_id=run_id,
            seed=args.seed,
            deterministic_reset=args.deterministic_reset,
            deterministic_reset_seed=args.deterministic_reset_seed,
            suite=spec.suite,
            base_task=spec.name,
            condition="clean",
            category="clean",
            task_id=None,
            task_name=spec.name,
            language=spec.language,
            log_path=clean_log,
            instruction_mode="task",
        )
        clean_dir = results_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_episodes_path = clean_dir / "episodes.json"
        clean_episodes_path.write_text(json.dumps(clean_episodes, indent=2), encoding="utf-8")

    pert_rate, pert_episodes_raw, pert_log = runtime.run_single_task(
        suite_name=spec.suite,
        task=pert_task,
        init_states=init_states,
        num_trials=args.num_pairs,
        run_id_note=f"paired-{args.condition}-{args.num_pairs}pair",
        log_dir=log_dir,
        instruction_mode=args.instruction_mode,
        base_language=spec.language,
    )
    pert_episodes = annotate_episodes(
        pert_episodes_raw,
        run_id=run_id,
        seed=args.seed,
        deterministic_reset=args.deterministic_reset,
        deterministic_reset_seed=args.deterministic_reset_seed,
        suite=spec.suite,
        base_task=spec.name,
        condition=args.condition,
        category=args.category,
        task_id=pert_task_id,
        task_name=pert_task.name,
        language=pert_task.language,
        log_path=pert_log,
        instruction_mode=args.instruction_mode,
    )
    pert_dir = results_dir / args.condition
    pert_dir.mkdir(parents=True, exist_ok=True)
    pert_episodes_path = pert_dir / "episodes.json"
    pert_episodes_path.write_text(json.dumps(pert_episodes, indent=2), encoding="utf-8")

    corrected_rate = None
    corrected_episodes = None
    corrected_episodes_path = None
    corrected_log = None
    recovery = None
    if runtime.corrector is not None:
        corrected_rate, corrected_episodes_raw, corrected_log = runtime.run_single_task(
            suite_name=spec.suite,
            task=pert_task,
            init_states=init_states,
            num_trials=args.num_pairs,
            run_id_note=f"paired-{args.condition}-corrected-{args.num_pairs}pair",
            log_dir=log_dir,
            instruction_mode=args.instruction_mode,
            base_language=spec.language,
            apply_correction=True,
        )
        corrected_episodes = annotate_episodes(
            corrected_episodes_raw,
            run_id=run_id,
            seed=args.seed,
            deterministic_reset=args.deterministic_reset,
            deterministic_reset_seed=args.deterministic_reset_seed,
            suite=spec.suite,
            base_task=spec.name,
            condition="corrected",
            category=f"Corrected {args.category}",
            task_id=pert_task_id,
            task_name=pert_task.name,
            language=pert_task.language,
            log_path=corrected_log,
            instruction_mode=args.instruction_mode,
        )
        corrected_dir = results_dir / "corrected"
        corrected_dir.mkdir(parents=True, exist_ok=True)
        corrected_episodes_path = corrected_dir / "episodes.json"
        corrected_episodes_path.write_text(json.dumps(corrected_episodes, indent=2), encoding="utf-8")
        recovery = rollout_recovery_metrics(pert_episodes, corrected_episodes)

    condition_summaries = []
    if clean_episodes is not None:
        condition_summaries.append(
            {
                "condition": "clean",
                "task_name": spec.name,
                "category": "clean",
                "seed": args.seed,
                "success_rate": clean_rate,
                "successes": sum(1 for episode in clean_episodes if episode["success"]),
                "num_trials": len(clean_episodes),
                "log_path": clean_log,
                "episodes_path": str(clean_episodes_path),
            }
        )
    condition_summaries.append(
        {
            "condition": args.condition,
            "task_name": pert_task.name,
            "category": args.category,
            "seed": args.seed,
            "success_rate": pert_rate,
            "successes": sum(1 for episode in pert_episodes if episode["success"]),
            "num_trials": len(pert_episodes),
            "log_path": pert_log,
            "episodes_path": str(pert_episodes_path),
            "instruction_mode": args.instruction_mode,
        }
    )
    if corrected_episodes is not None:
        condition_summaries.append(
            {
                "condition": "corrected",
                "source_condition": args.condition,
                "task_name": pert_task.name,
                "category": f"Corrected {args.category}",
                "seed": args.seed,
                "success_rate": corrected_rate,
                "successes": sum(1 for episode in corrected_episodes if episode["success"]),
                "num_trials": len(corrected_episodes),
                "log_path": corrected_log,
                "episodes_path": str(corrected_episodes_path),
                "instruction_mode": args.instruction_mode,
                "corrector": runtime.corrector_config,
            }
        )
    summary = {
        "mode": "perturbed_vs_corrected" if runtime.corrector is not None else "paired",
        "run_id": run_id,
        "suite": spec.suite,
        "seed": args.seed,
        "deterministic_reset": args.deterministic_reset,
        "deterministic_reset_seed": args.deterministic_reset_seed,
        "num_pairs": args.num_pairs,
        "base_task": spec.name,
        "conditions": condition_summaries,
        "rollout_recovery": recovery,
        "preferred_pert_task": preferred_pert_task_name,
        "used_variant_fallback": used_variant_fallback,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if clean_episodes is not None:
        print(
            f"  clean success: {condition_summaries[0]['successes']}/{condition_summaries[0]['num_trials']} "
            f"({clean_rate:.3f})",
            flush=True,
        )
    pert_summary = next(row for row in condition_summaries if row["condition"] == args.condition)
    print(
        f"  pert success:  {pert_summary['successes']}/{pert_summary['num_trials']} "
        f"({pert_rate:.3f})",
        flush=True,
    )
    if corrected_episodes is not None:
        corrected_summary = next(row for row in condition_summaries if row["condition"] == "corrected")
        print(
            f"  corrected success: {corrected_summary['successes']}/{corrected_summary['num_trials']} "
            f"({corrected_rate:.3f})",
            flush=True,
        )
        failure_recovery_rate = recovery["failure_recovery_rate"]
        recovery_text = "n/a" if failure_recovery_rate is None else f"{failure_recovery_rate:.3f}"
        print(
            f"  failure recovery: {recovery['num_recovered']}/"
            f"{recovery['num_pert_failures']} ({recovery_text})",
            flush=True,
        )
    print(f"  summary: {summary_path}", flush=True)
    return summary_path


def write_summary_list(output_root: pathlib.Path, summary_paths: list[pathlib.Path], batch_summary: dict[str, Any]) -> None:
    summary_paths = sorted(path.resolve() for path in summary_paths)
    (output_root / "summaries.txt").write_text(
        "".join(f"{path}\n" for path in summary_paths),
        encoding="utf-8",
    )
    (output_root / "batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--num-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-reset-seed", type=int, default=0)
    parser.add_argument("--policy-dir", type=pathlib.Path, default=pathlib.Path(os.environ.get("COSMOS_POLICY_MODEL_DIR", POLICY_DIR)))
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=ROOT / "experiments/phase8_full_rollout/main_suites_camera_seed7_20pair",
    )
    parser.add_argument("--condition", default="camera_viewpoints")
    parser.add_argument("--category", default="Camera Viewpoints")
    parser.add_argument("--pert-suffix", default="_view_50_0_100_0_0_initstate_0")
    parser.add_argument("--allow-variant-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instruction-mode", default="base", choices=["task", "base", "strict"])
    parser.add_argument("--run-id-prefix", default="phase8")
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet-step-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-limit", type=int, default=0, help="Debug helper: limit tasks per suite; 0 means all.")
    parser.add_argument(
        "--only-task",
        action="append",
        default=[],
        help="Run one explicit task as suite:task_name. May be repeated.",
    )
    parser.add_argument("--t5-extra-embeddings", type=pathlib.Path, default=None)
    parser.add_argument("--t5-fallback-to-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--corrector-checkpoint",
        type=pathlib.Path,
        default=None,
        help="compare perturbed and corrected rollouts using this Phase 8 checkpoint; clean is skipped",
    )
    parser.add_argument("--correction-alpha", type=float, default=1.0)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate task/variant selection without loading the model.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.policy_dir = args.policy_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser()
    if args.t5_extra_embeddings is not None:
        args.t5_extra_embeddings = args.t5_extra_embeddings.expanduser().resolve()
        if not args.t5_extra_embeddings.exists():
            raise FileNotFoundError(f"Missing extra T5 embeddings: {args.t5_extra_embeddings}")
    if args.corrector_checkpoint is not None:
        args.corrector_checkpoint = args.corrector_checkpoint.expanduser().resolve()
        if not args.corrector_checkpoint.is_file():
            raise FileNotFoundError(f"Missing corrector checkpoint: {args.corrector_checkpoint}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    only_tasks = parse_only_tasks(args.only_task)
    specs = read_base_tasks(args.suites, args.task_limit, only_tasks)
    if not specs:
        raise RuntimeError("No base tasks selected.")

    expected_rollouts = len(specs) * args.num_pairs * 2
    print("Phase 8 fast full rollout", flush=True)
    print(f"  suites: {', '.join(args.suites)}", flush=True)
    print(f"  base tasks: {len(specs)}", flush=True)
    rollout_modes = "pert+corrected" if args.corrector_checkpoint is not None else "clean+pert"
    print(f"  {rollout_modes} rollouts: {expected_rollouts}", flush=True)
    print(f"  output: {args.output_root}", flush=True)
    print(f"  save videos: {args.save_videos}", flush=True)

    if args.dry_run:
        lookup = VariantLookup()
        plan = []
        for spec in specs:
            preferred_pert_task_name = f"{spec.name}{args.pert_suffix}"
            pert_task_id, pert_task, used_variant_fallback = lookup.camera_variant_for_base(
                spec.suite,
                args.category,
                spec.name,
                preferred_pert_task_name,
                args.allow_variant_fallback,
            )
            plan.append(
                {
                    "suite": spec.suite,
                    "base_task": spec.name,
                    "clean_language": spec.language,
                    "preferred_pert_task": preferred_pert_task_name,
                    "pert_task_id": pert_task_id,
                    "pert_task": pert_task.name,
                    "used_variant_fallback": used_variant_fallback,
                }
            )
        dry_run_path = args.output_root / "dry_run_plan.json"
        dry_run_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Dry run OK. Wrote task plan to {dry_run_path}", flush=True)
        return

    runtime = EvalRuntime(args)
    lookup = VariantLookup()
    summary_paths: list[pathlib.Path] = []
    errors: list[dict[str, Any]] = []
    start = time.time()
    for idx, spec in enumerate(specs, 1):
        print(f"\n=== Task {idx}/{len(specs)} ===", flush=True)
        try:
            summary_paths.append(run_pair(runtime, lookup, spec, args))
        except Exception as exc:
            errors.append({"suite": spec.suite, "base_task": spec.name, "error": repr(exc)})
            print(f"ERROR [{spec.suite}] {spec.name}: {exc}", flush=True)
            if args.fail_fast:
                raise

    batch_summary = {
        "suites": args.suites,
        "condition": args.condition,
        "category": args.category,
        "pert_suffix": args.pert_suffix,
        "allow_variant_fallback": args.allow_variant_fallback,
        "num_pairs": args.num_pairs,
        "seed": args.seed,
        "deterministic_reset": args.deterministic_reset,
        "deterministic_reset_seed": args.deterministic_reset_seed,
        "corrector_checkpoint": str(args.corrector_checkpoint) if args.corrector_checkpoint else None,
        "correction_alpha": args.correction_alpha if args.corrector_checkpoint else None,
        "base_tasks": len(specs),
        "expected_rollouts": expected_rollouts,
        "summary_count": len(summary_paths),
        "summaries_txt": str(args.output_root / "summaries.txt"),
        "elapsed_s": time.time() - start,
        "errors": errors,
    }
    write_summary_list(args.output_root, summary_paths, batch_summary)
    print(f"\nDone. Wrote {len(summary_paths)} summary paths to {args.output_root / 'summaries.txt'}", flush=True)
    if errors:
        print(f"Errors: {len(errors)}; see {args.output_root / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
