"""The policy, seen from the LIBERO side: observation + instruction -> chunk.

One ``act`` call runs the whole best-of-N query -- parallel workers or serial,
autoregressive future-state and value prediction, the value-based pick among
the queries, and the action unnormalisation and VAE latent capture that live
inside ``get_action``.  The chunk it returns is the inference unit, the
recorder's unit and the training unit.

Nothing here rewrites inference or touches aloha; it calls the existing
``cosmos_utils`` entry points.  The unexecuted actions of the
chunk it returned are held here, so handing control back consumes the rest
of the chunk (``reset``).

Nothing here judges, schedules, retrieves, executes or records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ChunkSelection:
    """What the last ``act`` call chose among the queries."""

    actions: np.ndarray
    future_image_predictions: Any
    value: float
    seed: int


class PolicyAdapter:
    """Query the policy for one action chunk."""

    def __init__(
        self,
        cfg: Any,
        model: Any,
        dataset_stats: Any,
        *,
        planning_model: Any | None = None,
        worker_pool: Any | None = None,
        capture_latent: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.dataset_stats = dataset_stats
        self.planning_model = planning_model
        self.worker_pool = worker_pool
        # The caller decides: the alignment path and the vector-db export need
        # the VAE latent, a plain rollout does not.
        self.capture_latent = bool(capture_latent)
        self._log = log or (lambda _message: None)
        self._latent: list[Any] = []
        self.last: ChunkSelection | None = None
        self._cursor = 0

    def act(self, observation: Any, instruction: str) -> np.ndarray:
        """Return one action chunk for this observation and instruction."""
        cfg = self.cfg
        self._latent = []
        self._cursor = 0
        original_condition = self._install_latent_hook()
        try:
            return self._query(observation, instruction, num_queries=cfg.num_queries_best_of_n)
        finally:
            if original_condition is not None:
                self.model.get_data_and_condition = original_condition

    def _query(self, observation: Any, instruction: str, *, num_queries: int) -> np.ndarray:
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action,
            get_future_state_prediction,
            get_qvalue_prediction,
            get_value_prediction,
            query_model_parallel,
        )

        cfg = self.cfg
        if cfg.use_parallel_inference and num_queries > 1 and self.worker_pool is not None \
                and self.worker_pool.initialized:
            query_results = query_model_parallel(
                cfg, observation, instruction, self.worker_pool, cfg.parallel_timeout
            )
            self._log(f"Parallel queries completed: {len(query_results)} results")
        else:
            query_results = []
            for query_idx in range(num_queries):
                return_dict: dict[str, Any] = {}
                action_return_dict = get_action(
                    cfg,
                    self.model,
                    self.dataset_stats,
                    observation,
                    instruction,
                    seed=cfg.seed + query_idx,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=not (
                        cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                    ),
                )
                return_dict["actions"] = action_return_dict["actions"]
                if cfg.ar_future_prediction:
                    future_state_return_dict = get_future_state_prediction(
                        cfg,
                        model=self.planning_model if self.planning_model is not None else self.model,
                        data_batch=action_return_dict["data_batch"],
                        generated_latent_with_action=action_return_dict["generated_latent"],
                        orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                        future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                        future_wrist_image_latent_idx=action_return_dict["latent_indices"]["future_wrist_image_latent_idx"],
                        future_wrist_image2_latent_idx=action_return_dict["latent_indices"]["future_wrist_image2_latent_idx"],
                        future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                        future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                        seed=cfg.seed + query_idx,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                        use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                        num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                        future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                    )
                    return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                else:
                    return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]
                if cfg.ar_value_prediction:
                    value_return_dict = get_value_prediction(
                        cfg,
                        model=self.planning_model if self.planning_model is not None else self.model,
                        data_batch=action_return_dict["data_batch"],
                        future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                        use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                        num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                    )
                    return_dict["value_prediction"] = value_return_dict["value_prediction"]
                elif cfg.ar_qvalue_prediction:
                    value_return_dict = get_qvalue_prediction(
                        cfg,
                        model=self.planning_model if self.planning_model is not None else self.model,
                        data_batch=action_return_dict["data_batch"],
                        action_sample=action_return_dict["generated_latent"],
                        use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                        num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                    )
                    return_dict["value_prediction"] = value_return_dict["value_prediction"]
                else:
                    return_dict["value_prediction"] = action_return_dict["value_prediction"]
                query_results.append(return_dict)

        for query_idx, return_dict in enumerate(query_results):
            self._log(
                f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): "
                f"Predicted value = {return_dict['value_prediction']:.4f}"
            )
        # Highest predicted value wins the chunk.
        best_seed, best_return_dict = max(
            (
                (cfg.seed + query_idx, (d["actions"], d["future_image_predictions"], d["value_prediction"]))
                for query_idx, d in enumerate(query_results)
            ),
            key=lambda item: item[1][2],
        )
        self.last = ChunkSelection(
            actions=np.asarray(best_return_dict[0]),
            future_image_predictions=best_return_dict[1],
            value=float(best_return_dict[2]),
            seed=int(best_seed),
        )
        self._log(f"Selected seed {best_seed} with value = {best_return_dict[2]:.4f}")
        return self.last.actions

    def reset(self) -> None:
        """Hand control back: consume the chunk unexecuted actions.

        The query record (``last``) and the captured latent stay readable: the caller
        reads them right after a query, including when control was handed over.
        """
        if self.last is not None:
            self._cursor = len(self.last.actions)

    def has_actions(self) -> bool:
        """Whether the current chunk still holds an unexecuted action."""
        return self.last is not None and self._cursor < len(self.last.actions)

    def next_action(self) -> Any:
        """Return the next unexecuted action of the current chunk."""
        action = self.last.actions[self._cursor]
        self._cursor += 1
        return action

    @property
    def latent(self) -> Any | None:
        """The VAE latent captured by the last query, if capture was on."""
        return self._latent[0] if self._latent else None

    # ---- internals

    def _install_latent_hook(self) -> Any | None:
        """Wrap the model's data/condition hook for one query; returns the original."""
        if not self.capture_latent:
            return None
        original = self.model.get_data_and_condition

        def hook(data_batch: Any) -> Any:
            raw, latent, cond = original(data_batch)
            if not self._latent:
                self._latent.append(latent.detach().cpu())
            return raw, latent, cond

        self.model.get_data_and_condition = hook
        return original
