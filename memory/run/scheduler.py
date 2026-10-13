"""Hold the phase cursor, the held-object belief and the control ownership.

The scheduler answers one question per step -- who emits this step's action --
and applies the verdicts that come back: a phase ending (PhaseDecision) moves
the cursor and decides whether the next phase gets an intervention, an
intervention ending (ExecutionResult) hands control back.  It does not judge
phases, diagnose failures, compute t*, execute actions, map poses or record.

Rules carried over from the eval loop's phase hook, unchanged:
  * the run starts at the diagnosed phase, whose t* intervention already covers
    it, so only the phases after it get corrected again;
  * the held-object belief only follows a semantic completion: a completed Pick
    sets it, a completed Place clears it, a timeout advance leaves it alone;
  * a Place phase is never handed an intervention unless the object is believed
    held;
  * only the first intervention comes from the repair request; later phases are
    armed by the same rule without re-diagnosing or recomputing t*;
  * every control hand-over -- to an intervention, back to the policy, or past
    a phase -- drops the actions the policy has not executed yet.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from memory.types import ActionSource, ExecutionResult, PhaseSpec

_PLACE_SKILLS = {"PlaceIn", "PlaceOn"}


class RunScheduler:
    """Own the cursor, the belief and who holds control."""

    def __init__(
        self,
        arm: Callable[[PhaseSpec, Mapping[str, Any]], Any | None] | None = None,
        policy: Any | None = None,
        continuation: bool = True,
    ) -> None:
        """``arm`` prepares one phase's intervention and returns its handle.

        The entry point injects it (retrieve a candidate, prepare the executor);
        a handle answers step/observe/close.  Returning None means the phase
        could not be armed and falls back to the policy.

        ``policy`` holds the unexecuted actions of the last query; every
        control hand-over drops them through its ``reset``.

        ``continuation`` False keeps only the request intervention: the plan
        stays empty, so no completion window opens and no phase is armed again.
        """
        self._arm = arm
        self._policy = policy
        self._continuation = bool(continuation)
        self._reset()

    # ---- lifecycle

    def begin(
        self,
        *,
        task: str,
        phases: Sequence[PhaseSpec],
        request: Any | None = None,
    ) -> None:
        """Start one run.

        ``request`` is the repair request (RepairChecker's output); without one
        the run has no phase plan and the scheduler stays inert.  With one, the
        plan starts at the diagnosed phase, whose t* intervention the caller
        already replayed.
        """
        self._reset()
        self._task = str(task)
        self._request = request
        if request is None:
            return
        if not self._continuation:
            # Single intervention: the t* repair owns the diagnosed phase and
            # the run ends there, so no phase is corrected again.
            return
        plan = tuple(request.plan) if request.plan else tuple(phases)
        index = plan.index(request.phase) if request.phase in plan else 0
        self._plan = plan[index:]

    def decide(self, observation: Mapping[str, Any]) -> ActionSource:
        """Who emits this step's action."""
        if self._finished:
            return ActionSource("stop")
        if self._intervention is not None:
            return ActionSource("intervention")
        request = self._request
        if request is not None and not self._first_armed:
            # The first intervention repairs the diagnosed phase; it is armed
            # on the first step the policy does not cover.
            self._first_armed = True
            self._intervention = self._arm_phase(request.phase, observation)
            if self._intervention is not None:
                return ActionSource("intervention")
        return ActionSource("vla")

    def apply(
        self, verdict: Any, observation: Mapping[str, Any] | None = None
    ) -> str | None:
        """Apply one verdict; returns what was decided when a phase advanced.

        ``ExecutionResult`` closes the intervention on finished or timeout and
        keeps it on arrived (the segment still runs).  ``PhaseDecision`` with an
        advance moves the cursor, updates the belief and either arms the next
        phase or reports why not.
        """
        if isinstance(verdict, ExecutionResult):
            if verdict.outcome in {"finished", "timeout"}:
                self._release()
            return None
        if not getattr(verdict, "advance", False):
            return None
        return self._advance(verdict, observation)

    def finish(self, success: bool) -> None:
        """End the run; the intervention, if any, is dropped."""
        self._release()
        self._success = bool(success)
        self._finished = True

    # ---- state

    @property
    def task(self) -> str:
        return self._task

    @property
    def plan(self) -> tuple[PhaseSpec, ...]:
        return self._plan

    @property
    def phase_index(self) -> int:
        return self._phase_index

    @property
    def active_phase(self) -> PhaseSpec | None:
        if self._phase_index >= len(self._plan):
            return None
        return self._plan[self._phase_index]

    @property
    def holding(self) -> bool:
        return self._holding

    @property
    def intervention(self) -> Any | None:
        """The armed intervention handle, for the caller to step and observe."""
        return self._intervention

    # ---- internals

    def _reset(self) -> None:
        self._task = ""
        self._request: Any | None = None
        self._plan: tuple[PhaseSpec, ...] = ()
        self._phase_index = 0
        self._holding = False
        self._intervention: Any | None = None
        self._first_armed = False
        self._finished = False
        self._success = False

    def _advance(
        self, decision: Any, observation: Mapping[str, Any] | None
    ) -> str | None:
        """Move the cursor and decide the next phase's fate."""
        # A phase boundary ends the intervention still running and drops the
        # actions the policy has not executed yet.
        self._release()
        completed = self.active_phase
        self._phase_index += 1
        if decision.semantic_completed and completed is not None:
            # The belief only follows a semantic completion, never a timeout.
            if completed.skill == "Pick":
                self._holding = True
            elif completed.skill in _PLACE_SKILLS:
                self._holding = False
        next_phase = self.active_phase
        if next_phase is None:
            return "no_next_phase"
        if not self._holding and next_phase.skill in _PLACE_SKILLS:
            # A Place phase's memory assumes the object is in hand; a Pick that
            # advanced on timeout leaves that belief false, so the phase runs on
            # the policy instead.
            return "skipped_not_holding"
        self._intervention = self._arm_phase(next_phase, observation)
        return "armed" if self._intervention is not None else "arm_failed"

    def _arm_phase(
        self, phase: PhaseSpec, observation: Mapping[str, Any] | None
    ) -> Any | None:
        if self._arm is None or observation is None:
            return None
        intervention = self._arm(phase, observation)
        if intervention is not None:
            # Control passes to the intervention; the policy chunk is stale.
            self._drop_policy_chunk()
        return intervention

    def _release(self) -> None:
        """End the intervention, if one is held, and drop the policy chunk.

        Every hand-over goes through here: control changes owner only after
        the segment that was running has been closed.
        """
        if self._intervention is not None:
            close = getattr(self._intervention, "close", None)
            if callable(close):
                close()
            self._intervention = None
        self._drop_policy_chunk()

    def _drop_policy_chunk(self) -> None:
        """Drop the actions the policy has not executed yet."""
        if self._policy is not None:
            self._policy.reset()
