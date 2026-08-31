"""Common action-chunk lifecycle for test-time skill completion."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_ACTION_CHUNKS = 3


@dataclass(frozen=True)
class SkillDecision:
    """Decision returned at frame/chunk boundaries."""

    advance: bool
    semantic_completed: bool
    reason: str
    action_chunks: int


class TimedSkillCompletion(ABC):
    """Base lifecycle that combines a rule with an action-chunk budget.

    A concrete skill supplies ``_check_rule``.  A rule match advances with
    ``reason='rule'``; otherwise the skill advances with ``reason='timeout'``
    after the configured number of completed action chunks.  Timeout is an
    execution-boundary decision, not a semantic success.
    """

    def __init__(self, *, max_action_chunks: int = DEFAULT_MAX_ACTION_CHUNKS):
        if max_action_chunks < 1:
            raise ValueError("max_action_chunks must be at least one")
        self.max_action_chunks = int(max_action_chunks)
        self.reset()

    def reset(self) -> None:
        """Start a fresh skill execution."""
        self.action_chunks = 0
        self.semantic_completed = False
        self.advance_reason: str | None = None
        self._reset_rule()

    def _reset_rule(self) -> None:
        """Allow subclasses to reset their rule state."""

    def observe_frame(self, *args: Any, **kwargs: Any) -> SkillDecision:
        """Consume one frame and update semantic completion state."""
        if self.advance_reason is None and self._check_rule(*args, **kwargs):
            self.semantic_completed = True
            self.advance_reason = "rule"
        return self.decision

    def finish_action_chunk(self) -> SkillDecision:
        """Close one action chunk and apply the timeout fallback.

        Call this exactly once for every action chunk executed by the active
        skill.  The caller should clear any remaining queued actions before
        advancing when ``decision.advance`` is true.
        """
        self.action_chunks += 1
        if (
            self.advance_reason is None
            and self.action_chunks >= self.max_action_chunks
        ):
            self.advance_reason = "timeout"
        return self.decision

    @property
    def decision(self) -> SkillDecision:
        reason = self.advance_reason or "running"
        return SkillDecision(
            advance=reason != "running",
            semantic_completed=self.semantic_completed,
            reason=reason,
            action_chunks=self.action_chunks,
        )

    @abstractmethod
    def _check_rule(self, *args: Any, **kwargs: Any) -> bool:
        """Return whether the concrete skill's semantic rule is satisfied."""


class TimeoutOnlySkillCompletion(TimedSkillCompletion):
    """Completion implementation for skills without a semantic rule yet."""

    def _check_rule(self, *args: Any, **kwargs: Any) -> bool:
        return False
