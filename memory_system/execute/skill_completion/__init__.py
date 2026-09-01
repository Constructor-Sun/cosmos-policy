"""Rule-based skill-completion checks."""

from memory_system.execute.skill_completion.base import (
    DEFAULT_MAX_ACTION_CHUNKS,
    SkillDecision,
    TimedSkillCompletion,
    TimeoutOnlySkillCompletion,
)
from memory_system.execute.skill_completion.pick import (
    EMPTY_CLOSED_GAP,
    PickCompletionChecker,
    PickSkillCompletion,
)
from memory_system.execute.skill_completion.place import (
    DEFAULT_OPEN_FRAMES,
    PLACE_SKILLS,
    ReleaseSkillCompletion,
)

COMPLETION_REGISTRY: dict[str, type[TimedSkillCompletion]] = {
    "Pick": PickSkillCompletion,
    "PlaceIn": ReleaseSkillCompletion,
    "PlaceOn": ReleaseSkillCompletion,
    "Open": TimeoutOnlySkillCompletion,
    "Close": TimeoutOnlySkillCompletion,
    "TurnOn": TimeoutOnlySkillCompletion,
}

__all__ = [
    "COMPLETION_REGISTRY",
    "DEFAULT_MAX_ACTION_CHUNKS",
    "DEFAULT_OPEN_FRAMES",
    "EMPTY_CLOSED_GAP",
    "PLACE_SKILLS",
    "PickCompletionChecker",
    "PickSkillCompletion",
    "ReleaseSkillCompletion",
    "SkillDecision",
    "TimedSkillCompletion",
    "TimeoutOnlySkillCompletion",
]
