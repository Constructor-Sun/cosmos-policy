"""Skill transition session, intervention, and coordinator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from memory_system.execute.plan import PhaseSpec
from memory_system.execute.recovery.retrieval import mean_ee_states


def _arguments_key(arguments: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((str(key), str(value)) for key, value in (arguments or {}).items())
    )


@dataclass
class SkillTransitionSession:
    task_name: str
    demo_id: str
    held_item: str | None = None
    held_observation: Any | None = None
    pending_place_aligner: Any | None = None


@dataclass
class TransitionIntervention:
    kind: str
    gripper_action: float
    controller: Any | None = None
    step_budget: int = 0
    deferred_aligner: Any | None = None


class StrictReadyPoseResolver:
    """Resolve a next-phase ready pose using existing select() plus exact assertion.

    The memory query still uses ``FeasibleRecoveryMemory.select()``.  After the
    query we assert that the candidate matches the exact demo, planner step,
    skill, and arguments, and that there is exactly one such target.  On any
    mismatch we log and return ``None`` so the caller can resume next-phase VLA
    without falling back across demos or steps.
    """

    def __init__(
        self,
        memory: Any,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.memory = memory
        self.log = log or (lambda _msg: None)

    def resolve(
        self,
        task_name: str,
        demo_id: str,
        phase: PhaseSpec,
    ) -> dict[str, Any] | None:
        candidates = self.memory.select(
            task_name,
            phase.planner_step_id,
            phase.skill,
            phase.arguments,
        )
        matches = [
            candidate for candidate in candidates
            if (
                str(candidate.get("demo_id")) == str(demo_id)
                and int(candidate["planner_step_id"]) == int(phase.planner_step_id)
                and candidate["skill"] == phase.skill
                and _arguments_key(candidate.get("arguments", {}))
                == _arguments_key(phase.arguments)
            )
        ]
        if len(matches) != 1:
            self.log(
                "[SKILL_TRANSITION] ready-pose mismatch: "
                f"demo={demo_id} step={phase.planner_step_id} "
                f"skill={phase.skill} args={phase.arguments} "
                f"matches={len(matches)}"
            )
            return None
        return matches[0]


class SkillTransitionCoordinator:
    """Episode-local coordinator for skill effects and transition routing.

    The coordinator owns the session and applies skill effects after each phase
    advance.  It does not keep an active controller; the eval loop executes the
    returned ``TransitionIntervention``.  Concrete planners are injected from
    outside so this module stays a thin adapter.
    """

    def __init__(
        self,
        session: SkillTransitionSession,
        memory: Any | None = None,
        *,
        mode: str = "curobo",
        held_extractor: Callable[..., Any] | None = None,
        held_planner: Callable[..., Any] | None = None,
        motion_planner: Callable[..., Any] | None = None,
        spatial_candidate_selector: Callable[..., Any] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.session = session
        self.memory = memory
        self.mode = mode
        self.held_extractor = held_extractor
        self.held_planner = held_planner
        self.motion_planner = motion_planner
        self.spatial_candidate_selector = spatial_candidate_selector
        self.log = log or (lambda _msg: None)
        self.resolver = (
            StrictReadyPoseResolver(memory, log=self.log)
            if memory is not None else None
        )

    def apply_skill_effects(self, completed_phase: PhaseSpec) -> None:
        """Update episode-local held state after a phase advance."""
        skill = completed_phase.skill
        if skill == "Pick":
            self.session.held_item = completed_phase.arguments.get("item")
            self.session.held_observation = None
            self.session.pending_place_aligner = None
        elif skill in {"PlaceIn", "PlaceOn"}:
            self.session.held_item = None
            self.session.held_observation = None
            self.session.pending_place_aligner = None
        # Open / Close / TurnOn: held state unchanged.

    def _gripper_action(
        self,
        completed_phase: PhaseSpec,
        next_phase: PhaseSpec,
    ) -> float:
        del completed_phase, next_phase
        # Generic rule: hold the object -> closed; otherwise -> open.
        return 1.0 if self.session.held_item is not None else 0.0

    def on_phase_advance(
        self,
        *,
        observation: Any,
        completed_phase: PhaseSpec,
        next_phase: PhaseSpec | None,
        env: Any,
        cfg: Any,
        log_file: Any,
        episode_id: Any,
    ) -> TransitionIntervention | None:
        """Apply skill effects, assert ready pose, and route to a planner."""
        self.apply_skill_effects(completed_phase)

        if next_phase is None:
            return None

        if self.resolver is None:
            self.log("[SKILL_TRANSITION] no ready-pose memory configured; resume VLA")
            return None

        is_place = (
            self.session.held_item is not None
            and next_phase.skill in {"PlaceIn", "PlaceOn"}
        )
        targets = []
        if (
            is_place or next_phase.skill == "Pick"
        ) and self.spatial_candidate_selector is not None:
            try:
                targets = list(self.spatial_candidate_selector(
                    task_name=self.session.task_name,
                    phase=next_phase,
                    observation=observation,
                    env=env,
                    cfg=cfg,
                    log_file=log_file,
                ))
            except Exception as exc:
                self.log(f"[TARGET_3D] candidate selection failed: {exc}")

        if not targets:
            target = self.resolver.resolve(
                self.session.task_name,
                self.session.demo_id,
                next_phase,
            )
            if target is None:
                return None
            targets = [target]

        target = targets[0]
        ready_pose = (
            mean_ee_states(targets) if len(targets) > 1 else target["ee_states"]
        )
        target_demos = tuple(str(item.get("demo_id", "")) for item in targets)
        gripper_action = self._gripper_action(completed_phase, next_phase)

        if is_place:
            if self.mode == "simple":
                from memory_system.execute.planner.place_fine_aligner import (
                    PlaceFineAligner,
                )

                aligner = PlaceFineAligner(ready_pose)
                self.session.pending_place_aligner = aligner
                return TransitionIntervention(
                    kind="simple_place",
                    gripper_action=gripper_action,
                    controller=None,
                    step_budget=0,
                    deferred_aligner=aligner,
                )

            if self.held_extractor is None or self.held_planner is None:
                self.log(
                    "[SKILL_TRANSITION] held-object planner not configured; resume VLA"
                )
                return None

            held = self.held_extractor(
                observation=observation,
                env=env,
                cfg=cfg,
                item=self.session.held_item,
            )
            if held is None:
                self.log("[SKILL_TRANSITION] held-object extraction failed; resume VLA")
                return None
            self.session.held_observation = held

            candidate_demo = str(target.get("demo_id", self.session.demo_id))
            result = self.held_planner(
                held=held,
                ready_pose=ready_pose,
                observation=observation,
                env=env,
                cfg=cfg,
                task_name=self.session.task_name,
                demo_id=candidate_demo,
                episode_id=episode_id,
                log_file=log_file,
            )
            if result is None:
                self.log(f"[TARGET_3D] planner failed demos={target_demos}; resume VLA")
                return None
            self.log(f"[TARGET_3D] planner accepted demos={target_demos}")
            return TransitionIntervention(
                kind="held_object",
                gripper_action=gripper_action,
                controller=result.controller,
                step_budget=result.correction_steps,
            )

        # For Place -> Close, skip the motion planner and let the VLA continue
        # directly into the Close phase.
        if (
            completed_phase.skill in {"PlaceIn", "PlaceOn"}
            and next_phase.skill == "Close"
        ):
            return TransitionIntervention(
                kind="vla",
                gripper_action=gripper_action,
                controller=None,
                step_budget=0,
            )

        if self.motion_planner is not None:
            candidate_demo = str(target.get("demo_id", self.session.demo_id))
            result = self.motion_planner(
                ready_pose=ready_pose,
                observation=observation,
                env=env,
                cfg=cfg,
                task_name=self.session.task_name,
                demo_id=candidate_demo,
                episode_id=episode_id,
                log_file=log_file,
            )
            if result is None:
                if next_phase.skill == "Pick":
                    self.log(
                        f"[TARGET_3D] planner failed demos={target_demos}; resume VLA"
                    )
                return None
            if next_phase.skill == "Pick":
                self.log(f"[TARGET_3D] planner accepted demos={target_demos}")
            return TransitionIntervention(
                kind="motion",
                gripper_action=gripper_action,
                controller=result.controller,
                step_budget=result.correction_steps,
            )

        return TransitionIntervention(
            kind="vla",
            gripper_action=gripper_action,
            controller=None,
            step_budget=0,
        )
