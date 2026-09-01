"""Tests for skill transition session, effects, resolver, and coordinator."""
from __future__ import annotations

import unittest

from memory_system.execute.plan import PhaseSpec
from memory_system.execute.skill_transition import (
    SkillTransitionCoordinator,
    SkillTransitionSession,
    StrictReadyPoseResolver,
)


class FakeMemory:
    def __init__(self, targets):
        self.targets = list(targets)

    def select(self, task_name, planner_step_id, skill, arguments):
        del task_name, planner_step_id, skill, arguments
        return list(self.targets)


def target(
    demo_id="demo_0",
    step=2,
    skill="PlaceIn",
    arguments=None,
    ee_states=None,
):
    return {
        "demo_id": demo_id,
        "planner_step_id": step,
        "skill": skill,
        "arguments": dict(arguments) if arguments is not None else {"item": "mug"},
        "ee_states": ee_states if ee_states is not None else [0.0] * 6,
    }


class SkillEffectsTest(unittest.TestCase):
    def test_pick_sets_held_item_and_clears_old_held_state(self):
        session = SkillTransitionSession(
            task_name="task",
            demo_id="demo_0",
            held_item="old",
            held_observation="old_obs",
            pending_place_aligner="old_aligner",
        )
        coordinator = SkillTransitionCoordinator(session)
        coordinator.apply_skill_effects(PhaseSpec(1, "Pick", {"item": "mug"}))
        self.assertEqual(session.held_item, "mug")
        self.assertIsNone(session.held_observation)
        self.assertIsNone(session.pending_place_aligner)

    def test_place_clears_held_state(self):
        session = SkillTransitionSession(
            task_name="task",
            demo_id="demo_0",
            held_item="mug",
            held_observation="obs",
            pending_place_aligner="aligner",
        )
        coordinator = SkillTransitionCoordinator(session)
        coordinator.apply_skill_effects(PhaseSpec(2, "PlaceIn", {"item": "mug"}))
        self.assertIsNone(session.held_item)
        self.assertIsNone(session.held_observation)
        self.assertIsNone(session.pending_place_aligner)


class StrictReadyPoseResolverTest(unittest.TestCase):
    def test_unique_match_returns_candidate(self):
        memory = FakeMemory([target()])
        resolver = StrictReadyPoseResolver(memory)
        result = resolver.resolve("task", "demo_0", PhaseSpec(2, "PlaceIn", {"item": "mug"}))
        self.assertIsNotNone(result)
        self.assertEqual(result["demo_id"], "demo_0")

    def test_missing_match_returns_none(self):
        memory = FakeMemory([target(demo_id="demo_1")])
        resolver = StrictReadyPoseResolver(memory)
        self.assertIsNone(
            resolver.resolve("task", "demo_0", PhaseSpec(2, "PlaceIn", {"item": "mug"}))
        )

    def test_multiple_matches_return_none(self):
        memory = FakeMemory([target(), target()])
        resolver = StrictReadyPoseResolver(memory)
        self.assertIsNone(
            resolver.resolve("task", "demo_0", PhaseSpec(2, "PlaceIn", {"item": "mug"}))
        )


class CoordinatorRoutingTest(unittest.TestCase):
    def _coordinator(self, mode="curobo", **kwargs):
        session = SkillTransitionSession(task_name="task", demo_id="demo_0")
        return SkillTransitionCoordinator(
            session,
            FakeMemory([target()]),
            mode=mode,
            **kwargs,
        )

    def test_curobo_place_routes_to_held_planner(self):
        held = object()
        result = type("Result", (), {"controller": "controller", "correction_steps": 10})()

        def extractor(**kwargs):
            self.assertEqual(kwargs["item"], "mug")
            return held

        def planner(**kwargs):
            self.assertIs(kwargs["held"], held)
            return result

        coordinator = self._coordinator(
            held_extractor=extractor,
            held_planner=planner,
        )
        coordinator.session.held_item = "mug"
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(1, "Pick", {"item": "mug"}),
            next_phase=PhaseSpec(2, "PlaceIn", {"item": "mug"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "held_object")
        self.assertEqual(intervention.controller, "controller")
        self.assertEqual(intervention.step_budget, 10)
        self.assertEqual(intervention.gripper_action, 1.0)
        self.assertIs(coordinator.session.held_observation, held)

    def test_curobo_place_averages_top3_and_plans_once(self):
        candidates = [
            target(
                demo_id=f"demo_{i}",
                ee_states=[float(i)] * 3 + [0.0] * 3,
            )
            for i in range(1, 4)
        ]
        attempts = []
        result = type("Result", (), {"controller": "ok", "correction_steps": 4})()

        def planner(**kwargs):
            attempts.append((kwargs["demo_id"], list(kwargs["ready_pose"])))
            return result

        coordinator = self._coordinator(
            held_extractor=lambda **kwargs: object(),
            held_planner=planner,
            spatial_candidate_selector=lambda **kwargs: candidates,
        )
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(1, "Pick", {"item": "mug"}),
            next_phase=PhaseSpec(2, "PlaceIn", {"item": "mug"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )

        self.assertEqual(attempts, [("demo_1", [2.0] * 3 + [0.0] * 3)])
        self.assertEqual(intervention.controller, "ok")
        self.assertEqual(coordinator.session.demo_id, "demo_0")

    def test_simple_place_returns_deferred_aligner(self):
        session = SkillTransitionSession(task_name="task", demo_id="demo_0")
        coordinator = SkillTransitionCoordinator(
            session,
            FakeMemory([target(skill="PlaceOn")]),
            mode="simple",
        )
        coordinator.session.held_item = "mug"
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(1, "Pick", {"item": "mug"}),
            next_phase=PhaseSpec(2, "PlaceOn", {"item": "mug"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "simple_place")
        self.assertIsNone(intervention.controller)
        self.assertIsNotNone(intervention.deferred_aligner)
        self.assertIs(coordinator.session.pending_place_aligner, intervention.deferred_aligner)
        self.assertEqual(intervention.gripper_action, 1.0)

    def test_no_held_generic_motion(self):
        result = type("Result", (), {"controller": "motion_controller", "correction_steps": 20})()

        def motion_planner(**kwargs):
            return result

        session = SkillTransitionSession(task_name="task", demo_id="demo_0")
        coordinator = SkillTransitionCoordinator(
            session,
            FakeMemory([target(skill="Pick", step=2, arguments={"item": "mug"})]),
            motion_planner=motion_planner,
        )
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(1, "TurnOn", {}),
            next_phase=PhaseSpec(2, "Pick", {"item": "mug"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "motion")
        self.assertEqual(intervention.controller, "motion_controller")
        self.assertEqual(intervention.step_budget, 20)
        self.assertEqual(intervention.gripper_action, 0.0)

    def test_later_pick_averages_top3_and_plans_once(self):
        candidates = [
            target(
                demo_id=f"demo_{i}",
                step=3,
                skill="Pick",
                arguments={"item": "mug_b"},
                ee_states=[float(i)] * 3 + [0.0] * 3,
            )
            for i in range(1, 4)
        ]
        attempts = []

        def motion_planner(**kwargs):
            attempts.append((kwargs["demo_id"], list(kwargs["ready_pose"])))
            return None

        session = SkillTransitionSession(task_name="task", demo_id="demo_0")
        coordinator = SkillTransitionCoordinator(
            session,
            FakeMemory([
                target(
                    step=3,
                    skill="Pick",
                    arguments={"item": "mug_b"},
                )
            ]),
            motion_planner=motion_planner,
            spatial_candidate_selector=lambda **kwargs: candidates,
        )
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(2, "PlaceOn", {"item": "mug_a"}),
            next_phase=PhaseSpec(3, "Pick", {"item": "mug_b"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )

        self.assertEqual(attempts, [("demo_1", [2.0] * 3 + [0.0] * 3)])
        self.assertIsNone(intervention)
        self.assertEqual(session.demo_id, "demo_0")

    def test_planner_returning_none_resumes_vla(self):
        coordinator = self._coordinator(
            held_extractor=lambda **kwargs: object(),
            held_planner=lambda **kwargs: None,
        )
        coordinator.session.held_item = "mug"
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(1, "Pick", {"item": "mug"}),
            next_phase=PhaseSpec(2, "PlaceIn", {"item": "mug"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNone(intervention)

    def test_consecutive_two_pick_place_uses_correct_item_and_ready_pose(self):
        session = SkillTransitionSession(task_name="task", demo_id="demo_0")
        memory = FakeMemory(
            [
                target(
                    skill="PlaceOn",
                    step=2,
                    arguments={"item": "mug_a"},
                    ee_states=[1.0] * 6,
                ),
                target(
                    skill="Pick",
                    step=3,
                    arguments={"item": "mug_b"},
                    ee_states=[2.0] * 6,
                ),
                target(
                    skill="PlaceOn",
                    step=4,
                    arguments={"item": "mug_b"},
                    ee_states=[3.0] * 6,
                ),
            ]
        )
        coordinator = SkillTransitionCoordinator(
            session,
            memory,
            mode="simple",
        )

        # First Pick -> Place.
        first = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(1, "Pick", {"item": "mug_a"}),
            next_phase=PhaseSpec(2, "PlaceOn", {"item": "mug_a"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(first)
        self.assertEqual(session.held_item, "mug_a")
        self.assertEqual(first.kind, "simple_place")
        self.assertEqual(first.deferred_aligner.ready_pose.tolist(), [1.0] * 6)

        # First Place -> second Pick.
        second_pick = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(2, "PlaceOn", {"item": "mug_a"}),
            next_phase=PhaseSpec(3, "Pick", {"item": "mug_b"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(second_pick)
        self.assertIsNone(session.held_item)
        self.assertEqual(second_pick.kind, "vla")
        self.assertEqual(second_pick.gripper_action, 0.0)

        # Second Pick -> Place.
        second = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(3, "Pick", {"item": "mug_b"}),
            next_phase=PhaseSpec(4, "PlaceOn", {"item": "mug_b"}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(second)
        self.assertEqual(session.held_item, "mug_b")
        self.assertEqual(second.kind, "simple_place")
        self.assertEqual(second.deferred_aligner.ready_pose.tolist(), [3.0] * 6)

    def test_place_to_close_clears_held_and_uses_open_gripper(self):
        session = SkillTransitionSession(
            task_name="task",
            demo_id="demo_0",
            held_item="mug",
            held_observation="obs",
            pending_place_aligner="aligner",
        )
        coordinator = SkillTransitionCoordinator(
            session,
            FakeMemory([target(skill="Close", step=5, arguments={})]),
        )
        intervention = coordinator.on_phase_advance(
            observation={},
            completed_phase=PhaseSpec(2, "PlaceIn", {"item": "mug"}),
            next_phase=PhaseSpec(5, "Close", {}),
            env=None,
            cfg=None,
            log_file=None,
            episode_id=0,
        )
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "vla")
        self.assertEqual(intervention.gripper_action, 0.0)
        self.assertIsNone(session.held_item)
        self.assertIsNone(session.held_observation)
        self.assertIsNone(session.pending_place_aligner)


if __name__ == "__main__":
    unittest.main()
