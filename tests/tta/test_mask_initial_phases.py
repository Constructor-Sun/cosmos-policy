"""Regression tests for phase initialization, without loading a simulator or GPU."""
from types import SimpleNamespace

import pytest

from experiments.tta.mask_view.regenerate_robot_masks import (
    initialize_mask_phases, order_same_destination_placements,
    same_destination_placement_pairs, infer_observed_pair_order,
)
from memory_system.execute.plan import PhaseSpec


def fake_env(button_qpos=None):
    def joint_name2id(name):
        if button_qpos is None or name != "flat_stove_1_button":
            raise ValueError(name)
        return 0
    sim = SimpleNamespace(
        model=SimpleNamespace(joint_name2id=joint_name2id, jnt_qposadr=[0]),
        data=SimpleNamespace(qpos=[button_qpos]),
    )
    return SimpleNamespace(env=SimpleNamespace(sim=sim))


def pot_plan():
    return (
        PhaseSpec(0, "TurnOn", {"target": "flat_stove_1"}),
        PhaseSpec(1, "Pick", {"item": "moka_pot_1"}),
        PhaseSpec(2, "PlaceOn", {"item": "moka_pot_1", "target": "flat_stove_1_cook_region"}),
        PhaseSpec(3, "Pick", {"item": "moka_pot_2"}),
        PhaseSpec(4, "PlaceOn", {"item": "moka_pot_2", "target": "flat_stove_1_cook_region"}),
    )


def test_kscene8_stove_already_on_keeps_both_pot_sequences():
    phases = pot_plan()
    kept, skipped = initialize_mask_phases(phases, fake_env(1.0))
    assert kept == phases[1:]
    assert [p.planner_step_id for p in kept] == [1, 2, 3, 4]
    assert [p.skill for p in kept] == ["Pick", "PlaceOn", "Pick", "PlaceOn"]
    assert skipped == [{"planner_step_id": 0, "skill": "TurnOn",
                        "arguments": {"target": "flat_stove_1"},
                        "reason": "initial_state_already_satisfied"}]


@pytest.mark.parametrize("qpos", [0.0, 0.49, -1.0])
def test_stove_off_keeps_turnon(qpos):
    phases = pot_plan()
    assert initialize_mask_phases(phases, fake_env(qpos)) == (phases, [])


@pytest.mark.parametrize("qpos", [0.5, 1.0])
def test_turnon_threshold(qpos):
    assert initialize_mask_phases(pot_plan(), fake_env(qpos))[0] == pot_plan()[1:]


@pytest.mark.parametrize("qpos,should_skip", [(0.0, True), (1.0, False)])
def test_turnoff_skipped_only_when_already_off(qpos, should_skip):
    phase = PhaseSpec(0, "TurnOff", {"target": "flat_stove_1"})
    kept, skipped = initialize_mask_phases((phase,), fake_env(qpos))
    assert bool(skipped) == should_skip
    assert kept == (() if should_skip else (phase,))


def test_missing_joint_keeps_phase_conservatively():
    phases = pot_plan()
    assert initialize_mask_phases(phases, fake_env()) == (phases, [])


def test_open_close_are_not_dropped():
    phases = (PhaseSpec(0, "Open", {"target": "flat_stove_1"}),
              PhaseSpec(1, "Close", {"target": "flat_stove_1"}))
    assert initialize_mask_phases(phases, fake_env(1.0)) == (phases, [])


def test_initial_state_is_checked_per_episode():
    phases = pot_plan()
    assert initialize_mask_phases(phases, fake_env(1.0))[0] == phases[1:]
    assert initialize_mask_phases(phases, fake_env(0.0))[0] == phases


def test_actual_second_pot_first_reorders_whole_pairs_preserving_ids():
    phases = pot_plan()[1:]
    ordered = order_same_destination_placements(phases, "moka_pot_2")
    assert ordered == phases[2:] + phases[:2]
    assert [p.planner_step_id for p in ordered] == [3, 4, 1, 2]
    assert [p.arguments["item"] for p in ordered] == ["moka_pot_2", "moka_pot_2", "moka_pot_1", "moka_pot_1"]


@pytest.mark.parametrize("item", ["moka_pot_1", "unknown_object", None])
def test_unchanged_or_unknown_first_object_keeps_goal_order(item):
    phases = pot_plan()[1:]
    assert order_same_destination_placements(phases, item) == phases


def test_different_destinations_are_not_reordered():
    phases = list(pot_plan()[1:])
    phases[3] = PhaseSpec(4, "PlaceOn", {"item": "moka_pot_2", "target": "another_destination"})
    assert not same_destination_placement_pairs(phases)
    assert order_same_destination_placements(phases, "moka_pot_2") == tuple(phases)


def test_unsatisfied_control_precondition_is_not_reordered_away():
    phases = pot_plan()
    assert not same_destination_placement_pairs(phases)
    assert order_same_destination_placements(phases, "moka_pot_2") == phases


def test_nonindependent_plan_does_not_probe_cached_frames():
    kept, evidence = infer_observed_pair_order(
        pot_plan(), None, None, task_name="test", episode_id="test",
        object_query=None, point_cloud=None)
    assert kept == pot_plan()
    assert evidence is None


def test_order_probe_uses_cached_observations_without_simulator(monkeypatch):
    import memory_system.tta.phase_record as phase_record

    consumed = []

    class FakeRecorder:
        def __init__(self, phases, **kwargs):
            self.active_phase = phases[0]
            self.spans = [SimpleNamespace(actual_object=None, close_event_confirmed_step=None)]

        def begin(self, **kwargs):
            pass

        def observe(self, *, step, action, obs, target_points):
            consumed.append(obs["marker"])
            if step == 1:
                self.spans[0].actual_object = {"name": "moka_pot_2", "distance": 0.04}
                self.spans[0].close_event_confirmed_step = step

    monkeypatch.setattr(phase_record, "PhaseEventRecorder", FakeRecorder)
    observations = [{"marker": i} for i in range(3)]
    kept, evidence = infer_observed_pair_order(
        pot_plan()[1:], [None] * 3, observations,
        task_name="test", episode_id="test", object_query=None,
        point_cloud=SimpleNamespace(points=lambda obs, item: None))
    assert [p.planner_step_id for p in kept] == [3, 4, 1, 2]
    assert consumed == [0, 1]
    assert observations == [{"marker": i} for i in range(3)]
    assert evidence["source"] == "first_confirmed_interaction_in_cached_recording_replay"
    assert evidence["step"] == 1
    assert evidence["changed"] is True
