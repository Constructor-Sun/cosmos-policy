"""Ordering tests for the BDDL goal -> phase plan parser.

The rules in ``plan_from_goal_state`` are checked three ways: directly on the
goal shapes they were taken from, as a contract over every BDDL the local
libero package ships, and against the recorded LIBERO-10 demo order.  The demos
are the independent ground truth for phase order, because the plan deliberately
no longer derives from them (plan doc section 2).
"""
from __future__ import annotations

import json
import os
import unittest
from collections import Counter
from pathlib import Path

from memory_system.execute.plan import (
    _gates_placement,
    plan_from_bddl,
    plan_from_goal_state,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "skill_memory_test/libero_10/segments_ready_fixed16.json"

# Suites swept by the ordering contract below (directories under bddl_files).
CORPUS_SUITES = ("libero_10", "libero_90")

# The state predicates, which are ordered around the placements rather than by
# goal order.
STATE_SKILLS = {"Open", "Close", "TurnOn", "TurnOff"}

# Tasks whose demo order permutes independent placements: the demos pick the
# second object first, the goal names the first one first.  Both orders reach
# the goal state, so the plan keeps goal order.
ORDER_EXCEPTIONS = {
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": (
        "goal names moka_pot_1 first; the demos pick moka_pot_2 first"
    ),
}

HashableKey = tuple[str, tuple[tuple[str, str], ...]]


def _keys(phases) -> list[tuple[str, dict[str, str]]]:
    """Readable ``(skill, arguments)`` pairs for direct assertions."""
    return [(p.skill, dict(p.arguments)) for p in phases]


def _hashable(skill, arguments) -> HashableKey:
    return (str(skill), tuple(sorted((str(k), str(v)) for k, v in (arguments or {}).items())))


def _plan_keys(bddl_path: Path) -> list[HashableKey]:
    return [_hashable(p.skill, p.arguments) for p in plan_from_bddl(bddl_path)]


def _bddl_root() -> Path | None:
    """Directory holding the libero BDDL files (override for other checkouts)."""
    override = os.environ.get("COSMOS_LIBERO_BDDL_ROOT")
    if override:
        return Path(override) if Path(override).is_dir() else None
    try:
        from libero.libero import get_libero_path
    except Exception:  # noqa: BLE001 - missing libero just skips these tests
        return None
    root = Path(get_libero_path("bddl_files"))
    return root if root.is_dir() else None


def _demo_order(manifest_path: Path):
    """Most common executed phase order per task, plus pre-satisfied phases.

    Segments the labeller marked ``already_satisfied`` carry no execution order:
    KITCHEN_SCENE8's stove is lit from the start, and the ``Open`` the runtime
    synthesizes for a drawer that resets open is recorded the same way.  They
    are returned separately from the executed sequence.
    """
    payload = json.loads(Path(manifest_path).read_text())
    votes: dict[str, Counter] = {}
    satisfied: dict[str, set[HashableKey]] = {}
    for record in payload.get("records", []):
        if not record.get("valid"):
            continue
        task = str(record["task_name"])
        executed: list[HashableKey] = []
        for segment in record.get("segments", []):
            key = _hashable(segment.get("skill"), segment.get("arguments"))
            if segment.get("status") == "already_satisfied":
                satisfied.setdefault(task, set()).add(key)
            else:
                executed.append(key)
        votes.setdefault(task, Counter())[tuple(executed)] += 1

    orders: dict[str, list[HashableKey]] = {}
    disagreements: dict[str, int] = {}
    for task, counter in votes.items():
        orders[task] = list(counter.most_common(1)[0][0])
        if len(counter) > 1:
            disagreements[task] = len(counter)
    return orders, satisfied, disagreements


class GatesPlacementTest(unittest.TestCase):
    """Body-prefix matching, which is what KITCHEN_SCENE6 turns on."""

    def test_matches_a_region_of_the_named_body(self) -> None:
        self.assertTrue(_gates_placement("microwave_1", {"microwave_1_heating_region"}))

    def test_matches_the_target_itself(self) -> None:
        self.assertTrue(
            _gates_placement(
                "white_cabinet_1_bottom_region", {"white_cabinet_1_bottom_region"}
            )
        )

    def test_does_not_match_a_longer_instance_name(self) -> None:
        self.assertFalse(_gates_placement("plate_1", {"plate_10_region"}))

    def test_does_not_match_an_unrelated_body(self) -> None:
        self.assertFalse(_gates_placement("microwave_1", {"basket_1_contain_region"}))


class PlanFromGoalStateTest(unittest.TestCase):
    """One case per rule, using the goals the rules were derived from."""

    def test_close_naming_the_container_body_runs_after_the_placement(self) -> None:
        # KITCHEN_SCENE6, the regression: the goal names the body while the
        # placement names its heating region, so only the body prefix links them.
        goal = [
            ["in", "white_yellow_mug_1", "microwave_1_heating_region"],
            ["close", "microwave_1"],
        ]
        self.assertEqual(
            _keys(plan_from_goal_state(goal)),
            [
                ("Pick", {"item": "white_yellow_mug_1"}),
                (
                    "PlaceIn",
                    {"item": "white_yellow_mug_1", "target": "microwave_1_heating_region"},
                ),
                ("Close", {"target": "microwave_1"}),
            ],
        )

    def test_close_naming_the_region_runs_after_the_placement(self) -> None:
        # KITCHEN_SCENE4: the same relation written with the region twice.  BDDL
        # lists the Close first, which is why goal order cannot be trusted.
        goal = [
            ["close", "white_cabinet_1_bottom_region"],
            ["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"],
        ]
        self.assertEqual(
            [skill for skill, _ in _keys(plan_from_goal_state(goal))],
            ["Pick", "PlaceIn", "Close"],
        )

    def test_turnon_precedes_the_placement_on_its_body(self) -> None:
        # KITCHEN_SCENE3: TurnOn(flat_stove_1) gates flat_stove_1_cook_region and
        # is emitted first anyway (convention, matching the demos).
        goal = [
            ["turnon", "flat_stove_1"],
            ["on", "moka_pot_1", "flat_stove_1_cook_region"],
        ]
        self.assertEqual(
            _keys(plan_from_goal_state(goal)),
            [
                ("TurnOn", {"target": "flat_stove_1"}),
                ("Pick", {"item": "moka_pot_1"}),
                ("PlaceOn", {"item": "moka_pot_1", "target": "flat_stove_1_cook_region"}),
            ],
        )

    def test_open_precedes_the_placement_into_its_region(self) -> None:
        # libero_90 KITCHEN_SCENE1_open_the_top_drawer...: the drawer must admit
        # the bowl, so the Open is a precondition.
        goal = [
            ["open", "wooden_cabinet_1_top_region"],
            ["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"],
        ]
        self.assertEqual(
            [skill for skill, _ in _keys(plan_from_goal_state(goal))],
            ["Open", "Pick", "PlaceIn"],
        )

    def test_two_placements_keep_the_goal_order(self) -> None:
        # KITCHEN_SCENE8 shape: two placements on one destination, TurnOn last
        # in the goal but emitted first.
        goal = [
            ["on", "moka_pot_1", "flat_stove_1_cook_region"],
            ["on", "moka_pot_2", "flat_stove_1_cook_region"],
            ["turnon", "flat_stove_1"],
        ]
        self.assertEqual(
            _keys(plan_from_goal_state(goal)),
            [
                ("TurnOn", {"target": "flat_stove_1"}),
                ("Pick", {"item": "moka_pot_1"}),
                ("PlaceOn", {"item": "moka_pot_1", "target": "flat_stove_1_cook_region"}),
                ("Pick", {"item": "moka_pot_2"}),
                ("PlaceOn", {"item": "moka_pot_2", "target": "flat_stove_1_cook_region"}),
            ],
        )

    def test_standalone_state_goal(self) -> None:
        # "open the middle drawer of the cabinet": one predicate, no placement.
        self.assertEqual(
            _keys(plan_from_goal_state([["open", "wooden_cabinet_1_middle_region"]])),
            [("Open", {"target": "wooden_cabinet_1_middle_region"})],
        )

    def test_state_only_goal_keeps_the_goal_order(self) -> None:
        # libero_90 KITCHEN_SCENE4_close_the_bottom_drawer..._and_open_the_top_drawer
        goal = [
            ["close", "white_cabinet_1_bottom_region"],
            ["open", "white_cabinet_1_top_region"],
        ]
        self.assertEqual(
            [skill for skill, _ in _keys(plan_from_goal_state(goal))],
            ["Close", "Open"],
        )

    def test_close_not_gating_a_placement_keeps_its_goal_position(self) -> None:
        # libero_90 KITCHEN_SCENE10_close_the_top_drawer..._and_put_the_black_bowl
        # _on_top_of_it: the bowl goes on the cabinet top while the Close acts on
        # the drawer, so the two are independent and goal order stands.
        goal = [
            ["close", "wooden_cabinet_1_top_region"],
            ["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"],
        ]
        self.assertEqual(
            [skill for skill, _ in _keys(plan_from_goal_state(goal))],
            ["Close", "Pick", "PlaceOn"],
        )

    def test_unsupported_predicate_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            plan_from_goal_state([["near", "akita_black_bowl_1", "plate_1"]])
        self.assertIn("unsupported goal predicate", str(caught.exception))

    def test_empty_goal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_from_goal_state([])


class PlanOrderingContractTest(unittest.TestCase):
    """Sweep the shipped BDDL corpus and enforce the documented ordering.

    This re-checks the code against its own contract over ~8000 real goals: it
    catches parser crashes and bucket interleaving, not rule mistakes.  The demo
    comparison below is the independent check.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.bddl_root = _bddl_root()
        if cls.bddl_root is None:
            raise unittest.SkipTest("no libero bddl_files directory available")

    def test_contract_holds_for_every_shipped_bddl(self) -> None:
        checked = 0
        for suite in CORPUS_SUITES:
            for bddl in sorted((self.bddl_root / suite).glob("*.bddl")):
                with self.subTest(bddl=bddl.name):
                    self._assert_contract(bddl)
                checked += 1
        self.assertGreater(checked, 0)

    def _assert_contract(self, bddl_path: Path) -> None:
        phases = plan_from_bddl(bddl_path)
        self.assertEqual(
            [phase.planner_step_id for phase in phases], list(range(len(phases)))
        )
        keys = [(p.skill, dict(p.arguments)) for p in phases]
        for index, (skill, arguments) in enumerate(keys):
            site = str(arguments.get("target", ""))
            if skill == "Pick":
                # A Pick is always immediately followed by its own placement.
                self.assertLess(index + 1, len(keys), f"dangling Pick in {bddl_path}")
                next_skill, next_arguments = keys[index + 1]
                self.assertIn(next_skill, {"PlaceIn", "PlaceOn"})
                self.assertEqual(next_arguments.get("item"), arguments.get("item"))
            elif skill in STATE_SKILLS:
                for other_index, (other_skill, other_arguments) in enumerate(keys):
                    if other_skill not in {"PlaceIn", "PlaceOn"}:
                        continue
                    target = str(other_arguments.get("target", ""))
                    if not _gates_placement(site, {target}):
                        continue
                    if skill == "Close":
                        # A Close follows the placement it gates.
                        self.assertLess(
                            other_index,
                            index,
                            f"{bddl_path}: Close {site!r} precedes its placement",
                        )
                    else:
                        # A precondition precedes the placement it gates.
                        self.assertLess(
                            index,
                            other_index,
                            f"{bddl_path}: {skill} {site!r} follows its placement",
                        )


class PlanMatchesDemoOrderTest(unittest.TestCase):
    """The recorded demos are the ground truth for phase order.

    Two documented differences are asserted rather than tolerated silently: a
    phase the demos mark ``already_satisfied`` has no demo position, and
    ORDER_EXCEPTIONS lists permutations of independent placements.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not MANIFEST.exists():
            raise unittest.SkipTest(f"missing demo manifest {MANIFEST}")
        cls.bddl_root = _bddl_root()
        if cls.bddl_root is None:
            raise unittest.SkipTest("no libero bddl_files directory available")
        cls.orders, cls.satisfied, cls.disagreements = _demo_order(MANIFEST)

    def test_demos_agree_on_the_order(self) -> None:
        self.assertEqual(self.disagreements, {})

    def test_plan_matches_the_demo_order(self) -> None:
        self.assertTrue(self.orders, "demo manifest carried no usable records")
        for task, demo_order in sorted(self.orders.items()):
            with self.subTest(task=task):
                bddl = self.bddl_root / "libero_10" / f"{task}.bddl"
                self.assertTrue(bddl.exists(), bddl)
                plan = _plan_keys(bddl)
                demonstrated = set(demo_order)
                satisfied = self.satisfied.get(task, set())

                # Every demonstrated phase is planned ...
                self.assertLessEqual(
                    demonstrated, set(plan), f"{task}: demonstrated but unplanned"
                )
                # ... and the plan only adds preconditions the demos found
                # already satisfied (KITCHEN_SCENE8's stove is lit at reset).
                self.assertLessEqual(
                    set(plan) - demonstrated,
                    satisfied,
                    f"{task}: planned but neither demonstrated nor pre-satisfied",
                )
                # The reverse gap is the synthesized preconditions, which are
                # not goal predicates and so have no phase (an already-open
                # drawer, libero's ensure_open).
                for skill, _arguments in satisfied - set(plan):
                    self.assertIn(skill, STATE_SKILLS)

                demonstrated_order = [key for key in plan if key in demonstrated]
                if task in ORDER_EXCEPTIONS:
                    # The exception must still be needed; a stale one rots.
                    self.assertNotEqual(demonstrated_order, list(demo_order))
                    continue
                self.assertEqual(demonstrated_order, list(demo_order))
