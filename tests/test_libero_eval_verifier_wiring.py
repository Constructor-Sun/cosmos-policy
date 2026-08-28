import ast
from pathlib import Path
import unittest


EVAL = (
    Path(__file__).resolve().parents[1]
    / "cosmos_policy/experiments/robot/libero/run_libero_eval.py"
)


class EvalInitialAlignmentWiringTest(unittest.TestCase):
    def test_removed_runtime_is_not_wired(self):
        source = EVAL.read_text()

        for removed in (
            "enable_phase_verifier",
            "enable_phase_3d",
            "enable_feasible_3d",
            "enable_phase_recovery",
            "enable_feasible_recovery",
            "ExecutionMonitor",
            "PhaseRecoverySelector",
            "FeasibleRecoverySelector",
            "_phase_check_and_recover",
        ):
            self.assertNotIn(removed, source)

    def test_one_selector_drives_the_initial_alignment_path(self):
        source = EVAL.read_text()

        tree = ast.parse(source)
        constructors = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_create_initial_alignment_selector"
        ]
        self.assertEqual(len(constructors), 1)

        starters = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_maybe_start_initial_alignment"
        ]
        self.assertEqual(len(starters), 1)
        starter_source = ast.get_source_segment(source, starters[0])
        self.assertIn("_policy_step_count != 0", starter_source)
        self.assertIn("initial_alignment_selector.select", starter_source)
        self.assertIn("action_queue.clear()", starter_source)
        self.assertIn("alignment_task_name", source)

    def test_eval_uses_package_imports(self):
        source = EVAL.read_text()
        self.assertNotIn("_BIN", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("from execute.", source)
        self.assertNotIn("from bin.", source)
        self.assertIn("memory_system.execute", source)
        self.assertIn("memory_system.execute.initial_alignment", source)
        self.assertIn("memory_system.execute.curobo_planner", source)
        self.assertIn("skill_memory_test", source)


if __name__ == "__main__":
    unittest.main()
