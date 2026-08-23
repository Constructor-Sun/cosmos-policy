import ast
from pathlib import Path
import re
import unittest


EVAL = (
    Path(__file__).resolve().parents[1]
    / "cosmos_policy/experiments/robot/libero/run_libero_eval.py"
)


class EvalVerifierWiringTest(unittest.TestCase):
    def test_one_flag_constructs_one_sequential_monitor(self):
        source = EVAL.read_text()
        flags = re.findall(r"^\s+(enable_\w*verifier)\s*:", source, re.MULTILINE)
        self.assertEqual(flags, ["enable_phase_verifier"])

        tree = ast.parse(source)
        constructors = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_create_execution_monitor"
        ]
        self.assertEqual(len(constructors), 1)
        self.assertNotIn("_create_phase_monitor", source)
        self.assertNotIn("_create_feasible_region_verifier", source)
        # Step 4: eval wiring must use memory_system.execute instead of bin/execute.
        self.assertNotIn("_BIN", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("from execute.", source)
        self.assertNotIn("from bin.", source)
        self.assertIn("memory_system.execute", source)
        self.assertIn("skill_memory_test", source)


if __name__ == "__main__":
    unittest.main()
