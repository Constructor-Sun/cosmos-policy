from pathlib import Path
import sys
import unittest

from PIL import Image, ImageDraw


UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))

from visualize_libero_skill_segments import (  # noqa: E402
    READY_EVENT,
    SKILL_COLORS,
    dim_color,
    draw_timeline,
    terminal_boundary,
)


def segment(**updates):
    value = {
        "skill": "Pick",
        "start": 10,
        "end": 50,
        "success_start": 50,
        "success_end": 53,
        "terminal_start": 30,
        "boundary_method": "final_close_edge",
    }
    value.update(updates)
    return value


class ReadyBoundaryVisualizationTest(unittest.TestCase):
    def test_terminal_boundary_accepts_only_in_segment_values(self):
        self.assertEqual(terminal_boundary(segment()), 30)
        self.assertIsNone(terminal_boundary(segment(terminal_start=None)))
        self.assertIsNone(terminal_boundary(segment(terminal_start=51)))

    def test_timeline_splits_approach_and_terminal_colors(self):
        image = Image.new("RGB", (120, 70), (0, 0, 0))
        draw_timeline(
            ImageDraw.Draw(image),
            [segment()],
            frame=5,
            total_frames=100,
            left=10,
            top=0,
            width=100,
            height=60,
            scale=1,
        )
        self.assertEqual(image.getpixel((30, 25)), dim_color(SKILL_COLORS["Pick"]))
        self.assertEqual(image.getpixel((50, 25)), SKILL_COLORS["Pick"])
        self.assertEqual(image.getpixel((40, 20)), READY_EVENT)

    def test_old_manifest_keeps_the_original_skill_color(self):
        image = Image.new("RGB", (120, 70), (0, 0, 0))
        draw_timeline(
            ImageDraw.Draw(image),
            [segment(terminal_start=None)],
            frame=5,
            total_frames=100,
            left=10,
            top=0,
            width=100,
            height=60,
            scale=1,
        )
        self.assertEqual(image.getpixel((30, 25)), SKILL_COLORS["Pick"])


if __name__ == "__main__":
    unittest.main()
