"""Unit tests for MemoryTemplateMatcher with synthetic images."""
import numpy as np
import pytest

from memory_system.execute.planner.held_object.mask_matcher import MemoryTemplateMatcher


def _make_template():
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    for row in range(40, 120, 10):
        for col in range(60, 140, 10):
            if ((row // 10) + (col // 10)) % 2 == 0:
                img[row : row + 10, col : col + 10] = 30
            mask[row : row + 10, col : col + 10] = 1
    return {
        "crop_rgb": img[40:120, 60:140].copy(),
        "crop_mask": mask[40:120, 60:140].copy(),
        "task_name": "synthetic",
        "demo_id": "demo_0",
        "planner_step_id": 0,
        "skill": "Pick",
    }


def test_match_finds_translated_template():
    template = _make_template()
    current = np.full((200, 200, 3), 220, dtype=np.uint8)
    current[100:180, 120:200] = template["crop_rgb"]
    result = MemoryTemplateMatcher().match(current, template)
    assert result is not None
    ys, xs = np.nonzero(result.mask)
    assert len(ys) > 100
    assert ys.min() >= 90 and ys.max() <= 190
    assert xs.min() >= 110 and xs.max() <= 210


def test_match_returns_none_without_object():
    template = _make_template()
    current = np.full((200, 200, 3), 220, dtype=np.uint8)
    result = MemoryTemplateMatcher().match(current, template)
    assert result is None
