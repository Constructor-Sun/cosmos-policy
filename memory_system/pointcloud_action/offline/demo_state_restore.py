"""Restore the LIBERO runtime state omitted from flattened HDF states."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import robosuite

from memory_system.pointcloud_action.offline.extraction import LIBERO_PLUS


def _repair_asset_paths(xml_string: str) -> str:
    """Map absolute paths from the recording host to this checkout."""
    tree = ET.fromstring(xml_string)
    robosuite_root = Path(robosuite.__file__).resolve().parent
    libero_root = LIBERO_PLUS / "libero/libero"
    roots = (
        ("/robosuite/", robosuite_root),
        ("/libero/libero/", libero_root),
        ("/chiliocosm/assets/", libero_root / "assets"),
    )
    unresolved = []
    for elem in tree.iter():
        old = elem.get("file")
        if not old or Path(old).is_file():
            continue
        normalized = old.replace("\\", "/")
        replacement = None
        for marker, root in roots:
            if marker not in normalized:
                continue
            candidate = root / normalized.rsplit(marker, 1)[1]
            if candidate.is_file():
                replacement = candidate
                break
        if replacement is None and "/assets/" in normalized:
            suffix = normalized.rsplit("/assets/", 1)[1]
            for root in (
                libero_root / "assets",
                robosuite_root / "models/assets",
            ):
                candidate = root / suffix
                if candidate.is_file():
                    replacement = candidate
                    break
        if replacement is None:
            unresolved.append(old)
        else:
            elem.set("file", str(replacement))
    if unresolved:
        raise FileNotFoundError(
            f"unresolved demo XML assets ({len(unresolved)}): {unresolved[:3]}"
        )
    return ET.tostring(tree, encoding="unicode")


def _align_object_names(env, xml_string: str) -> str:
    """Align legacy/base XML prefixes with current LIBERO model metadata.

    Some older demos use e.g. ``salad_dressing_1`` while LIBERO-plus expects
    ``new_salad_dressing_1``. Their joint / geom topology is identical. Only
    XML name references are changed; recorded mesh and texture paths remain
    untouched so the physical model stays faithful to the demo.
    """
    tree = ET.fromstring(xml_string)
    values = [
        value
        for elem in tree.iter()
        for key, value in elem.attrib.items()
        if key != "file"
    ]
    for obj in env.env.model.mujoco_objects:
        current = str(getattr(obj, "name", ""))
        if not current:
            continue
        alternate = current[4:] if current.startswith("new_") else f"new_{current}"
        if any(current in value for value in values):
            continue
        if not any(alternate in value for value in values):
            continue
        for elem in tree.iter():
            for key, value in tuple(elem.attrib.items()):
                if key != "file" and alternate in value:
                    elem.set(key, value.replace(alternate, current))
    return ET.tostring(tree, encoding="unicode")


def reset_from_demo_xml(env, model_xml: str) -> None:
    """Reset ``env`` with one demo's exact XML and compatible metadata."""
    env.env.deterministic_reset = False
    env.reset()
    xml = _repair_asset_paths(model_xml)
    xml = _align_object_names(env, xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()


def restore_gripper_runtime(env, actions: np.ndarray, frame: int) -> None:
    """Reconstruct the stateful gripper command accumulator at ``frame``."""
    robot = env.env.robots[0]
    gripper = robot.gripper
    gripper.current_action[:] = 0.0
    grip_dim = int(robot.action_dim - robot.controller.control_dim)
    if grip_dim <= 0:
        return
    repeats = int(round(env.env.control_timestep / env.env.model_timestep))
    for action in np.asarray(actions)[: int(frame)]:
        gripper_action = np.asarray(action)[-grip_dim:]
        for _ in range(repeats):
            gripper.format_action(gripper_action)


def restore_demo_frame(
    env,
    state: np.ndarray,
    actions: np.ndarray,
    frame: int,
):
    """Restore a demo frame sufficiently completely for raw action replay."""
    frame = int(frame)
    env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
    env.sim.forward()

    base = env.env
    base.cur_time = float(env.sim.data.time)
    if getattr(base, "control_timestep", 0):
        base.timestep = int(round(base.cur_time / base.control_timestep))
    base.done = False

    controller = base.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    restore_gripper_runtime(env, actions, frame)

    base._post_process()
    base._update_observables(force=True)
    return base._get_observations()
