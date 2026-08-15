#!/usr/bin/env python3
"""Convert two HDF5 end-effector poses into a calibrated Cosmos offset.

The LIBERO rollout HDF5 proprio layout used by this repository is:

    [gripper_qpos(2), eef_position_xyz(3), eef_quaternion_xyzw(4)]

Translations are expressed as vectors in the world / robot-base frame.
Rotations are world-frame rotation vectors, matching robosuite's OSC input.

The source episode's following action chunk is used to estimate world-frame
action-to-displacement scales. Rotation actions are composed as rotations in
time order; translation actions remain additive vectors.

Example:

    python bin/compute_cosmos_offset.py \
        --source-hdf5 /path/to/episode_1.hdf5 \
        --target-hdf5 /path/to/episode_2.hdf5 \
        --source-t 160 \
        --target-t 80 160 176
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

POSITION_SLICE = slice(2, 5)
QUATERNION_SLICE = slice(5, 9)
TRANSLATION_NAMES = ("x", "y", "z")
ROTATION_NAMES = ("x", "y", "z")
OSC_ROTATION_SCALE_RAD = 0.5


@dataclass(frozen=True)
class EpisodeData:
    actions: np.ndarray
    proprio: np.ndarray


@dataclass(frozen=True)
class Calibration:
    action_sum: np.ndarray
    position_delta_world: np.ndarray
    rotation_delta_world: np.ndarray
    position_scale: np.ndarray
    rotation_scale: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-hdf5",
        "--ep1",
        dest="source_hdf5",
        type=pathlib.Path,
        required=True,
        help="Source episode HDF5 used for the intervention and calibration",
    )
    parser.add_argument(
        "--target-hdf5",
        "--ep2",
        dest="target_hdf5",
        type=pathlib.Path,
        required=True,
        help="Target/reference episode HDF5",
    )
    parser.add_argument(
        "--source-t",
        "--t1",
        dest="source_t",
        type=int,
        required=True,
        help="Source episode timestep at which the offset starts",
    )
    parser.add_argument(
        "--target-t",
        "--t2",
        dest="target_t",
        type=int,
        nargs="+",
        required=True,
        help="One or more target episode timesteps",
    )
    parser.add_argument(
        "--calibration-steps",
        "--duration",
        dest="calibration_steps",
        type=int,
        default=16,
        help="Number of source actions used for calibration (default: 16)",
    )
    parser.add_argument(
        "--rollout-wait-steps",
        type=int,
        default=10,
        help=(
            "Unrecorded stabilization steps before HDF5 index 0; added to source-t "
            "when emitting COSMOS_OFFSET_START_T (default: 10)"
        ),
    )
    parser.add_argument(
        "--translation-axes",
        default="x,y",
        help="Eligible world translation axes, comma-separated (default: x,y)",
    )
    parser.add_argument(
        "--rotation-axes",
        default="x,z",
        help="Eligible world rotation-vector axes, comma-separated (default: x,z)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal places in COSMOS_OFFSET_AMOUNT (default: 2)",
    )
    parser.add_argument(
        "--shell-only",
        action="store_true",
        help="Print only copyable COSMOS_OFFSET_* assignments",
    )
    return parser.parse_args()


def load_episode(path: pathlib.Path) -> EpisodeData:
    if not path.is_file():
        raise ValueError(f"HDF5 does not exist: {path}")

    with h5py.File(path, "r") as handle:
        missing = [name for name in ("actions", "proprio") if name not in handle]
        if missing:
            raise ValueError(f"{path} is missing dataset(s): {', '.join(missing)}")
        actions = np.asarray(handle["actions"])
        proprio = np.asarray(handle["proprio"])

    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected actions shaped (T, >=6), got {actions.shape} in {path}")
    if proprio.ndim != 2 or proprio.shape[1] < 9:
        raise ValueError(f"Expected proprio shaped (T, >=9), got {proprio.shape} in {path}")
    if not np.all(np.isfinite(actions[:, :6])):
        raise ValueError(f"Non-finite action values in {path}")
    if not np.all(np.isfinite(proprio[:, :9])):
        raise ValueError(f"Non-finite proprio values in {path}")
    return EpisodeData(actions=actions, proprio=proprio)


def pose_at(episode: EpisodeData, timestep: int) -> tuple[np.ndarray, Rotation]:
    length = episode.proprio.shape[0]
    if timestep < 0 or timestep >= length:
        raise ValueError(f"Timestep {timestep} is outside [0, {length - 1}]")

    position = episode.proprio[timestep, POSITION_SLICE].astype(np.float64, copy=True)
    quaternion = episode.proprio[timestep, QUATERNION_SLICE].astype(np.float64, copy=True)
    if np.linalg.norm(quaternion) < 1e-8:
        raise ValueError(f"Near-zero quaternion at timestep {timestep}")
    return position, Rotation.from_quat(quaternion)


def parse_axis_list(raw: str, valid_names: tuple[str, ...], label: str) -> tuple[int, ...]:
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError(f"{label} cannot be empty")
    unknown = [name for name in names if name not in valid_names]
    if unknown:
        raise ValueError(
            f"Unknown {label}: {', '.join(unknown)}; choose from {', '.join(valid_names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate value in {label}: {raw}")
    return tuple(valid_names.index(name) for name in names)


def calibrate(
    episode: EpisodeData,
    source_t: int,
    steps: int,
) -> tuple[np.ndarray, Rotation, Calibration]:
    if steps <= 0:
        raise ValueError("calibration-steps must be positive")
    end_t = source_t + steps
    if end_t >= episode.proprio.shape[0]:
        raise ValueError(
            f"Calibration endpoint {end_t} is outside source proprio range "
            f"[0, {episode.proprio.shape[0] - 1}]"
        )
    if end_t > episode.actions.shape[0]:
        raise ValueError(
            f"Calibration action slice [{source_t}:{end_t}] exceeds "
            f"source actions length {episode.actions.shape[0]}"
        )

    source_position, source_rotation = pose_at(episode, source_t)
    end_position, end_rotation = pose_at(episode, end_t)
    action_window = episode.actions[source_t:end_t, :6]
    action_sum = action_window.sum(axis=0)
    rotation_matrix = np.eye(3)
    for rotation_action in action_window[:, 3:6]:
        rotation_matrix = (
            Rotation.from_rotvec(OSC_ROTATION_SCALE_RAD * rotation_action).as_matrix() @ rotation_matrix
        )
    action_sum[3:6] = Rotation.from_matrix(rotation_matrix).as_rotvec() / OSC_ROTATION_SCALE_RAD

    position_delta_world = end_position - source_position
    rotation_delta_world = (end_rotation * source_rotation.inv()).as_rotvec()

    small_action_axes = np.flatnonzero(np.abs(action_sum) < 1e-8)
    if small_action_axes.size:
        names = TRANSLATION_NAMES + ROTATION_NAMES
        bad = ", ".join(names[index] for index in small_action_axes)
        raise ValueError(f"Cannot calibrate near-zero summed action axis/axes: {bad}")

    calibration = Calibration(
        action_sum=action_sum,
        position_delta_world=position_delta_world,
        rotation_delta_world=rotation_delta_world,
        position_scale=position_delta_world / action_sum[:3],
        rotation_scale=rotation_delta_world / action_sum[3:6],
    )
    return source_position, source_rotation, calibration


def choose_largest(values: np.ndarray, eligible: tuple[int, ...]) -> int:
    return max(eligible, key=lambda index: abs(float(values[index])))


def compute_offset(
    target: EpisodeData,
    target_t: int,
    source_position: np.ndarray,
    source_rotation: Rotation,
    calibration: Calibration,
    translation_axes: tuple[int, ...],
    rotation_axes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    target_position, target_rotation = pose_at(target, target_t)
    position_delta_world = target_position - source_position
    position_delta_local = source_rotation.inv().apply(position_delta_world)
    rotation_delta_world = (target_rotation * source_rotation.inv()).as_rotvec()

    translation_axis = choose_largest(position_delta_world, translation_axes)
    rotation_axis = choose_largest(rotation_delta_world, rotation_axes)
    if abs(calibration.position_scale[translation_axis]) < 1e-8:
        raise ValueError(
            f"Near-zero position scale for selected {TRANSLATION_NAMES[translation_axis]} axis"
        )
    if abs(calibration.rotation_scale[rotation_axis]) < 1e-8:
        raise ValueError(
            f"Near-zero rotation scale for selected {ROTATION_NAMES[rotation_axis]} axis"
        )

    amount = np.zeros(6, dtype=np.float64)
    amount[translation_axis] = (
        position_delta_world[translation_axis] / calibration.position_scale[translation_axis]
    )
    amount[3 + rotation_axis] = (
        rotation_delta_world[rotation_axis] / calibration.rotation_scale[rotation_axis]
    )
    return (
        amount,
        position_delta_world,
        position_delta_local,
        rotation_delta_world,
        translation_axis,
        rotation_axis,
    )


def vector_text(values: np.ndarray, precision: int) -> str:
    threshold = 0.5 * 10.0 ** (-precision)
    parts: list[str] = []
    for value in values:
        value = 0.0 if abs(float(value)) < threshold else float(value)
        text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        parts.append("0" if text in ("", "-0") else text)
    return ",".join(parts)


def array_text(values: np.ndarray, precision: int = 6) -> str:
    return np.array2string(
        np.asarray(values),
        precision=precision,
        suppress_small=False,
        separator=", ",
    )


def main() -> None:
    args = parse_args()
    if args.precision < 0:
        raise SystemExit("error: precision must be non-negative")
    if args.rollout_wait_steps < 0:
        raise SystemExit("error: rollout-wait-steps must be non-negative")

    offset_start_t = args.source_t + args.rollout_wait_steps

    try:
        translation_axes = parse_axis_list(
            args.translation_axes, TRANSLATION_NAMES, "translation axis"
        )
        rotation_axes = parse_axis_list(args.rotation_axes, ROTATION_NAMES, "rotation axis")
        source = load_episode(args.source_hdf5)
        target = load_episode(args.target_hdf5)
        source_position, source_rotation, calibration = calibrate(
            source, args.source_t, args.calibration_steps
        )

        results = []
        for target_t in args.target_t:
            result = compute_offset(
                target,
                target_t,
                source_position,
                source_rotation,
                calibration,
                translation_axes,
                rotation_axes,
            )
            results.append((target_t, result))
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    if not args.shell_only:
        source_rotvec = source_rotation.as_rotvec()
        print(f"Source HDF5: {args.source_hdf5}")
        print(f"Target HDF5: {args.target_hdf5}")
        print(f"Source HDF5 t: {args.source_t}")
        print(
            f"Rollout offset t: {offset_start_t} "
            f"(HDF5 t + {args.rollout_wait_steps} unrecorded wait steps)"
        )
        print(f"Source EE xyz (world): {array_text(source_position)}")
        print(f"Source EE rotvec (world rad): {array_text(source_rotvec)}")
        print()
        print(
            f"Calibration window: [{args.source_t}:{args.source_t + args.calibration_steps}] "
            f"({args.calibration_steps} actions)"
        )
        print(f"Action sum: {array_text(calibration.action_sum)}")
        print(
            "Observed position delta (world m): "
            f"{array_text(calibration.position_delta_world)}"
        )
        print(
            "Observed rotation-vector delta (world xyz rad): "
            f"{array_text(calibration.rotation_delta_world)}"
        )
        print(f"Position scale (world m/action): {array_text(calibration.position_scale)}")
        print(f"Rotation scale (world rad/action): {array_text(calibration.rotation_scale)}")

    for result_index, (target_t, result) in enumerate(results):
        (
            amount,
            position_delta_world,
            position_delta_local,
            rotation_delta_world,
            translation_axis,
            rotation_axis,
        ) = result
        amount_text = vector_text(amount, args.precision)

        if args.shell_only:
            if len(results) > 1:
                print(f"# target_t={target_t}")
        else:
            print()
            print(f"Target t: {target_t}")
            print(f"Position delta (world m): {array_text(position_delta_world)}")
            print(f"Position delta (source-local m, diagnostic): {array_text(position_delta_local)}")
            print(f"Rotation-vector delta (world xyz rad): {array_text(rotation_delta_world)}")
            print(
                "Selected axes: "
                f"world {TRANSLATION_NAMES[translation_axis]}, "
                f"world {ROTATION_NAMES[rotation_axis]}"
            )
            print("Calibrated total action offset:")

        print(f"COSMOS_OFFSET_START_T={offset_start_t}")
        print(f"COSMOS_OFFSET_DURATION={args.calibration_steps}")
        print(f'COSMOS_OFFSET_AMOUNT="{amount_text}"')

        if args.shell_only and result_index + 1 < len(results):
            print()

    if not args.shell_only:
        print()
        print(
            "NOTE: The calibrated amount is in total controller action units; "
            "the rollout divides it by COSMOS_OFFSET_DURATION."
        )
        print("Calibration is approximate, so validate the resulting rollout visually.")


if __name__ == "__main__":
    main()
