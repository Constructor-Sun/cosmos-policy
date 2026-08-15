#!/usr/bin/env python3
"""
Compute the relative 6D transform between two end-effector poses, expressed in
both world frame and the end-effector's local frame at the source timestep.

The world-frame result is what you would use to reason about COSMOS_OFFSET_AMOUNT
-- it tells you the delta [dx, dy, dz, drx, dry, drz] needed in action
space, assuming a perfect 1:1 controller (which the OSC controller is NOT; see
note at the bottom).

Usage:
    python bin/transform_ee_pose.py \
        --ep1 experiments/.../episode_data--...ep=1...hdf5 \
        --ep2 experiments/.../episode_data--...ep=2...hdf5 \
        --t1 160 --t2 176
"""

from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

OSC_ROTATION_SCALE_RAD = 0.5


def load_pose(hdf5_path: pathlib.Path, t: int) -> tuple[np.ndarray, Rotation]:
    """Return (position_xyz, Rotation) at timestep t from an episode HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        proprio = f["proprio"][:]                 # (T, 9)
        if t < 0 or t >= proprio.shape[0]:
            raise IndexError(f"t={t} out of range [0, {proprio.shape[0] - 1}]")
        pos = proprio[t, 2:5].copy()               # world-frame xyz
        quat_xyzw = proprio[t, 5:9].copy()         # scipy format: x,y,z,w
    return pos, Rotation.from_quat(quat_xyzw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep1", type=pathlib.Path, required=True, help="Source episode HDF5")
    parser.add_argument("--ep2", type=pathlib.Path, required=True, help="Target episode HDF5")
    parser.add_argument("--t1", type=int, required=True, help="Timestep in ep1 (source)")
    parser.add_argument("--t2", type=int, required=True, help="Timestep in ep2 (target)")
    args = parser.parse_args()

    # ── Load poses ──────────────────────────────────────────────────
    p_src, R_src = load_pose(args.ep1, args.t1)   # A
    p_tgt, R_tgt = load_pose(args.ep2, args.t2)   # B

    print("=" * 72)
    print(f"Source: {args.ep1.name}")
    print(f"  t={args.t1}")
    print(f"  pos (world xyz):     {p_src.round(4).tolist()}")
    print(f"  rotvec (world xyz, °): {np.rad2deg(R_src.as_rotvec()).round(2).tolist()}")
    print()
    print(f"Target: {args.ep2.name}")
    print(f"  t={args.t2}")
    print(f"  pos (world xyz):     {p_tgt.round(4).tolist()}")
    print(f"  rotvec (world xyz, °): {np.rad2deg(R_tgt.as_rotvec()).round(2).tolist()}")

    # ── 1. Relative transform in WORLD frame ────────────────────────
    dp_world = p_tgt - p_src                           # position delta
    dR_world = R_tgt * R_src.inv()                     # rotation delta

    print()
    print("─" * 72)
    print("1. WORLD-frame relative transform (A → B)")
    print(f"   Δpos   = {dp_world.round(4).tolist()} m")
    print(f"   Δrotvec = {np.rad2deg(dR_world.as_rotvec()).round(2).tolist()}°")

    # ── 2. Relative transform in END-EFFECTOR LOCAL frame at A ──────
    # R_src columns = local axes expressed in world:
    #   col 0 → local x (forward)
    #   col 1 → local y (left)
    #   col 2 → local z (up)
    R_mat = R_src.as_matrix()
    dp_local = R_src.inv().apply(dp_world)             # R^T @ dp_world
    dR_local = R_src.inv() * dR_world * R_src          # conjugate

    print()
    print("─" * 72)
    print("2. LOCAL-frame relative transform (A → B)")
    print("   Local axes at source pose (world direction):")
    print(f"     x (forward): {R_mat[:, 0].round(3).tolist()}")
    print(f"     y (left):    {R_mat[:, 1].round(3).tolist()}")
    print(f"     z (up):      {R_mat[:, 2].round(3).tolist()}")
    print()
    print(f"   Δpos_local [前后, 左右, 上下] = {dp_local.round(4).tolist()} m")
    print(f"   Δrotvec_local               = {np.rad2deg(dR_local.as_rotvec()).round(2).tolist()}°")

    # ── 3. Theoretical COSMOS_OFFSET_AMOUNT ─────────────────────────
    # Position:  additive world-frame translation vector
    # Rotation:  world-frame rotation vector used by the OSC controller
    print()
    print("─" * 72)
    print("3. Theoretical COSMOS_OFFSET_AMOUNT (world-frame convention)")
    for name, value in zip(("dx", "dy", "dz"), dp_world):
        print(f"   {name:3s} = {value:+.4f}   # world translation (m)")
    for name, value in zip(("drx", "dry", "drz"), dR_world.as_rotvec()):
        print(f"   {name:3s} = {value:+.4f}   # world rotation vector (rad)")

    # ── 4. Verification ─────────────────────────────────────────────
    p_check = p_src + dp_world
    R_check = dR_world * R_src
    pos_err = float(np.linalg.norm(p_check - p_tgt))
    rot_err = float((R_check * R_tgt.inv()).magnitude())

    print()
    print("─" * 72)
    print("4. Verification")
    print(f"   A + transform = B  →  pos error = {pos_err:.2e} m,  rot error = {rot_err:.2e} rad")

    # ── 5. Actual action calibration at t1 (for reference) ──────────
    with h5py.File(args.ep1, "r") as f:
        actions = f["actions"][:]
        proprio = f["proprio"][:]

    if args.t1 + 16 < proprio.shape[0]:
        action_window = actions[args.t1:args.t1 + 16, :6]
        act16 = action_window.sum(axis=0)
        rotation_matrix = np.eye(3)
        for rotation_action in action_window[:, 3:6]:
            rotation_matrix = (
                Rotation.from_rotvec(OSC_ROTATION_SCALE_RAD * rotation_action).as_matrix() @ rotation_matrix
            )
        act16[3:6] = Rotation.from_matrix(rotation_matrix).as_rotvec() / OSC_ROTATION_SCALE_RAD
        dp16 = proprio[args.t1 + 16, 2:5] - proprio[args.t1, 2:5]
        R16_s = Rotation.from_quat(proprio[args.t1, 5:9])
        R16_e = Rotation.from_quat(proprio[args.t1 + 16, 5:9])
        dr16 = R16_e * R16_s.inv()

        print()
        print("─" * 72)
        print("5. Action→displacement calibration at source t1 (16-step window)")
        print(f"   Action total [dx,dy,dz,drx,dry,drz] = {act16.round(2).tolist()}")
        print(f"   Actual Δpos                     = {dp16.round(4).tolist()} m")
        print(f"   Actual Δrotvec                  = {np.rad2deg(dr16.as_rotvec()).round(2).tolist()}°")

        pos_scale = dp16 / (act16[:3] + 1e-8)
        rot_scale = dr16.as_rotvec() / (act16[3:6] + 1e-8)
        print(f"   Pos scale (m / action unit):  {pos_scale.round(6).tolist()}")
        print(f"   Rot scale (rad / action unit): {rot_scale.round(6).tolist()}")

        # Calibrated estimate
        calibrated_dpos = dp_world / pos_scale
        calibrated_drot = dR_world.as_rotvec() / rot_scale
        print()
        print(f"   → Calibrated COSMOS_OFFSET (pos) = {calibrated_dpos.round(1).tolist()}")
        print(f"   → Calibrated COSMOS_OFFSET (rot) = {calibrated_drot.round(1).tolist()}")

    print()
    print("=" * 72)
    print("NOTE: Action units ≠ meters/radians. The OSC controller applies")
    print("a configuration-dependent gain. The 'calibrated' values above use")
    print("the local scaling at the source timestep and are approximate.")
    print("Rotation actions are especially unreliable near wrist singularities.")


if __name__ == "__main__":
    main()
