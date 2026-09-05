"""MuJoCo-based numerical IK for the LIBERO Franka arm."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _arm_qpos_addresses(sim) -> list[int]:
    addresses = []
    for index in range(sim.model.njnt):
        name = sim.model.joint_names[index]
        if name.startswith("robot0_joint"):
            addresses.append(int(sim.model.jnt_qposadr[index]))
    return addresses


def _body_ids(sim) -> tuple[int, int]:
    hand_id = sim.model.body_name2id("robot0_right_hand")
    grip_id = sim.model.body_name2id("gripper0_eef")
    return hand_id, grip_id


def _forward_pose(sim, q: np.ndarray, arm_addr, hand_id, grip_id) -> tuple[np.ndarray, np.ndarray]:
    for address, value in zip(arm_addr, q):
        sim.data.qpos[address] = float(value)
    sim.forward()
    hand_mat = np.asarray(sim.data.body_xmat[hand_id], dtype=np.float64).reshape(3, 3).copy()
    grip_pos = np.asarray(sim.data.body_xpos[grip_id], dtype=np.float64).reshape(3).copy()
    return grip_pos, hand_mat


def solve_ee_ik(
    sim,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    init_q: np.ndarray | None = None,
    max_iter: int = 100,
) -> np.ndarray | None:
    """Solve 7 arm joint angles for a grip-position + hand-orientation target."""
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    arm_addr = _arm_qpos_addresses(sim)
    hand_id, grip_id = _body_ids(sim)
    lower = []
    upper = []
    for index in range(sim.model.njnt):
        name = sim.model.joint_names[index]
        if name.startswith("robot0_joint"):
            lower.append(float(sim.model.jnt_range[index, 0]))
            upper.append(float(sim.model.jnt_range[index, 1]))
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    if init_q is None:
        init_q = np.clip(np.asarray(sim.data.qpos[arm_addr], dtype=np.float64), lower, upper)
    else:
        init_q = np.clip(np.asarray(init_q, dtype=np.float64).reshape(7), lower, upper)

    def residual(q: np.ndarray) -> np.ndarray:
        grip_pos, hand_mat = _forward_pose(sim, q, arm_addr, hand_id, grip_id)
        pos_err = grip_pos - target_pos
        rot_err = (
            Rotation.from_matrix(target_rot)
            * Rotation.from_matrix(hand_mat).inv()
        ).as_rotvec()
        return np.concatenate([pos_err, rot_err])

    result = least_squares(
        residual,
        init_q,
        bounds=(lower, upper),
        max_nfev=max_iter,
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
    )
    if not result.success and np.linalg.norm(result.fun) > 1e-3:
        return None
    q_sol = np.clip(result.x, lower, upper)
    # Restore to solution pose so caller can use it directly.
    _forward_pose(sim, q_sol, arm_addr, hand_id, grip_id)
    return q_sol
