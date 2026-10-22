"""
Numerical Inverse Kinematics (IK) solvers for serial-link manipulators
using Damped Least Squares (Levenberg-Marquardt) Jacobian iteration.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np

from mechakinetics.kinematics import SerialManipulator


@dataclass
class IKResult:
    """Result container returned by numerical inverse kinematics solvers."""
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    error_history: list[float]


def solve_ik_position(
    arm: SerialManipulator,
    target_xyz: Sequence[float],
    q_init: Optional[Sequence[float]] = None,
    joint_limits: Optional[Sequence[Tuple[float, float]]] = None,
    damping: float = 0.05,
    max_step: float = 0.35,
    tol: float = 1e-4,
    max_iter: int = 150,
) -> IKResult:
    """
    Solve 3D position-only Inverse Kinematics using Damped Least Squares (DLS)
    with step-norm clamping and optional box constraints on joint limits.
    """
    target = np.asarray(target_xyz, dtype=float).reshape(3)
    q = np.zeros(arm.dof, dtype=float) if q_init is None else np.asarray(q_init, dtype=float).copy()

    if len(q) != arm.dof:
        raise ValueError(f"Expected q_init of length {arm.dof}, got {len(q)}.")

    bounds_low: Optional[np.ndarray] = None
    bounds_high: Optional[np.ndarray] = None
    if joint_limits is not None:
        if len(joint_limits) != arm.dof:
            raise ValueError(f"Expected {arm.dof} joint limit pairs, got {len(joint_limits)}.")
        bounds_low = np.array([ b[0] for b in joint_limits ], dtype=float)
        bounds_high = np.array([ b[1] for b in joint_limits ], dtype=float)
        q = np.clip(q, bounds_low, bounds_high)

    error_history: list[float] = []
    lam2 = damping ** 2
    eye3 = np.eye(3, dtype=float)

    for iteration in range(1, max_iter + 1):
        T_curr = arm.forward_kinematics(q)
        p_curr = T_curr[:3, 3]
        err = target - p_curr
        err_norm = float(np.linalg.norm(err))
        error_history.append(err_norm)

        if err_norm <= tol:
            return IKResult(
                q=q,
                success=True,
                iterations=iteration,
                position_error=err_norm,
                error_history=error_history,
            )

        J_v = arm.space_jacobian(q)[:3, :]
        dq = J_v.T @ np.linalg.solve(J_v @ J_v.T + lam2 * eye3, err)

        # Clamp maximum joint step per iteration to prevent erratic jumps near singularities
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step:
            dq = dq * (max_step / step_norm)

        q = q + dq

        if bounds_low is not None and bounds_high is not None:
            q = np.clip(q, bounds_low, bounds_high)
        else:
            # Wrap revolute joints cleanly into [-pi, pi]
            for i, link in enumerate(arm.links):
                if not link.is_prismatic:
                    q[i] = (q[i] + np.pi) % (2.0 * np.pi) - np.pi

    final_pos = arm.forward_kinematics(q)[:3, 3]
    final_err = float(np.linalg.norm(target - final_pos))
    return IKResult(
        q=q,
        success=final_err <= tol,
        iterations=max_iter,
        position_error=final_err,
        error_history=error_history,
    )