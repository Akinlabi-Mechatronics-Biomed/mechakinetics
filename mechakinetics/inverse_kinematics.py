"""
Numerical Inverse Kinematics (IK) solvers for serial-link manipulators
using Damped Least Squares (Levenberg-Marquardt) Jacobian iteration.
"""

from dataclasses import dataclass
from typing import Optional, Sequence
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
    damping: float = 0.05,
    tol: float = 1e-4,
    max_iter: int = 150,
) -> IKResult:
    """
    Solve 3D position-only Inverse Kinematics using Damped Least Squares (DLS):
        dq = J_v^T (J_v J_v^T + lambda^2 I)^(-1) * e_pos
    """
    target = np.asarray(target_xyz, dtype=float).reshape(3)
    q = np.zeros(arm.dof, dtype=float) if q_init is None else np.asarray(q_init, dtype=float).copy()

    if len(q) != arm.dof:
        raise ValueError(f"Expected q_init of length {arm.dof}, got {len(q)}.")

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
        q = q + dq

    final_pos = arm.forward_kinematics(q)[:3, 3]
    final_err = float(np.linalg.norm(target - final_pos))
    return IKResult(
        q=q,
        success=final_err <= tol,
        iterations=max_iter,
        position_error=final_err,
        error_history=error_history,
    )