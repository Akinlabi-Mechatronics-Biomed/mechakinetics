"""
Forward kinematics and geometric Jacobian computation for serial-link
robotic manipulators using standard Denavit-Hartenberg (DH) parameters.
"""

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from mechakinetics.transforms import dh_matrix


@dataclass
class DHLink:
    """Single revolute or prismatic joint defined via standard DH parameters."""
    d: float
    a: float
    alpha: float
    theta_offset: float = 0.0
    is_prismatic: bool = False

    def transform(self, q: float) -> np.ndarray:
        if self.is_prismatic:
            return dh_matrix(self.theta_offset, self.d + q, self.a, self.alpha)
        return dh_matrix(q + self.theta_offset, self.d, self.a, self.alpha)


class SerialManipulator:
    """N-DOF serial-link robotic arm defined by a chain of DH links."""

    def __init__(self, links: Sequence[DHLink], name: str = "SerialArm") -> None:
        if not links:
            raise ValueError("SerialManipulator requires at least one DHLink.")
        self.links: List[DHLink] = list(links)
        self.name = name

    @property
    def dof(self) -> int:
        return len(self.links)

    def forward_kinematics(self, q: Sequence[float]) -> np.ndarray:
        """
        Compute the 4x4 end-effector pose matrix T_0_N for joint configuration q.
        """
        if len(q) != self.dof:
            raise ValueError(f"Expected {self.dof} joint values, got {len(q)}.")

        T = np.eye(4, dtype=float)
        for link, joint_val in zip(self.links, q):
            T = T @ link.transform(float(joint_val))
        return T

    def joint_frames(self, q: Sequence[float]) -> List[np.ndarray]:
        """
        Return cumulative 4x4 transformation matrices [T_0, T_1, ..., T_N]
        for every frame from base to end-effector.
        """
        if len(q) != self.dof:
            raise ValueError(f"Expected {self.dof} joint values, got {len(q)}.")

        frames = [np.eye(4, dtype=float)]
        T = np.eye(4, dtype=float)
        for link, joint_val in zip(self.links, q):
            T = T @ link.transform(float(joint_val))
            frames.append(T.copy())
        return frames

    def space_jacobian(self, q: Sequence[float]) -> np.ndarray:
        """
        Compute the 6xN geometric Jacobian J(q) mapping joint velocities
        to end-effector spatial twist [vx, vy, vz, wx, wy, wz]^T in base frame.
        """
        frames = self.joint_frames(q)
        p_end = frames[-1][:3, 3]
        J = np.zeros((6, self.dof), dtype=float)

        for i, link in enumerate(self.links):
            T_prev = frames[i]
            z_prev = T_prev[:3, 2]
            p_prev = T_prev[:3, 3]

            if link.is_prismatic:
                J[:3, i] = z_prev
                J[3:, i] = 0.0
            else:
                J[:3, i] = np.cross(z_prev, p_end - p_prev)
                J[3:, i] = z_prev

        return J