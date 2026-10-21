"""
3D spatial rotations and homogeneous transformation matrices (SE(3))
for robotic manipulators and mobile platforms.
"""

import numpy as np


def rot_x(theta_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix about the X-axis by theta_rad."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ], dtype=float)


def rot_y(theta_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix about the Y-axis by theta_rad."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [  c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,   c],
    ], dtype=float)


def rot_z(theta_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix about the Z-axis by theta_rad."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Construct a 4x4 SE(3) homogeneous transformation matrix from a 3x3
    rotation matrix and a 3-element translation vector.
    """
    rot = np.asarray(rotation, dtype=float)
    trans = np.asarray(translation, dtype=float).reshape(3)
    if rot.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {rot.shape}")

    T = np.eye(4, dtype=float)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Compute the standard Denavit-Hartenberg (DH) 4x4 transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=float)