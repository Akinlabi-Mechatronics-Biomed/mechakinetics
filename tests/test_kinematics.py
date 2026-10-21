import numpy as np
import pytest

from mechakinetics.transforms import rot_x, rot_y, rot_z, homogeneous_transform
from mechakinetics.kinematics import DHLink, SerialManipulator


def test_rotation_orthogonality():
    for angle in (-1.2, 0.0, np.pi / 4, np.pi / 2):
        for fn in (rot_x, rot_y, rot_z):
            R = fn(angle)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(R), 1.0)


def test_planar_2dof_forward_kinematics():
    arm = SerialManipulator([
        DHLink(d=0.0, a=1.0, alpha=0.0),
        DHLink(d=0.0, a=0.5, alpha=0.0),
    ], name="Planar2R")

    # Stretched along +X axis (q1=0, q2=0) -> x = 1.5, y = 0.0
    T0 = arm.forward_kinematics([0.0, 0.0])
    np.testing.assert_allclose(T0[:3, 3], [1.5, 0.0, 0.0], atol=1e-12)

    # Rotated 90 deg at base (q1=pi/2, q2=0) -> x = 0.0, y = 1.5
    T1 = arm.forward_kinematics([np.pi / 2, 0.0])
    np.testing.assert_allclose(T1[:3, 3], [0.0, 1.5, 0.0], atol=1e-12)


def test_planar_2dof_jacobian():
    arm = SerialManipulator([
        DHLink(d=0.0, a=1.0, alpha=0.0),
        DHLink(d=0.0, a=1.0, alpha=0.0),
    ])
    J = arm.space_jacobian([0.0, np.pi / 2])
    np.testing.assert_allclose(J[:3, 0], [-1.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(J[:3, 1], [-1.0, 0.0, 0.0], atol=1e-12)