import numpy as np

from mechakinetics.kinematics import DHLink, SerialManipulator
from mechakinetics.inverse_kinematics import solve_ik_position
from mechakinetics.mobile import (
    DifferentialDriveKinematics,
    MecanumDriveKinematics,
    PlanarPose,
    integrate_odometry,
)


def test_solve_ik_3dof_planar():
    arm = SerialManipulator([
        DHLink(d=0.0, a=0.8, alpha=0.0),
        DHLink(d=0.0, a=0.6, alpha=0.0),
        DHLink(d=0.0, a=0.4, alpha=0.0),
    ])
    target = np.array([1.1, 0.7, 0.0])
    result = solve_ik_position(arm, target, q_init=[0.1, 0.2, 0.1], tol=1e-5)

    assert result.success
    ee_actual = arm.forward_kinematics(result.q)[:3, 3]
    np.testing.assert_allclose(ee_actual, target, atol=1e-4)


def test_solve_ik_respects_joint_limits():
    arm = SerialManipulator([
        DHLink(d=0.0, a=1.0, alpha=0.0),
        DHLink(d=0.0, a=1.0, alpha=0.0),
    ])
    limits = [(-0.5, 0.5), (-0.5, 0.5)]
    result = solve_ik_position(arm, [0.0, 1.9, 0.0], joint_limits=limits)

    for q_val, (low, high) in zip(result.q, limits):
        assert low - 1e-9 <= q_val <= high + 1e-9


def test_mecanum_forward_inverse_roundtrip():
    mecanum = MecanumDriveKinematics(wheel_radius=0.05, half_length=0.15, half_width=0.12)
    desired_twist = np.array([0.6, -0.35, 0.8])  # vx, vy, omega

    wheel_speeds = mecanum.inverse_velocity(*desired_twist)
    recovered_twist = mecanum.forward_velocity(wheel_speeds)
    np.testing.assert_allclose(recovered_twist, desired_twist, atol=1e-12)


def test_differential_drive_and_odometry():
    diff = DifferentialDriveKinematics(wheel_radius=0.04, track_width=0.24)
    wheels = diff.inverse_velocity(vx=0.5, omega=0.0)
    twist = diff.forward_velocity(wheels)
    np.testing.assert_allclose(twist, [0.5, 0.0, 0.0], atol=1e-12)

    pose = PlanarPose(0.0, 0.0, 0.0)
    pose_after = integrate_odometry(pose, twist, dt=2.0)
    np.testing.assert_allclose(pose_after.as_array(), [1.0, 0.0, 0.0], atol=1e-12)