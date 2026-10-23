"""
MechaKinetics: Robotics Kinematics, Trajectory Planning & Actuator Sizing Toolkit.
"""

from mechakinetics.transforms import rot_x, rot_y, rot_z, homogeneous_transform, dh_matrix
from mechakinetics.kinematics import DHLink, SerialManipulator
from mechakinetics.inverse_kinematics import IKResult, solve_ik_position
from mechakinetics.mobile import (
    PlanarPose,
    DifferentialDriveKinematics,
    MecanumDriveKinematics,
    integrate_odometry,
)
from mechakinetics.trajectory import (
    TrajectorySample,
    quintic_polynomial_trajectory,
    trapezoidal_velocity_trajectory,
)
from mechakinetics.actuator_sizing import (
    DrivetrainParameters,
    SizingReport,
    evaluate_drivetrain_sizing,
)

__version__ = "0.1.0"

__all__ = [
    "rot_x",
    "rot_y",
    "rot_z",
    "homogeneous_transform",
    "dh_matrix",
    "DHLink",
    "SerialManipulator",
    "IKResult",
    "solve_ik_position",
    "PlanarPose",
    "DifferentialDriveKinematics",
    "MecanumDriveKinematics",
    "integrate_odometry",
]