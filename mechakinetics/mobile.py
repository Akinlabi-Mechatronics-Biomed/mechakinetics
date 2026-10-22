"""
Forward, inverse velocity kinematics, and planar SE(2) dead-reckoning
odometry for Differential Drive and 4-Wheel Mecanum holonomic mobile robots.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class PlanarPose:
    """2D mobile robot pose (x, y, yaw_rad) in the world frame."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)


class DifferentialDriveKinematics:
    """Two-wheel non-holonomic differential drive kinematics model."""

    def __init__(self, wheel_radius: float, track_width: float) -> None:
        if wheel_radius <= 0 or track_width <= 0:
            raise ValueError("wheel_radius and track_width must be strictly positive.")
        self.r = float(wheel_radius)
        self.L = float(track_width)

    def inverse_velocity(self, vx: float, omega: float) -> np.ndarray:
        """
        Convert chassis twist (vx [m/s], omega [rad/s]) to wheel angular
        velocities [w_left, w_right] in rad/s.
        """
        w_left = (vx - 0.5 * self.L * omega) / self.r
        w_right = (vx + 0.5 * self.L * omega) / self.r
        return np.array([w_left, w_right], dtype=float)

    def forward_velocity(self, wheel_speeds: Sequence[float]) -> np.ndarray:
        """
        Convert wheel speeds [w_left, w_right] (rad/s) to chassis twist
        [vx, vy=0.0, omega] in the body frame.
        """
        w_l, w_r = float(wheel_speeds[0]), float(wheel_speeds[1])
        vx = 0.5 * self.r * (w_l + w_r)
        omega = (self.r / self.L) * (w_r - w_l)
        return np.array([vx, 0.0, omega], dtype=float)


class MecanumDriveKinematics:
    """
    4-Wheel Mecanum omnidirectional (holonomic) kinematics model.
    Wheel order: [front_left (FL), front_right (FR), rear_left (RL), rear_right (RR)].
    """

    def __init__(self, wheel_radius: float, half_length: float, half_width: float) -> None:
        if wheel_radius <= 0 or half_length <= 0 or half_width <= 0:
            raise ValueError("Geometry parameters must be strictly positive.")
        self.r = float(wheel_radius)
        self.lx = float(half_length)
        self.ly = float(half_width)
        self.geom_factor = self.lx + self.ly

    def inverse_velocity(self, vx: float, vy: float, omega: float) -> np.ndarray:
        """
        Convert desired body-frame planar twist (vx, vy, omega) into the 4 wheel
        angular speeds [w_fl, w_fr, w_rl, w_rr] in rad/s.
        """
        g = self.geom_factor
        w_fl = (vx - vy - g * omega) / self.r
        w_fr = (vx + vy + g * omega) / self.r
        w_rl = (vx + vy - g * omega) / self.r
        w_rr = (vx - vy + g * omega) / self.r
        return np.array([w_fl, w_fr, w_rl, w_rr], dtype=float)

    def forward_velocity(self, wheel_speeds: Sequence[float]) -> np.ndarray:
        """
        Convert 4 wheel angular speeds [w_fl, w_fr, w_rl, w_rr] (rad/s) into
        body-frame twist [vx, vy, omega].
        """
        if len(wheel_speeds) != 4:
            raise ValueError(f"Expected 4 wheel speeds, got {len(wheel_speeds)}.")
        w_fl, w_fr, w_rl, w_rr = [float(w) for w in wheel_speeds]
        vx = (self.r / 4.0) * (w_fl + w_fr + w_rl + w_rr)
        vy = (self.r / 4.0) * (-w_fl + w_fr + w_rl - w_rr)
        omega = (self.r / (4.0 * self.geom_factor)) * (-w_fl + w_fr - w_rl + w_rr)
        return np.array([vx, vy, omega], dtype=float)


def integrate_odometry(pose: PlanarPose, body_twist: Sequence[float], dt: float) -> PlanarPose:
    """
    Integrate body-frame twist [vx, vy, omega] over timestep dt (seconds)
    into the global world frame using exact midpoint Runge-Kutta integration.
    """
    vx, vy, omega = float(body_twist[0]), float(body_twist[1]), float(body_twist[2])
    theta_mid = pose.theta + 0.5 * omega * dt
    c, s = np.cos(theta_mid), np.sin(theta_mid)

    dx_world = (vx * c - vy * s) * dt
    dy_world = (vx * s + vy * c) * dt
    new_theta = pose.theta + omega * dt

    return PlanarPose(
        x=pose.x + dx_world,
        y=pose.y + dy_world,
        theta=float((new_theta + np.pi) % (2.0 * np.pi) - np.pi),
    )