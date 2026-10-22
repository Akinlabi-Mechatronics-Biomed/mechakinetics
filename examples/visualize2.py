"""
Day 2 Visualizer:
1. 3-DOF Planar Arm tracking a circular trajectory via numerical Inverse Kinematics.
2. 4WD Mecanum Omnidirectional Rover executing a holonomic figure-8 trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt

from mechakinetics.kinematics import DHLink, SerialManipulator
from mechakinetics.inverse_kinematics import solve_ik_position
from mechakinetics.mobile import MecanumDriveKinematics, PlanarPose, integrate_odometry


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("MechaKinetics — Numerical Inverse Kinematics & Holonomic Mecanum Odometry", fontsize=13, fontweight="bold")

    # =========================================================================
    # Left Plot: 3-DOF Arm Tracking Circular Path with Damped Least Squares IK
    # =========================================================================
    arm = SerialManipulator([
        DHLink(d=0.0, a=0.75, alpha=0.0),
        DHLink(d=0.0, a=0.60, alpha=0.0),
        DHLink(d=0.0, a=0.40, alpha=0.0),
    ])

    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    circle_x = 0.85 + 0.35 * np.cos(angles)
    circle_y = 0.55 + 0.35 * np.sin(angles)

    ax1.plot(circle_x, circle_y, "k--", linewidth=1.5, label="Desired Circular Path")

    q_curr = np.array([1.1, -0.8, -0.5])  # Clean "elbow-up" posture
    ee_actual_pts = []

    for i, (tx, ty) in enumerate(zip(circle_x, circle_y)):
        res = solve_ik_position(arm, [tx, ty, 0.0], q_init=q_curr, tol=1e-4)
        q_curr = res.q
        frames = arm.joint_frames(q_curr)
        pts = np.array([T[:3, 3] for T in frames])
        ee_actual_pts.append(pts[-1])

        if i % 6 == 0:
            ax1.plot(pts[:, 0], pts[:, 1], "-o", alpha=0.65, linewidth=2.2, label=f"IK Pose {i // 6 + 1}")

    ee_actual_pts = np.array(ee_actual_pts)
    ax1.scatter(ee_actual_pts[:, 0], ee_actual_pts[:, 1], color="crimson", s=18, zorder=5, label="Solved EE Positions")
    ax1.plot(0, 0, "ks", markersize=9, label="Base")
    ax1.set_title("DLS Inverse Kinematics: Circle Trajectory Tracking")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_xlim(-0.3, 1.6)
    ax1.set_ylim(-0.45, 1.35)
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8)

    # =========================================================================
    # Right Plot: 4WD Mecanum Rover Holonomic Odometry & Heading Vectors
    # =========================================================================
    mecanum = MecanumDriveKinematics(wheel_radius=0.05, half_length=0.18, half_width=0.15)
    pose = PlanarPose(0.0, 0.0, 0.0)
    dt = 0.05
    t_steps = np.arange(0.0, 12.6, dt)

    poses = []
    for t in t_steps:
        # Command simultaneous forward velocity, lateral strafing, and yaw rotation
        vx_body = 0.35 * np.cos(0.5 * t)
        vy_body = 0.35 * np.sin(1.0 * t)
        omega_body = 0.15 * np.cos(0.5 * t)

        wheels = mecanum.inverse_velocity(vx_body, vy_body, omega_body)
        recovered_twist = mecanum.forward_velocity(wheels)
        pose = integrate_odometry(pose, recovered_twist, dt)
        poses.append(pose.as_array())

    poses_arr = np.array(poses)
    ax2.plot(poses_arr[:, 0], poses_arr[:, 1], color="#1b9e77", linewidth=2.5, label="Mecanum Trajectory")

    # Draw robot heading arrows along trajectory
    step_idx = np.arange(0, len(poses_arr), 20)
    ax2.quiver(
        poses_arr[step_idx, 0],
        poses_arr[step_idx, 1],
        np.cos(poses_arr[step_idx, 2]),
        np.sin(poses_arr[step_idx, 2]),
        color="#d95f02",
        scale=12,
        width=0.006,
        label="Chassis Heading",
    )

    ax2.plot(poses_arr[0, 0], poses_arr[0, 1], "go", markersize=8, label="Start")
    ax2.set_title("4WD Mecanum Holonomic Locomotion & Odometry")
    ax2.set_xlabel("World X (m)")
    ax2.set_ylabel("World Y (m)")
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()