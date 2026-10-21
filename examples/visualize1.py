"""
Interactive visualization of Day 1 MechaKinetics modules:
1. 3-DOF Planar Manipulator poses, workspace cloud, and Jacobian velocity vector.
2. 4-DOF 3D Spatial Manipulator with SE(3) joint coordinate frames.
"""

import numpy as np
import matplotlib.pyplot as plt

from mechakinetics.kinematics import DHLink, SerialManipulator


def extract_joint_positions(arm: SerialManipulator, q: list[float]) -> np.ndarray:
    """Extract (N+1, 3) array of [x, y, z] coordinates for base and all joints."""
    frames = arm.joint_frames(q)
    return np.array([T[:3, 3] for T in frames])


def main() -> None:
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("MechaKinetics Day 1 — Forward Kinematics & Space Jacobian Visualizer", fontsize=14, fontweight="bold")

    # =========================================================================
    # Plot 1: 3-DOF Planar Arm + Workspace Cloud + Jacobian Twist Vector
    # =========================================================================
    ax1 = fig.add_subplot(1, 2, 1)
    planar_arm = SerialManipulator([
        DHLink(d=0.0, a=0.8, alpha=0.0),
        DHLink(d=0.0, a=0.6, alpha=0.0),
        DHLink(d=0.0, a=0.4, alpha=0.0),
    ], name="Planar-3R")

    # Sample random joint configurations to plot reachable workspace cloud
    rng = np.random.default_rng(42)
    samples = rng.uniform(-np.pi, np.pi, size=(600, 3))
    ee_points = np.array([planar_arm.forward_kinematics(q)[:3, 3] for q in samples])
    ax1.scatter(ee_points[:, 0], ee_points[:, 1], s=5, alpha=0.2, color="steelblue", label="Reachable Workspace")

    # Plot 3 distinct arm configurations
    configs = [
        ([0.2, 0.5, 0.4], "#2b5c8f", "Pose A (q=[0.2, 0.5, 0.4])"),
        ([0.9, -0.7, -0.5], "#d95f02", "Pose B (q=[0.9, -0.7, -0.5])"),
        ([1.6, 0.8, -0.6], "#1b9e77", "Pose C (q=[1.6, 0.8, -0.6])"),
    ]

    for q, color, label in configs:
        pts = extract_joint_positions(planar_arm, q)
        ax1.plot(pts[:, 0], pts[:, 1], "-o", linewidth=3, markersize=7, color=color, label=label)

    # Compute and draw end-effector linear velocity vector v = J * q_dot at Pose A
    q_active = configs[0][0]
    q_dot = np.array([0.5, 0.8, -0.4])  # rad/s joint speeds
    J = planar_arm.space_jacobian(q_active)
    v_ee = J[:3, :] @ q_dot
    ee_pos = extract_joint_positions(planar_arm, q_active)[-1]

    ax1.quiver(
        ee_pos[0], ee_pos[1], v_ee[0], v_ee[1],
        angles="xy", scale_units="xy", scale=1.5, color="crimson", width=0.008,
        label=r"EE Velocity $\mathbf{v} = J(\mathbf{q})\dot{\mathbf{q}}$"
    )

    ax1.plot(0, 0, "ks", markersize=10, label="Base Origin")
    ax1.set_title("3-DOF Planar Arm: Poses, Workspace & Jacobian Velocity")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-2.0, 2.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower left", fontsize=8)

    # =========================================================================
    # Plot 2: 4-DOF 3D Spatial Anthropomorphic Arm with SE(3) Coordinate Frames
    # =========================================================================
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    spatial_arm = SerialManipulator([
        DHLink(d=0.4, a=0.0, alpha=np.pi / 2),   # Waist turret
        DHLink(d=0.0, a=0.6, alpha=0.0),         # Shoulder pitch
        DHLink(d=0.0, a=0.5, alpha=0.0),         # Elbow pitch
        DHLink(d=0.0, a=0.25, alpha=-np.pi / 2), # Wrist
    ], name="Spatial-4DOF")

    q_3d = [np.deg2rad(35), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(20)]
    frames = spatial_arm.joint_frames(q_3d)
    pts_3d = np.array([T[:3, 3] for T in frames])

    # Plot arm links
    ax2.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], "-o", color="#2b2d42", linewidth=4, markersize=7, label="Arm Links")

    # Draw local X (Red), Y (Green), Z (Blue) axes at each joint frame
    axis_len = 0.14
    for T in frames:
        origin = T[:3, 3]
        R = T[:3, :3]
        ax2.quiver(*origin, *(R[:, 0] * axis_len), color="r", linewidth=1.5)
        ax2.quiver(*origin, *(R[:, 1] * axis_len), color="g", linewidth=1.5)
        ax2.quiver(*origin, *(R[:, 2] * axis_len), color="b", linewidth=1.5)

    ax2.set_title("4-DOF 3D Spatial Arm & Joint SE(3) Frames (RGB = XYZ)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_zlim(0.0, 1.2)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()