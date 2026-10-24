"""
MechaKinetics Interactive Engineering Web Studio (Streamlit App)
Run locally with:
    streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from mechakinetics.kinematics import DHLink, SerialManipulator
from mechakinetics.inverse_kinematics import solve_ik_position
from mechakinetics.mobile import MecanumDriveKinematics, PlanarPose, integrate_odometry
from mechakinetics.trajectory import (
    quintic_polynomial_trajectory,
    trapezoidal_velocity_trajectory,
)
from mechakinetics.actuator_sizing import (
    DrivetrainParameters,
    evaluate_drivetrain_sizing,
)

st.set_page_config(page_title="MechaKinetics Studio", page_icon="🦾", layout="wide")

st.title("🦾 MechaKinetics — Robotics Kinematics & Drivetrain Sizing Studio")
st.markdown(
    "Interactive engineering workbench for **serial-link manipulator kinematics**, "
    "**holonomic Mecanum locomotion**, **motion trajectory profiling**, and **servo/gearbox sizing**."
)

tab_ik, tab_sizing, tab_mecanum = st.tabs([
    "1. Manipulator FK & Inverse Kinematics",
    "2. Trajectory & Servo Actuator Sizing",
    "3. 4WD Mecanum Holonomic Simulator",
])

# =============================================================================
# TAB 1: Manipulator Forward & Inverse Kinematics
# =============================================================================
with tab_ik:
    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        st.subheader("3-DOF Planar Arm Geometry")
        l1 = st.slider("Link 1 Length a₁ [m]", 0.2, 1.2, 0.75, 0.05)
        l2 = st.slider("Link 2 Length a₂ [m]", 0.2, 1.2, 0.60, 0.05)
        l3 = st.slider("Link 3 Length a₃ [m]", 0.1, 1.0, 0.40, 0.05)

        st.subheader("Target End-Effector Position")
        tx = st.slider("Target X [m]", -1.5, 1.7, 0.95, 0.05)
        ty = st.slider("Target Y [m]", -1.5, 1.7, 0.65, 0.05)
        damping = st.slider("DLS Damping λ", 0.01, 0.25, 0.05, 0.01)

    arm = SerialManipulator([
        DHLink(d=0.0, a=l1, alpha=0.0),
        DHLink(d=0.0, a=l2, alpha=0.0),
        DHLink(d=0.0, a=l3, alpha=0.0),
    ])

    ik_res = solve_ik_position(arm, [tx, ty, 0.0], q_init=[0.8, -0.6, -0.3], damping=damping, tol=1e-4)
    frames = arm.joint_frames(ik_res.q)
    pts = np.array([T[:3, 3] for T in frames])

    with col_ctrl:
        if ik_res.success:
            st.success(f"Converged in {ik_res.iterations} iterations (Error: {ik_res.position_error*1000:.2f} mm)")
        else:
            st.warning(f"Target outside reachable envelope (Closest error: {ik_res.position_error:.3f} m)")
        st.write("**Solved Joint Angles [deg]:**", np.round(np.rad2deg(ik_res.q), 2))

    with col_plot:
        fig_ik, (ax_arm, ax_err) = plt.subplots(1, 2, figsize=(10.5, 4.5))
        reach = l1 + l2 + l3
        circle = plt.Circle((0, 0), reach, color="steelblue", alpha=0.08, label="Max Reach Envelope")
        ax_arm.add_patch(circle)
        ax_arm.plot(pts[:, 0], pts[:, 1], "-o", lw=3.5, markersize=8, color="#1f77b4", label="Solved Arm Configuration")
        ax_arm.scatter([tx], [ty], color="crimson", s=90, marker="X", zorder=5, label="Target (X, Y)")
        ax_arm.plot(0, 0, "ks", markersize=10, label="Base")
        ax_arm.set_xlim(-reach - 0.2, reach + 0.2)
        ax_arm.set_ylim(-reach - 0.2, reach + 0.2)
        ax_arm.set_aspect("equal")
        ax_arm.set_xlabel("X [m]")
        ax_arm.set_ylabel("Y [m]")
        ax_arm.set_title("2D Manipulator IK Solution")
        ax_arm.grid(True, linestyle="--", alpha=0.5)
        ax_arm.legend(loc="lower left", fontsize=8)

        ax_err.plot(range(1, len(ik_res.error_history) + 1), ik_res.error_history, "-o", color="#d95f02", markersize=4)
        ax_err.set_yscale("log")
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("Position Error Norm [m]")
        ax_err.set_title("DLS Convergence History")
        ax_err.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_ik)

# =============================================================================
# TAB 2: Trajectory & Servo Actuator Sizing
# =============================================================================
with tab_sizing:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Motion & Load Parameters")
        move_deg = st.slider("Output Rotation Angle [deg]", 30.0, 360.0, 180.0, 15.0)
        duration = st.slider("Move Duration [s]", 0.3, 3.0, 1.2, 0.1)
        gear_ratio = st.slider("Gearbox Reduction Ratio N:1", 1.0, 30.0, 12.0, 1.0)
        load_inertia = st.number_input("Load Inertia J_L [kg·m²]", value=0.035, min_value=0.001, step=0.005, format="%.4f")
        load_torque = st.slider("External Load Torque [N·m]", 0.0, 10.0, 2.0, 0.5)

    traj_q = quintic_polynomial_trajectory(0.0, np.deg2rad(move_deg), duration=duration, num_steps=200)
    traj_t = trapezoidal_velocity_trajectory(0.0, np.deg2rad(move_deg), v_max=3.8, a_max=10.5, num_steps=200)

    params = DrivetrainParameters(
        motor_rotor_inertia=1.8e-4,
        gearbox_inertia=0.4e-4,
        load_inertia=load_inertia,
        gear_ratio=gear_ratio,
        constant_load_torque=load_torque,
    )
    rep = evaluate_drivetrain_sizing(params, traj_q)

    with c1:
        st.metric("Peak Motor Torque", f"{rep.peak_motor_torque_nm:.3f} N·m")
        st.metric("Continuous RMS Torque", f"{rep.rms_motor_torque_nm:.3f} N·m")
        st.metric("Peak Motor Speed", f"{rep.peak_motor_speed_rpm:.0f} RPM")
        st.metric("Inertia Mismatch Ratio (J_refl / J_m)", f"{rep.inertia_mismatch_ratio:.2f}")

    with c2:
        fig_sz, (ax_v, ax_ts) = plt.subplots(1, 2, figsize=(10.5, 4.4))
        ax_v.plot(traj_q.t, traj_q.velocity, lw=2.2, label="Quintic Velocity [rad/s]")
        ax_v.plot(traj_t.t, traj_t.velocity, "--", lw=2.0, label="Trapezoidal Velocity [rad/s]")
        ax_v.set_xlabel("Time [s]")
        ax_v.set_ylabel("Load Velocity [rad/s]")
        ax_v.set_title("Output Shaft Velocity Profile")
        ax_v.grid(True, linestyle="--", alpha=0.5)
        ax_v.legend()

        ax_ts.plot(np.abs(rep.motor_speed_rpm_profile), np.abs(rep.motor_torque_profile), color="seagreen", lw=2.3, label="Dynamic Locus")
        ax_ts.axhline(rep.rms_motor_torque_nm, color="royalblue", linestyle="-.", label=f"RMS Torque ({rep.rms_motor_torque_nm:.2f} N·m)")
        ax_ts.scatter([rep.peak_motor_speed_rpm], [rep.peak_motor_torque_nm], color="crimson", s=60, zorder=5, label="Peak Operating Point")
        ax_ts.set_xlabel("Motor Speed [RPM]")
        ax_ts.set_ylabel("Motor Torque [N·m]")
        ax_ts.set_title(f"Motor Torque-Speed Operating Curve (N={gear_ratio:.0f}:1)")
        ax_ts.grid(True, linestyle="--", alpha=0.5)
        ax_ts.legend(fontsize=8)
        st.pyplot(fig_sz)

# =============================================================================
# TAB 3: 4WD Mecanum Holonomic Simulator
# =============================================================================
with tab_mecanum:
    m1, m2 = st.columns([1, 2])
    with m1:
        st.subheader("Body Twist Commands")
        vx_cmd = st.slider("Forward Speed vx [m/s]", -1.0, 1.0, 0.4, 0.05)
        vy_cmd = st.slider("Lateral Strafe Speed vy [m/s]", -1.0, 1.0, 0.3, 0.05)
        omega_cmd = st.slider("Yaw Rate ω [rad/s]", -1.5, 1.5, 0.25, 0.05)

        mec = MecanumDriveKinematics(wheel_radius=0.05, half_length=0.18, half_width=0.15)
        wheels = mec.inverse_velocity(vx_cmd, vy_cmd, omega_cmd)
        st.write("**Computed Wheel Speeds [rad/s]:**")
        st.json({
            "Front-Left (FL)": round(float(wheels[0]), 2),
            "Front-Right (FR)": round(float(wheels[1]), 2),
            "Rear-Left (RL)": round(float(wheels[2]), 2),
            "Rear-Right (RR)": round(float(wheels[3]), 2),
        })

    with m2:
        pose = PlanarPose(0.0, 0.0, 0.0)
        pts_m = [pose.as_array()]
        for _ in range(120):
            pose = integrate_odometry(pose, [vx_cmd, vy_cmd, omega_cmd], dt=0.05)
            pts_m.append(pose.as_array())
        pts_m = np.array(pts_m)

        fig_m, ax_m = plt.subplots(figsize=(8, 4.4))
        ax_m.plot(pts_m[:, 0], pts_m[:, 1], color="#1b9e77", lw=2.5, label="Integrated SE(2) Trajectory (6s)")
        idx = np.arange(0, len(pts_m), 15)
        ax_m.quiver(pts_m[idx, 0], pts_m[idx, 1], np.cos(pts_m[idx, 2]), np.sin(pts_m[idx, 2]), color="#d95f02", scale=14, label="Chassis Heading")
        ax_m.plot(0, 0, "go", markersize=8, label="Start (0,0)")
        ax_m.set_aspect("equal")
        ax_m.set_xlabel("World X [m]")
        ax_m.set_ylabel("World Y [m]")
        ax_m.set_title("Holonomic Mecanum Dead-Reckoning Trajectory")
        ax_m.grid(True, linestyle="--", alpha=0.5)
        ax_m.legend()
        st.pyplot(fig_m)