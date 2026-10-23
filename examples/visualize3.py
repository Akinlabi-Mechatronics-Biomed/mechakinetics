"""
Day 3 Visualizer:
1. Quintic Minimum-Jerk vs. Trapezoidal Velocity Motion Profiles (Pos, Vel, Accel).
2. Servo Motor Torque-Speed Operating Locus & Gear Ratio Inertia Matching Curve.
"""

import numpy as np
import matplotlib.pyplot as plt

from mechakinetics.trajectory import (
    quintic_polynomial_trajectory,
    trapezoidal_velocity_trajectory,
)
from mechakinetics.actuator_sizing import (
    DrivetrainParameters,
    evaluate_drivetrain_sizing,
)


def main() -> None:
    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle(
        "MechaKinetics — Motion Trajectory Profiling & Mechatronic Actuator Sizing",
        fontsize=13,
        fontweight="bold",
    )

    # Generate trajectories for a 180-deg (pi rad) joint rotation
    quintic = quintic_polynomial_trajectory(0.0, np.pi, duration=1.2, num_steps=250)
    trap = trapezoidal_velocity_trajectory(0.0, np.pi, v_max=3.8, a_max=10.5, num_steps=250)

    # Left column: Position, Velocity, Acceleration comparison
    ax_pos = fig.add_subplot(3, 2, 1)
    ax_vel = fig.add_subplot(3, 2, 3, sharex=ax_pos)
    ax_acc = fig.add_subplot(3, 2, 5, sharex=ax_pos)

    ax_pos.plot(quintic.t, quintic.position, color="#1f77b4", lw=2.2, label="Quintic Min-Jerk")
    ax_pos.plot(trap.t, trap.position, "--", color="#ff7f0e", lw=2.0, label="Trapezoidal LSPB")
    ax_pos.set_ylabel("Pos [rad]")
    ax_pos.set_title("Joint Trajectory Comparison (0 to π rad)")
    ax_pos.grid(True, linestyle="--", alpha=0.5)
    ax_pos.legend(loc="lower right", fontsize=8)

    ax_vel.plot(quintic.t, quintic.velocity, color="#1f77b4", lw=2.2)
    ax_vel.plot(trap.t, trap.velocity, "--", color="#ff7f0e", lw=2.0)
    ax_vel.set_ylabel("Vel [rad/s]")
    ax_vel.grid(True, linestyle="--", alpha=0.5)

    ax_acc.plot(quintic.t, quintic.acceleration, color="#1f77b4", lw=2.2)
    ax_acc.plot(trap.t, trap.acceleration, "--", color="#ff7f0e", lw=2.0)
    ax_acc.set_ylabel("Acc [rad/s²]")
    ax_acc.set_xlabel("Time [s]")
    ax_acc.grid(True, linestyle="--", alpha=0.5)

    # Right-Top Plot: Motor Torque-Speed Operating Locus vs Motor Envelope
    ax_ts = fig.add_subplot(2, 2, 2)
    params = DrivetrainParameters(
        motor_rotor_inertia=1.8e-4,
        gearbox_inertia=0.4e-4,
        load_inertia=0.035,
        gear_ratio=12.0,
        constant_load_torque=2.0,
    )
    report = evaluate_drivetrain_sizing(params, quintic)

    # Plot dynamic operating locus (RPM vs Torque)
    ax_ts.plot(
        np.abs(report.motor_speed_rpm_profile),
        np.abs(report.motor_torque_profile),
        color="#2ca02c",
        lw=2.3,
        label="Dynamic Operating Trajectory",
    )
    ax_ts.axhline(
        report.rms_motor_torque_nm,
        color="royalblue",
        linestyle="-.",
        lw=1.8,
        label=f"Required RMS Torque ({report.rms_motor_torque_nm:.2f} N·m)",
    )
    ax_ts.scatter(
        [report.peak_motor_speed_rpm],
        [report.peak_motor_torque_nm],
        color="crimson",
        s=55,
        zorder=5,
        label=f"Peak Point ({report.peak_motor_speed_rpm:.0f} RPM, {report.peak_motor_torque_nm:.2f} N·m)",
    )

    # Sample reference 60W Robotics Servo Motor envelope (0.45 N*m peak, 0.25 N*m cont)
    rpm_env = np.array([0, 650, 1000])
    peak_env = np.array([0.42, 0.42, 0.28])
    cont_env = np.array([0.24, 0.24, 0.16])
    ax_ts.plot(rpm_env, peak_env, "r--", lw=1.5, label="Ref Servo Peak Limit")
    ax_ts.plot(rpm_env, cont_env, "k:", lw=1.5, label="Ref Servo Continuous Limit")

    ax_ts.set_title(f"Motor Torque-Speed Locus (Gear Ratio N={params.gear_ratio:.0f}:1)")
    ax_ts.set_xlabel("Motor Speed [RPM]")
    ax_ts.set_ylabel("Motor Shaft Torque [N·m]")
    ax_ts.set_xlim(0, 1050)
    ax_ts.set_ylim(0, 0.50)
    ax_ts.grid(True, linestyle="--", alpha=0.5)
    ax_ts.legend(loc="upper right", fontsize=7.5)

    # Right-Bottom Plot: Gear Ratio Sweep (Peak/RMS Torque & Inertia Mismatch)
    ax_gear = fig.add_subplot(2, 2, 4)
    ratios = np.linspace(4.0, 25.0, 35)
    peak_torques, rms_torques, mismatches = [], [], []

    for r in ratios:
        p = DrivetrainParameters(
            motor_rotor_inertia=1.8e-4,
            gearbox_inertia=0.4e-4,
            load_inertia=0.035,
            gear_ratio=float(r),
            constant_load_torque=2.0,
        )
        rep = evaluate_drivetrain_sizing(p, quintic)
        peak_torques.append(rep.peak_motor_torque_nm)
        rms_torques.append(rep.rms_motor_torque_nm)
        mismatches.append(rep.inertia_mismatch_ratio)

    ax_gear.plot(ratios, peak_torques, color="crimson", lw=2.2, label="Peak Motor Torque [N·m]")
    ax_gear.plot(ratios, rms_torques, color="royalblue", lw=2.2, label="RMS Motor Torque [N·m]")
    ax_gear.axvline(
        np.sqrt(0.035 / (1.8e-4 + 0.4e-4)),
        color="darkgreen",
        linestyle="--",
        label="Exact Inertia Match (J_refl = J_m)",
    )
    ax_gear.set_title("Gear Ratio Optimization: Torque vs. Reduction Ratio N")
    ax_gear.set_xlabel("Gear Reduction Ratio N:1")
    ax_gear.set_ylabel("Motor Torque [N·m]")
    ax_gear.grid(True, linestyle="--", alpha=0.5)
    ax_gear.legend(loc="upper right", fontsize=8)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.09, hspace=0.42, wspace=0.24)
    plt.show()

if __name__ == "__main__":
    main()