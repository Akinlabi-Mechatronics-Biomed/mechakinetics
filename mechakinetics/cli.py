"""
Command-Line Interface (CLI) for MechaKinetics engineering calculations.
Usage:
    python -m mechakinetics.cli size-motor --gear-ratio 10 --load-inertia 0.02 --move-rad 3.14 --duration 1.2
    python -m mechakinetics.cli mecanum --vx 0.5 --vy -0.2 --omega 0.4
"""

import argparse
import sys

from mechakinetics.mobile import MecanumDriveKinematics
from mechakinetics.trajectory import quintic_polynomial_trajectory
from mechakinetics.actuator_sizing import DrivetrainParameters, evaluate_drivetrain_sizing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mechakinetics",
        description="MechaKinetics CLI: Robotics kinematics, motion profiling, and actuator sizing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand 1: Motor & Gearbox Sizing
    p_size = sub.add_parser("size-motor", help="Compute RMS/peak motor torque and inertia ratio for a move.")
    p_size.add_argument("--move-rad", type=float, default=3.14159, help="Output rotation angle [rad]")
    p_size.add_argument("--duration", type=float, default=1.0, help="Move duration [s]")
    p_size.add_argument("--gear-ratio", type=float, default=10.0, help="Gear reduction ratio N:1")
    p_size.add_argument("--load-inertia", type=float, default=0.025, help="Load inertia J_L [kg*m^2]")
    p_size.add_argument("--motor-inertia", type=float, default=0.00015, help="Motor rotor inertia J_m [kg*m^2]")
    p_size.add_argument("--load-torque", type=float, default=1.5, help="Constant gravity/friction load torque [N*m]")

    # Subcommand 2: Mecanum Wheel Speed Calculator
    p_mec = sub.add_parser("mecanum", help="Compute 4WD Mecanum wheel speeds from chassis twist.")
    p_mec.add_argument("--vx", type=float, required=True, help="Forward velocity [m/s]")
    p_mec.add_argument("--vy", type=float, required=True, help="Lateral strafe velocity [m/s]")
    p_mec.add_argument("--omega", type=float, required=True, help="Yaw rate [rad/s]")
    p_mec.add_argument("--wheel-radius", type=float, default=0.05, help="Wheel radius [m]")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "size-motor":
        traj = quintic_polynomial_trajectory(0.0, args.move_rad, args.duration, num_steps=200)
        params = DrivetrainParameters(
            motor_rotor_inertia=args.motor_inertia,
            gearbox_inertia=0.2 * args.motor_inertia,
            load_inertia=args.load_inertia,
            gear_ratio=args.gear_ratio,
            constant_load_torque=args.load_torque,
        )
        report = evaluate_drivetrain_sizing(params, traj)
        print("=== MechaKinetics Actuator Sizing Report ===")
        print(f"  Gear Ratio (N)           : {report.gear_ratio:.2f}:1")
        print(f"  Reflected Load Inertia   : {report.reflected_load_inertia:.6f} kg*m^2")
        print(f"  Inertia Mismatch Ratio   : {report.inertia_mismatch_ratio:.2f} (Target < 5.0 for high dynamic response)")
        print(f"  Peak Motor Torque        : {report.peak_motor_torque_nm:.3f} N*m")
        print(f"  Continuous RMS Torque    : {report.rms_motor_torque_nm:.3f} N*m")
        print(f"  Peak Motor Speed         : {report.peak_motor_speed_rpm:.1f} RPM")
        print(f"  Peak Mechanical Power    : {report.peak_mechanical_power_w:.1f} W")
        return 0

    if args.command == "mecanum":
        mec = MecanumDriveKinematics(wheel_radius=args.wheel_radius, half_length=0.18, half_width=0.15)
        w = mec.inverse_velocity(args.vx, args.vy, args.omega)
        print("=== 4WD Mecanum Wheel Angular Speeds [rad/s] ===")
        print(f"  Front-Left  (FL): {w[0]:+8.3f} rad/s")
        print(f"  Front-Right (FR): {w[1]:+8.3f} rad/s")
        print(f"  Rear-Left   (RL): {w[2]:+8.3f} rad/s")
        print(f"  Rear-Right  (RR): {w[3]:+8.3f} rad/s")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())