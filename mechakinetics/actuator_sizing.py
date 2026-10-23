"""
Mechatronic actuator and gearbox sizing toolkit:
Computes reflected load inertia, inertia mismatch ratio, motor shaft torque
time-series, RMS continuous torque, peak acceleration torque, and peak RPM.
"""

from dataclasses import dataclass
import numpy as np

from mechakinetics.trajectory import TrajectorySample


@dataclass
class DrivetrainParameters:
    """Physical parameters of the motor, gearbox, and driven mechanical load."""
    motor_rotor_inertia: float       # J_m [kg*m^2]
    gearbox_inertia: float           # J_g [kg*m^2] at motor shaft
    load_inertia: float              # J_L [kg*m^2] at output shaft
    gear_ratio: float                # N (motor turns per 1 output turn, N >= 1)
    gearbox_efficiency: float = 0.90 # eta in (0, 1]
    viscous_damping: float = 0.01    # B_L [N*m*s/rad] at output shaft
    constant_load_torque: float = 0.0 # Gravity / friction load torque at output [N*m]


@dataclass
class SizingReport:
    """Summary engineering metrics for selecting a servo/BLDC motor and drive."""
    gear_ratio: float
    reflected_load_inertia: float
    total_motor_inertia: float
    inertia_mismatch_ratio: float
    peak_motor_torque_nm: float
    rms_motor_torque_nm: float
    peak_motor_speed_rpm: float
    peak_mechanical_power_w: float
    motor_torque_profile: np.ndarray
    motor_speed_rpm_profile: np.ndarray


def evaluate_drivetrain_sizing(
    params: DrivetrainParameters,
    load_trajectory: TrajectorySample,
) -> SizingReport:
    """
    Evaluate motor shaft torque, speed, power, and inertia matching for a given
    output load trajectory and drivetrain configuration.
    """
    if params.gear_ratio <= 0.0:
        raise ValueError("gear_ratio must be strictly positive.")
    if not (0.0 < params.gearbox_efficiency <= 1.0):
        raise ValueError("gearbox_efficiency must be in (0.0, 1.0].")

    N = float(params.gear_ratio)
    eta = float(params.gearbox_efficiency)

    # Reflected load inertia seen by the motor shaft: J_refl = J_L / N^2
    j_reflected = params.load_inertia / (N ** 2)
    j_motor_side = params.motor_rotor_inertia + params.gearbox_inertia
    if j_motor_side <= 0.0:
        raise ValueError("Sum of motor_rotor_inertia and gearbox_inertia must be positive.")

    j_total = j_motor_side + j_reflected
    mismatch_ratio = j_reflected / j_motor_side

    # Convert load velocity (rad/s) and acceleration (rad/s^2) to motor shaft
    w_load = load_trajectory.velocity
    alpha_load = load_trajectory.acceleration

    w_motor = w_load * N
    alpha_motor = alpha_load * N

    # External torque at load shaft (viscous + constant gravity/friction)
    tau_load_ext = params.constant_load_torque + params.viscous_damping * w_load

    # Total required motor electromagnetic torque across trajectory
    tau_inertial = j_total * alpha_motor
    tau_reflected_ext = tau_load_ext / (N * eta)
    tau_motor = tau_inertial + tau_reflected_ext

    # Compute engineering summary metrics
    peak_torque = float(np.max(np.abs(tau_motor)))
    rms_torque = float(np.sqrt(np.trapezoid(tau_motor**2, load_trajectory.t) / (load_trajectory.t[-1] - load_trajectory.t[0])))
    speed_rpm = w_motor * (60.0 / (2.0 * np.pi))
    peak_rpm = float(np.max(np.abs(speed_rpm)))
    mech_power = tau_motor * w_motor
    peak_power = float(np.max(np.abs(mech_power)))

    return SizingReport(
        gear_ratio=N,
        reflected_load_inertia=j_reflected,
        total_motor_inertia=j_total,
        inertia_mismatch_ratio=mismatch_ratio,
        peak_motor_torque_nm=peak_torque,
        rms_motor_torque_nm=rms_torque,
        peak_motor_speed_rpm=peak_rpm,
        peak_mechanical_power_w=peak_power,
        motor_torque_profile=tau_motor,
        motor_speed_rpm_profile=speed_rpm,
    )