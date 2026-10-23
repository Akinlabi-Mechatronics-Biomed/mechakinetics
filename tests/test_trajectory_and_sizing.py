import numpy as np

from mechakinetics.trajectory import (
    quintic_polynomial_trajectory,
    trapezoidal_velocity_trajectory,
)
from mechakinetics.actuator_sizing import (
    DrivetrainParameters,
    evaluate_drivetrain_sizing,
)


def test_quintic_boundary_conditions():
    traj = quintic_polynomial_trajectory(q_start=0.5, q_end=2.5, duration=1.5, num_steps=150)

    assert np.isclose(traj.position[0], 0.5)
    assert np.isclose(traj.position[-1], 2.5)
    assert np.isclose(traj.velocity[0], 0.0, atol=1e-9)
    assert np.isclose(traj.velocity[-1], 0.0, atol=1e-9)
    assert np.isclose(traj.acceleration[0], 0.0, atol=1e-9)
    assert np.isclose(traj.acceleration[-1], 0.0, atol=1e-9)


def test_trapezoidal_trajectory_respects_velocity_bounds():
    v_max = 1.2
    a_max = 3.0
    traj = trapezoidal_velocity_trajectory(0.0, 2.0, v_max=v_max, a_max=a_max, num_steps=200)

    assert np.isclose(traj.position[0], 0.0)
    assert np.isclose(traj.position[-1], 2.0)
    assert np.max(np.abs(traj.velocity)) <= v_max + 1e-6
    assert np.max(np.abs(traj.acceleration)) <= a_max + 1e-6


def test_drivetrain_sizing_inertia_reduction():
    traj = quintic_polynomial_trajectory(0.0, np.pi, duration=1.0, num_steps=200)
    params_n5 = DrivetrainParameters(
        motor_rotor_inertia=1e-4,
        gearbox_inertia=2e-5,
        load_inertia=0.03,
        gear_ratio=5.0,
    )
    params_n15 = DrivetrainParameters(
        motor_rotor_inertia=1e-4,
        gearbox_inertia=2e-5,
        load_inertia=0.03,
        gear_ratio=15.0,
    )

    rep5 = evaluate_drivetrain_sizing(params_n5, traj)
    rep15 = evaluate_drivetrain_sizing(params_n15, traj)

    # Reflected inertia scales with 1 / N^2 -> 9x smaller for N=15 vs N=5
    assert np.isclose(rep5.reflected_load_inertia / rep15.reflected_load_inertia, 9.0)
    assert rep15.rms_motor_torque_nm <= rep15.peak_motor_torque_nm