"""
Point-to-point motion trajectory generators for robotic joints and linear axes:
- Quintic polynomial (minimum-jerk) smooth trajectories with boundary conditions.
- Trapezoidal velocity (LSPB - Linear Segment with Parabolic Blend) profiles.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectorySample:
    """Time-indexed trajectory arrays for position, velocity, acceleration, and jerk."""
    t: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


def quintic_polynomial_trajectory(
    q_start: float,
    q_end: float,
    duration: float,
    num_steps: int = 100,
    v_start: float = 0.0,
    v_end: float = 0.0,
    a_start: float = 0.0,
    a_end: float = 0.0,
) -> TrajectorySample:
    """
    Generate a 5th-order (quintic) polynomial trajectory q(t) satisfying
    initial and final position, velocity, and acceleration constraints.
    """
    if duration <= 0.0:
        raise ValueError("Trajectory duration must be strictly positive.")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    T = float(duration)
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    c0 = float(q_start)
    c1 = float(v_start)
    c2 = 0.5 * float(a_start)

    # Solve 3x3 linear system for [c3, c4, c5]
    A = np.array([
        [T3,      T4,       T5],
        [3 * T2,  4 * T3,   5 * T4],
        [6 * T,   12 * T2,  20 * T3],
    ], dtype=float)

    b = np.array([
        q_end - (c0 + c1 * T + c2 * T2),
        v_end - (c1 + 2.0 * c2 * T),
        a_end - (2.0 * c2),
    ], dtype=float)

    c3, c4, c5 = np.linalg.solve(A, b)

    t = np.linspace(0.0, T, num_steps)
    pos = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    vel = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    acc = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    jrk = 6 * c3 + 24 * c4 * t + 60 * c5 * t**2

    return TrajectorySample(t=t, position=pos, velocity=vel, acceleration=acc, jerk=jrk)


def trapezoidal_velocity_trajectory(
    q_start: float,
    q_end: float,
    v_max: float,
    a_max: float,
    num_steps: int = 100,
) -> TrajectorySample:
    """
    Generate a rest-to-rest Trapezoidal (or Triangular if short move) velocity
    profile constrained by maximum velocity v_max and acceleration a_max.
    """
    if v_max <= 0.0 or a_max <= 0.0:
        raise ValueError("v_max and a_max must be strictly positive.")

    dq = float(q_end - q_start)
    sign = 1.0 if dq >= 0.0 else -1.0
    dist = abs(dq)

    if dist < 1e-12:
        t = np.zeros(num_steps)
        zeros = np.zeros(num_steps)
        return TrajectorySample(t, np.full(num_steps, q_start), zeros, zeros, zeros)

    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2

    if 2.0 * d_acc >= dist:
        # Triangular profile (never reaches v_max)
        t_acc = np.sqrt(dist / a_max)
        t_cruise = 0.0
        v_peak = a_max * t_acc
    else:
        # Full trapezoidal profile with constant-velocity cruise phase
        d_cruise = dist - 2.0 * d_acc
        t_cruise = d_cruise / v_max
        v_peak = v_max

    total_time = 2.0 * t_acc + t_cruise
    t = np.linspace(0.0, total_time, num_steps)

    pos = np.zeros_like(t)
    vel = np.zeros_like(t)
    acc = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti <= t_acc:
            pos[i] = 0.5 * a_max * ti**2
            vel[i] = a_max * ti
            acc[i] = a_max
        elif ti <= t_acc + t_cruise:
            dt_c = ti - t_acc
            pos[i] = d_acc + v_peak * dt_c
            vel[i] = v_peak
            acc[i] = 0.0
        else:
            dt_d = total_time - ti
            pos[i] = dist - 0.5 * a_max * dt_d**2
            vel[i] = a_max * dt_d
            acc[i] = -a_max

    pos = q_start + sign * pos
    vel = sign * vel
    acc = sign * acc
    jrk = np.gradient(acc, t)

    return TrajectorySample(t=t, position=pos, velocity=vel, acceleration=acc, jerk=jrk)