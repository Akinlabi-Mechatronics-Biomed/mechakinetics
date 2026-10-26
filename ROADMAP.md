# MechaKinetics Engineering Roadmap (`v0.2.0` – `v0.3.0`)

This document tracks the upcoming software architecture, mathematical modeling, and validation milestones (~26 hours of planned development) for **MechaKinetics**.

---

## Milestone 1: Full 6-DOF Pose IK & Redundant Manipulators (`v0.2.0-alpha` — ~7 Hours)
- **Full $SE(3)$ Position + Orientation Inverse Kinematics (`Issue #1`):**
  - Extend `solve_ik_position` into `solve_ik_pose` using the full $6 \times N$ geometric Jacobian and $SO(3)$ logarithmic map / quaternion orientation error vectors.
  - Add configurable task-space weighting matrices $W_x \in \mathbb{R}^{6 \times 6}$ to balance translational vs. rotational tracking accuracy.
- **Null-Space Posture Optimization (`Issue #2`):**
  - For redundant manipulators ($N > 6$ or planar $N > 2$), project secondary objectives (joint-limit avoidance and manipulability maximization $\sqrt{\det(J J^T)}$ into the Jacobian null space $(I - J^\dagger J)\dot{\mathbf{q}}_0$.

## Milestone 2: Rigid-Body Inverse Dynamics & Closed-Loop Control (`v0.2.0` — ~8 Hours)
- **Recursive Newton-Euler (RNE) Inverse Dynamics Solver (`Issue #3`):**
  - Add link mass, center-of-mass (CoM) vector, and $3 \times 3$ rotational inertia tensor attributes to `DHLink`.
  - Implement outward velocity/acceleration propagation and inward wrench recursion to compute required joint torques $\boldsymbol{\tau}(t)$ along any `TrajectorySample`.
- **Discrete-Time PID & Computed-Torque Control Simulator (`Issue #4`):**
  - Implement a closed-loop trajectory tracking simulator (`mechakinetics/control.py`) with anti-windup PID, gravity compensation, actuator torque saturation, and Gaussian encoder noise injection.

## Milestone 3: Advanced Motion Profiling & Robot Description Import (`v0.3.0-alpha` — ~6 Hours)
- **7-Segment S-Curve (Bounded-Jerk) Trajectory Generator (`Issue #5`):**
  - Implement industrial jerk-limited S-curve point-to-point motion profiling and multi-waypoint cubic spline interpolation with continuous acceleration.
- **URDF & YAML Robot Description Loader (`Issue #6`):**
  - Build `mechakinetics/io.py` to import/export robot kinematic chains and drivetrain specifications from standard `.yaml` and `.urdf` files, with CLI support (`mechakinetics inspect-robot arm.yaml`).

## Milestone 4: Collision Checking & Thermal/Fatigue Drivetrain Analysis (`v0.3.0` — ~6 Hours)
- **2D/3D Capsule Collision Detection & RRT\* Path Planner (`Issue #7`):**
  - Add link capsule vs. spherical obstacle minimum-distance queries and a sampling-based RRT* collision-free joint-space planner.
- **Motor Winding Thermal $I^2R$ & Gearbox Fatigue Estimator:**
  - Extend `actuator_sizing.py` to model motor winding temperature rise over repetitive duty cycles and estimate gearbox bearing $L_{10}$ life.
