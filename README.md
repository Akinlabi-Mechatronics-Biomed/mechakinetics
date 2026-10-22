# MechaKinetics 🦾⚙️

**MechaKinetics** is an open-source Python engineering toolkit for robotic manipulator kinematics, omnidirectional mobile robot locomotion, smooth trajectory generation, and mechatronic actuator/drivetrain sizing.
![MechaKinetics Visualizer](Figure_1.png) [MechaKinetics Visualizer](Figure_2.png)
## Who It's For
Built for mechatronics engineers, mechanical designers, and robotics developers who need a lightweight, well-tested Python library and CLI for kinematic verification, motion profiling, and motor/gearbox selection before CAD freeze or firmware deployment.

## Current Features
- **Spatial Transformations (`mechakinetics.transforms`):** $SO(3)$ rotation matrices, $SE(3)$ homogeneous transforms, and standard Denavit-Hartenberg (DH) link matrices.
- **Serial Manipulator Kinematics (`mechakinetics.kinematics`):** Forward kinematics ($T_0^N$), intermediate link frame extraction, and $6 \times N$ geometric space Jacobians for arbitrary revolute/prismatic chains.

## Installation & How to Run

```bash
git clone https://github.com/Akinlabi-Mechatronics-Biomed/mechakinetics.git
cd mechakinetics
pip install -e ".[dev]"
pytest
```

## Roadmap (Planned Work)
- [x] Numerical Inverse Kinematics solver (Damped Least Squares / Levenberg-Marquardt with joint limit clamping)
- [x] Holonomic & Non-Holonomic Mobile Robot Kinematics (Mecanum 4WD, Differential Drive, SE(2) odometry)
- [ ] Point-to-point Trajectory Planners (Trapezoidal velocity & Quintic polynomial minimum-jerk profiles)
- [ ] Mechatronic Drivetrain & Actuator Sizing module (reflected inertia matching, RMS/peak torque, gear ratio optimization)
- [ ] Interactive CLI and Streamlit visualization dashboard

## License
MIT License — see [LICENSE](LICENSE) for details.