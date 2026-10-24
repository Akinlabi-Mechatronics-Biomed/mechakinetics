from mechakinetics.cli import main


def test_cli_size_motor_command(capsys):
    exit_code = main([
        "size-motor",
        "--move-rad", "3.14",
        "--duration", "1.0",
        "--gear-ratio", "10.0",
        "--load-inertia", "0.02",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "MechaKinetics Actuator Sizing Report" in captured.out
    assert "Peak Motor Torque" in captured.out


def test_cli_mecanum_command(capsys):
    exit_code = main([
        "mecanum",
        "--vx", "0.5",
        "--vy", "-0.2",
        "--omega", "0.3",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "4WD Mecanum Wheel Angular Speeds" in captured.out