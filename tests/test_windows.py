import numpy as np
import pandas as pd
import pytest

from f1telemetry.features.windows import build_sequences


def _synthetic_telemetry(n: int = 120, driver: str = "VER", lap: int = 1) -> pd.DataFrame:
    distance = np.linspace(0, 3000, n)
    session_time = pd.to_timedelta(np.arange(n) * 0.1, unit="s")
    return pd.DataFrame(
        {
            "Distance": distance,
            "Driver": driver,
            "LapNumber": lap,
            "Speed": 250 + np.sin(distance / 300),
            "Throttle": 80.0,
            "Brake": 0,
            "nGear": 6,
            "RPM": 10000.0,
            "SessionTime": session_time,
        }
    )


def test_build_sequences_shape():
    telemetry = _synthetic_telemetry(n=120)
    X, meta = build_sequences(telemetry, window_size=50, stride=25)

    assert X.shape[1] == 50
    assert X.shape[2] == 6  # default channels (incl. derived Acceleration)
    assert len(meta) == len(X)


def test_build_sequences_multiple_laps():
    t1 = _synthetic_telemetry(n=120, driver="VER", lap=1)
    t2 = _synthetic_telemetry(n=120, driver="VER", lap=2)
    telemetry = pd.concat([t1, t2], ignore_index=True)

    X, meta = build_sequences(telemetry, window_size=50, stride=25)
    assert set(meta["LapNumber"].unique()) == {1, 2}


def test_build_sequences_skips_short_laps():
    telemetry = _synthetic_telemetry(n=10)
    X, meta = build_sequences(telemetry, window_size=50, stride=25)
    assert len(X) == 0


def test_build_sequences_missing_channel_raises():
    telemetry = _synthetic_telemetry(n=120).drop(columns=["RPM"])
    with pytest.raises(ValueError):
        build_sequences(telemetry, window_size=50)


def test_acceleration_channel_derivation_is_correct():
    # Constant speed increase of 36 km/h per second == 10 m/s^2 constant acceleration.
    n = 60
    session_time = pd.to_timedelta(np.arange(n) * 1.0, unit="s")
    telemetry = pd.DataFrame(
        {
            "Distance": np.linspace(0, 3000, n),
            "Driver": "VER",
            "LapNumber": 1,
            "Speed": 100 + 36 * np.arange(n),
            "Throttle": 80.0,
            "Brake": 0,
            "nGear": 6,
            "RPM": 10000.0,
            "SessionTime": session_time,
        }
    )

    X, meta = build_sequences(
        telemetry,
        window_size=50,
        stride=50,
        channels=["Speed", "Acceleration"],
    )

    assert X.shape[2] == 2
    acceleration = X[0][:, 1]
    # First sample in each lap has no prior row, so its diff-based value is 0.
    assert acceleration[0] == pytest.approx(0.0)
    assert acceleration[1:] == pytest.approx(10.0, abs=1e-6)


def test_acceleration_missing_session_time_raises():
    telemetry = _synthetic_telemetry(n=120).drop(columns=["SessionTime"])
    with pytest.raises(ValueError):
        build_sequences(telemetry, window_size=50, channels=["Speed", "Acceleration"])


def test_lateral_acceleration_channel_derivation_is_correct():
    # Uniform circular motion: constant speed v = R*omega implies constant
    # centripetal (lateral) acceleration = v^2 / R = R * omega^2.
    n = 200
    dt = 0.02
    t = np.arange(n) * dt
    radius = 100.0
    omega = 0.05
    theta = omega * t

    telemetry = pd.DataFrame(
        {
            "Distance": np.linspace(0, 3000, n),
            "Driver": "VER",
            "LapNumber": 1,
            "Speed": (radius * omega) * 3.6,  # m/s -> km/h, constant
            "Throttle": 80.0,
            "Brake": 0,
            "nGear": 6,
            "RPM": 10000.0,
            "SessionTime": pd.to_timedelta(t, unit="s"),
            "X": radius * np.cos(theta),
            "Y": radius * np.sin(theta),
        }
    )

    X, meta = build_sequences(
        telemetry,
        window_size=180,
        stride=180,
        channels=["Speed", "LateralAcceleration"],
    )

    expected = radius * omega**2
    lateral = X[0][:, 1]
    # First couple of samples are finite-difference transients (no prior
    # velocity/acceleration to compute from); check the steady-state region.
    assert lateral[3:].mean() == pytest.approx(expected, rel=1e-4)


def test_lateral_acceleration_missing_position_raises():
    telemetry = _synthetic_telemetry(n=120)  # no X/Y columns
    with pytest.raises(ValueError):
        build_sequences(telemetry, window_size=50, channels=["Speed", "LateralAcceleration"])
