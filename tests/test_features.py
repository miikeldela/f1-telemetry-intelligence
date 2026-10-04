import numpy as np
import pandas as pd
import pytest

from f1telemetry.features.build_features import build_telemetry_features


def _synthetic_telemetry(
    n: int = 100, driver: str = "VER", lap: int = 1, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = np.linspace(0, 5000, n)
    speed = 250 + 30 * np.sin(distance / 500) + rng.normal(0, 2, n)
    throttle = np.clip(80 + 20 * np.sin(distance / 400), 0, 100)
    brake = (np.sin(distance / 300) < -0.7).astype(int)
    gear = np.clip((speed // 40).astype(int), 1, 8)
    rpm = 8000 + speed * 20
    return pd.DataFrame(
        {
            "Distance": distance,
            "Driver": driver,
            "LapNumber": lap,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "nGear": gear,
            "RPM": rpm,
        }
    )


def test_build_telemetry_features_shape_and_columns():
    telemetry = _synthetic_telemetry()
    features = build_telemetry_features(telemetry, window=10)

    assert len(features) > 0
    expected_cols = [
        "Speed_roll_mean",
        "Speed_roll_std",
        "Speed_delta",
        "brake_transitions",
        "gear_shifts",
    ]
    for col in expected_cols:
        assert col in features.columns
    assert not features.isna().any().any()


def test_build_telemetry_features_multiple_laps_drivers():
    t1 = _synthetic_telemetry(driver="VER", lap=1, seed=1)
    t2 = _synthetic_telemetry(driver="HAM", lap=1, seed=2)
    telemetry = pd.concat([t1, t2], ignore_index=True)

    features = build_telemetry_features(telemetry, window=10)
    assert set(features["Driver"].unique()) == {"VER", "HAM"}


def test_build_telemetry_features_missing_columns_raises():
    with pytest.raises(ValueError):
        build_telemetry_features(pd.DataFrame({"Speed": [1, 2, 3]}))
