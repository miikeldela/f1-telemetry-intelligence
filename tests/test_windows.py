import numpy as np
import pandas as pd
import pytest

from f1telemetry.features.windows import build_sequences


def _synthetic_telemetry(n: int = 120, driver: str = "VER", lap: int = 1) -> pd.DataFrame:
    distance = np.linspace(0, 3000, n)
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
        }
    )


def test_build_sequences_shape():
    telemetry = _synthetic_telemetry(n=120)
    X, meta = build_sequences(telemetry, window_size=50, stride=25)

    assert X.shape[1] == 50
    assert X.shape[2] == 5  # default channels
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
