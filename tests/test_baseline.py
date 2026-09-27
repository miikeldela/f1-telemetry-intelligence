import numpy as np
import pandas as pd

from f1telemetry.features.build_features import build_telemetry_features
from f1telemetry.models.baseline import BaselineAnomalyModel


def _telemetry_with_anomaly(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = np.linspace(0, 5000, n)
    speed = 250 + 10 * np.sin(distance / 500) + rng.normal(0, 1, n)
    speed[n // 2] += 150  # inject a clear, unrealistic speed spike
    throttle = np.clip(80 + 20 * np.sin(distance / 400), 0, 100)
    brake = (np.sin(distance / 300) < -0.7).astype(int)
    gear = np.clip((speed // 40).astype(int), 1, 8)
    rpm = 8000 + speed * 20
    return pd.DataFrame(
        {
            "Distance": distance,
            "Driver": "VER",
            "LapNumber": 1,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "nGear": gear,
            "RPM": rpm,
        }
    )


def test_baseline_model_flags_injected_anomaly():
    telemetry = _telemetry_with_anomaly()
    features = build_telemetry_features(telemetry, window=10)

    model = BaselineAnomalyModel(contamination=0.05).fit(features)
    scored = model.score(features)

    assert scored["is_anomaly"].sum() > 0
    top_idx = scored["anomaly_score"].idxmax()
    assert abs(top_idx - len(scored) // 2) < 15
