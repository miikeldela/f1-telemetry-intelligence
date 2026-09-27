"""Feature engineering on raw FastF1 telemetry."""
from __future__ import annotations

import pandas as pd

TELEMETRY_COLUMNS = ["Speed", "Throttle", "Brake", "nGear", "RPM"]
ID_COLUMNS = ["Distance", "Driver", "LapNumber"]


def _rolling_features(group: pd.DataFrame, window: int) -> pd.DataFrame:
    group = group.sort_values("Distance").reset_index(drop=True)
    feats = pd.DataFrame(index=group.index)
    feats["Distance"] = group["Distance"]
    feats["Driver"] = group["Driver"]
    feats["LapNumber"] = group["LapNumber"]

    for col in TELEMETRY_COLUMNS:
        if col not in group:
            continue
        series = group[col].astype(float)
        feats[f"{col}_roll_mean"] = series.rolling(window, min_periods=1).mean()
        feats[f"{col}_roll_std"] = series.rolling(window, min_periods=1).std().fillna(0.0)
        feats[f"{col}_delta"] = series.diff().fillna(0.0)

    # Braking / gear-shift transitions: useful anomaly signals on their own
    if "Brake" in group:
        feats["brake_transitions"] = (
            group["Brake"].astype(bool).astype(int).diff().abs().fillna(0)
        )
    if "nGear" in group:
        feats["gear_shifts"] = group["nGear"].diff().fillna(0).ne(0).astype(int)

    return feats


def build_telemetry_features(telemetry: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Compute rolling-window features per driver/lap from raw telemetry.

    Expects columns: Distance, Driver, LapNumber, and any of TELEMETRY_COLUMNS.
    Returns one row per telemetry sample with derived features, ready for
    an anomaly-detection model (see f1telemetry.models.baseline).
    """
    missing = set(ID_COLUMNS) - set(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing required columns: {missing}")

    groups = [
        _rolling_features(g, window)
        for _, g in telemetry.groupby(["Driver", "LapNumber"], sort=False)
    ]
    features = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    return features.dropna().reset_index(drop=True)
