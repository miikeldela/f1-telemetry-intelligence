"""Turn per-lap telemetry into fixed-length sequences for sequence models."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM"]


def build_sequences(
    telemetry: pd.DataFrame,
    window_size: int = 50,
    stride: int = 25,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Slice per-driver/lap telemetry into fixed-length overlapping windows.

    Args:
        telemetry: long-format telemetry with Distance, Driver, LapNumber and
            the requested channel columns (e.g. from get_all_laps_telemetry).
        window_size: number of samples per sequence.
        stride: step between consecutive window start points.
        channels: telemetry columns to include as sequence features.

    Returns:
        X: array of shape (n_windows, window_size, n_channels).
        meta: one row per window (Driver, LapNumber, start_index,
            start_distance), aligned with X's first axis.
    """
    channels = channels or DEFAULT_CHANNELS
    missing = set(channels) - set(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing channels: {missing}")

    # Group by session too when the telemetry carries session tags (added by
    # ingest.py), so laps from different races/sessions never get merged into
    # the same (Driver, LapNumber) group.
    session_tags = [c for c in ("Year", "GrandPrix", "SessionType") if c in telemetry.columns]
    group_keys = session_tags + ["Driver", "LapNumber"]

    windows: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for key, group in telemetry.groupby(group_keys, sort=False):
        key_dict = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        group = group.sort_values("Distance").reset_index(drop=True)
        values = group[channels].astype(float).to_numpy()

        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, stride):
            windows.append(values[start : start + window_size])
            meta_rows.append(
                {
                    **key_dict,
                    "start_index": start,
                    "start_distance": group["Distance"].iloc[start],
                }
            )

    if not windows:
        return np.empty((0, window_size, len(channels))), pd.DataFrame(meta_rows)

    X = np.stack(windows)
    meta = pd.DataFrame(meta_rows)
    return X, meta
