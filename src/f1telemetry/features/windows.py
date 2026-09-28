"""Turn per-lap telemetry into fixed-length sequences for sequence models."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Acceleration and LateralAcceleration aren't raw FastF1 channels - they're
# derived below (F1 doesn't expose raw accelerometer/g-force data publicly).
# Brake, note, is boolean (on/off) in FastF1, not a pressure value, so it
# can't tell a light dab from a full lockup on its own - the derived
# channels are partly there to compensate for that gap.
DEFAULT_CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM", "Acceleration"]

# Channels available to add via --channels but not in DEFAULT_CHANNELS.
# LateralAcceleration needs X/Y position data, which is only present in
# telemetry pulled with the newer get_telemetry()-based ingest (not the
# older get_car_data()-only pulls).
DERIVED_CHANNELS = {"Acceleration", "LateralAcceleration"}


def _add_derived_channels(group: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Compute any requested derived channels not already present in group."""
    if "Acceleration" in channels and "Acceleration" not in group.columns:
        if "SessionTime" not in group.columns:
            raise ValueError(
                "Acceleration channel requested but telemetry has no SessionTime "
                "column to derive it from."
            )
        dt_seconds = group["SessionTime"].diff().dt.total_seconds()
        dv_ms = group["Speed"].diff() * (1000.0 / 3600.0)  # km/h -> m/s
        acceleration = (dv_ms / dt_seconds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        group = group.assign(Acceleration=acceleration)

    if "LateralAcceleration" in channels and "LateralAcceleration" not in group.columns:
        missing = {"X", "Y", "SessionTime"} - set(group.columns)
        if missing:
            raise ValueError(
                f"LateralAcceleration channel requested but telemetry is missing {missing} "
                "(needs position data - re-ingest with the get_telemetry()-based ingest.py)."
            )
        dt = group["SessionTime"].diff().dt.total_seconds()
        vx = group["X"].diff() / dt
        vy = group["Y"].diff() / dt
        ax = vx.diff() / dt
        ay = vy.diff() / dt
        speed = np.sqrt(vx**2 + vy**2)
        # Centripetal/lateral acceleration = component of (ax, ay) perpendicular
        # to the velocity vector (vx, vy), i.e. |v x a| / |v|.
        lateral = ((vx * ay - vy * ax) / speed).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        group = group.assign(LateralAcceleration=lateral)

    return group


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
        channels: telemetry columns to include as sequence features. Any
            entry in DERIVED_CHANNELS ("Acceleration", "LateralAcceleration")
            is computed automatically rather than required to already exist.

    Returns:
        X: array of shape (n_windows, window_size, n_channels).
        meta: one row per window (Driver, LapNumber, start_index,
            start_distance, and start_date/end_date if the telemetry has a
            Date column), aligned with X's first axis.
    """
    channels = channels or DEFAULT_CHANNELS
    raw_channels_needed = set(channels) - DERIVED_CHANNELS
    missing = raw_channels_needed - set(telemetry.columns)
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
        group = _add_derived_channels(group, channels)
        values = group[channels].astype(float).to_numpy()

        if len(values) < window_size:
            continue

        has_date = "Date" in group.columns
        has_session_time = "SessionTime" in group.columns

        for start in range(0, len(values) - window_size + 1, stride):
            windows.append(values[start : start + window_size])
            end = start + window_size - 1
            row = {
                **key_dict,
                "start_index": start,
                "start_distance": group["Distance"].iloc[start],
            }
            if has_date:
                # Absolute timestamp, mainly for display/debugging.
                row["start_date"] = group["Date"].iloc[start]
                row["end_date"] = group["Date"].iloc[end]
            if has_session_time:
                # Session-relative time (a Timedelta from session start) -
                # this is the basis FastF1's race_control_messages['Time']
                # uses too, so it's what to match scored windows against
                # real session events.
                row["start_session_time"] = group["SessionTime"].iloc[start]
                row["end_session_time"] = group["SessionTime"].iloc[end]
            meta_rows.append(row)

    if not windows:
        return np.empty((0, window_size, len(channels))), pd.DataFrame(meta_rows)

    X = np.stack(windows)
    meta = pd.DataFrame(meta_rows)
    return X, meta
