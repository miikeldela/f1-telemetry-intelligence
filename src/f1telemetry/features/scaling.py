"""Scaling strategies for window sequences before they go into the model.

Two options:
  - "global": one StandardScaler fit across every window from every
    session pooled together (the original approach).
  - "per_session": each session is standardized against its OWN mean/std,
    not the pooled one.

Why per_session exists: different circuits have wildly different normal
speed/braking/gear profiles. A single global scaler lets "which track is
this" dominate a window's reconstruction error far more than "is this lap
unusual for this track" - the kind of cross-session non-stationarity an
ARIMA-style model would handle via differencing/seasonal decomposition,
which a vanilla LSTM autoencoder does not do on its own. Per-session
scaling is a cheap, partial stand-in for that.

It's recomputed fresh from whatever data is passed in (training or
scoring) rather than persisted as a fitted object, so it also works
unmodified on sessions the model never saw during training - there is no
scaler to go stale or to mismatch at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SESSION_COLUMNS = ("Year", "GrandPrix", "SessionType")


def _scale_group(x_group: np.ndarray) -> np.ndarray:
    n_windows, window_size, n_channels = x_group.shape
    flat = x_group.reshape(-1, n_channels)
    scaler = StandardScaler().fit(flat)
    return scaler.transform(flat).reshape(n_windows, window_size, n_channels)


def scale_global(X: np.ndarray, scaler: StandardScaler | None = None) -> np.ndarray:
    """One scaler fit across all of X (or reused, if provided)."""
    n_windows, window_size, n_channels = X.shape
    flat = X.reshape(-1, n_channels)
    if scaler is None:
        scaler = StandardScaler().fit(flat)
    return scaler.transform(flat).reshape(n_windows, window_size, n_channels)


def scale_per_session(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Standardize each window relative to its own session's mean/std.

    meta must be the DataFrame returned alongside X by build_sequences,
    unmodified (same row order/index, one row per window in X's first
    axis) - this relies on meta's positional index matching X's rows.
    Falls back to global scaling if meta carries no session tags at all
    (e.g. synthetic telemetry in tests with a single implicit session).
    """
    session_cols = [c for c in SESSION_COLUMNS if c in meta.columns]
    if not session_cols:
        return scale_global(X)

    scaled = np.empty_like(X, dtype="float64")
    for _, group in meta.groupby(session_cols, sort=False):
        idx = group.index.to_numpy()
        scaled[idx] = _scale_group(X[idx])
    return scaled
