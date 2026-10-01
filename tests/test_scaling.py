"""Tests for per-session vs. global scaling (see features/scaling.py).

The core thing to prove: with a single pooled (global) scaler, a
low-magnitude session's variance collapses toward zero relative to a
high-magnitude session, while per-session scaling normalizes each
session independently regardless of the other sessions present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from f1telemetry.features.scaling import scale_global, scale_per_session

WINDOW_SIZE = 5
N_CHANNELS = 2


def _make_session(
    n_windows: int, mean: float, scale: float, rng: np.random.Generator
) -> np.ndarray:
    return rng.normal(loc=mean, scale=scale, size=(n_windows, WINDOW_SIZE, N_CHANNELS))


def _meta_for(session_sizes: dict[str, int]) -> pd.DataFrame:
    rows = []
    for session_name, n_windows in session_sizes.items():
        rows.extend({"GrandPrix": session_name} for _ in range(n_windows))
    return pd.DataFrame(rows)


def test_per_session_scaling_normalizes_each_session_independently():
    rng = np.random.default_rng(0)

    # Two "sessions": one with a much larger natural scale than the other,
    # like a high-speed circuit vs. a tight street circuit.
    big_session = _make_session(50, mean=200.0, scale=50.0, rng=rng)
    small_session = _make_session(50, mean=1.0, scale=0.2, rng=rng)
    X = np.concatenate([big_session, small_session], axis=0)
    meta = _meta_for({"BigTrack": 50, "SmallTrack": 50})

    X_per_session = scale_per_session(X, meta)

    big_scaled = X_per_session[:50]
    small_scaled = X_per_session[50:]
    assert np.isclose(big_scaled.mean(), 0.0, atol=0.2)
    assert np.isclose(big_scaled.std(), 1.0, atol=0.2)
    assert np.isclose(small_scaled.mean(), 0.0, atol=0.2)
    assert np.isclose(small_scaled.std(), 1.0, atol=0.2)


def test_global_scaling_lets_the_dominant_session_swamp_the_other():
    rng = np.random.default_rng(1)

    big_session = _make_session(50, mean=200.0, scale=50.0, rng=rng)
    small_session = _make_session(50, mean=1.0, scale=0.2, rng=rng)
    X = np.concatenate([big_session, small_session], axis=0)

    X_global = scale_global(X)
    small_scaled = X_global[50:]

    # The small session's own variance, once squashed through a scaler fit
    # on the (much larger) pooled range, should be far below 1.0 - this is
    # exactly the collapse that makes real anomalies within that session
    # nearly invisible to reconstruction error.
    assert small_scaled.std() < 0.1


def test_scale_per_session_falls_back_to_global_without_session_columns():
    rng = np.random.default_rng(2)
    X = _make_session(20, mean=10.0, scale=2.0, rng=rng)
    meta = pd.DataFrame({"Driver": ["VER"] * 20})  # no session-tag columns

    result = scale_per_session(X, meta)
    expected = scale_global(X)
    np.testing.assert_allclose(result, expected)


def test_scale_per_session_preserves_shape():
    rng = np.random.default_rng(3)
    X = _make_session(30, mean=5.0, scale=1.0, rng=rng)
    meta = _meta_for({"Monaco": 10, "Spa": 20})

    result = scale_per_session(X, meta)
    assert result.shape == X.shape
