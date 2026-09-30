"""Unit tests for ingest.py's telemetry-fetching helpers, using stubbed lap
objects instead of real FastF1 network calls - these need to run offline
and fast in CI.
"""
from __future__ import annotations

import pandas as pd
import pytest

from f1telemetry.data.ingest import _full_telemetry_with_distance


class _FakeLapWithPosition:
    """Stands in for a FastF1 Lap whose get_telemetry() succeeds and
    already includes position (X/Y/Z/DRS) columns and Distance."""

    def get_telemetry(self):
        return pd.DataFrame(
            {
                "Speed": [200.0, 210.0],
                "Distance": [0.0, 10.0],
                "X": [0.0, 1.0],
                "Y": [0.0, 1.0],
                "DRS": [0, 0],
            }
        )


class _FakeLapWithoutDistance:
    """get_telemetry() succeeds but the returned frame has no Distance
    column yet - add_distance() should be called to add one."""

    def get_telemetry(self):
        return _DistanceAddable({"Speed": [200.0, 210.0], "X": [0.0, 1.0], "Y": [0.0, 1.0]})


class _DistanceAddable(pd.DataFrame):
    def add_distance(self):
        out = self.copy()
        out["Distance"] = [0.0, 10.0]
        return pd.DataFrame(out)


class _FakeLapPositionUnavailable:
    """get_telemetry() fails (as happens for some partial/incomplete laps);
    should fall back to get_car_data()."""

    def get_telemetry(self):
        raise RuntimeError("no position data for this lap")

    def get_car_data(self):
        return _DistanceAddable({"Speed": [200.0, 210.0]})


def test_full_telemetry_uses_get_telemetry_when_available():
    telemetry = _full_telemetry_with_distance(_FakeLapWithPosition())
    assert "X" in telemetry.columns
    assert "Distance" in telemetry.columns


def test_full_telemetry_adds_distance_when_missing():
    telemetry = _full_telemetry_with_distance(_FakeLapWithoutDistance())
    assert "Distance" in telemetry.columns
    assert list(telemetry["Distance"]) == [0.0, 10.0]


def test_full_telemetry_falls_back_to_car_data_on_failure():
    telemetry = _full_telemetry_with_distance(_FakeLapPositionUnavailable())
    assert "X" not in telemetry.columns  # car-only data, no position
    assert "Distance" in telemetry.columns
    assert list(telemetry["Speed"]) == [200.0, 210.0]
