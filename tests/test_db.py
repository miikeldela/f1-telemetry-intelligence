"""Unit tests for the MongoDB access layer, against mongomock (no live
MongoDB required).
"""
from __future__ import annotations

import mongomock
import pytest

from f1telemetry.api.db import query_windows, summarize, upsert_windows


@pytest.fixture()
def collection():
    client = mongomock.MongoClient()
    coll = client["f1telemetry-test"]["anomaly_windows"]
    coll.create_index(
        [
            ("Year", 1),
            ("GrandPrix", 1),
            ("SessionType", 1),
            ("Driver", 1),
            ("LapNumber", 1),
            ("start_index", 1),
        ],
        unique=True,
        name="window_identity",
    )
    return coll


def _doc(**overrides):
    base = {
        "Year": 2026,
        "GrandPrix": "Testville",
        "SessionType": "R",
        "Driver": "VER",
        "LapNumber": 1.0,
        "start_index": 0,
        "anomaly_error": 0.01,
        "is_anomaly": False,
    }
    base.update(overrides)
    return base


def test_upsert_then_query_roundtrip(collection):
    docs = [_doc(start_index=0, is_anomaly=False), _doc(start_index=10, is_anomaly=True)]
    written = upsert_windows(collection, docs)
    assert written == 2

    all_docs = query_windows(collection)
    assert len(all_docs) == 2

    anomalies = query_windows(collection, is_anomaly=True)
    assert len(anomalies) == 1
    assert anomalies[0]["start_index"] == 10


def test_upsert_is_idempotent_on_identity_key(collection):
    doc = _doc(start_index=0, anomaly_error=0.01)
    upsert_windows(collection, [doc])
    # Re-score the same window with a different error - should replace, not duplicate.
    upsert_windows(collection, [_doc(start_index=0, anomaly_error=0.99, is_anomaly=True)])

    all_docs = query_windows(collection)
    assert len(all_docs) == 1
    assert all_docs[0]["anomaly_error"] == 0.99
    assert all_docs[0]["is_anomaly"] is True


def test_query_filters_by_driver_and_session(collection):
    upsert_windows(
        collection,
        [
            _doc(Driver="VER", start_index=0),
            _doc(Driver="HAM", start_index=1),
            _doc(Driver="VER", GrandPrix="Otherville", start_index=2),
        ],
    )

    ver_testville = query_windows(collection, driver="VER", grand_prix="Testville")
    assert len(ver_testville) == 1
    assert ver_testville[0]["start_index"] == 0


def test_summarize_counts_per_session(collection):
    upsert_windows(
        collection,
        [
            _doc(start_index=0, is_anomaly=False),
            _doc(start_index=1, is_anomaly=True),
            _doc(start_index=2, is_anomaly=True),
        ],
    )

    summary = summarize(collection)
    assert len(summary) == 1
    row = summary[0]
    assert row["GrandPrix"] == "Testville"
    assert row["n_windows"] == 3
    assert row["n_anomalies"] == 2
    assert row["anomaly_rate"] == pytest.approx(2 / 3)
