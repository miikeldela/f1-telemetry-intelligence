"""MongoDB access layer.

Kept as a thin wrapper around a pymongo collection so main.py's endpoints
never touch pymongo directly - that makes it trivial to swap in mongomock
for tests (see tests/test_api.py) via FastAPI's dependency_overrides.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pymongo.collection import Collection

from f1telemetry.api.config import Settings

ANOMALY_COLLECTION = "anomaly_windows"


def get_collection(settings: Settings) -> Collection:
    from pymongo import MongoClient

    client = MongoClient(settings.mongo_uri)
    db = client[settings.mongo_db]
    collection = db[ANOMALY_COLLECTION]
    _ensure_indexes(collection)
    return collection


def _ensure_indexes(collection: Collection) -> None:
    # One document per window, keyed by session + driver + lap + position in
    # the lap - re-scoring the same telemetry file is then an upsert, not a
    # pile of duplicates.
    collection.create_index(
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
    collection.create_index("is_anomaly")


def upsert_windows(collection: Collection, documents: Iterable[dict[str, Any]]) -> int:
    """Upsert each window document by its identity key. Returns count written."""
    n = 0
    for doc in documents:
        key = {
            k: doc.get(k)
            for k in ("Year", "GrandPrix", "SessionType", "Driver", "LapNumber", "start_index")
        }
        collection.replace_one(key, doc, upsert=True)
        n += 1
    return n


def query_windows(
    collection: Collection,
    *,
    year: int | None = None,
    grand_prix: str | None = None,
    session_type: str | None = None,
    driver: str | None = None,
    is_anomaly: bool | None = None,
    limit: int = 100,
    skip: int = 0,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if year is not None:
        query["Year"] = year
    if grand_prix is not None:
        query["GrandPrix"] = grand_prix
    if session_type is not None:
        query["SessionType"] = session_type
    if driver is not None:
        query["Driver"] = driver
    if is_anomaly is not None:
        query["is_anomaly"] = is_anomaly

    cursor = collection.find(query, {"_id": 0}).skip(skip).limit(limit)
    return list(cursor)


def summarize(collection: Collection) -> list[dict[str, Any]]:
    """Per-session summary: window count and anomaly count/rate."""
    pipeline = [
        {
            "$group": {
                "_id": {"Year": "$Year", "GrandPrix": "$GrandPrix", "SessionType": "$SessionType"},
                "n_windows": {"$sum": 1},
                "n_anomalies": {"$sum": {"$cond": ["$is_anomaly", 1, 0]}},
            }
        },
        {"$sort": {"_id.Year": 1, "_id.GrandPrix": 1}},
    ]
    results = []
    for row in collection.aggregate(pipeline):
        key = row["_id"]
        results.append(
            {
                "Year": key.get("Year"),
                "GrandPrix": key.get("GrandPrix"),
                "SessionType": key.get("SessionType"),
                "n_windows": row["n_windows"],
                "n_anomalies": row["n_anomalies"],
                "anomaly_rate": row["n_anomalies"] / row["n_windows"] if row["n_windows"] else 0.0,
            }
        )
    return results
