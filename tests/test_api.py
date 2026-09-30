"""API tests using mongomock and a tiny untrained model - no live Mongo or
MLflow run needed, so this runs in CI the same as any other test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from f1telemetry.api.inference import ModelBundle
from f1telemetry.api.main import app, get_db_collection, get_model_bundle
from f1telemetry.models.lstm_autoencoder import LSTMAutoencoder

CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM"]


def _synthetic_telemetry(n_laps: int = 2, n: int = 80) -> pd.DataFrame:
    frames = []
    for lap in range(1, n_laps + 1):
        distance = np.linspace(0, 3000, n)
        frames.append(
            pd.DataFrame(
                {
                    "Distance": distance,
                    "Driver": "VER",
                    "LapNumber": lap,
                    "Speed": 250 + np.sin(distance / 300),
                    "Throttle": 80.0,
                    "Brake": 0,
                    "nGear": 6,
                    "RPM": 10000.0,
                    "SessionTime": pd.to_timedelta(np.arange(n) * 0.1 + lap * 100, unit="s"),
                    "Date": pd.Timestamp("2026-01-01")
                    + pd.to_timedelta(np.arange(n) * 0.1 + lap * 100, unit="s"),
                    "Year": 2026,
                    "GrandPrix": "Testville",
                    "SessionType": "R",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def fake_bundle() -> ModelBundle:
    torch_model = LSTMAutoencoder(n_channels=len(CHANNELS), hidden_size=4, num_layers=1)
    return ModelBundle(run_id="fake-run-id", model=torch_model, scaler=None, channels=CHANNELS)


@pytest.fixture()
def mongo_collection():
    import mongomock

    client = mongomock.MongoClient()
    collection = client["f1telemetry-test"]["anomaly_windows"]
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
    return collection


@pytest.fixture()
def client(fake_bundle, mongo_collection):
    app.dependency_overrides[get_model_bundle] = lambda: fake_bundle
    app.dependency_overrides[get_db_collection] = lambda: mongo_collection
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_with_model_configured(fake_bundle):
    app.dependency_overrides[get_model_bundle] = lambda: fake_bundle
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_run_id"] == "fake-run-id"
    assert body["channels"] == CHANNELS


def test_score_and_query_roundtrip(client, tmp_path):
    telemetry = _synthetic_telemetry()
    telemetry_path = tmp_path / "telemetry.parquet"
    telemetry.to_parquet(telemetry_path)

    score_response = client.post(
        "/score",
        json={"telemetry_path": str(telemetry_path), "window_size": 20, "stride": 10},
    )
    assert score_response.status_code == 200
    body = score_response.json()
    assert body["n_windows"] > 0
    assert body["n_anomalies"] > 0
    assert body["channels"] == CHANNELS
    assert body["written_to_db"] == body["n_windows"]

    list_response = client.get("/anomalies", params={"driver": "VER"})
    assert list_response.status_code == 200
    windows = list_response.json()
    assert len(windows) == body["n_windows"]
    assert all(w["Driver"] == "VER" for w in windows)

    anomalies_only = client.get("/anomalies", params={"isAnomaly": True})
    assert anomalies_only.status_code == 200
    assert all(w["is_anomaly"] for w in anomalies_only.json())

    summary_response = client.get("/anomalies/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert len(summary) == 1
    assert summary[0]["GrandPrix"] == "Testville"
    assert summary[0]["n_windows"] == body["n_windows"]


def test_score_missing_file_returns_404(client):
    response = client.post("/score", json={"telemetry_path": "data/raw/does_not_exist.parquet"})
    assert response.status_code == 404


def test_score_without_model_configured_returns_503(mongo_collection):
    app.dependency_overrides[get_db_collection] = lambda: mongo_collection
    with TestClient(app) as test_client:
        response = test_client.post("/score", json={"telemetry_path": "data/raw/anything.parquet"})
    app.dependency_overrides.clear()

    assert response.status_code == 503
