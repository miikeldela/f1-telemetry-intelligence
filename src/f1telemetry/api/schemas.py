"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_run_id: str | None
    channels: list[str] | None = None


class ScoreRequest(BaseModel):
    """Score a telemetry parquet file already on the server (e.g. produced by
    ingest.py / combine_telemetry.py) and persist the resulting per-window
    anomaly scores to MongoDB.
    """

    telemetry_path: str
    window_size: int | None = None
    stride: int | None = None
    top_fraction: float | None = None


class ScoreResponse(BaseModel):
    n_windows: int
    n_anomalies: int
    threshold: float
    channels: list[str]
    written_to_db: int


class AnomalyWindow(BaseModel):
    Year: int | None = None
    GrandPrix: str | None = None
    SessionType: str | None = None
    Driver: str | None = None
    LapNumber: float | None = None
    start_index: int
    start_distance: float | None = None
    start_session_time: str | None = None
    anomaly_error: float
    is_anomaly: bool


class SessionSummary(BaseModel):
    Year: int | None = None
    GrandPrix: str | None = None
    SessionType: str | None = None
    n_windows: int
    n_anomalies: int
    anomaly_rate: float
