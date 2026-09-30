"""API configuration, loaded from environment variables (or a .env file).

Kept as a small pydantic-settings model rather than scattered os.environ
calls, so every setting the service needs is declared in one place and
FastAPI's dependency-injection (see deps.py) can override it cleanly in
tests.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Mongo
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "f1telemetry"

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    # The MLflow run ID to serve. Required for real scoring; the API starts
    # without it (so /health and docs still work) but /score fails clearly
    # until it's set. Update this after picking a winner from the overnight
    # channel-ablation sweep.
    model_run_id: str | None = None

    # Windowing - should match how the served model was trained. These are
    # defaults; a request can still override window_size/stride explicitly.
    window_size: int = 20
    stride: int = 10
    top_fraction: float = 0.05


def get_settings() -> Settings:
    return Settings()
