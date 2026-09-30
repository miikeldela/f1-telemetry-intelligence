"""FastAPI service: scores telemetry with a trained anomaly model and
persists/serves the results from MongoDB.

Run locally:
    export MODEL_RUN_ID=<run id from mlflow>
    uvicorn f1telemetry.api.main:app --reload

Or via Docker Compose (see docker-compose.yml):
    docker compose up --build

Endpoints:
    GET  /health              service + model status
    POST /score                score a telemetry parquet file, persist to Mongo
    GET  /anomalies            query stored per-window results
    GET  /anomalies/summary    per-session window/anomaly counts
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pymongo.collection import Collection

from f1telemetry.api import db as db_module
from f1telemetry.api.config import Settings, get_settings
from f1telemetry.api.inference import (
    ModelBundle,
    ModelNotConfiguredError,
    load_model_bundle,
    score_telemetry,
)
from f1telemetry.api.schemas import (
    AnomalyWindow,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    SessionSummary,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Telemetry Anomaly API",
    description="Serves an LSTM-autoencoder anomaly model over F1 telemetry, backed by MongoDB.",
    version="0.1.0",
)

# Simple process-local caches so repeated requests don't reconnect to Mongo
# or reload the (fairly large) PyTorch model on every call. Tests never hit
# these - they override get_db_collection/get_model_bundle directly.
_db_cache: dict[str, Collection] = {}
_bundle_cache: ModelBundle | None = None


def get_db_collection(settings: Settings = Depends(get_settings)) -> Collection:
    key = f"{settings.mongo_uri}::{settings.mongo_db}"
    if key not in _db_cache:
        _db_cache[key] = db_module.get_collection(settings)
    return _db_cache[key]


def get_model_bundle(settings: Settings = Depends(get_settings)) -> ModelBundle | None:
    """Returns the loaded model bundle, or None if it's not configured or
    failed to load. Deliberately never raises: /health needs to report a
    "not configured"/"degraded" status rather than fail outright, and
    /score decides for itself whether a missing bundle is fatal (it is).
    Kept as a single dependency (rather than one strict + one lenient
    variant) so overriding this one function in tests covers both
    endpoints - see tests/test_api.py.
    """
    global _bundle_cache
    if _bundle_cache is None or _bundle_cache.run_id != settings.model_run_id:
        try:
            _bundle_cache = load_model_bundle(settings)
        except ModelNotConfiguredError:
            return None
        except Exception as exc:  # noqa: BLE001 - a bad run ID shouldn't crash the process
            logger.warning("Failed to load model bundle: %s", exc)
            return None
    return _bundle_cache


def _row_to_document(row: dict) -> dict:
    """Coerce a meta-row dict into Mongo-safe types (no Timedelta/NaT/NaN)."""
    doc = dict(row)
    for key in ("start_session_time", "end_session_time"):
        if key in doc and pd.notna(doc[key]):
            doc[key] = str(doc[key])
        elif key in doc:
            doc[key] = None
    for key in ("start_date", "end_date"):
        if key in doc and pd.notna(doc[key]):
            doc[key] = pd.Timestamp(doc[key]).isoformat()
        elif key in doc:
            doc[key] = None
    if "LapNumber" in doc and pd.notna(doc["LapNumber"]):
        doc["LapNumber"] = float(doc["LapNumber"])
    if "Year" in doc and pd.notna(doc["Year"]):
        doc["Year"] = int(doc["Year"])
    doc["is_anomaly"] = bool(doc.get("is_anomaly", False))
    doc["anomaly_error"] = float(doc.get("anomaly_error", 0.0))
    doc["start_index"] = int(doc["start_index"])
    if "start_distance" in doc and pd.notna(doc["start_distance"]):
        doc["start_distance"] = float(doc["start_distance"])
    return doc


@app.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    bundle: ModelBundle | None = Depends(get_model_bundle),
) -> HealthResponse:
    if bundle is not None:
        return HealthResponse(status="ok", model_run_id=bundle.run_id, channels=bundle.channels)
    if not settings.model_run_id:
        return HealthResponse(status="ok (no model configured)", model_run_id=None, channels=None)
    return HealthResponse(
        status="degraded (model failed to load)",
        model_run_id=settings.model_run_id,
    )


@app.post("/score", response_model=ScoreResponse)
def score(
    request: ScoreRequest,
    settings: Settings = Depends(get_settings),
    bundle: ModelBundle | None = Depends(get_model_bundle),
    collection: Collection = Depends(get_db_collection),
) -> ScoreResponse:
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model configured or it failed to load - set MODEL_RUN_ID to a valid "
                "MLflow run ID."
            ),
        )

    path = Path(request.telemetry_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"telemetry_path not found: {path}")

    telemetry = pd.read_parquet(path)
    window_size = request.window_size or settings.window_size
    stride = request.stride or settings.stride
    top_fraction = request.top_fraction or settings.top_fraction

    try:
        meta, threshold = score_telemetry(bundle, telemetry, window_size, stride, top_fraction)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(meta) == 0:
        raise HTTPException(
            status_code=422,
            detail="No windows produced - check window_size/stride against this telemetry.",
        )

    documents = [_row_to_document(row) for row in meta.to_dict(orient="records")]
    written = db_module.upsert_windows(collection, documents)

    return ScoreResponse(
        n_windows=len(meta),
        n_anomalies=int(meta["is_anomaly"].sum()),
        threshold=threshold,
        channels=bundle.channels,
        written_to_db=written,
    )


@app.get("/anomalies", response_model=list[AnomalyWindow])
def list_anomalies(
    year: int | None = None,
    grand_prix: str | None = Query(default=None, alias="grandPrix"),
    session_type: str | None = Query(default=None, alias="sessionType"),
    driver: str | None = None,
    is_anomaly: bool | None = Query(default=None, alias="isAnomaly"),
    limit: int = 100,
    skip: int = 0,
    collection: Collection = Depends(get_db_collection),
) -> list[dict]:
    return db_module.query_windows(
        collection,
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        driver=driver,
        is_anomaly=is_anomaly,
        limit=limit,
        skip=skip,
    )


@app.get("/anomalies/summary", response_model=list[SessionSummary])
def anomalies_summary(collection: Collection = Depends(get_db_collection)) -> list[dict]:
    return db_module.summarize(collection)
