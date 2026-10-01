"""Loads a trained model bundle from MLflow and scores telemetry with it.

Reuses the same windowing/scaling/reconstruction-error logic as
train_lstm.py and validate_anomalies.py, rather than re-implementing it, so
the API can never silently drift from how the model was trained and
validated offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd

from f1telemetry.api.config import Settings
from f1telemetry.features.scaling import scale_per_session
from f1telemetry.features.windows import DEFAULT_CHANNELS, build_sequences
from f1telemetry.models.lstm_autoencoder import reconstruction_errors


@dataclass
class ModelBundle:
    run_id: str
    model: object
    scaler: object
    channels: list[str]
    scaling_strategy: str = "global"


class ModelNotConfiguredError(RuntimeError):
    """Raised when the API is asked to score without MODEL_RUN_ID set."""


def load_model_bundle(settings: Settings) -> ModelBundle:
    if not settings.model_run_id:
        raise ModelNotConfiguredError(
            "No MODEL_RUN_ID configured - set it to an MLflow run ID before calling /score "
            "(see the channel-ablation sweep results for which run to pick)."
        )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run_id = settings.model_run_id

    run_info = mlflow.get_run(run_id)
    channels_param = run_info.data.params.get("channels")
    channels = channels_param.split(",") if channels_param else DEFAULT_CHANNELS
    # Older runs predate the per-session fix and never logged this param -
    # they were always trained with a single pooled ("global") scaler.
    scaling_strategy = run_info.data.params.get("scaling_strategy", "global")

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    try:
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
    except Exception:  # noqa: BLE001 - older runs may not have a persisted scaler
        scaler = None

    return ModelBundle(
        run_id=run_id,
        model=model,
        scaler=scaler,
        channels=channels,
        scaling_strategy=scaling_strategy,
    )


def _scale(X: np.ndarray, scaler=None) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    n_windows, window_size, n_channels = X.shape
    flat = X.reshape(-1, n_channels)
    if scaler is None:
        scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(n_windows, window_size, n_channels)
    return scaled


def score_telemetry(
    bundle: ModelBundle,
    telemetry: pd.DataFrame,
    window_size: int,
    stride: int,
    top_fraction: float,
) -> tuple[pd.DataFrame, float]:
    """Score telemetry with the bundle's model.

    Returns (meta, threshold): meta is one row per window with an
    'anomaly_error' and boolean 'is_anomaly' column added; threshold is the
    reconstruction-error cutoff used (top_fraction by reconstruction error).
    """
    X, meta = build_sequences(
        telemetry, window_size=window_size, stride=stride, channels=bundle.channels
    )
    if len(X) == 0:
        return meta, float("nan")

    if bundle.scaling_strategy == "per_session":
        X_scaled = scale_per_session(X, meta)
    else:
        X_scaled = _scale(X, bundle.scaler)
    meta = meta.copy()
    meta["anomaly_error"] = reconstruction_errors(bundle.model, X_scaled)

    threshold = float(meta["anomaly_error"].quantile(1 - top_fraction))
    meta["is_anomaly"] = meta["anomaly_error"] >= threshold
    return meta, threshold
