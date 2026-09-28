"""Train the LSTM autoencoder on full-session telemetry, tracked with MLflow.

Usage:
    python -m f1telemetry.data.ingest 2024 Bahrain --session R --mode full --out data/raw/telemetry_full.parquet
    python -m f1telemetry.models.train_lstm data/raw/telemetry_full.parquet

Then inspect runs with:
    mlflow ui
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.pytorch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from f1telemetry.features.windows import DEFAULT_CHANNELS, build_sequences
from f1telemetry.models.lstm_autoencoder import (
    TrainConfig,
    reconstruction_errors,
    train_autoencoder,
)


def _scale_sequences(X, scaler: StandardScaler | None = None):
    n_windows, window_size, n_channels = X.shape
    flat = X.reshape(-1, n_channels)
    if scaler is None:
        scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(n_windows, window_size, n_channels)
    return scaled, scaler


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM autoencoder on telemetry windows.")
    parser.add_argument(
        "telemetry_path", help="Parquet file with full-session telemetry (see ingest.py --mode full)"
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--experiment", default="f1-telemetry-lstm-autoencoder")
    parser.add_argument(
        "--register-as", default=None, help="If set, register the trained model under this name"
    )
    args = parser.parse_args()

    telemetry = pd.read_parquet(args.telemetry_path)
    X, meta = build_sequences(telemetry, window_size=args.window_size, stride=args.stride)
    if len(X) == 0:
        raise SystemExit(
            "No windows produced - check --window-size against your laps' sample counts."
        )

    X_scaled, _scaler = _scale_sequences(X)
    config = TrainConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "window_size": args.window_size,
                "stride": args.stride,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "n_windows": len(X_scaled),
                "n_laps": meta[["Driver", "LapNumber"]].drop_duplicates().shape[0],
                "channels": ",".join(DEFAULT_CHANNELS),
            }
        )

        model, history = train_autoencoder(X_scaled, config)

        for epoch, (train_loss, val_loss) in enumerate(
            zip(history["train_loss"], history["val_loss"])
        ):
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        errors = reconstruction_errors(model, X_scaled)
        mlflow.log_metrics(
            {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "mean_reconstruction_error": float(errors.mean()),
                "p99_reconstruction_error": float(pd.Series(errors).quantile(0.99)),
            }
        )

        # Use the plain pickle format rather than the newer 'pt2' (torch.export)
        # format: pt2 requires a strict TensorSpec signature and is fussier
        # about tracing an LSTM than this project needs right now.
        input_example = X_scaled[:1].astype("float32")
        log_kwargs = dict(
            name="model",
            input_example=input_example,
            serialization_format="pickle",
        )
        if args.register_as:
            log_kwargs["registered_model_name"] = args.register_as
        mlflow.pytorch.log_model(model, **log_kwargs)

        run_id = mlflow.active_run().info.run_id
        print(f"Trained on {len(X_scaled)} windows from {len(telemetry)} telemetry rows.")
        print(f"Final val loss: {history['val_loss'][-1]:.5f}")
        print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    _main()
