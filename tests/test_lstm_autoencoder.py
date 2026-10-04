import numpy as np

from f1telemetry.models.lstm_autoencoder import (
    TrainConfig,
    reconstruction_errors,
    train_autoencoder,
)


def _synthetic_windows(
    n_normal: int = 40, window_size: int = 20, n_channels: int = 3, seed: int = 0
):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, window_size)
    base = np.stack([np.sin(t + i) for i in range(n_channels)], axis=-1)
    return np.stack([base + rng.normal(0, 0.05, base.shape) for _ in range(n_normal)])


def test_train_autoencoder_runs_and_reduces_loss():
    X = _synthetic_windows()
    config = TrainConfig(hidden_size=8, epochs=10, batch_size=8)

    model, history = train_autoencoder(X, config)

    assert len(history["train_loss"]) == config.epochs
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_reconstruction_error_flags_outlier_window():
    X = _synthetic_windows(n_normal=40, window_size=20, n_channels=3)
    config = TrainConfig(hidden_size=8, epochs=15, batch_size=8)
    model, _ = train_autoencoder(X, config)

    normal_errors = reconstruction_errors(model, X)

    outlier = X[:1].copy()
    outlier[0] += 5.0  # clearly out of distribution
    outlier_error = reconstruction_errors(model, outlier)[0]

    assert outlier_error > normal_errors.mean() + 2 * normal_errors.std()
