"""LSTM autoencoder for telemetry sequence anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder.

    Encodes a (window_size, n_channels) sequence into a fixed-size hidden
    state, then reconstructs the sequence from it. Trained to minimise
    reconstruction error on (mostly normal) telemetry windows; a high
    reconstruction error at inference time flags an anomalous window.
    """

    def __init__(self, n_channels: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_channels)
        seq_len = x.size(1)
        _, (hidden, _) = self.encoder(x)
        # Repeat the final hidden state across time as the decoder's input.
        latent = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden)
        decoded, _ = self.decoder(latent)
        return self.output_layer(decoded)


@dataclass
class TrainConfig:
    hidden_size: int = 32
    num_layers: int = 1
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 42


def train_autoencoder(
    X: np.ndarray, config: TrainConfig | None = None
) -> tuple[LSTMAutoencoder, dict[str, list[float]]]:
    """Train an LSTM autoencoder on windows of shape (n_windows, window_size, n_channels).

    Returns the trained model and a history dict with 'train_loss' / 'val_loss'
    per epoch, ready to be logged to MLflow.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)

    n_samples, _, n_channels = X.shape
    n_val = max(1, int(n_samples * config.val_fraction))
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(n_samples)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_tensor[train_idx]), batch_size=config.batch_size, shuffle=True
    )
    val_tensor = X_tensor[val_idx]

    model = LSTMAutoencoder(
        n_channels=n_channels, hidden_size=config.hidden_size, num_layers=config.num_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(config.epochs):
        model.train()
        epoch_losses = []
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_reconstruction = model(val_tensor)
            val_loss = loss_fn(val_reconstruction, val_tensor).item()

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)

    return model, history


def reconstruction_errors(model: LSTMAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-window mean squared reconstruction error, higher = more anomalous."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        reconstruction = model(x_tensor)
        errors = torch.mean((reconstruction - x_tensor) ** 2, dim=(1, 2))
    return errors.numpy()
