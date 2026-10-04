"""Baseline anomaly-detection model over engineered telemetry features."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import IsolationForest

NON_FEATURE_COLUMNS = {"Distance", "Driver", "LapNumber"}


@dataclass
class BaselineAnomalyModel:
    """Isolation Forest baseline for telemetry anomaly detection."""

    contamination: float = 0.05
    random_state: int = 42
    n_estimators: int = 200
    _model: IsolationForest | None = field(default=None, init=False, repr=False)
    _feature_columns: list[str] | None = field(default=None, init=False, repr=False)

    def _feature_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
        return features[cols]

    def fit(self, features: pd.DataFrame) -> BaselineAnomalyModel:
        x = self._feature_matrix(features)
        self._feature_columns = list(x.columns)
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        self._model.fit(x)
        return self

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("Call fit() before score().")
        x = features[self._feature_columns]
        keep = [c for c in NON_FEATURE_COLUMNS if c in features.columns]
        result = features[keep].copy()
        result["anomaly_score"] = -self._model.score_samples(x)  # higher = more anomalous
        result["is_anomaly"] = self._model.predict(x) == -1
        return result
