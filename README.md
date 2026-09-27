# F1 Telemetry Intelligence

Anomaly detection and analytics on Formula 1 telemetry, built on [FastF1](https://github.com/theOehrly/Fast-F1).

The goal: take raw car telemetry (speed, throttle, brake, gear, RPM) from real
F1 sessions and turn it into engineered features and a baseline
anomaly-detection model, then grow it into a small production-style ML
system — experiment tracking, a serving API, tests and CI.

## Status

Work in progress. Current milestone: data ingestion, feature engineering and
a baseline model.

## Project layout

```
src/f1telemetry/
  data/       FastF1 session loading and telemetry extraction
  features/   Rolling-window feature engineering on telemetry
  models/     Baseline anomaly-detection model (Isolation Forest)
tests/        Unit tests (synthetic data, no network required)
notebooks/    Exploratory analysis
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Usage

Pull telemetry for a session and save it locally:

```bash
python -m f1telemetry.data.ingest 2024 Bahrain --session R --out data/raw/telemetry.parquet
```

Then, in Python:

```python
import pandas as pd
from f1telemetry.features.build_features import build_telemetry_features
from f1telemetry.models.baseline import BaselineAnomalyModel

telemetry = pd.read_parquet("data/raw/telemetry.parquet")
features = build_telemetry_features(telemetry)

model = BaselineAnomalyModel().fit(features)
scored = model.score(features)
scored.sort_values("anomaly_score", ascending=False).head(20)
```

## Tests

```bash
pytest
```

## Roadmap

- [x] Repo, data ingestion, feature engineering, baseline anomaly model
- [ ] LSTM autoencoder + experiment tracking (MLflow: runs, comparison, registry)
- [ ] Serving API (FastAPI) + Docker, results stored in MongoDB
- [ ] Test suite + GitHub Actions CI, session replay
- [ ] Public release
