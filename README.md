# F1 Telemetry Intelligence

Anomaly detection and analytics on Formula 1 telemetry, built on [FastF1](https://github.com/theOehrly/Fast-F1).

The goal: take raw car telemetry (speed, throttle, brake, gear, RPM, DRS,
position) from real F1 sessions and turn it into engineered features and
sequence models for anomaly detection, then grow it into a small
production-style ML system — experiment tracking, a serving API backed by
MongoDB, tests and CI.

## Status

Core pipeline, LSTM autoencoder + MLflow tracking, and the FastAPI/MongoDB
serving layer are in place. See [Findings](#findings) below for an honest
account of where the model currently stands against real race-control
incidents — it's not a finished success story yet, and the README says so.

## Project layout

```
src/f1telemetry/
  data/       FastF1 session loading and telemetry extraction
  features/   Rolling-window features + sequence windowing (incl. derived
              Acceleration / LateralAcceleration channels)
  models/     Baseline anomaly model (Isolation Forest) and an LSTM
              autoencoder, trained/tracked via MLflow
  api/        FastAPI service that scores telemetry and serves results
              from MongoDB
scripts/      Data-fetching, combining, and anomaly-validation utilities
tests/        Unit + API tests (synthetic data / mongomock, no network
              or live services required)
notebooks/    Exploratory analysis
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Fetch telemetry

One session:

```bash
python -m f1telemetry.data.ingest 2024 Bahrain --session R --mode full --out data/raw/bahrain.parquet
```

A full season (skips races already fetched):

```bash
python scripts/fetch_season.py 2026
```

Combine several sessions/a season directory into one file:

```bash
python scripts/combine_telemetry.py bahrain monza --dir data/raw/season_2026 --out data/raw/telemetry_all.parquet
```

### 2. Train the LSTM autoencoder (tracked in MLflow)

```bash
python -m f1telemetry.models.train_lstm data/raw/telemetry_all.parquet \
    --window-size 20 --stride 10 --hidden-size 64 --num-layers 2 --epochs 50 \
    --register-as f1-anomaly-lstm

mlflow ui   # inspect runs, compare configs, browse the model registry
```

Override the channel set for an ablation experiment:

```bash
python -m f1telemetry.models.train_lstm data/raw/telemetry_all.parquet \
    --channels Speed Throttle Brake nGear RPM LateralAcceleration
```

### 3. Validate against real race-control incidents

```bash
python scripts/validate_anomalies.py data/raw/telemetry_all.parquet --run-id <RUN_ID>
```

This clusters race-control messages (flags, safety cars, etc.) into events,
checks whether the model's flagged anomaly windows land near them
driver-matched, and — critically — compares that hit rate against a
random-baseline control (same driver pool, same per-event window count) with
a z-score, rather than reporting a raw match percentage on its own.

### 4. Replay a session live

```bash
python scripts/replay_session.py data/raw/telemetry_all.parquet \
    --run-id <RUN_ID> --year 2024 --gp Bahrain --session R --driver VER --speed 100
```

Plays the session's scored windows back in chronological order at an
accelerated pace (100x by default), printing each window as it "arrives"
and flagging anomalies live, annotated against the nearest real
race-control message when one is close in time. This stands in for wiring
the model to a live timing feed for real-time trackside monitoring.

### 5. Serve results via the API

```bash
export MODEL_RUN_ID=<run id>          # $env:MODEL_RUN_ID="<run id>" on PowerShell
uvicorn f1telemetry.api.main:app --reload
```

or with Docker Compose (API + MongoDB together):

```bash
MODEL_RUN_ID=<run id> docker compose up --build
```

Then:

```bash
curl -X POST localhost:8000/score \
    -H "Content-Type: application/json" \
    -d '{"telemetry_path": "data/raw/telemetry_all.parquet"}'

curl "localhost:8000/anomalies?isAnomaly=true&limit=20"
curl "localhost:8000/anomalies/summary"
```

## Tests

```bash
pytest
```

API tests use `mongomock` and a tiny untrained model, so the full suite runs
without a live MongoDB or a real MLflow run.

## Findings

The core question this project asks: **does unsupervised reconstruction
error on car telemetry line up with what race control actually flags as an
incident?** Validated with a random-baseline control (not just a raw match
rate, which is misleading given how dense both incidents and flagged
windows are across a session), the honest answer is **no, not with this
approach** - and the evidence for that has gotten stronger, not weaker, as
more data was added.

Trained on 17 sessions (2 historical + 15 completed 2026-season races,
~13.3M telemetry rows, 1.3M windows) with the baseline 5-channel set
(Speed/Throttle/Brake/nGear/RPM), the model's flagged anomalies matched
real, driver-matched race-control events **less often than chance**: 301 /
1084 events (28%) vs. a random-baseline expectation of 339.9 ± 11.0 (31%),
**z = -3.55**. With over a thousand events in the comparison, that's not
noise - it's a statistically solid result, just not the one the project set
out to find.

A channel-ablation sweep testing whether a derived `Acceleration` and/or
`LateralAcceleration` channel (from position data) closes that gap is
still in progress; an earlier, smaller-sample pass with just `Acceleration`
added showed no improvement either. The working hypothesis is that this is
less a missing-signal problem and more a task-framing one: many
race-control events (track limits, procedural penalties, unsafe releases)
likely have no telemetry-visible signature at all, which would mean raw
reconstruction error is only ever a partial proxy for "incident," however
the model is tuned. That's the honest conclusion this project is prepared
to report, rather than iterating until a p-value looks better.

## Roadmap

- [x] Repo, data ingestion, feature engineering, baseline anomaly model
- [x] LSTM autoencoder + experiment tracking (MLflow: runs, comparison, registry)
- [x] Serving API (FastAPI) + Docker, results stored in MongoDB
- [x] Test suite (incl. API tests via mongomock) + GitHub Actions CI
      (test job + a Docker-build job) + session replay script
- [ ] Public release
