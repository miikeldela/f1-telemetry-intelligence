"""Streamlit dashboard for the F1 telemetry anomaly model.

Run:
    streamlit run dashboard/app.py

Reads telemetry straight from a parquet file and a trained model straight
from MLflow (same helpers the API and validate_anomalies.py use), so there's
nothing to keep in sync with the rest of the project - one scoring path,
several ways to look at the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1telemetry.api.config import Settings  # noqa: E402
from f1telemetry.api.inference import ModelBundle, load_model_bundle, score_telemetry  # noqa: E402

st.set_page_config(page_title="F1 Telemetry Anomaly Dashboard", layout="wide")


@st.cache_resource(show_spinner="Loading model from MLflow...")
def _load_bundle(run_id: str, mlflow_tracking_uri: str) -> ModelBundle:
    settings = Settings(model_run_id=run_id, mlflow_tracking_uri=mlflow_tracking_uri)
    return load_model_bundle(settings)


@st.cache_data(show_spinner="Reading telemetry parquet...")
def _load_telemetry(telemetry_path: str) -> pd.DataFrame:
    return pd.read_parquet(telemetry_path)


@st.cache_data(show_spinner="Scoring windows (this can take a while on a full season)...")
def _score(
    _bundle: ModelBundle,
    bundle_run_id: str,
    telemetry_path: str,
    window_size: int,
    stride: int,
    top_fraction: float,
) -> tuple[pd.DataFrame, float]:
    # telemetry_path/window_size/stride/top_fraction/bundle_run_id are the
    # real cache key; _bundle is prefixed with an underscore so Streamlit
    # doesn't try (and fail) to hash the model object itself.
    telemetry = _load_telemetry(telemetry_path)
    return score_telemetry(_bundle, telemetry, window_size, stride, top_fraction)


def _session_summary(meta: pd.DataFrame) -> pd.DataFrame:
    session_cols = [c for c in ("Year", "GrandPrix", "SessionType") if c in meta.columns]
    if not session_cols:
        return pd.DataFrame()
    grouped = meta.groupby(session_cols).agg(
        n_windows=("is_anomaly", "size"), n_anomalies=("is_anomaly", "sum")
    )
    grouped["anomaly_rate"] = grouped["n_anomalies"] / grouped["n_windows"]
    return grouped.reset_index().sort_values(session_cols)


st.title("F1 Telemetry Anomaly Dashboard")
st.caption(
    "Scores telemetry with the trained LSTM autoencoder and shows where it "
    "flags anomalies, session by session and lap by lap."
)

with st.sidebar:
    st.header("Configuration")
    telemetry_path = st.text_input("Telemetry parquet path", value="data/raw/telemetry_all.parquet")
    run_id = st.text_input("MLflow run ID", value="", help="From `mlflow ui` or the sweep logs")
    mlflow_uri = st.text_input("MLflow tracking URI", value="sqlite:///mlflow.db")
    window_size = st.number_input("Window size", min_value=5, max_value=200, value=20)
    stride = st.number_input("Stride", min_value=1, max_value=200, value=10)
    top_fraction = st.slider("Top fraction flagged as anomaly", 0.01, 0.25, 0.05)
    run_clicked = st.button("Load & score", type="primary")

if not run_id:
    st.info("Enter an MLflow run ID in the sidebar and click **Load & score** to begin.")
    st.stop()

if not Path(telemetry_path).exists():
    st.error(f"Telemetry file not found: {telemetry_path}")
    st.stop()

if run_clicked or "meta" not in st.session_state or st.session_state.get("run_id") != run_id:
    bundle = _load_bundle(run_id, mlflow_uri)
    meta, threshold = _score(
        bundle, run_id, telemetry_path, int(window_size), int(stride), float(top_fraction)
    )
    st.session_state["meta"] = meta
    st.session_state["threshold"] = threshold
    st.session_state["bundle"] = bundle
    st.session_state["run_id"] = run_id

meta: pd.DataFrame = st.session_state["meta"]
threshold: float = st.session_state["threshold"]
bundle: ModelBundle = st.session_state["bundle"]

if len(meta) == 0:
    st.warning("No windows produced - check window_size/stride against this telemetry.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Channels", ", ".join(bundle.channels))
col2.metric("Total windows", f"{len(meta):,}")
col3.metric("Anomaly threshold (reconstruction error)", f"{threshold:.4f}")

st.subheader("Session overview")
summary = _session_summary(meta)
st.dataframe(
    summary.style.format({"anomaly_rate": "{:.1%}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Lap telemetry with flagged anomaly windows")

session_cols = [c for c in ("Year", "GrandPrix", "SessionType") if c in meta.columns]
if session_cols:
    session_options = (
        meta[session_cols].drop_duplicates().sort_values(session_cols).itertuples(index=False)
    )
    session_labels = {" / ".join(str(v) for v in row): row for row in session_options}
    session_label = st.selectbox("Session", list(session_labels.keys()))
    session_values = session_labels[session_label]
    session_mask = pd.Series(True, index=meta.index)
    for col, val in zip(session_cols, session_values, strict=True):
        session_mask &= meta[col] == val
    session_meta = meta[session_mask]
else:
    session_meta = meta

has_driver_col = "Driver" in session_meta.columns
drivers = sorted(session_meta["Driver"].dropna().unique()) if has_driver_col else []
if not drivers:
    st.warning("No Driver column in this telemetry.")
    st.stop()

driver = st.selectbox("Driver", drivers)
driver_meta = session_meta[session_meta["Driver"] == driver]
laps = sorted(driver_meta["LapNumber"].dropna().unique())
lap = st.selectbox("Lap", laps)
lap_meta = driver_meta[driver_meta["LapNumber"] == lap].sort_values("start_distance")

channel = st.selectbox(
    "Channel to plot",
    [c for c in bundle.channels if c in ("Speed", "Throttle", "RPM", "nGear")] or bundle.channels,
)

telemetry = _load_telemetry(telemetry_path)
lap_row_mask = (telemetry["Driver"] == driver) & (telemetry["LapNumber"] == lap)
if session_cols:
    for col, val in zip(session_cols, session_values, strict=True):
        lap_row_mask &= telemetry[col] == val
lap_telemetry = telemetry[lap_row_mask].sort_values("Distance").reset_index(drop=True)

if lap_telemetry.empty or channel not in lap_telemetry.columns:
    st.warning("No raw telemetry found for this driver/lap/channel combination.")
else:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lap_telemetry["Distance"],
            y=lap_telemetry[channel],
            mode="lines",
            name=channel,
            line=dict(color="#4C78A8"),
        )
    )

    # start_index is this window's position within the same Distance-sorted
    # lap telemetry build_sequences() windowed over, so we can look up its
    # exact start/end distance here instead of approximating a width.
    anomaly_windows = lap_meta[lap_meta["is_anomaly"]]
    last_idx = len(lap_telemetry) - 1
    for _, window in anomaly_windows.iterrows():
        start_idx = int(window["start_index"])
        end_idx = min(start_idx + int(window_size) - 1, last_idx)
        if start_idx > last_idx:
            continue
        fig.add_vrect(
            x0=lap_telemetry["Distance"].iloc[start_idx],
            x1=lap_telemetry["Distance"].iloc[end_idx],
            fillcolor="#E45756",
            opacity=0.25,
            line_width=0,
        )

    fig.update_layout(
        title=f"{driver} - Lap {int(lap)} - {channel} (shaded = flagged anomaly window)",
        xaxis_title="Distance (m)",
        yaxis_title=channel,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    n_flagged = int(anomaly_windows.shape[0])
    if n_flagged:
        st.caption(f"{n_flagged} anomaly window(s) flagged on this lap.")
    else:
        st.caption("No anomaly windows flagged on this lap.")
