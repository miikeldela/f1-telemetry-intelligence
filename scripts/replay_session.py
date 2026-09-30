"""Replay a session's telemetry through the trained model in chronological
order, at an accelerated real-time pace - a stand-in for the "real-time
trackside decisions" angle in the job posting: this is what wiring the
model to a live timing feed would look and behave like, using recorded
telemetry as the stand-in "live" feed.

Usage:
    python scripts/replay_session.py data/raw/telemetry_all.parquet \
        --run-id <RUN_ID> --year 2024 --gp Bahrain --session R --driver VER \
        --speed 100

--speed is how much faster than real time to play back (100 = 1 real second
of the session per ~0.01s here). Each window is printed as it "arrives",
flagged windows are marked, and (best-effort) annotated with the nearest
real race-control event if one is close in time - so a reviewer watching
the console can see the model's calls line up (or not) with what actually
happened, live, rather than only in a post-hoc table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1telemetry.api.config import Settings  # noqa: E402
from f1telemetry.api.inference import load_model_bundle, score_telemetry  # noqa: E402
from f1telemetry.data.ingest import load_session  # noqa: E402


def _nearest_incident(incidents: pd.DataFrame, when: pd.Timestamp, tolerance: pd.Timedelta):
    if incidents.empty or "Time" not in incidents.columns:
        return None
    nearby = incidents[(incidents["Time"] - when).abs() <= tolerance]
    if nearby.empty:
        return None
    row = nearby.iloc[0]
    return str(row.get("Message", row.get("Category", "incident")))[:80]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a session's telemetry live, window by window."
    )
    parser.add_argument("telemetry_path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--gp", required=True)
    parser.add_argument("--session", default="R")
    parser.add_argument(
        "--driver", default=None, help="Limit playback to one driver code, e.g. VER"
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument(
        "--speed", type=float, default=50.0, help="Playback speed multiplier (default 50x)"
    )
    parser.add_argument(
        "--incident-tolerance-seconds",
        type=float,
        default=5.0,
        help="Annotate a window with a real race-control message within this many seconds",
    )
    args = parser.parse_args()

    print(f"Loading model bundle for run {args.run_id}...")
    settings = Settings(model_run_id=args.run_id)
    bundle = load_model_bundle(settings)
    print(f"Channels: {','.join(bundle.channels)}")

    telemetry = pd.read_parquet(args.telemetry_path)
    mask = pd.Series(True, index=telemetry.index)
    if "Year" in telemetry.columns:
        mask &= telemetry["Year"] == args.year
    if "GrandPrix" in telemetry.columns:
        mask &= telemetry["GrandPrix"] == args.gp
    if "SessionType" in telemetry.columns:
        mask &= telemetry["SessionType"] == args.session
    if args.driver:
        mask &= telemetry["Driver"] == args.driver
    telemetry = telemetry[mask]

    if telemetry.empty:
        raise SystemExit("No telemetry rows matched year/gp/session/driver filters.")

    print(f"Scoring {len(telemetry)} telemetry rows...")
    meta, threshold = score_telemetry(
        bundle, telemetry, args.window_size, args.stride, args.top_fraction
    )
    if len(meta) == 0 or "start_date" not in meta.columns:
        raise SystemExit(
            "No windows produced, or telemetry has no Date column to replay in real time - "
            "re-ingest with the current ingest.py (which preserves Date) if needed."
        )

    meta = meta.sort_values("start_date").reset_index(drop=True)
    print(f"{len(meta)} windows queued, threshold={threshold:.4f}. Fetching race control log...")

    try:
        session = load_session(args.year, args.gp, args.session)
        incidents = session.race_control_messages
    except Exception as exc:  # noqa: BLE001 - replay should still work without live annotation
        print(f"Could not load race control messages ({exc}); playing back without annotations.")
        incidents = pd.DataFrame()

    tolerance = pd.Timedelta(seconds=args.incident_tolerance_seconds)

    print(f"\n=== Replaying {args.year} {args.gp} {args.session} at {args.speed:.0f}x ===\n")
    last_time = None
    for _, row in meta.iterrows():
        current_time = row["start_date"]
        if last_time is not None:
            real_gap = (current_time - last_time).total_seconds()
            time.sleep(max(0.0, real_gap) / args.speed)
        last_time = current_time

        flag = "ANOMALY" if row["is_anomaly"] else "       "
        driver = row.get("Driver", "?")
        lap = row.get("LapNumber", "?")
        line = (
            f"[{current_time.strftime('%H:%M:%S.%f')[:-3]}] {flag}  "
            f"driver={driver} lap={lap} error={row['anomaly_error']:.4f}"
        )

        if row["is_anomaly"]:
            note = _nearest_incident(incidents, current_time, tolerance)
            if note:
                line += f"   <-- race control: {note}"
        print(line)

    print("\n=== Replay finished ===")


if __name__ == "__main__":
    _main()
