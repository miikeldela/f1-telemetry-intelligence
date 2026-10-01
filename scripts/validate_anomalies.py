"""Check whether the LSTM autoencoder's flagged anomalies line up with real
race events (flags, safety cars, etc.) pulled from FastF1's race control log.

v2: the naive per-message matching in v1 turned out to be a bad test - with
a generous tolerance, almost any random sample of windows "matches" some
incident purely because incidents and windows are both dense across a
session (see the random-baseline comparison). This version:

  1. Clusters consecutive race-control messages (same driver, close in time)
     into single events, since one safety car period can generate 5-8
     separate messages that shouldn't each count as an independent test.
  2. Matches driver-specific events only against that driver's own anomaly
     windows, not the whole field's - a stricter, more meaningful test.
  3. Uses a tighter default tolerance.
  4. Still runs the random-baseline comparison, now built the same way
     (same driver pool, same per-event window count) for a fair comparison.

Usage:
    python scripts/validate_anomalies.py data/raw/telemetry_combined.parquet --run-id <RUN_ID>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from f1telemetry.api.config import Settings
from f1telemetry.api.inference import load_model_bundle, score_telemetry
from f1telemetry.data.ingest import load_session

INCIDENT_KEYWORDS = ("FLAG", "SAFETY CAR", "VIRTUAL SAFETY CAR", "RED FLAG", "INCIDENT")


def _get_incidents(year: int, gp: str, session_type: str):
    session = load_session(year, gp, session_type)  # cached, should be fast
    race_control = session.race_control_messages

    if len(race_control) == 0:
        return race_control, session

    if "Category" in race_control.columns:
        incidents = race_control[
            race_control["Category"].astype(str).str.upper().isin(
                {"FLAG", "SAFETYCAR", "SAFETY CAR", "DRS"}
            )
        ]
    elif "Message" in race_control.columns:
        pattern = "|".join(INCIDENT_KEYWORDS)
        messages = race_control["Message"].astype(str).str.upper()
        incidents = race_control[messages.str.contains(pattern)]
    else:
        incidents = race_control

    return incidents, session


def _get_driver_code(session, racing_number) -> str | None:
    """Map a race-control RacingNumber (car number) to the 3-letter driver
    code our telemetry uses (session.laps['Driver']). Best-effort: FastF1's
    exact API for this wasn't verified by running it, so this fails soft."""
    if pd.isna(racing_number):
        return None
    try:
        info = session.get_driver(str(int(racing_number)))
        return info.get("Abbreviation") if hasattr(info, "get") else info["Abbreviation"]
    except Exception:  # noqa: BLE001
        return None


def _cluster_incidents(incidents: pd.DataFrame, merge_gap: pd.Timedelta) -> pd.DataFrame:
    """Merge consecutive messages (same driver, close in time) into events,
    so one safety-car period counts once instead of once per message."""
    if len(incidents) == 0:
        return pd.DataFrame(columns=["event_start", "event_end", "RacingNumber"])

    incidents = incidents.sort_values("Time").reset_index(drop=True)
    events: list[dict] = []
    current: dict | None = None

    for _, row in incidents.iterrows():
        racing_number = row.get("RacingNumber")
        same_driver = (
            current is not None
            and (
                (pd.isna(racing_number) and pd.isna(current["RacingNumber"]))
                or racing_number == current["RacingNumber"]
            )
        )
        within_gap = current is not None and (row["Time"] - current["event_end"]) <= merge_gap
        if current is not None and same_driver and within_gap:
            current["event_end"] = row["Time"]
        else:
            if current is not None:
                events.append(current)
            current = {
                "event_start": row["Time"],
                "event_end": row["Time"],
                "RacingNumber": racing_number,
            }
    if current is not None:
        events.append(current)

    return pd.DataFrame(events)


def _driver_pool(session_meta: pd.DataFrame, driver_code: str | None) -> pd.DataFrame:
    """This event's relevant window pool: one driver's windows if we could
    resolve one, otherwise the whole field (whole-field incidents, e.g. a
    session-wide safety car, don't have a single driver to check)."""
    if driver_code is None or "Driver" not in session_meta.columns:
        return session_meta
    filtered = session_meta[session_meta["Driver"] == driver_code]
    return filtered if len(filtered) > 0 else session_meta


def _match_event(pool: pd.DataFrame, event: pd.Series, tolerance: pd.Timedelta) -> bool:
    window_start = event["event_start"] - tolerance
    window_end = event["event_end"] + tolerance
    nearby = pool[(pool["start_date"] >= window_start) & (pool["start_date"] <= window_end)]
    return len(nearby) > 0


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate anomalies against real race incidents.")
    parser.add_argument("telemetry_path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=60.0)
    parser.add_argument("--n-trials", type=int, default=200)
    args = parser.parse_args()

    # Reuse the exact same model-loading/scaling/scoring path as the API
    # and dashboard (f1telemetry.api.inference), so this script can never
    # silently drift from how telemetry is actually scored in production -
    # e.g. it now picks up per_session vs. global scaling automatically
    # from whatever the run was trained with, instead of always doing a
    # fresh global StandardScaler fit like this script used to.
    settings = Settings(model_run_id=args.run_id)
    bundle = load_model_bundle(settings)
    print(
        f"Using channels from run {args.run_id}: {','.join(bundle.channels)} "
        f"(scaling={bundle.scaling_strategy})"
    )

    telemetry = pd.read_parquet(args.telemetry_path)
    meta, threshold = score_telemetry(
        bundle,
        telemetry,
        window_size=args.window_size,
        stride=args.stride,
        top_fraction=args.top_fraction,
    )
    if len(meta) == 0:
        raise SystemExit("No windows produced - check --window-size/--stride against the data.")

    print(
        f"\nFlagged {meta['is_anomaly'].sum()} / {len(meta)} windows as anomalies "
        f"(top {args.top_fraction:.0%} by reconstruction error, threshold={threshold:.4f})."
    )

    if "start_date" not in meta.columns:
        print("\nNo Date column in this telemetry - re-run ingest.py to regenerate it, then retry.")
        return

    session_cols = [c for c in ["Year", "GrandPrix", "SessionType"] if c in meta.columns]
    if not session_cols:
        print(
            "\nNo session tags in this telemetry - cannot look up race control "
            "messages per session."
        )
        return

    tolerance = pd.Timedelta(seconds=args.tolerance_seconds)
    merge_gap = pd.Timedelta(seconds=args.merge_gap_seconds)

    sessions_data = []  # (session_meta, events, driver_lookup)
    for session_key, session_meta in meta.groupby(session_cols):
        session_key = session_key if isinstance(session_key, tuple) else (session_key,)
        year, gp, session_type = (
            dict(zip(session_cols, session_key)).get(c)
            for c in ["Year", "GrandPrix", "SessionType"]
        )

        incidents, session = _get_incidents(year, gp, session_type)
        if len(incidents) == 0 or "Time" not in incidents.columns:
            print(f"\n{year} {gp} {session_type}: no usable incidents found, skipping.")
            continue

        events = _cluster_incidents(incidents, merge_gap)
        n_driver_specific = events["RacingNumber"].notna().sum()
        print(
            f"\n{year} {gp} {session_type}: {len(incidents)} raw messages -> "
            f"{len(events)} clustered events ({n_driver_specific} driver-specific, "
            f"{len(events) - n_driver_specific} field-wide)"
        )

        driver_lookup = {
            rn: _get_driver_code(session, rn) for rn in events["RacingNumber"].dropna().unique()
        }
        resolved = sum(1 for v in driver_lookup.values() if v is not None)
        if driver_lookup:
            print(
                f"  Resolved {resolved} / {len(driver_lookup)} driver numbers "
                "to telemetry driver codes."
            )

        sessions_data.append((session_meta, events, driver_lookup))

    total_events = sum(len(events) for _, events, _ in sessions_data)
    if total_events == 0:
        print("\nNo events found across any session - nothing to validate against.")
        return

    def _run_pass(use_random: bool, rng: np.random.Generator | None = None) -> int:
        matched = 0
        for session_meta, events, driver_lookup in sessions_data:
            for _, event in events.iterrows():
                driver_code = (
                    driver_lookup.get(event["RacingNumber"])
                    if pd.notna(event["RacingNumber"])
                    else None
                )
                pool = _driver_pool(session_meta, driver_code)
                n_flagged = int(pool["is_anomaly"].sum())
                if n_flagged == 0 or len(pool) == 0:
                    continue
                if use_random:
                    candidate = pool.sample(
                        n=min(n_flagged, len(pool)),
                        replace=False,
                        random_state=rng.integers(1_000_000_000),
                    )
                else:
                    candidate = pool[pool["is_anomaly"]]
                if _match_event(candidate, event, tolerance):
                    matched += 1
        return matched

    real_matched = _run_pass(use_random=False)
    print(
        f"\nModel: {real_matched} / {total_events} events "
        f"({real_matched / total_events:.0%}) had a flagged anomaly window "
        f"within {args.tolerance_seconds:.0f}s (driver-matched where possible)."
    )

    print(f"\nRunning random-baseline comparison ({args.n_trials} trials)...")
    rng = np.random.default_rng(42)
    baseline_matches = np.array([_run_pass(use_random=True, rng=rng) for _ in range(args.n_trials)])

    baseline_mean = baseline_matches.mean()
    baseline_std = baseline_matches.std()
    z_score = (real_matched - baseline_mean) / (baseline_std + 1e-9)

    print(
        f"Random baseline: {baseline_mean:.1f} +/- {baseline_std:.1f} events matched "
        f"out of {total_events} ({baseline_mean / total_events:.0%})."
    )
    print(f"Model vs. baseline: {real_matched} vs {baseline_mean:.1f} (z = {z_score:.2f})")

    if z_score > 2:
        verdict = "The model matches real incidents meaningfully more than chance."
    elif z_score > 1:
        verdict = "Weak signal above chance - directionally promising but not strong evidence yet."
    else:
        verdict = (
            "Not distinguishable from random flagging - do not claim this validates the model."
        )
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    _main()
