"""Fetch full-race telemetry for every completed race of a season.

Usage:
    python scripts/fetch_season.py 2026
    python scripts/fetch_season.py 2026 --out-dir data/raw/season_2026

Skips races whose output file already exists, so it's safe to re-run or
resume after an interruption. Looks up the season via FastF1's event
schedule; a round that fails to load (postponed, cancelled, sprint-only
weekend without a plain 'R' session, transient network hiccup, etc.) is
logged and skipped rather than aborting the whole run - this can take a
long time for a full season and is meant to run unattended overnight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fastf1
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1telemetry.data.ingest import enable_cache, get_all_laps_telemetry, load_session  # noqa: E402


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch full telemetry for every completed race in a season."
    )
    parser.add_argument("year", type=int)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--session", default="R", help="Session to fetch per round (default: R)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"data/raw/season_{args.year}")
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_cache()
    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    now = pd.Timestamp.now(tz="UTC")

    event_dates = pd.to_datetime(schedule["EventDate"], utc=True)
    completed = schedule[event_dates < now]
    print(f"{len(completed)} / {len(schedule)} rounds of {args.year} appear completed as of now.")

    n_saved, n_skipped, n_failed = 0, 0, 0

    for _, event in completed.iterrows():
        round_no = int(event["RoundNumber"])
        gp = event["EventName"]
        out_path = out_dir / f"{args.year}_r{round_no:02d}_{args.session}.parquet"

        if out_path.exists():
            print(f"[{round_no:02d}] {gp}: already have {out_path.name}, skipping.")
            n_skipped += 1
            continue

        print(f"[{round_no:02d}] {gp}: loading {args.session}...")
        try:
            session = load_session(args.year, gp, args.session)
            df = get_all_laps_telemetry(session)
            if df.empty:
                print(f"[{round_no:02d}] {gp}: no telemetry returned, skipping save.")
                n_failed += 1
                continue
            df["Year"] = args.year
            df["GrandPrix"] = gp
            df["SessionType"] = args.session
            df.to_parquet(out_path, index=False)
            print(f"[{round_no:02d}] {gp}: saved {len(df)} rows -> {out_path}")
            n_saved += 1
        except Exception as exc:  # noqa: BLE001 - one bad round shouldn't kill an overnight run
            print(f"[{round_no:02d}] {gp}: FAILED ({exc}), skipping.")
            n_failed += 1

    print(f"\nDone. Saved {n_saved}, skipped (already had) {n_skipped}, failed {n_failed}.")


if __name__ == "__main__":
    _main()
