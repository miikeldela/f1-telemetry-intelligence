"""Combine several per-session telemetry parquet files into one.

Usage:
    python scripts/combine_telemetry.py madrid bahrain monza

Reads data/raw/<name>.parquet for each name given and writes
data/raw/telemetry_combined.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")


def main() -> None:
    names = sys.argv[1:]
    if not names:
        raise SystemExit("Usage: python scripts/combine_telemetry.py <name1> <name2> ...")

    frames = []
    for name in names:
        path = RAW_DIR / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")
        frames.append(pd.read_parquet(path))

    combined = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "telemetry_combined.parquet"
    combined.to_parquet(out_path, index=False)

    session_cols = [c for c in ["Year", "GrandPrix", "SessionType"] if c in combined.columns]
    n_sessions = combined[session_cols].drop_duplicates().shape[0] if session_cols else 1

    print(f"{len(combined)} total rows across {n_sessions} sessions -> {out_path}")


if __name__ == "__main__":
    main()
