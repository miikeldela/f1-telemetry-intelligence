"""Combine several per-session telemetry parquet files into one.

Usage:
    python scripts/combine_telemetry.py madrid bahrain monza
    python scripts/combine_telemetry.py madrid bahrain monza --dir data/raw/season_2026
    python scripts/combine_telemetry.py --dir data/raw/season_2026 \
        --out data/raw/telemetry_all.parquet

Reads data/raw/<name>.parquet for each name given, plus every *.parquet file
under --dir (if given), and writes the combined result (default:
data/raw/telemetry_combined.parquet, override with --out).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-session telemetry parquet files.")
    parser.add_argument(
        "names", nargs="*", help="Names of data/raw/<name>.parquet files to include"
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Directory whose *.parquet files should all be included (repeatable)",
    )
    parser.add_argument("--out", default=str(RAW_DIR / "telemetry_combined.parquet"))
    args = parser.parse_args()

    paths: list[Path] = []
    for name in args.names:
        path = RAW_DIR / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")
        paths.append(path)

    for dir_arg in args.dir:
        dir_path = Path(dir_arg)
        if not dir_path.exists():
            raise SystemExit(f"Missing directory: {dir_path}")
        found = sorted(dir_path.glob("*.parquet"))
        if not found:
            print(f"Warning: no .parquet files found in {dir_path}")
        paths.extend(found)

    if not paths:
        raise SystemExit(
            "Usage: python scripts/combine_telemetry.py <name1> <name2> ... "
            "[--dir <dir>] [--out <path>]"
        )

    # De-duplicate while preserving order, in case a name and --dir overlap.
    seen = set()
    unique_paths = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)

    print(f"Combining {len(unique_paths)} file(s):")
    for p in unique_paths:
        print(f"  - {p}")

    frames = [pd.read_parquet(p) for p in unique_paths]
    combined = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    session_cols = [c for c in ["Year", "GrandPrix", "SessionType"] if c in combined.columns]
    n_sessions = combined[session_cols].drop_duplicates().shape[0] if session_cols else 1

    print(f"{len(combined)} total rows across {n_sessions} sessions -> {out_path}")


if __name__ == "__main__":
    main()
