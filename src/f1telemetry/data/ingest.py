"""FastF1 session and telemetry loading utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

# repo_root/data/cache
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "cache"


def enable_cache(cache_dir: Path | str = CACHE_DIR) -> None:
    """Enable FastF1's on-disk cache so repeated loads don't re-download."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session(year: int, gp: str, session_type: str = "R") -> fastf1.core.Session:
    """Load a FastF1 session (laps + telemetry) for a given year/GP/session.

    session_type: one of 'FP1', 'FP2', 'FP3', 'Q', 'S', 'R' (race).
    """
    enable_cache()
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def get_driver_lap_telemetry(
    session: fastf1.core.Session, driver: str, lap: str | int = "fastest"
) -> pd.DataFrame:
    """Return car telemetry (with distance) for one driver's lap.

    driver: three-letter FastF1 driver code, e.g. 'VER', 'HAM'.
    lap: 'fastest' or a lap number.
    """
    driver_laps = session.laps.pick_driver(driver)
    if lap == "fastest":
        target_lap = driver_laps.pick_fastest()
    else:
        target_lap = driver_laps[driver_laps["LapNumber"] == lap].iloc[0]

    telemetry = target_lap.get_car_data().add_distance()
    telemetry["Driver"] = driver
    telemetry["LapNumber"] = target_lap["LapNumber"]
    return telemetry


def get_all_drivers_telemetry(
    session: fastf1.core.Session, lap: str | int = "fastest"
) -> pd.DataFrame:
    """Return telemetry for every driver's chosen lap in one long DataFrame."""
    frames: list[pd.DataFrame] = []
    for driver in session.laps["Driver"].unique():
        try:
            frames.append(get_driver_lap_telemetry(session, driver, lap=lap))
        except Exception as exc:  # noqa: BLE001 - keep pulling other drivers on one failure
            logger.warning("Skipping %s: %s", driver, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_lap_telemetry(session: fastf1.core.Session, driver: str, lap_number: int) -> pd.DataFrame:
    """Return car telemetry (with distance) for one specific lap."""
    driver_laps = session.laps.pick_driver(driver)
    matches = driver_laps[driver_laps["LapNumber"] == lap_number]
    if matches.empty:
        raise ValueError(f"No lap {lap_number} found for driver {driver}")
    target_lap = matches.iloc[0]

    telemetry = target_lap.get_car_data().add_distance()
    telemetry["Driver"] = driver
    telemetry["LapNumber"] = lap_number
    return telemetry


def get_all_laps_telemetry(
    session: fastf1.core.Session, drivers: list[str] | None = None
) -> pd.DataFrame:
    """Return telemetry for every completed lap of every (or selected) driver.

    Unlike get_all_drivers_telemetry (one lap per driver), this pulls the full
    set of laps, which is what a sequence model like an LSTM autoencoder needs
    to have enough training data. Telemetry for a session is already cached
    locally after the first load, so this does not mean many more downloads.
    """
    if drivers is None:
        drivers = list(session.laps["Driver"].unique())

    frames: list[pd.DataFrame] = []
    for driver in drivers:
        driver_laps = session.laps.pick_driver(driver)
        for _, lap in driver_laps.iterlaps():
            lap_number = lap["LapNumber"]
            try:
                telemetry = lap.get_car_data().add_distance()
                telemetry["Driver"] = driver
                telemetry["LapNumber"] = lap_number
                frames.append(telemetry)
            except Exception as exc:  # noqa: BLE001 - keep pulling other laps on one failure
                logger.warning("Skipping %s lap %s: %s", driver, lap_number, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pull FastF1 telemetry for a session.")
    parser.add_argument("year", type=int)
    parser.add_argument("gp", type=str, help="Grand Prix name or round number, e.g. 'Bahrain'")
    parser.add_argument("--session", default="R", help="FP1/FP2/FP3/Q/S/R (default: R)")
    parser.add_argument(
        "--mode",
        choices=["fastest", "full"],
        default="fastest",
        help=(
            "'fastest': one lap per driver (default, quick exploration). "
            "'full': every lap of every driver, needed for sequence models "
            "like the LSTM autoencoder."
        ),
    )
    parser.add_argument("--lap", default="fastest", help="'fastest' or a lap number (only used with --mode fastest)")
    parser.add_argument("--out", default="data/raw/telemetry.parquet")
    args = parser.parse_args()

    s = load_session(args.year, args.gp, args.session)

    if args.mode == "full":
        df = get_all_laps_telemetry(s)
    else:
        lap_arg: str | int = int(args.lap) if str(args.lap).isdigit() else args.lap  # noqa: E501
        df = get_all_drivers_telemetry(s, lap=lap_arg)

    # Tag every row with its session, so telemetry from different races/sessions
    # can be safely concatenated later without a (Driver, LapNumber) collision
    # (e.g. VER's "Lap 1" in two different races would otherwise merge into one
    # group when building sequences).
    df["Year"] = args.year
    df["GrandPrix"] = args.gp
    df["SessionType"] = args.session

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} telemetry rows to {out_path}")


if __name__ == "__main__":
    _main()
