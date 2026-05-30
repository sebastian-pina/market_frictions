"""
Download SPY MBP-10 for FOMC Event-Window strategy — 2021, 2022, 2023, 2025.

Adds to existing 2024 data (already downloaded via download_fomc.py).
2023 and 2025 were already downloaded; script skips existing files.
Only MBP-10 is needed; trades schema not required for the analysis.
"""

import os
import time
from pathlib import Path
import pandas as pd

_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import databento as db

API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    raise EnvironmentError("DATABENTO_API_KEY not set. Check your .env file.")

BASE_DIR = Path(__file__).parent / "data" / "raw"
DATASET  = "XNAS.ITCH"

# -----------------------------------------------------------------------
# 2021 — Fed on hold (zero rates), taper announced Nov 2021
# -----------------------------------------------------------------------
FOMC_2021 = [
    "2021-01-27",   # holds
    "2021-03-17",   # holds
    "2021-04-28",   # holds
    "2021-06-16",   # holds (dot plot shift — first hike pulled forward)
    "2021-07-28",   # holds
    "2021-09-22",   # holds (taper signal)
    "2021-11-03",   # taper begins
    "2021-12-15",   # taper accelerated + 3 hikes projected for 2022
]
CONTROL_2021 = [
    "2021-01-20",
    "2021-03-10",
    "2021-04-21",
    "2021-06-09",
    "2021-07-21",
    "2021-09-15",
    "2021-10-27",
    "2021-12-08",
]

# -----------------------------------------------------------------------
# 2022 — Aggressive hiking cycle (four 75 bps hikes)
# -----------------------------------------------------------------------
FOMC_2022 = [
    "2022-01-26",   # holds (pre-hike)
    "2022-03-16",   # +25 bps (first hike)
    "2022-05-04",   # +50 bps
    "2022-06-15",   # +75 bps (surprise — consensus was 50)
    "2022-07-27",   # +75 bps
    "2022-09-21",   # +75 bps
    "2022-11-02",   # +75 bps
    "2022-12-14",   # +50 bps
]
CONTROL_2022 = [
    "2022-01-19",
    "2022-03-09",
    "2022-04-27",
    "2022-06-08",
    "2022-07-20",
    "2022-09-14",
    "2022-10-26",
    "2022-12-07",
]

# -----------------------------------------------------------------------
# 2023 — End of hiking cycle + pivot to holds
# -----------------------------------------------------------------------
FOMC_2023 = [
    "2023-02-01",   # +25 bps
    "2023-03-22",   # +25 bps
    "2023-05-03",   # +25 bps (last hike)
    "2023-06-14",   # holds
    "2023-07-26",   # +25 bps (surprise extra hike)
    "2023-09-20",   # holds
    "2023-11-01",   # holds
    "2023-12-13",   # holds (dovish pivot — dot plot)
]
CONTROL_2023 = [
    "2023-01-25",
    "2023-03-15",
    "2023-04-26",
    "2023-06-07",
    "2023-07-19",
    "2023-09-13",
    "2023-10-25",
    "2023-12-06",
]

# -----------------------------------------------------------------------
# 2025 — Post-cut stabilization
# -----------------------------------------------------------------------
FOMC_2025 = [
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
]
CONTROL_2025 = [
    "2025-01-22",
    "2025-03-12",
    "2025-04-30",
    "2025-06-11",
    "2025-07-23",
    "2025-09-10",
    "2025-10-22",
    "2025-12-03",
]

ALL_DATES = FOMC_2021 + CONTROL_2021 + FOMC_2022 + CONTROL_2022 + FOMC_2023 + CONTROL_2023 + FOMC_2025 + CONTROL_2025

client = db.Historical(key=API_KEY)


def session_times_et(date_str: str):
    tz = "America/New_York"
    return (
        pd.Timestamp(f"{date_str} 09:30:00", tz=tz),
        pd.Timestamp(f"{date_str} 16:00:00", tz=tz),
    )


def save_path(symbol: str, schema: str, date_str: str) -> Path:
    p = BASE_DIR / symbol / schema
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{date_str}.parquet"


def download_day(symbol: str, schema: str, date_str: str) -> None:
    out = save_path(symbol, schema, date_str)
    if out.exists():
        print(f"  SKIP  {symbol}/{schema}/{date_str}")
        return
    start_et, end_et = session_times_et(date_str)
    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[symbol],
            schema=schema,
            start=start_et,
            end=end_et,
            stype_in="raw_symbol",
        )
        df = data.to_df()
        if df.empty:
            print(f"  EMPTY {symbol}/{schema}/{date_str}")
            return
        df.to_parquet(out, index=True)
        print(f"  OK    {symbol}/{schema}/{date_str}  ({len(df):,} rows, {out.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  ERROR {symbol}/{schema}/{date_str}: {e}")
    time.sleep(0.3)


if __name__ == "__main__":
    try:
        c_mbp = client.metadata.get_cost(
            dataset=DATASET, symbols=["SPY"], schema="mbp-10",
            start="2024-01-31T14:30:00", end="2024-01-31T21:00:00",
            stype_in="raw_symbol",
        )
        n_new = sum(
            1 for d in ALL_DATES
            if not save_path("SPY", "mbp-10", d).exists()
        )
        print(f"Estimated cost: ~${c_mbp * n_new:.2f} USD  ({n_new} new downloads)\n")
    except Exception:
        pass

    print(f"Downloading to: {BASE_DIR.resolve()}\n")

    for year_label, dates in [
        ("=== 2021 SPY MBP-10 ===", FOMC_2021 + CONTROL_2021),
        ("=== 2022 SPY MBP-10 ===", FOMC_2022 + CONTROL_2022),
        ("=== 2023 SPY MBP-10 ===", FOMC_2023 + CONTROL_2023),
        ("=== 2025 SPY MBP-10 ===", FOMC_2025 + CONTROL_2025),
    ]:
        print(year_label)
        for d in dates:
            download_day("SPY", "mbp-10", d)
        print()

    print("Done.")
