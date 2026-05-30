"""
Download SPY MBP-10 + trades for FOMC Event-Window strategy.

FOMC announcements are at 2:00 PM ET — within the regular session.
We download the full trading day to establish intraday baselines.

Date structure:
  FOMC_TRAIN  : 4 FOMC days in-sample  (Jan–Jun 2024)
  FOMC_TEST   : 4 FOMC days OOS        (Jul–Dec 2024)
  CONTROL     : 8 non-FOMC days (same weekday, adjacent week) as placebo

Total: 16 days × 2 schemas (mbp-10 + trades) = 32 downloads ~ $11 USD
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
# FOMC 2024 dates (announcement at 2:00 PM ET each day)
# -----------------------------------------------------------------------
FOMC_TRAIN = [
    "2024-01-31",   # Fed holds — no change expected
    "2024-03-20",   # Fed holds
    "2024-05-01",   # Fed holds
    "2024-06-12",   # Fed holds
]

FOMC_TEST = [
    "2024-07-31",   # Fed holds
    "2024-09-18",   # -50 bps surprise cut  ← high-signal event
    "2024-11-07",   # -25 bps cut
    "2024-12-18",   # -25 bps cut
]

# Non-FOMC control days: same weekday, week before each FOMC day
CONTROL = [
    "2024-01-24",   # control for Jan 31
    "2024-03-13",   # control for Mar 20
    "2024-04-24",   # control for May 1
    "2024-06-05",   # control for Jun 12
    "2024-07-24",   # control for Jul 31
    "2024-09-11",   # control for Sep 18
    "2024-10-30",   # control for Nov 7
    "2024-12-11",   # control for Dec 18
]

ALL_DATES = FOMC_TRAIN + FOMC_TEST + CONTROL

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
    # Estimate cost first
    try:
        c_mbp = client.metadata.get_cost(dataset=DATASET, symbols=["SPY"],
                    schema="mbp-10", start="2024-01-31T14:30:00",
                    end="2024-01-31T21:00:00", stype_in="raw_symbol")
        c_trd = client.metadata.get_cost(dataset=DATASET, symbols=["SPY"],
                    schema="trades", start="2024-01-31T14:30:00",
                    end="2024-01-31T21:00:00", stype_in="raw_symbol")
        n_new = sum(
            1 for d in ALL_DATES
            for schema in ["mbp-10", "trades"]
            if not save_path("SPY", schema, d).exists()
        )
        print(f"Estimated cost for {n_new} new downloads: ~${(c_mbp + c_trd) * n_new / 2:.2f} USD\n")
    except Exception:
        pass

    print(f"Downloading to: {BASE_DIR.resolve()}\n")

    print("=== SPY MBP-10 ===")
    for d in ALL_DATES:
        download_day("SPY", "mbp-10", d)

    print("\n=== SPY trades ===")
    for d in ALL_DATES:
        download_day("SPY", "trades", d)

    print("\nDone.")
