"""
Download MBP-10 and trades data from Databento for the final project.

Symbols:
  SPY  — leader in the Lead-Lag strategy
  AAPL — lagger in Lead-Lag AND the single symbol for Factor Zoo

Schemas:
  mbp-10  → 10-level limit order book snapshots (both symbols)
  trades  → executed trades only for AAPL (signed OFI / VPIN features)

Date split:
  TRAIN  2024-01-02 → 2024-02-29  (~40 trading days)
  TEST   2024-03-01 → 2024-04-30  (~43 trading days)

  We download ~20 training days + ~10 test days to keep credit usage low.
  Adjust TRAIN_DATES / TEST_DATES lists below if you want more.

Usage:
  Set DATABENTO_API_KEY as a Windows environment variable, then run:
      python download_data.py

  Downloaded files land in data/raw/<symbol>/<schema>/<date>.parquet
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

# Load .env if present (no extra package needed)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import databento as db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "DATABENTO_API_KEY not set. Add it to FinalProject/.env:\n"
        "  DATABENTO_API_KEY=your-key\n"
    )

BASE_DIR = Path(__file__).parent / "data" / "raw"

# 20 training days: every other trading day in Jan-Feb 2024
TRAIN_DATES = [
    "2024-01-02", "2024-01-04", "2024-01-08", "2024-01-10", "2024-01-12",
    "2024-01-16", "2024-01-18", "2024-01-22", "2024-01-24", "2024-01-26",
    "2024-02-01", "2024-02-05", "2024-02-07", "2024-02-09", "2024-02-12",
    "2024-02-14", "2024-02-20", "2024-02-22", "2024-02-26", "2024-02-28",
]

# 10 test days: Mar-Apr 2024
TEST_DATES = [
    "2024-03-04", "2024-03-07", "2024-03-11", "2024-03-14", "2024-03-18",
    "2024-04-01", "2024-04-04", "2024-04-08", "2024-04-11", "2024-04-15",
]

ALL_DATES = TRAIN_DATES + TEST_DATES

SYMBOLS_MBP10 = ["SPY", "AAPL"]   # lead-lag needs both; factor zoo uses AAPL
SYMBOLS_TRADES = ["AAPL"]          # signed trades for Factor Zoo features

DATASET = "XNAS.ITCH"              # Nasdaq ITCH — covers SPY and AAPL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

client = db.Historical(key=API_KEY)


def session_times_et(date_str: str):
    """Return regular session start/end as timezone-aware pandas Timestamps (ET)."""
    tz = "America/New_York"
    start = pd.Timestamp(f"{date_str} 09:30:00", tz=tz)
    end   = pd.Timestamp(f"{date_str} 16:00:00", tz=tz)
    return start, end


def save_path(symbol: str, schema: str, date_str: str) -> Path:
    p = BASE_DIR / symbol / schema
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{date_str}.parquet"


def download_day(symbol: str, schema: str, date_str: str) -> None:
    out = save_path(symbol, schema, date_str)
    if out.exists():
        print(f"  SKIP  {symbol}/{schema}/{date_str}  (already downloaded)")
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
        print(f"  OK    {symbol}/{schema}/{date_str}  ({len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  ERROR {symbol}/{schema}/{date_str}: {e}")

    time.sleep(0.3)   # be polite to the API


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Downloading to: {BASE_DIR.resolve()}\n")

    print("=== MBP-10 (order book) ===")
    for symbol in SYMBOLS_MBP10:
        for d in ALL_DATES:
            download_day(symbol, "mbp-10", d)

    print("\n=== TRADES (signed flow) ===")
    for symbol in SYMBOLS_TRADES:
        for d in ALL_DATES:
            download_day(symbol, "trades", d)

    print("\nDone.")
