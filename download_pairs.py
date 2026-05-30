"""
Download MBP-10 data for the Pairs Trading strategy: GOOG vs GOOGL.
Same date split as the other strategies.

Usage:
    python download_pairs.py
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import databento as db

API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    raise EnvironmentError("DATABENTO_API_KEY not set. Check your .env file.")

BASE_DIR = Path(__file__).parent / "data" / "raw"
DATASET  = "XNAS.ITCH"

TRAIN_DATES = [
    "2024-01-02", "2024-01-04", "2024-01-08", "2024-01-10", "2024-01-12",
    "2024-01-16", "2024-01-18", "2024-01-22", "2024-01-24", "2024-01-26",
    "2024-02-01", "2024-02-05", "2024-02-07", "2024-02-09", "2024-02-12",
    "2024-02-14", "2024-02-20", "2024-02-22", "2024-02-26", "2024-02-28",
]
TEST_DATES = [
    "2024-03-04", "2024-03-07", "2024-03-11", "2024-03-14", "2024-03-18",
    "2024-04-01", "2024-04-04", "2024-04-08", "2024-04-11", "2024-04-15",
]

client = db.Historical(key=API_KEY)


def save_path(symbol: str, date_str: str) -> Path:
    p = BASE_DIR / symbol / "mbp-10"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{date_str}.parquet"


def download_day(symbol: str, date_str: str) -> None:
    out = save_path(symbol, date_str)
    if out.exists():
        print(f"  SKIP  {symbol}/{date_str}")
        return
    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[symbol],
            schema="mbp-10",
            start=f"{date_str}T09:30:00",
            end=f"{date_str}T16:00:00",
            stype_in="raw_symbol",
        )
        df = data.to_df()
        if df.empty:
            print(f"  EMPTY {symbol}/{date_str}")
            return
        df.to_parquet(out, index=True)
        print(f"  OK    {symbol}/{date_str}  ({len(df):,} rows, {out.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  ERROR {symbol}/{date_str}: {e}")
    time.sleep(0.3)


if __name__ == "__main__":
    print(f"Downloading GOOG + GOOGL MBP-10 to {BASE_DIR.resolve()}\n")
    for symbol in ["GOOG", "GOOGL"]:
        print(f"=== {symbol} ===")
        for d in TRAIN_DATES + TEST_DATES:
            download_day(symbol, d)
    print("\nDone.")
