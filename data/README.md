# Data

Raw order book data (Databento MBP-10 parquet files) is **not included** in this repository due to file size (~24 GB total).

## Structure

```
data/raw/
    AAPL/mbp-10/        # AAPL L2 order book snapshots (2.5 GB, 47 files)
        2024-01-02.parquet
        2024-01-03.parquet
        ...
    SPY/mbp-10/         # SPY L2 order book snapshots (21 GB, 126 files)
        2021-01-27.parquet
        2021-03-17.parquet
        ...
```

## AAPL dates

Training: Jan 2 – Feb 26, 2024 (18 trading days)  
Validation: Feb 28 – Apr 1, 2024 (7 trading days)  
Out-of-sample: Oct 1 – Oct 23, 2024 (17 trading days)

## SPY dates

All 40 FOMC announcement days from 2021-01-27 through 2025-12-10, plus matched control days.

## Download

Data was sourced from Databento via the XNAS ITCH MBP-10 feed. See `download_data.py` and `download_fomc.py` for the download scripts.

## Processed outputs

The following processed CSV files **are** included in the repository:

- `fomc_surprise_table.csv` — per-event FOMC PnL, DGS2 surprise, z-score, and signal classification for all 40 events
- `AAPL_ZOO/factor_zoo_aapl_final_metrics.csv` — per-model, per-period performance metrics (IC, Sharpe, PnL, hit rate, avg hold)
- `factor_zoo_nvda_final_metrics.csv` — same for NVDA
