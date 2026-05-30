# FOMC Event-Window Strategy

Contrarian intraday strategy around FOMC policy announcements using SPY (SPDR S&P 500 ETF).  
**40 events · 2021–2025 · Five monetary policy regimes**

---

## Core Idea

At exactly 2:00 PM ET, the FOMC statement is released. Dozens of algorithmic systems fire simultaneously into a deliberately thin order book, producing a price overshoot beyond the statement's information content. We enter the *opposite* direction at 2:01 PM and exit at 2:15 PM, capturing the partial reversion.

**Two-track filter** — the key innovation:

| Meeting type | Filter | Rationale |
|---|---|---|
| SEP days (Mar/Jun/Sep/Dec) | Trade only when `\|z\| ≤ 1.0` (DGS2 z-score) | Dot-plot can deliver path surprise that overwhelms reversion |
| Non-SEP days | Always trade | Only the rate decision is new; almost always well-priced |

**Result:** 29 traded events · Sharpe = 2.51 · p = 0.018 · Hit rate = 69%

---

## Data

Raw L2 order book data (Databento XNAS ITCH MBP-10) is **not included** in this repo (~21 GB).

**Download from Google Drive:**  
[https://drive.google.com/drive/folders/1ANaOOUpS8bslB7QcqvbI_1iaDI7gILIz](https://drive.google.com/drive/folders/1ANaOOUpS8bslB7QcqvbI_1iaDI7gILIz)

Place files at:
```
data/raw/SPY/mbp-10/
    2021-01-27.parquet
    2021-03-17.parquet
    ... (40 FOMC dates + matched control days)
```

**FRED data** (auto-fetched at runtime via `urllib`):
- `DGS2` — 2-Year Treasury yield (SEP-day filter)
- `DGS1MO` — 1-Month Treasury yield
- `EFFR` — Effective Fed Funds Rate

---

## Folder Structure

```
FOMC/
├── README.md                    ← this file
├── fomc_surprise_table.csv      ← per-event results (40 rows)
├── report_fomc_full.pdf         ← full academic report
├── report_fomc_full.qmd         ← Quarto source (full report)
├── exec_summary_fomc.pdf        ← 2-page executive summary
├── exec_summary_fomc.qmd        ← Quarto source (exec summary)
├── figures/                     ← all 9 publication figures
│   ├── fomc_spread_ratio.png
│   ├── fomc_ret_pnl_scatter.png
│   ├── fomc_vs_control.png
│   ├── fomc_zscore_analysis.png
│   ├── fomc_regime_pnl.png
│   ├── fomc_tc_trend.png
│   ├── fomc_sep_vs_ois.png
│   ├── fomc_ois_surprise.png
│   └── fomc_two_track.png
└── code/
    ├── fomc_ois_filter.py       ← MAIN PIPELINE (run this first)
    ├── generate_fomc_report.py  ← figure generator (run this second)
    ├── download_fomc.py         ← download SPY data from Databento
    ├── download_fomc_extended.py← download FOMC + control days
    ├── _run_fomc.py             ← basic trade runner (exploratory)
    ├── _run_fomc_extended.py    ← runner with control days
    ├── _run_sector_fomc.py      ← sector ETF analysis
    ├── compute_fomc_stats.py    ← aggregate stats from CSV
    └── fomc_event.ipynb         ← exploratory notebook
```

---

## Code Descriptions

### `fomc_ois_filter.py` — Main Pipeline

The complete analysis pipeline. Run from the `FinalProject/` root directory.

**What it does:**
1. Fetches DGS2, DGS1MO, and EFFR from FRED via HTTP
2. Classifies each of the 40 FOMC events as SEP or non-SEP
3. Loads SPY L2 parquet files for each event date
4. Computes the contrarian trade (fade 2:00 PM 1-min return, hold 14 min to 2:15 PM)
5. Applies the two-track surprise filter:
   - SEP days: z-score = `(DGS2_fomc − DGS2_prev) × 100 / σ_30d(ΔDG2)`
   - Non-SEP days: always trade
6. Prints the full event table and backtest comparison
7. Saves `fomc_surprise_table.csv` and `figures/fomc_ois_surprise.png`

**Key parameters:**
```python
NOTIONAL   = 100_000   # position size ($)
HOLD_MINS  = 15        # holding period (minutes)
Z_SMALL    = 1.0       # filter threshold for SEP days
```

**Run:**
```bash
python FOMC/code/fomc_ois_filter.py
```
*(Must use Anaconda Python — requires `fastparquet` for reading Databento parquets)*

---

### `generate_fomc_report.py` — Figure Generator

Reads `fomc_surprise_table.csv` and generates all 9 publication figures into `FOMC/figures/`.

**Figures produced:**

| Filename | Description |
|---|---|
| `fomc_spread_ratio.png` | Pre-FOMC spread widening vs. control days (2021–2025) |
| `fomc_ret_pnl_scatter.png` | 2:00 PM bar return vs. strategy PnL, colored by regime |
| `fomc_vs_control.png` | Annual PnL: FOMC strategy vs. matched control (placebo test) |
| `fomc_zscore_analysis.png` | DGS2 z-score histogram + year-by-year scatter |
| `fomc_regime_pnl.png` | Regime-conditioned cumulative PnL + annual bar chart |
| `fomc_tc_trend.png` | Average TC per trade and spread ratio trend (2021–2025) |
| `fomc_sep_vs_ois.png` | Fed dot-plot median vs. DGS2 market path + guidance surprise bars |
| `fomc_ois_surprise.png` | Two-track surprise per event (SEP=circle, non-SEP=triangle) |
| `fomc_two_track.png` | Two-track decomposition: cumulative PnL and annual breakdown |

**Run:**
```bash
python FOMC/code/generate_fomc_report.py
```

---

### `download_fomc.py` / `download_fomc_extended.py` — Data Downloaders

Download SPY L2 order book data from the Databento API for all FOMC event dates and (extended version) matched control days. Requires a Databento API key in `.env`.

---

### `_run_fomc.py` — Basic Trade Runner

Exploratory script that runs the contrarian trade logic event-by-event without the surprise filter. Useful for inspecting individual events.

### `_run_fomc_extended.py` — Extended Runner

Adds matched control days to the analysis. Source of the control-day PnL figures used in the placebo test.

### `_run_sector_fomc.py` — Sector Analysis

Applies the same FOMC fade strategy to sector ETFs (XLF, XLK, XLE, etc.) instead of SPY.

### `compute_fomc_stats.py` — Stats Helper

Computes Sharpe, hit rate, and p-values from `fomc_surprise_table.csv`. Useful for quick sanity checks.

### `fomc_event.ipynb` — Exploratory Notebook

Jupyter notebook for inspecting intraday SPY dynamics around individual FOMC events.

---

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/sebastian-pina/market_frictions.git
cd market_frictions

# 2. Download raw data from Google Drive → place in data/raw/SPY/mbp-10/

# 3. Run the pipeline (Anaconda Python required)
& "C:\Users\<you>\anaconda3\python.exe" FOMC/code/fomc_ois_filter.py

# 4. Generate figures
& "C:\Users\<you>\anaconda3\python.exe" FOMC/code/generate_fomc_report.py

# 5. Render PDFs (requires Quarto + LaTeX)
cd FOMC
quarto render report_fomc_full.qmd --to pdf
quarto render exec_summary_fomc.qmd --to pdf
```

---

## Key Results

| Filter | N | Total PnL | Sharpe | p-value | Hit Rate |
|---|---|---|---|---|---|
| No filter (all 40) | 40 | +$1,567 | 0.73 | 0.467 | 57% |
| **Two-track (SEP\|z\|≤1.0 + all non-SEP)** | **29** | **+$3,614** | **2.51** | **0.018** | **69%** |
| Excluded SEP events (\|z\|>1.0) | 11 | −$2,046 | −1.51 | 0.162 | 27% |

| Track | N | Total PnL | Sharpe | p-value |
|---|---|---|---|---|
| Non-SEP (always traded) | 22 | +$3,226 | 3.95 | 0.001 |
| SEP \|z\|≤1.0 (traded) | 7 | +$388 | 0.31 | 0.765 |

| Year | PnL | Regime |
|---|---|---|
| 2021 | +$835 | ZLB Hold |
| 2022 | −$787 | Hiking |
| 2023 | +$290 | Transition |
| 2024 | +$252 | Cutting |
| 2025 | +$977 | Post-cut |

---

## Notes on the Surprise Filter

**SEP-day z-score:**  
`z = (DGS2_fomc − DGS2_prev) × 100 / σ_30d`  
where `DGS2_fomc` = 2Y Treasury yield at close on FOMC day and `DGS2_prev` = yield on the prior business day.  
The DGS2 is published at 5 PM ET, so this is a post-hoc classification. In live trading, approximate using intraday yield changes or OIS rates before 2:00 PM.

**Non-SEP days:** No filter needed. The rate decision alone (no dot-plot) is almost always fully priced by the short end of the curve.
