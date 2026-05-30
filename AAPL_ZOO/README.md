# AAPL Factor Zoo

Microstructure factor model for predicting 1-minute mid-price returns in AAPL (and NVDA comparison).  
**Training: Jan–Feb 2024 · Validation: Mar–Apr 2024 · OOS: Oct 2024**

---

## Core Idea

Can limit order book (L2) microstructure factors predict the next 1-minute return in AAPL? We build a "factor zoo" — a set of features derived from the real-time order book — and test multiple machine learning models to find which factors carry genuine predictive power versus which are spurious in-sample fits.

**Key question:** Does the *spread* itself predict returns? A wide bid-ask spread (rational adverse-selection widening by market makers) predicts that informed flow is incoming — so the next mid-price move should be directional, not mean-reverting.

---

## Data

Raw L2 order book data (Databento XNAS ITCH MBP-10) is **not included** in this repo (~2.5 GB for AAPL).

**Download from Google Drive:**  
[https://drive.google.com/drive/folders/1ANaOOUpS8bslB7QcqvbI_1iaDI7gILIz](https://drive.google.com/drive/folders/1ANaOOUpS8bslB7QcqvbI_1iaDI7gILIz)

Place files at:
```
data/raw/AAPL/mbp-10/
    2024-01-02.parquet     ← training start
    2024-01-03.parquet
    ...
    2024-10-23.parquet     ← OOS end
```

**Date splits:**

| Period | Dates | Days |
|---|---|---|
| Training | Jan 2 – Feb 26, 2024 | 38 |
| Validation | Feb 28 – Apr 1, 2024 | 23 |
| Out-of-sample | Oct 1 – Oct 23, 2024 | 17 |

---

## Folder Structure

```
AAPL_ZOO/
├── README.md                          ← this file
├── factor_zoo_aapl_final_metrics.csv  ← per-model, per-period metrics (AAPL)
├── factor_zoo_nvda_final_metrics.csv  ← same for NVDA
├── factor_zoo_spy_final_metrics.csv   ← same for SPY
├── factor_zoo_metrics.csv             ← combined metrics (early run)
├── factor_zoo_nvda_metrics.csv        ← NVDA metrics (early run)
├── factor_zoo_spy_metrics.csv         ← SPY metrics (early run)
├── report_fz_aapl_full.pdf            ← full AAPL report
├── report_fz_aapl_full.qmd            ← Quarto source
├── exec_summary_fz_aapl.pdf           ← executive summary
├── exec_summary_fz_aapl.qmd           ← Quarto source
├── figures/                           ← all publication figures
│   ├── fz_aapl_ic.png
│   ├── fz_aapl_pnl.png
│   ├── fz_aapl_importance.png
│   ├── fz_aapl_inventory.png
│   └── fz_aapl_rolling_sharpe.png
└── code/
    ├── factor_zoo_final.py            ← MAIN AAPL PIPELINE (run this)
    ├── factor_zoo_nvda_final.py       ← NVDA pipeline
    ├── factor_zoo_spy_final.py        ← SPY intraday pipeline
    ├── generate_fz_report.py          ← figure generator
    ├── download_data.py               ← download AAPL data from Databento
    ├── download_oct2024.py            ← download Oct 2024 specifically
    ├── factor_zoo.py                  ← earlier AAPL version
    ├── factor_zoo_multi.py            ← multi-asset runner
    ├── factor_zoo_nvda.py             ← earlier NVDA version
    ├── factor_zoo_spy.py              ← earlier SPY version
    ├── factor_zoo_oct2024.py          ← Oct 2024 focused analysis
    ├── factor_zoo_final_v2.py         ← variant with different configs
    ├── _run_nvda.py                   ← NVDA runner script
    └── factor_zoo.ipynb               ← exploratory notebook
```

---

## Microstructure Factors

All features are computed from the 10-level bid/ask order book at 1-minute resolution:

| Factor | Description | Hypothesis |
|---|---|---|
| `ofi` | Order Flow Imbalance: `(bid_sz − ask_sz) / (bid_sz + ask_sz)` | Positive OFI → buy pressure → price up |
| `spread_bps` | Bid-ask spread in bps: `(ask − bid) / mid × 10000` | Wide spread → market maker expects informed flow |
| `depth_imb` | Depth imbalance across top 5 levels | Total book skew → directional pressure |
| `rv` | Realized variance: sum of squared 1-sec returns in the minute | High volatility → momentum or mean-reversion regime |
| `vpin` | Volume-Synchronized Probability of Informed Trading | VPIN spike → informed flow incoming |
| `microprice` | Quantity-weighted mid: `(ask×bid_sz + bid×ask_sz) / (bid_sz + ask_sz)` | Better estimate of true fair value than mid |
| `price_impact` | Price move per unit of OFI | Measures book resilience |
| `book_pressure` | Ratio of top-3 to total depth | How concentrated is liquidity near the touch |

**Target variable:** `ret_next` = 1-minute mid-price return (next bar)

---

## Models

| Model | Description |
|---|---|
| OLS | Baseline linear regression on all factors |
| Ridge | L2-regularized OLS; controls for multicollinearity |
| Lasso | L1-regularized OLS; performs automatic factor selection |
| Random Forest | Ensemble of decision trees; captures non-linear interactions |
| XGBoost | Gradient-boosted trees; typically best performer |

**Evaluation metrics** (per model, per period):
- **IC** (Information Coefficient): Spearman correlation between predicted and realized returns
- **Sharpe**: Annualized Sharpe of a long-short portfolio sorted by predicted return
- **PnL**: Simulated net PnL after transaction costs
- **Hit rate**: Fraction of correctly-signed predictions

---

## Code Descriptions

### `factor_zoo_final.py` — Main AAPL Pipeline

The definitive AAPL analysis. Run from the `FinalProject/` root directory.

**What it does:**
1. Loads all AAPL L2 parquet files for training / validation / OOS periods
2. Resamples tick data to 1-minute bars
3. Computes all 8 microstructure factors
4. Fits 5 models (OLS, Ridge, Lasso, RF, XGBoost) on training data
5. Evaluates each model on validation and OOS with IC, Sharpe, PnL, hit rate
6. Saves `AAPL_ZOO/factor_zoo_aapl_final_metrics.csv`

**Run:**
```bash
python AAPL_ZOO/code/factor_zoo_final.py
```
*(Must use Anaconda Python — requires `fastparquet`, `xgboost`, `sklearn`)*

---

### `factor_zoo_nvda_final.py` — NVDA Pipeline

Identical pipeline applied to NVDA. Uses different date splits reflecting NVDA's higher volatility. Outputs `AAPL_ZOO/factor_zoo_nvda_final_metrics.csv`.

Key finding: NVDA shows higher `spread_bps` importance than AAPL, consistent with its higher intraday volatility and thinner book relative to price.

---

### `factor_zoo_spy_final.py` — SPY Pipeline

Same pipeline applied to SPY intraday data. SPY has much tighter spreads and higher depth — useful as a baseline for comparing microstructure factor strength across instruments.

---

### `generate_fz_report.py` — Figure Generator

Reads the metrics CSVs and generates all publication figures into `AAPL_ZOO/figures/`:

| Filename | Description |
|---|---|
| `fz_aapl_ic.png` | IC by model and period (bar chart) |
| `fz_aapl_pnl.png` | Simulated PnL by model across periods |
| `fz_aapl_importance.png` | Feature importance (RF/XGBoost) |
| `fz_aapl_rolling_sharpe.png` | Rolling 20-day Sharpe of best model |
| `fz_aapl_inventory.png` | Factor autocorrelation / stationarity check |

**Run:**
```bash
python AAPL_ZOO/code/generate_fz_report.py
```

---

### `download_data.py` / `download_oct2024.py` — Data Downloaders

Download AAPL L2 order book data from the Databento API. Requires a Databento API key in `.env` at the project root:
```
DATABENTO_API_KEY=your_key_here
```

---

### `factor_zoo.py`, `factor_zoo_multi.py`, etc. — Earlier Versions

Exploratory and intermediate versions of the pipeline. `factor_zoo.py` is the original single-asset AAPL prototype. `factor_zoo_multi.py` runs AAPL + NVDA + SPY in sequence. These are kept for reference but `factor_zoo_final.py` is the canonical version.

### `factor_zoo_oct2024.py` — October 2024 Deep Dive

Focused analysis of the October 2024 out-of-sample period, which coincides with elevated pre-election volatility. Checks whether factor loadings shift in high-vol regimes.

### `_run_nvda.py` — NVDA Runner

Shell runner that executes `factor_zoo_nvda_final.py` with logging.

### `factor_zoo.ipynb` — Exploratory Notebook

Jupyter notebook for interactive exploration: plotting order book dynamics, computing factors for single days, inspecting model residuals.

---

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/sebastian-pina/market_frictions.git
cd market_frictions

# 2. Download raw data from Google Drive → place in data/raw/AAPL/mbp-10/

# 3. Run the pipeline (Anaconda Python required)
& "C:\Users\<you>\anaconda3\python.exe" AAPL_ZOO/code/factor_zoo_final.py

# 4. Generate figures
& "C:\Users\<you>\anaconda3\python.exe" AAPL_ZOO/code/generate_fz_report.py

# 5. Render PDFs (requires Quarto + LaTeX)
cd AAPL_ZOO
quarto render report_fz_aapl_full.qmd --to pdf
quarto render exec_summary_fz_aapl.qmd --to pdf
```

---

## Key Results

| Model | Period | IC | Sharpe | Hit Rate |
|---|---|---|---|---|
| XGBoost | Training | 0.041 | 1.82 | 53% |
| XGBoost | Validation | 0.018 | 0.74 | 51% |
| XGBoost | OOS | 0.012 | 0.41 | 51% |
| Random Forest | OOS | 0.009 | 0.33 | 50% |
| OLS | OOS | 0.003 | 0.11 | 50% |

**Top factors (by feature importance):**
1. `spread_bps` — most important predictor; wide spread → direction continuation
2. `ofi` — order flow imbalance; standard microstructure signal
3. `depth_imb` — book-level pressure; captures multi-level supply/demand
4. `vpin` — informed trading probability; spikes precede larger moves

**Comparison AAPL vs NVDA:**
- NVDA shows higher `spread_bps` importance (wider spreads, higher volatility)
- AAPL shows more stable IC across periods (deeper, more liquid book)
- Both show IC decay from training → OOS, consistent with alpha decay in liquid markets

---

## Dependencies

```
pandas, numpy, scipy, matplotlib, scikit-learn, xgboost, fastparquet
```

Install with Anaconda:
```bash
conda install pandas numpy scipy matplotlib scikit-learn
conda install -c conda-forge xgboost fastparquet
```
