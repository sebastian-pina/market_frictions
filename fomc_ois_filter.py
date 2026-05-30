"""
FOMC Rate-Path Surprise Filter
================================
Measures the 2-year Treasury (DGS2) change on each FOMC day as a proxy
for the rate-path surprise: both the rate decision AND the dot-plot / guidance.

The key idea: DGS2 is the most Fed-sensitive tenor. A large move on FOMC day
means the Fed delivered something the market did NOT price the day before.
That is precisely the situation where our contrarian fade fails -- the initial
move is not overshoot, it is new fundamental information being processed.

Surprise = DGS2(fomc_date) - DGS2(business day before fomc_date)

Since DGS2 is published at 5 PM ET (after the trade is placed at 2:01 PM),
the correct implementation uses the T-1 rate as "expected" and the T rate
as "realized" -- consistent with the user's insight to use the implied curve
from the day before.
"""
import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import urllib.request
from pathlib import Path
from scipy import stats

Path('figures').mkdir(exist_ok=True)
DATA_DIR = Path('data/raw')
ET       = 'America/New_York'
BOOK_COLS    = ['bid_px_00','ask_px_00','bid_sz_00','ask_sz_00']
HOLD_MINS    = 15
NOTIONAL     = 100_000

# ── FOMC / Control date lists (from _run_fomc_extended.py) ───────────────────
FOMC_2021 = ['2021-01-27','2021-03-17','2021-04-28','2021-06-16',
             '2021-07-28','2021-09-22','2021-11-03','2021-12-15']
FOMC_2022 = ['2022-01-26','2022-03-16','2022-05-04','2022-06-15',
             '2022-07-27','2022-09-21','2022-11-02','2022-12-14']
FOMC_2023 = ['2023-02-01','2023-03-22','2023-05-03','2023-06-14',
             '2023-07-26','2023-09-20','2023-11-01','2023-12-13']
FOMC_2024 = ['2024-01-31','2024-03-20','2024-05-01','2024-06-12',
             '2024-07-31','2024-09-18','2024-11-07','2024-12-18']
FOMC_2025 = ['2025-01-29','2025-03-19','2025-05-07','2025-06-18',
             '2025-07-30','2025-09-17','2025-10-29','2025-12-10']
ALL_FOMC  = FOMC_2021 + FOMC_2022 + FOMC_2023 + FOMC_2024 + FOMC_2025

REGIME = {
    '2021-01-27':'hold',       '2021-03-17':'hold',       '2021-04-28':'hold',
    '2021-06-16':'hold',       '2021-07-28':'hold',       '2021-09-22':'hold',
    '2021-11-03':'taper',      '2021-12-15':'taper',
    '2022-01-26':'hike+25',    '2022-03-16':'hike+25',    '2022-05-04':'hike+50',
    '2022-06-15':'hike+75',    '2022-07-27':'hike+75',    '2022-09-21':'hike+75',
    '2022-11-02':'hike+75',    '2022-12-14':'hike+50',
    '2023-02-01':'hike+25',    '2023-03-22':'hike+25',    '2023-05-03':'hike+25',
    '2023-06-14':'hold',       '2023-07-26':'hike+25',    '2023-09-20':'hold',
    '2023-11-01':'hold',       '2023-12-13':'hold',
    '2024-01-31':'hold',       '2024-03-20':'hold',       '2024-05-01':'hold',
    '2024-06-12':'hold',       '2024-07-31':'hold',       '2024-09-18':'cut-50',
    '2024-11-07':'cut-25',     '2024-12-18':'cut-25+hawk',
    '2025-01-29':'hold',       '2025-03-19':'hold',       '2025-05-07':'hold',
    '2025-06-18':'hold',       '2025-07-30':'?',          '2025-09-17':'?',
    '2025-10-29':'?',          '2025-12-10':'?',
}


# ── Trade engine (copied from _run_fomc_extended.py) ─────────────────────────
def load_day_bars(date_str):
    p = DATA_DIR / 'SPY' / 'mbp-10' / f'{date_str}.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p, engine='fastparquet', columns=BOOK_COLS).sort_index()
    df.index = df.index.tz_convert(ET)
    bid   = df['bid_sz_00'].astype(np.int64)
    ask   = df['ask_sz_00'].astype(np.int64)
    denom = (bid + ask).replace(0, np.nan)
    tick  = pd.DataFrame({
        'mid':    (df['bid_px_00'] + df['ask_px_00']) / 2,
        'spread': df['ask_px_00'] - df['bid_px_00'],
        'ofi':    (bid - ask) / denom,
    }, index=df.index)
    bars = tick.resample('1min').agg(
        mid_open  = ('mid',    'first'),
        mid_close = ('mid',    'last'),
        spread    = ('spread', 'mean'),
        ofi       = ('ofi',    'mean'),
    ).dropna(subset=['mid_open'])
    bars['ret_bps'] = bars['mid_close'].pct_change() * 1e4
    del df, tick
    return bars


def trade_day(date_str, bars):
    mask = bars.index.time == pd.Timestamp('14:00').time()
    if not mask.any():
        return None
    entry_bar   = bars[mask].iloc[0]
    entry_time  = bars[mask].index[0]
    ret_14      = entry_bar['ret_bps']
    if ret_14 == 0 or pd.isna(ret_14):
        return None
    direction    = -1 if ret_14 > 0 else 1
    entry_price  = entry_bar['mid_close']
    entry_spread = entry_bar['spread']
    future = bars[bars.index >= entry_time + pd.Timedelta(minutes=HOLD_MINS)]
    if future.empty:
        return None
    exit_bar  = future.iloc[0]
    shares    = NOTIONAL / entry_price
    gross_pnl = direction * (exit_bar['mid_close'] - entry_price) * shares
    tc        = (entry_spread / 2 + exit_bar['spread'] / 2) * shares
    return {
        'date':      date_str,
        'year':      int(date_str[:4]),
        'direction': direction,
        'ret_14_bps': round(ret_14, 2),
        'gross_pnl': round(gross_pnl, 2),
        'tc':        round(tc, 2),
        'net_pnl':   round(gross_pnl - tc, 2),
    }


# ── 1. Fetch DGS2 from FRED ─────────────────────────────────────────────────
# We use DGS2 rather than fed-funds futures (Kuttner 1999) because:
# (a) DGS2 responds to BOTH the rate decision AND forward guidance (dot plot)
# (b) The Kuttner measure only captures the "target factor" -- it would have
#     classified Dec-18-2024 as ~zero surprise (25bp cut was fully priced),
#     missing the hawkish dot-plot path surprise entirely.
# DGS2 is the 2-year equivalent of the "path factor" in Gurkaynak-Sack-Swanson (2005).
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2'

print('Fetching DGS2 (2-Year Treasury) from FRED...')
try:
    with urllib.request.urlopen(FRED_URL, timeout=20) as r:
        dgs2 = pd.read_csv(
            r, index_col='observation_date', parse_dates=True, na_values=['.']
        )['DGS2'].dropna()
    dgs2 = dgs2.loc['2020-01-01':]
    # Rolling 30-day std of daily changes (for z-score normalization)
    dgs2_diff = dgs2.diff() * 100          # daily changes in bps
    roll_std  = dgs2_diff.rolling(30, min_periods=10).std()
    print(f'  Loaded {len(dgs2)} observations ({dgs2.index[0].date()} - {dgs2.index[-1].date()})')
except Exception as e:
    dgs2 = None
    print(f'  FRED fetch failed: {e} -- surprise will be NaN for all events')


# ── 2. Compute per-event PnL and DGS2 surprise ───────────────────────────────
print('\nComputing per-event PnL and DGS2 surprise...')
records = []
for fomc_str in ALL_FOMC:
    fomc_ts = pd.Timestamp(fomc_str)
    rec = {
        'date':    fomc_str,
        'year':    fomc_ts.year,
        'regime':  REGIME.get(fomc_str, '?'),
    }

    # DGS2 surprise: FOMC day rate minus previous business day rate
    if dgs2 is not None:
        prev_idx  = dgs2.index[dgs2.index < fomc_ts]
        fomc_rate = dgs2.get(fomc_ts, np.nan)
        if len(prev_idx) > 0 and not np.isnan(fomc_rate):
            prev_rate            = dgs2[prev_idx[-1]]
            raw_bps              = (fomc_rate - prev_rate) * 100
            vol                  = roll_std.get(fomc_ts, np.nan)
            z_score              = raw_bps / vol if (not np.isnan(vol) and vol > 0) else np.nan
            rec['dgs2_prev']     = round(prev_rate, 3)
            rec['dgs2_fomc']     = round(fomc_rate, 3)
            rec['surprise_bps']  = round(raw_bps, 1)
            rec['roll_std_bps']  = round(vol, 2) if not np.isnan(vol) else np.nan
            rec['z_score']       = round(z_score, 2) if not np.isnan(z_score) else np.nan
        else:
            rec['dgs2_prev'] = rec['dgs2_fomc'] = rec['surprise_bps'] = np.nan
            rec['roll_std_bps'] = rec['z_score'] = np.nan
    else:
        for k in ['dgs2_prev','dgs2_fomc','surprise_bps','roll_std_bps','z_score']:
            rec[k] = np.nan

    # PnL from trade engine
    try:
        bars = load_day_bars(fomc_str)
        if bars is not None:
            t = trade_day(fomc_str, bars)
            if t:
                rec['net_pnl']   = t['net_pnl']
                rec['gross_pnl'] = t['gross_pnl']
                rec['tc']        = t['tc']
                rec['ret_14']    = t['ret_14_bps']
    except Exception:
        pass

    records.append(rec)

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
for c in ['net_pnl','gross_pnl','tc','ret_14','surprise_bps','dgs2_prev','dgs2_fomc']:
    if c not in df.columns:
        df[c] = np.nan

# ── 3. Classify events ────────────────────────────────────────────────────────
# Primary: z-score (normalized by rolling 30d vol) -> regime-invariant threshold
# Fallback: raw bps if z-score unavailable
Z_SMALL  = 1.0   # |z| <= 1.0 -> "anticipated" -> trade
Z_LARGE  = 1.5   # |z| >= 1.5 -> "surprise" -> skip
# Raw bps thresholds (used as reference / fallback)
THRESH_SMALL  = 5.0
THRESH_MEDIUM = 12.0

def classify(row):
    z = row.get('z_score', np.nan)
    if not pd.isna(z):
        if abs(z) <= Z_SMALL:  return 'small'
        if abs(z) < Z_LARGE:   return 'medium'
        return 'large'
    # fallback to raw bps
    s = row.get('surprise_bps', np.nan)
    if pd.isna(s): return 'unknown'
    return 'small' if abs(s) <= THRESH_SMALL else ('medium' if abs(s) <= THRESH_MEDIUM else 'large')

df['signal_class'] = df.apply(classify, axis=1)


# ── 4. Print full event table ─────────────────────────────────────────────────
print('\n' + '='*92)
print(f'{"Date":<12} {"Regime":<18} {"Raw bps":>8} {"Std30d":>7} {"Z-score":>8} {"Class":<8} {"Net PnL":>9}')
print('-'*92)
for _, r in df.iterrows():
    pnl_str  = f'${r["net_pnl"]:+,.0f}' if not pd.isna(r['net_pnl']) else '     N/A'
    surp_str = f'{r["surprise_bps"]:+.1f}' if not pd.isna(r['surprise_bps']) else '   N/A'
    std_str  = f'{r["roll_std_bps"]:.2f}'  if not pd.isna(r.get("roll_std_bps", np.nan)) else '   N/A'
    z_str    = f'{r["z_score"]:+.2f}'      if not pd.isna(r.get("z_score", np.nan)) else '   N/A'
    print(f'{str(r["date"].date()):<12} {r["regime"]:<18} {surp_str:>8} {std_str:>7} '
          f'{z_str:>8} {r["signal_class"]:<8} {pnl_str:>9}')


# ── 5. Filtered backtest comparison ──────────────────────────────────────────
def summarize(sub, label):
    pnl = sub['net_pnl'].dropna()
    if len(pnl) < 3:
        print(f'  {label}: too few observations ({len(pnl)})')
        return
    total  = pnl.sum()
    mean   = pnl.mean()
    std    = pnl.std()
    sharpe = mean / std * np.sqrt(len(pnl)) if std > 0 else np.nan
    t, p   = stats.ttest_1samp(pnl, 0)
    hit    = (pnl > 0).mean()
    print(f'  {label:<42}  N={len(pnl):2d}  Total=${total:+,.0f}  '
          f'Sharpe={sharpe:.2f}  t={t:.2f}  p={p:.3f}  Hit={hit:.0%}')

print('\n' + '='*80)
print('FILTERED BACKTEST COMPARISON')
print('='*80)
summarize(df,                                                'All 40 events (unfiltered)')
summarize(df[df['year'] != 2022],                            'Ex-2022 (original regime filter)')
summarize(df[df['signal_class'] == 'small'],                 f'Z-score small (|z| <= {Z_SMALL})')
summarize(df[df['signal_class'].isin(['small','medium'])],   f'Z-score small+medium (|z| < {Z_LARGE})')

for label, date_str in [('Dec-18-2024','2024-12-18'), ('Jun-16-2021','2021-06-16')]:
    row = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
    if not row.empty:
        r = row.iloc[0]
        action = 'SKIP' if r['signal_class'] != 'small' else 'TRADE'
        z_val = r.get('z_score', np.nan)
        print(f'\n  {label}: raw={r["surprise_bps"]:+.1f} bps | '
              f'z={z_val:+.2f} | class={r["signal_class"].upper()} -> {action}')

print('\nPnL by class (z-score classification):')
by_class = df.groupby('signal_class')['net_pnl'].agg(['count','sum','mean']).round(0)
print(by_class.to_string())


# ── 6. Figures ────────────────────────────────────────────────────────────────
COLOR = {'small':'#2196F3', 'medium':'#FF9800', 'large':'#F44336', 'unknown':'#9E9E9E'}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: DGS2 surprise bar chart per event
ax = axes[0]
x   = range(len(df))
col = df['signal_class'].map(COLOR)
ax.bar(x, df['surprise_bps'].fillna(0), color=col, edgecolor='white', linewidth=0.4)
ax.axhline(0,              color='black',   linewidth=0.7)
ax.axhline( THRESH_SMALL,  color='#2196F3', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-THRESH_SMALL,  color='#2196F3', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline( THRESH_MEDIUM, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-THRESH_MEDIUM, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xticks(list(x))
ax.set_xticklabels([d[:7] for d in df['date'].dt.strftime('%Y-%m-%d')], rotation=90, fontsize=5.5)
ax.set_ylabel('2Y Treasury change on FOMC day (bps)')
ax.set_title('Rate-Path Surprise per FOMC Event\n(DGS2 change: day-before vs FOMC day)')
patches = [
    mpatches.Patch(color='#2196F3', label=f'Small (|s| <= {THRESH_SMALL} bps) -> trade'),
    mpatches.Patch(color='#FF9800', label=f'Medium ({THRESH_SMALL}<|s|<={THRESH_MEDIUM}) -> skip'),
    mpatches.Patch(color='#F44336', label=f'Large (|s| > {THRESH_MEDIUM} bps) -> skip'),
]
ax.legend(handles=patches, fontsize=7, loc='upper left')

# Right: scatter surprise vs PnL
ax2 = axes[1]
mask = df['net_pnl'].notna() & df['surprise_bps'].notna()
if mask.sum() > 5:
    xs = df.loc[mask,'surprise_bps']
    ys = df.loc[mask,'net_pnl']
    cs = df.loc[mask,'signal_class'].map(COLOR)
    ax2.scatter(xs, ys, c=cs, s=55, edgecolors='white', linewidth=0.4, zorder=3)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    for t_val, c_val in [(THRESH_SMALL,'#2196F3'), (THRESH_MEDIUM,'#FF9800')]:
        ax2.axvline( t_val, color=c_val, linestyle='--', linewidth=1, alpha=0.7)
        ax2.axvline(-t_val, color=c_val, linestyle='--', linewidth=1, alpha=0.7)
    m, b_int, r, p, _ = stats.linregress(xs, ys)
    xs_line = np.linspace(xs.min(), xs.max(), 100)
    ax2.plot(xs_line, m*xs_line + b_int, 'k--', linewidth=1, alpha=0.7,
             label=f'OLS: {m:.0f}$/bps  R²={r**2:.2f}')
    # Annotate Dec-18
    d18_row = df[(df['date'].dt.strftime('%Y-%m-%d')=='2024-12-18') & mask]
    if not d18_row.empty:
        ax2.annotate('Dec-18\n2024',
                     xy=(d18_row['surprise_bps'].values[0], d18_row['net_pnl'].values[0]),
                     xytext=(8, 30), textcoords='offset points', fontsize=7,
                     arrowprops=dict(arrowstyle='->', lw=0.8))
    ax2.set_xlabel('2Y Treasury surprise on FOMC day (bps)')
    ax2.set_ylabel('Strategy net PnL ($)')
    ax2.set_title('Rate-Path Surprise vs Strategy PnL')
    ax2.legend(fontsize=8)
else:
    ax2.text(0.5, 0.5, 'PnL data not available\n(SPY parquet files missing)',
             ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='gray')
    ax2.set_title('Surprise vs PnL')

plt.tight_layout()
plt.savefig('figures/fomc_ois_surprise.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nFigure saved: figures/fomc_ois_surprise.png')

df.to_csv('fomc_surprise_table.csv', index=False)
print('Table saved: fomc_surprise_table.csv')
