"""
FOMC Rate-Path Surprise Filter — Two-Track Framework
======================================================
SEP days (Mar/Jun/Sep/Dec):  z = (SEP_median - DGS2_prev)*100 / max(roll_std_DGS2, 8 bps)
    SEP_median  = dot-plot year-end FFR median, released with the statement at 14:00
    DGS2_prev   = 2Y Treasury yield at prior-day close (FRED, published ~17:00 day before)
    Both inputs are available at 14:01 when the trade fires. DGS2_fomc (end-of-day on
    meeting date, published 17:00) is recorded for reference but NOT used in the filter.
    Floor of 8 bps on rolling vol prevents ZLB-era (2021) z-score inflation.

Non-SEP days: always trade (no filter).
    The rate decision alone is the only new information; it is almost always fully priced.
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

Path('FOMC/figures').mkdir(exist_ok=True)
DATA_DIR = Path('data/raw')
ET       = 'America/New_York'
BOOK_COLS    = ['bid_px_00','ask_px_00','bid_sz_00','ask_sz_00']
HOLD_MINS      = 15
NOTIONAL       = 100_000
SEP_FLOOR_STD  = 8.0   # bps — floor on rolling DGS2 vol; prevents ZLB inflation

# ── FOMC / Control date lists ─────────────────────────────────────────────────
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
    '2025-06-18':'hold',       '2025-07-30':'hold',       '2025-09-17':'hold',
    '2025-10-29':'hold',       '2025-12-10':'cut-25',
}

# ── SEP dot-plot: year-end FFR median projection at each quarterly meeting ────
# Source: FOMC Summary of Economic Projections (released same time as statement)
SEP_DATA = {
    '2021-03-17': 0.125, '2021-06-16': 0.125, '2021-09-22': 0.125, '2021-12-15': 0.875,
    '2022-03-16': 1.875, '2022-06-15': 3.375, '2022-09-21': 4.375, '2022-12-14': 5.125,
    '2023-03-22': 5.125, '2023-06-14': 5.625, '2023-09-20': 5.625, '2023-12-13': 4.625,
    '2024-03-20': 4.625, '2024-06-12': 5.125, '2024-09-18': 4.375, '2024-12-18': 3.875,
    '2025-03-19': 3.875, '2025-06-18': 3.875,
}
SEP_DATES = set(SEP_DATA.keys())


def regime_to_move_bps(regime):
    """Extract the actual rate change in bps from the regime string."""
    if 'hike' in regime:
        try: return int(regime.split('+')[1].split('+')[0])
        except: return 0
    if 'cut' in regime:
        try: return -int(regime.split('-')[1].split('+')[0])
        except: return 0
    if regime in ('hold', 'taper'): return 0
    return None  # '?' = unknown future meeting


# ── Trade engine ──────────────────────────────────────────────────────────────
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


# ── 1a. Fetch DGS2 (2-Year Treasury) from FRED — used for SEP-day track ──────
FRED_DGS2_URL   = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2'
FRED_DGS1MO_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO'
FRED_EFFR_URL   = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=EFFR'

print('Fetching DGS2 (2-Year Treasury) from FRED...')
try:
    with urllib.request.urlopen(FRED_DGS2_URL, timeout=20) as r:
        dgs2 = pd.read_csv(
            r, index_col='observation_date', parse_dates=True, na_values=['.']
        )['DGS2'].dropna()
    dgs2 = dgs2.loc['2020-01-01':]
    dgs2_diff     = dgs2.diff() * 100
    roll_std_dgs2 = dgs2_diff.rolling(30, min_periods=10).std()
    print(f'  Loaded {len(dgs2)} obs ({dgs2.index[0].date()} – {dgs2.index[-1].date()})')
except Exception as e:
    dgs2 = None
    print(f'  DGS2 fetch failed: {e}')

# ── 1b. Fetch DGS1MO (1-Month Treasury) — implied move for non-SEP track ─────
print('Fetching DGS1MO (1-Month Treasury) from FRED...')
try:
    with urllib.request.urlopen(FRED_DGS1MO_URL, timeout=20) as r:
        dgs1mo = pd.read_csv(
            r, index_col='observation_date', parse_dates=True, na_values=['.']
        )['DGS1MO'].dropna()
    dgs1mo = dgs1mo.loc['2020-01-01':]
    dgs1mo_diff     = dgs1mo.diff() * 100
    roll_std_dgs1mo = dgs1mo_diff.rolling(30, min_periods=10).std()
    print(f'  Loaded {len(dgs1mo)} obs')
except Exception as e:
    dgs1mo = None
    print(f'  DGS1MO fetch failed: {e}')

# ── 1c. Fetch EFFR (Effective Fed Funds Rate) — current rate for non-SEP ─────
print('Fetching EFFR from FRED...')
try:
    with urllib.request.urlopen(FRED_EFFR_URL, timeout=20) as r:
        effr = pd.read_csv(
            r, index_col='observation_date', parse_dates=True, na_values=['.']
        )['EFFR'].dropna()
    effr = effr.loc['2020-01-01':]
    print(f'  Loaded {len(effr)} obs')
except Exception as e:
    effr = None
    print(f'  EFFR fetch failed: {e}')


# ── 2. Per-event PnL and surprise (two-track) ─────────────────────────────────
print('\nComputing per-event PnL and surprise...')
records = []
for fomc_str in ALL_FOMC:
    fomc_ts = pd.Timestamp(fomc_str)
    rec = {
        'date':          fomc_str,
        'year':          fomc_ts.year,
        'regime':        REGIME.get(fomc_str, '?'),
        'surprise_type': 'SEP' if fomc_str in SEP_DATES else 'target',
    }

    # ── Surprise measure ─────────────────────────────────────────────────────
    if fomc_str in SEP_DATES:
        # SEP track: guidance gap = SEP_median - DGS2_prev (both available at 14:01)
        # DGS2_fomc is end-of-day (published ~17:00 on meeting date) — recorded for
        # reference only; not used in filter to preserve real-time discipline.
        if dgs2 is not None:
            prev_idx  = dgs2.index[dgs2.index < fomc_ts]
            fomc_rate = dgs2.get(fomc_ts, np.nan)   # informational only
            if len(prev_idx) > 0:
                prev_rate  = dgs2[prev_idx[-1]]
                sep_median = SEP_DATA[fomc_str]
                raw_bps    = (sep_median - prev_rate) * 100   # guidance gap, bps
                vol_raw    = roll_std_dgs2.get(prev_idx[-1], np.nan)  # prev-day vol
                vol        = max(vol_raw, SEP_FLOOR_STD) if not np.isnan(vol_raw) else SEP_FLOOR_STD
                z_score    = raw_bps / vol
                rec['dgs2_prev']    = round(prev_rate, 3)
                rec['dgs2_fomc']    = round(fomc_rate, 3) if not np.isnan(fomc_rate) else np.nan
                rec['sep_median']   = round(sep_median, 3)
                rec['surprise_bps'] = round(raw_bps, 1)
                rec['roll_std_bps'] = round(vol, 2)
                rec['z_score']      = round(z_score, 2)
            else:
                for k in ['dgs2_prev','dgs2_fomc','sep_median','surprise_bps','roll_std_bps','z_score']:
                    rec[k] = np.nan
        else:
            for k in ['dgs2_prev','dgs2_fomc','sep_median','surprise_bps','roll_std_bps','z_score']:
                rec[k] = np.nan
    else:
        # Non-SEP track: target surprise = actual_move - implied_move
        # implied_move = (DGS1MO_prev - EFFR_prev) * 100  [in bps]
        if dgs1mo is not None and effr is not None:
            prev_1mo_idx  = dgs1mo.index[dgs1mo.index < fomc_ts]
            prev_effr_idx = effr.index[effr.index < fomc_ts]
            if len(prev_1mo_idx) > 0 and len(prev_effr_idx) > 0:
                dgs1mo_prev      = dgs1mo[prev_1mo_idx[-1]]
                effr_prev        = effr[prev_effr_idx[-1]]
                implied_move_bps = (dgs1mo_prev - effr_prev) * 100
                actual_move_bps  = regime_to_move_bps(rec['regime'])
                rec['dgs1mo_prev']      = round(dgs1mo_prev, 3)
                rec['effr_prev']        = round(effr_prev, 3)
                rec['implied_move_bps'] = round(implied_move_bps, 1)
                rec['actual_move_bps']  = actual_move_bps
                if actual_move_bps is not None:
                    raw_bps = actual_move_bps - implied_move_bps
                    vol     = roll_std_dgs1mo.get(fomc_ts, np.nan)
                    if pd.isna(vol) and len(prev_1mo_idx) > 0:
                        vol = roll_std_dgs1mo.get(prev_1mo_idx[-1], np.nan)
                    z_score = raw_bps / vol if (not pd.isna(vol) and vol > 0) else np.nan
                    rec['surprise_bps'] = round(raw_bps, 1)
                    rec['roll_std_bps'] = round(vol, 2) if not pd.isna(vol) else np.nan
                    rec['z_score']      = round(z_score, 2) if not np.isnan(z_score) else np.nan
                else:
                    for k in ['surprise_bps','roll_std_bps','z_score']:
                        rec[k] = np.nan
            else:
                for k in ['surprise_bps','roll_std_bps','z_score']:
                    rec[k] = np.nan
        else:
            for k in ['surprise_bps','roll_std_bps','z_score']:
                rec[k] = np.nan

    # ── PnL from trade engine ────────────────────────────────────────────────
    try:
        bars = load_day_bars(fomc_str)
        if bars is not None:
            t = trade_day(fomc_str, bars)
            if t:
                rec['net_pnl']   = t['net_pnl']
                rec['gross_pnl'] = t['gross_pnl']
                rec['tc']        = t['tc']
                rec['ret_14']    = t['ret_14_bps']
    except Exception as ex:
        print(f'  Trade engine error on {fomc_str}: {ex}')

    records.append(rec)

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])

# Ensure all expected columns exist
for c in ['net_pnl','gross_pnl','tc','ret_14','surprise_bps','dgs2_prev','dgs2_fomc',
          'sep_median','dgs1mo_prev','effr_prev','implied_move_bps','actual_move_bps',
          'roll_std_bps','z_score']:
    if c not in df.columns:
        df[c] = np.nan


# ── 3. Classify events ────────────────────────────────────────────────────────
Z_SMALL        = 1.0
Z_LARGE        = 1.5
THRESH_SMALL   = 5.0
THRESH_MEDIUM  = 12.0

def classify(row):
    z = row.get('z_score', np.nan)
    if not pd.isna(z):
        if abs(z) <= Z_SMALL: return 'small'
        if abs(z) <  Z_LARGE: return 'medium'
        return 'large'
    s = row.get('surprise_bps', np.nan)
    if pd.isna(s): return 'unknown'
    return 'small' if abs(s) <= THRESH_SMALL else ('medium' if abs(s) <= THRESH_MEDIUM else 'large')

df['signal_class'] = df.apply(classify, axis=1)


# ── 4. Print full event table ─────────────────────────────────────────────────
print('\n' + '='*100)
print(f'{"Date":<12} {"Type":<7} {"Regime":<18} {"Surp bps":>9} {"Std30d":>7} {"Z-score":>8} {"Class":<8} {"Net PnL":>9}')
print('-'*100)
for _, r in df.iterrows():
    pnl_str  = f'${r["net_pnl"]:+,.0f}' if not pd.isna(r.get('net_pnl', np.nan)) else '     N/A'
    surp_str = f'{r["surprise_bps"]:+.1f}' if not pd.isna(r.get('surprise_bps', np.nan)) else '   N/A'
    std_str  = f'{r["roll_std_bps"]:.2f}' if not pd.isna(r.get('roll_std_bps', np.nan)) else '   N/A'
    z_str    = f'{r["z_score"]:+.2f}'     if not pd.isna(r.get('z_score', np.nan)) else '   N/A'
    stype    = r.get('surprise_type', '?')
    print(f'{str(r["date"].date()):<12} {stype:<7} {r["regime"]:<18} {surp_str:>9} {std_str:>7} '
          f'{z_str:>8} {r["signal_class"]:<8} {pnl_str:>9}')


# ── 5. Filtered backtest comparison ──────────────────────────────────────────
def summarize(sub, label):
    pnl = sub['net_pnl'].dropna()
    if len(pnl) < 3:
        print(f'  {label}: too few obs ({len(pnl)})')
        return
    total  = pnl.sum()
    mean   = pnl.mean()
    std    = pnl.std()
    sharpe = mean / std * np.sqrt(len(pnl)) if std > 0 else np.nan
    t, p   = stats.ttest_1samp(pnl, 0)
    hit    = (pnl > 0).mean()
    print(f'  {label:<46}  N={len(pnl):2d}  Total=${total:+,.0f}  '
          f'Sharpe={sharpe:.2f}  t={t:.2f}  p={p:.3f}  Hit={hit:.0%}')

print('\n' + '='*80)
print('FILTERED BACKTEST COMPARISON')
print('='*80)
summarize(df,                                                'All 40 events (unfiltered)')
summarize(df[df['year'] != 2022],                            'Ex-2022 (regime filter)')
summarize(df[(df['year'] != 2022) &
             ~((df['date'].dt.strftime('%Y-%m-%d') == '2024-12-18'))],
                                                             'Ex-2022 ex Dec-18-2024')

print('\n--- Two-track z-score filter ---')
summarize(df[df['signal_class'] == 'small'],                 f'Two-track |z|<=1.0 (small, trade)')
summarize(df[df['signal_class'].isin(['medium','large'])],   f'Two-track |z|>1.0  (medium+large, skip)')

print('\n--- SEP-only filter (trade all non-SEP) ---')
sep_small    = df[(df['surprise_type']=='SEP') & (df['signal_class']=='small')]
non_sep_all  = df[df['surprise_type']=='target']
sep_filtered = pd.concat([sep_small, non_sep_all]).sort_values('date')
sep_excluded = df[(df['surprise_type']=='SEP') & (df['signal_class'].isin(['medium','large']))]
summarize(sep_filtered,  'SEP |z|<=1.0 + all non-SEP (29 events)')
summarize(sep_excluded,  'Excluded SEP events (|z|>1.0)')
summarize(sep_small,     'SEP small only (7 events)')
summarize(non_sep_all,   'Non-SEP all (22 events)')

print('\nPnL by surprise class:')
by_class = df.groupby('signal_class')['net_pnl'].agg(['count','sum','mean']).round(0)
print(by_class.to_string())

print('\nPnL by surprise type:')
by_type = df.groupby('surprise_type')['net_pnl'].agg(['count','sum','mean']).round(0)
print(by_type.to_string())

print('\nFilter performance by type:')
for stype in ['SEP', 'target']:
    sub = df[df['surprise_type'] == stype]
    small = sub[sub['signal_class'] == 'small']
    large = sub[sub['signal_class'].isin(['medium','large'])]
    print(f'  {stype}: small={len(small)} (total=${small["net_pnl"].sum():+,.0f}), '
          f'medium+large={len(large)} (total=${large["net_pnl"].sum():+,.0f})')

for label, date_str in [('Dec-18-2024','2024-12-18'), ('Jun-16-2021','2021-06-16')]:
    row = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
    if not row.empty:
        r = row.iloc[0]
        action = 'SKIP' if r['signal_class'] != 'small' else 'TRADE'
        z_val = r.get('z_score', np.nan)
        z_str = f'{z_val:+.2f}' if not pd.isna(z_val) else 'N/A'
        print(f'\n  {label}: surprise={r.get("surprise_bps", np.nan):+.1f} bps | '
              f'z={z_str} | class={r["signal_class"].upper()} -> {action}')


# ── 6. Figures ────────────────────────────────────────────────────────────────
COLOR = {'small':'#2196F3', 'medium':'#FF9800', 'large':'#F44336', 'unknown':'#9E9E9E'}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: surprise bar chart, colored by classification; shape indicates track
ax = axes[0]
x   = range(len(df))
col = df['signal_class'].map(COLOR)
ax.bar(x, df['surprise_bps'].fillna(0), color=col, edgecolor='white', linewidth=0.4)
ax.axhline(0,              color='black',   linewidth=0.7)
ax.axhline( THRESH_SMALL,  color='#2196F3', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-THRESH_SMALL,  color='#2196F3', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline( THRESH_MEDIUM, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-THRESH_MEDIUM, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)

# Mark non-SEP events with a small dot at the top of each bar
for i, (_, r) in enumerate(df.iterrows()):
    if r.get('surprise_type') == 'target' and not pd.isna(r.get('surprise_bps', np.nan)):
        y_mark = r['surprise_bps'] + (3 if r['surprise_bps'] >= 0 else -3)
        ax.scatter(i, y_mark, marker='^', s=15, color='black', zorder=5, linewidths=0)

ax.set_xticks(list(x))
ax.set_xticklabels([d[:7] for d in df['date'].dt.strftime('%Y-%m-%d')], rotation=90, fontsize=5.5)
ax.set_ylabel('Surprise measure (bps)')
ax.set_title('FOMC Surprise per Event — Two-Track\n(SEP: SEP_median−DGS2_prev | Non-SEP: always trade; ○ = SEP)')
patches = [
    mpatches.Patch(color='#2196F3', label=f'Small (|z|≤{Z_SMALL}) → trade'),
    mpatches.Patch(color='#FF9800', label=f'Medium ({Z_SMALL}<|z|<{Z_LARGE}) → skip'),
    mpatches.Patch(color='#F44336', label=f'Large (|z|≥{Z_LARGE}) → skip'),
]
ax.legend(handles=patches, fontsize=7, loc='upper left')

# Right: scatter surprise vs PnL
ax2 = axes[1]
mask = df['net_pnl'].notna() & df['surprise_bps'].notna()
if mask.sum() > 5:
    for stype, marker, label_sfx in [('SEP','o','SEP'), ('target','^','Non-SEP')]:
        m_sub = mask & (df['surprise_type'] == stype)
        if m_sub.sum() > 0:
            xs_s = df.loc[m_sub,'surprise_bps']
            ys_s = df.loc[m_sub,'net_pnl']
            cs_s = df.loc[m_sub,'signal_class'].map(COLOR)
            ax2.scatter(xs_s, ys_s, c=cs_s, s=55, marker=marker,
                       edgecolors='white', linewidth=0.4, zorder=3, label=label_sfx)
    xs_all = df.loc[mask,'surprise_bps']
    ys_all = df.loc[mask,'net_pnl']
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    m_ols, b_int, r_val, p_val, _ = stats.linregress(xs_all, ys_all)
    xs_line = np.linspace(xs_all.min(), xs_all.max(), 100)
    ax2.plot(xs_line, m_ols*xs_line + b_int, 'k--', linewidth=1, alpha=0.7,
             label=f'OLS: {m_ols:.0f}$/bps  R²={r_val**2:.2f}')
    d18_row = df[(df['date'].dt.strftime('%Y-%m-%d')=='2024-12-18') & mask]
    if not d18_row.empty:
        ax2.annotate('Dec-18\n2024',
                     xy=(d18_row['surprise_bps'].values[0], d18_row['net_pnl'].values[0]),
                     xytext=(8, 30), textcoords='offset points', fontsize=7,
                     arrowprops=dict(arrowstyle='->', lw=0.8))
    ax2.set_xlabel('Surprise (bps) — guidance gap (SEP_median−DGS2_prev) for SEP; N/A for non-SEP')
    ax2.set_ylabel('Strategy net PnL ($)')
    ax2.set_title('Surprise vs Strategy PnL\n(○ SEP events, ▲ Non-SEP events)')
    ax2.legend(fontsize=8)
else:
    ax2.text(0.5, 0.5, 'PnL data not available', ha='center', va='center',
             transform=ax2.transAxes, fontsize=10, color='gray')
    ax2.set_title('Surprise vs PnL')

plt.tight_layout()
plt.savefig('FOMC/figures/fomc_ois_surprise.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nFigure saved: FOMC/figures/fomc_ois_surprise.png')

# ── Save CSV ──────────────────────────────────────────────────────────────────
col_order = ['date','year','regime','surprise_type',
             'dgs2_prev','sep_median','dgs2_fomc',
             'dgs1mo_prev','effr_prev','implied_move_bps','actual_move_bps',
             'surprise_bps','roll_std_bps','z_score','signal_class',
             'ret_14','gross_pnl','tc','net_pnl']
out_cols = [c for c in col_order if c in df.columns]
df[out_cols].to_csv('FOMC/fomc_surprise_table.csv', index=False)
print('Table saved: FOMC/fomc_surprise_table.csv')
