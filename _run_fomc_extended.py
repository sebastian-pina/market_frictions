import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

DATA_DIR = Path('data/raw')
Path('figures').mkdir(exist_ok=True)
ET = 'America/New_York'

# ── Date definitions ─────────────────────────────────────────────────────────
FOMC_2021 = ['2021-01-27','2021-03-17','2021-04-28','2021-06-16',
             '2021-07-28','2021-09-22','2021-11-03','2021-12-15']
CTRL_2021 = ['2021-01-20','2021-03-10','2021-04-21','2021-06-09',
             '2021-07-21','2021-09-15','2021-10-27','2021-12-08']

FOMC_2022 = ['2022-01-26','2022-03-16','2022-05-04','2022-06-15',
             '2022-07-27','2022-09-21','2022-11-02','2022-12-14']
CTRL_2022 = ['2022-01-19','2022-03-09','2022-04-27','2022-06-08',
             '2022-07-20','2022-09-14','2022-10-26','2022-12-07']

FOMC_2023 = ['2023-02-01','2023-03-22','2023-05-03','2023-06-14',
             '2023-07-26','2023-09-20','2023-11-01','2023-12-13']
CTRL_2023 = ['2023-01-25','2023-03-15','2023-04-26','2023-06-07',
             '2023-07-19','2023-09-13','2023-10-25','2023-12-06']

FOMC_2024 = ['2024-01-31','2024-03-20','2024-05-01','2024-06-12',
             '2024-07-31','2024-09-18','2024-11-07','2024-12-18']
CTRL_2024 = ['2024-01-24','2024-03-13','2024-04-24','2024-06-05',
             '2024-07-24','2024-09-11','2024-10-30','2024-12-11']

FOMC_2025 = ['2025-01-29','2025-03-19','2025-05-07','2025-06-18',
             '2025-07-30','2025-09-17','2025-10-29','2025-12-10']
CTRL_2025 = ['2025-01-22','2025-03-12','2025-04-30','2025-06-11',
             '2025-07-23','2025-09-10','2025-10-22','2025-12-03']

ALL_FOMC    = FOMC_2021 + FOMC_2022 + FOMC_2023 + FOMC_2024 + FOMC_2025
ALL_CONTROL = CTRL_2021 + CTRL_2022 + CTRL_2023 + CTRL_2024 + CTRL_2025

HOLD_MINS = 15
NOTIONAL  = 100_000
BOOK_COLS = ['bid_px_00','ask_px_00','bid_sz_00','ask_sz_00']

# Known high-information events
SURPRISE_DATES = {
    '2024-09-18': '-50bps surprise',
    '2023-12-13': 'dovish pivot (dot plot)',
    '2025-09-17': 'unknown',
}


def load_day_bars(date_str):
    p = DATA_DIR / 'SPY' / 'mbp-10' / f'{date_str}.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
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
    entry_bar  = bars[mask].iloc[0]
    entry_time = bars[mask].index[0]
    ret_14 = entry_bar['ret_bps']
    if ret_14 == 0 or pd.isna(ret_14):
        return None
    direction   = -1 if ret_14 > 0 else 1
    entry_price  = entry_bar['mid_close']
    entry_spread = entry_bar['spread']
    future = bars[bars.index >= entry_time + pd.Timedelta(minutes=HOLD_MINS)]
    if future.empty:
        return None
    exit_bar   = future.iloc[0]
    shares     = NOTIONAL / entry_price
    gross_pnl  = direction * (exit_bar['mid_close'] - entry_price) * shares
    tc         = (entry_spread / 2 + exit_bar['spread'] / 2) * shares
    return {
        'date':        date_str,
        'year':        date_str[:4],
        'direction':   direction,
        'ret_14_bps':  round(ret_14, 2),
        'entry_price': round(entry_price, 4),
        'exit_price':  round(exit_bar['mid_close'], 4),
        'gross_pnl':   round(gross_pnl, 2),
        'tc':          round(tc, 2),
        'net_pnl':     round(gross_pnl - tc, 2),
        'surprise':    date_str in SURPRISE_DATES,
    }


def load_and_trade(dates, label):
    trades = []
    for d in dates:
        try:
            bars = load_day_bars(d)
            t = trade_day(d, bars)
            if t:
                trades.append(t)
        except FileNotFoundError:
            print(f'  Missing {d}')
    if not trades:
        print(f'  {label}: no trades')
        return None
    df = pd.DataFrame(trades)
    sharpe = (df['net_pnl'].mean() / df['net_pnl'].std() * np.sqrt(8)
              if df['net_pnl'].std() > 0 else float('nan'))
    tstat, pval = stats.ttest_1samp(df['net_pnl'], 0)
    print(f'\n  [{label}]  N={len(df)}  PnL=${df["net_pnl"].sum():.0f}  '
          f'Hit={(df["net_pnl"]>0).mean():.0%}  '
          f'Sharpe={sharpe:.2f}  '
          f't={tstat:.2f}  p={pval:.3f}  '
          f'AvgTC=${df["tc"].mean():.1f}')
    print(df[['date','ret_14_bps','direction','gross_pnl','tc','net_pnl']].to_string(index=False))
    return df


# ── Per-year results ──────────────────────────────────────────────────────────
print('\n' + '='*60)
print('  SPY FOMC CONTRARIAN FADE — Extended (2023-2025)')
print('='*60)

print('\n--- Pre-FOMC spread widening (13:30-14:00 vs 11:00-13:00) ---')
for year, dates in [('2021', FOMC_2021), ('2022', FOMC_2022), ('2023', FOMC_2023),
                    ('2024', FOMC_2024), ('2025', FOMC_2025)]:
    ratios = []
    for d in dates:
        try:
            b    = load_day_bars(d)
            base = b.between_time('11:00','13:00')['spread'].mean()
            pre  = b.between_time('13:30','14:00')['spread'].mean()
            if base > 0:
                ratios.append(pre / base)
        except FileNotFoundError:
            pass
    if ratios:
        print(f'  {year}: spread ratio pre/baseline = {np.mean(ratios):.3f}  '
              f'(+{(np.mean(ratios)-1)*100:.1f}%)')

print('\n--- Control spread widening ---')
for year, dates in [('2021', CTRL_2021), ('2022', CTRL_2022), ('2023', CTRL_2023),
                    ('2024', CTRL_2024), ('2025', CTRL_2025)]:
    ratios = []
    for d in dates:
        try:
            b    = load_day_bars(d)
            base = b.between_time('11:00','13:00')['spread'].mean()
            pre  = b.between_time('13:30','14:00')['spread'].mean()
            if base > 0:
                ratios.append(pre / base)
        except FileNotFoundError:
            pass
    if ratios:
        print(f'  {year}: spread ratio pre/baseline = {np.mean(ratios):.3f}  '
              f'(+{(np.mean(ratios)-1)*100:.1f}%)')

# Per-year FOMC
results = {}
for year, dates in [('2021', FOMC_2021), ('2022', FOMC_2022), ('2023', FOMC_2023),
                    ('2024', FOMC_2024), ('2025', FOMC_2025)]:
    print(f'\n{"="*60}')
    print(f'  FOMC {year}')
    print('='*60)
    results[year] = load_and_trade(dates, year)

# Per-year Control
ctrl_results = {}
for year, dates in [('ctrl_2021', CTRL_2021), ('ctrl_2022', CTRL_2022),
                    ('ctrl_2023', CTRL_2023), ('ctrl_2024', CTRL_2024), ('ctrl_2025', CTRL_2025)]:
    ctrl_results[year] = load_and_trade(dates, year)

# ── Aggregate across all years ────────────────────────────────────────────────
print('\n' + '='*60)
print('  AGGREGATE: All 40 FOMC events (2021-2025)')
print('='*60)
all_fomc_dfs  = [df for df in results.values() if df is not None]
all_ctrl_dfs  = [df for df in ctrl_results.values() if df is not None]

if all_fomc_dfs:
    agg = pd.concat(all_fomc_dfs).sort_values('date')
    sharpe_agg = (agg['net_pnl'].mean() / agg['net_pnl'].std() * np.sqrt(8)
                  if agg['net_pnl'].std() > 0 else float('nan'))
    tstat, pval = stats.ttest_1samp(agg['net_pnl'], 0)
    print(f'\n  FOMC  N={len(agg)}  PnL=${agg["net_pnl"].sum():.0f}  '
          f'Hit={(agg["net_pnl"]>0).mean():.0%}  '
          f'Sharpe={sharpe_agg:.2f}  '
          f't={tstat:.2f}  p={pval:.3f}  '
          f'AvgTC=${agg["tc"].mean():.1f}')

if all_ctrl_dfs:
    ctrl = pd.concat(all_ctrl_dfs).sort_values('date')
    sharpe_ctrl = (ctrl['net_pnl'].mean() / ctrl['net_pnl'].std() * np.sqrt(8)
                   if ctrl['net_pnl'].std() > 0 else float('nan'))
    tstat_c, pval_c = stats.ttest_1samp(ctrl['net_pnl'], 0)
    print(f'  Ctrl  N={len(ctrl)}  PnL=${ctrl["net_pnl"].sum():.0f}  '
          f'Hit={(ctrl["net_pnl"]>0).mean():.0%}  '
          f'Sharpe={sharpe_ctrl:.2f}  '
          f't={tstat_c:.2f}  p={pval_c:.3f}  '
          f'AvgTC=${ctrl["tc"].mean():.1f}')

# ── Excluding Dec-18 2024 (known regime break) ───────────────────────────────
if all_fomc_dfs:
    agg_ex = agg[agg['date'] != '2024-12-18']
    sharpe_ex = (agg_ex['net_pnl'].mean() / agg_ex['net_pnl'].std() * np.sqrt(8)
                 if agg_ex['net_pnl'].std() > 0 else float('nan'))
    tstat_ex, pval_ex = stats.ttest_1samp(agg_ex['net_pnl'], 0)
    print(f'\n  FOMC ex-Dec18  N={len(agg_ex)}  PnL=${agg_ex["net_pnl"].sum():.0f}  '
          f'Hit={(agg_ex["net_pnl"]>0).mean():.0%}  '
          f'Sharpe={sharpe_ex:.2f}  '
          f't={tstat_ex:.2f}  p={pval_ex:.3f}')

# ── Cumulative PnL plot ───────────────────────────────────────────────────────
if all_fomc_dfs:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: cumulative PnL by year
    ax = axes[0]
    colors = {'2021': '#9C27B0', '2022': '#F44336', '2023': '#2196F3',
              '2024': '#FF9800', '2025': '#4CAF50'}
    cum_total = 0
    for year, df in [(y, results[y]) for y in ['2021','2022','2023','2024','2025']
                     if results.get(y) is not None]:
        cum = df['net_pnl'].cumsum() + cum_total
        ax.plot(range(cum_total, cum_total + len(cum)), cum.values,
                marker='o', label=year, color=colors[year])
        cum_total += len(df)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('SPY FOMC Contrarian Fade — Cumulative Net PnL (2021–2025)', fontsize=13)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Net PnL ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: per-event bar chart
    ax2 = axes[1]
    agg_sorted = agg.sort_values('date')
    bar_colors = ['#4CAF50' if x > 0 else '#F44336' for x in agg_sorted['net_pnl']]
    ax2.bar(range(len(agg_sorted)), agg_sorted['net_pnl'], color=bar_colors, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(range(len(agg_sorted)))
    ax2.set_xticklabels(agg_sorted['date'].str[2:], rotation=45, ha='right', fontsize=7)
    ax2.set_title('Per-Event Net PnL', fontsize=11)
    ax2.set_ylabel('Net PnL ($)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/fomc_extended_pnl.png', dpi=150)
    plt.close()
    print('\nFigure saved: figures/fomc_extended_pnl.png')

print('\nDone.')
