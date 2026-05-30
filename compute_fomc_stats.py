import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('fomc_surprise_table.csv')
df['date'] = pd.to_datetime(df['date'])

def summarize(sub, label):
    pnl = sub['net_pnl'].dropna()
    n = len(pnl)
    if n < 3:
        print(f'  {label}: too few ({n})')
        return
    total = pnl.sum()
    mean = pnl.mean()
    std = pnl.std()
    sharpe = mean / std * np.sqrt(n) if std > 0 else np.nan
    t, p = stats.ttest_1samp(pnl, 0)
    hit = (pnl > 0).mean()
    print(f'  {label:<50}  N={n:2d}  Total=${total:+,.0f}  Sharpe={sharpe:.2f}  p={p:.3f}  Hit={hit:.0%}')

print('=== REGIME-CONDITIONED ANALYSIS ===')
all_2022 = df['year'] == 2022
summarize(df, 'All 40 events (unfiltered)')
summarize(df[~all_2022], 'Hold/Cut regimes (ex-2022), 32 events')
dec18 = df['date'].dt.strftime('%Y-%m-%d') == '2024-12-18'
summarize(df[~all_2022 & ~dec18], 'Hold/Cut ex Dec-18-2024, 31 events')
summarize(df[all_2022], 'Hiking (2022), 8 events')

print()
print('=== OIS Z-SCORE FILTER ANALYSIS ===')
small = df['signal_class'] == 'small'
not_small = ~small
summarize(df[small], '|z| <= 1.0 (small surprise)')
summarize(df[not_small], '|z| > 1.0 (medium/large surprise)')
summarize(df, 'No filter (all 40)')

print()
print('=== PER-YEAR STATS ===')
for yr in sorted(df['year'].unique()):
    ysub = df[df['year'] == yr]
    pnl = ysub['net_pnl'].dropna()
    n = len(pnl)
    total = pnl.sum()
    std = pnl.std()
    mean = pnl.mean()
    sharpe = mean/std*np.sqrt(n) if (std > 0 and n > 1) else np.nan
    hit = (pnl > 0).mean()
    tc_mean = ysub['tc'].dropna().mean()
    print(f'  {yr}: N={n}  Total=${total:+,.0f}  Sharpe={sharpe:.2f}  Hit={hit:.0%}  AvgTC=${tc_mean:.1f}')

print()
print('=== INDIVIDUAL EVENT DATA ===')
for _, r in df.iterrows():
    pnl_str = f'${r["net_pnl"]:+,.0f}' if not pd.isna(r['net_pnl']) else 'N/A'
    print(f'  {str(r["date"].date())}  {r["regime"]:<20}  z={r["z_score"]:+.2f}  class={r["signal_class"]:<8}  {pnl_str}')
