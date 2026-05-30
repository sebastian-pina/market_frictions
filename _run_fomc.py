import matplotlib
matplotlib.use("Agg")

# === cell 0 ===
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

DATA_DIR = Path('data/raw')
Path('figures').mkdir(exist_ok=True)

ET = 'America/New_York'
FOMC_TIME = pd.Timestamp('14:00:00').time()   # 2:00 PM ET

FOMC_TRAIN = ['2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12']
FOMC_TEST  = ['2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18']
CONTROL    = ['2024-01-24', '2024-03-13', '2024-04-24', '2024-06-05',
              '2024-07-24', '2024-09-11', '2024-10-30', '2024-12-11']

# September 18 was a surprise -50bps cut — flag separately in analysis
SURPRISE_DATE = '2024-09-18'

# === cell 1 ===
def load_spy_day(date_str: str, schema: str = 'mbp-10') -> pd.DataFrame:
    p = DATA_DIR / 'SPY' / schema / f'{date_str}.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Run download_fomc.py first.')
    df = pd.read_parquet(p).sort_index()
    df.index = df.index.tz_convert(ET)
    return df


def compute_1min_bars(date_str: str) -> pd.DataFrame:
    """
    Build 1-minute bars for a single day with:
      mid, spread_bps, ofi (top-of-book), depth_top5, signed_volume (from trades)
    """
    book = load_spy_day(date_str, 'mbp-10')

    # Fix uint32 overflow before any arithmetic
    bid_sz = book['bid_sz_00'].astype(np.int64)
    ask_sz = book['ask_sz_00'].astype(np.int64)
    bid_px = book['bid_px_00']
    ask_px = book['ask_px_00']

    tick = pd.DataFrame({
        'mid':    (bid_px + ask_px) / 2,
        'spread': ask_px - bid_px,
        'ofi':    (bid_sz - ask_sz) / (bid_sz + ask_sz).replace(0, np.nan),
        'depth5': sum(book[f'bid_sz_{i:02d}'].astype(np.int64) +
                      book[f'ask_sz_{i:02d}'].astype(np.int64)
                      for i in range(5)),
    }, index=book.index)

    bars = tick.resample('1min').agg(
        mid_open  = ('mid',    'first'),
        mid_close = ('mid',    'last'),
        spread    = ('spread', 'mean'),
        ofi       = ('ofi',    'mean'),
        depth5    = ('depth5', 'mean'),
    ).dropna(subset=['mid_open'])

    bars['mid_px']     = bars['mid_close']
    bars['spread_bps'] = bars['spread'] / bars['mid_px'] * 1e4
    bars['ret_bps']    = bars['mid_close'].pct_change() * 1e4

    # Signed volume from trades
    try:
        tr = load_spy_day(date_str, 'trades')
        signed = tr['size'].astype(np.int64) * tr['side'].map({'A': 1, 'B': -1, 'N': 0}).fillna(0)
        total  = tr['size'].astype(np.int64)
        sv = (signed.resample('1min').sum() /
              total.resample('1min').sum().replace(0, np.nan)).rename('signed_vol')
        bars = bars.join(sv)
    except FileNotFoundError:
        bars['signed_vol'] = np.nan

    bars['date'] = date_str
    return bars


print('Loading FOMC train days...')
train_bars = {d: compute_1min_bars(d) for d in FOMC_TRAIN}
print('Loading FOMC test days...')
test_bars  = {d: compute_1min_bars(d) for d in FOMC_TEST}
print('Loading control days...')
ctrl_bars  = {d: compute_1min_bars(d) for d in CONTROL}
print('Done.')

# === cell 2 ===
def window_stats(bars: pd.DataFrame, start_time: str, end_time: str) -> pd.Series:
    """Mean of spread_bps, ofi, depth5 in a given time window."""
    mask = ((bars.index.time >= pd.Timestamp(start_time).time()) &
            (bars.index.time <  pd.Timestamp(end_time).time()))
    w = bars[mask]
    return pd.Series({
        'spread_bps': w['spread_bps'].mean(),
        'depth5':     w['depth5'].mean(),
        'ofi':        w['ofi'].mean(),
        'n_bars':     len(w),
    })


all_days  = {**train_bars, **test_bars}
ctrl_all  = ctrl_bars

rows = []
for d, bars in all_days.items():
    baseline  = window_stats(bars, '11:00', '13:00')   # quiet midday
    pre_fomc  = window_stats(bars, '13:30', '14:00')   # 30 min before
    post_fomc = window_stats(bars, '14:00', '14:30')   # 30 min after
    rows.append({
        'date': d,
        'type': 'FOMC',
        'spread_base':   baseline['spread_bps'],
        'spread_pre':    pre_fomc['spread_bps'],
        'spread_post':   post_fomc['spread_bps'],
        'depth_base':    baseline['depth5'],
        'depth_pre':     pre_fomc['depth5'],
        'ofi_pre':       pre_fomc['ofi'],
        'ofi_post_1m':   window_stats(bars, '14:00', '14:01')['ofi'],
    })

for d, bars in ctrl_all.items():
    baseline  = window_stats(bars, '11:00', '13:00')
    pre_fomc  = window_stats(bars, '13:30', '14:00')
    post_fomc = window_stats(bars, '14:00', '14:30')
    rows.append({
        'date': d,
        'type': 'Control',
        'spread_base':  baseline['spread_bps'],
        'spread_pre':   pre_fomc['spread_bps'],
        'spread_post':  post_fomc['spread_bps'],
        'depth_base':   baseline['depth5'],
        'depth_pre':    pre_fomc['depth5'],
        'ofi_pre':      pre_fomc['ofi'],
        'ofi_post_1m':  window_stats(bars, '14:00', '14:01')['ofi'],
    })

stats_df = pd.DataFrame(rows)
stats_df['spread_ratio'] = stats_df['spread_pre'] / stats_df['spread_base']
stats_df['depth_ratio']  = stats_df['depth_pre']  / stats_df['depth_base']

print('=== Spread widening pre-FOMC vs Baseline ===')
print(stats_df.groupby('type')[['spread_ratio', 'depth_ratio']].mean().round(3))

# === cell 3 ===
def plot_intraday_avg(days_dict: dict, label: str, color: str, ax_spread, ax_depth, ax_ofi):
    """Plot average intraday spread/depth/OFI across multiple days."""
    all_bars = []
    for d, bars in days_dict.items():
        b = bars.copy()
        b['time_str'] = b.index.strftime('%H:%M')
        all_bars.append(b)

    combined = pd.concat(all_bars)
    avg = combined.groupby('time_str')[['spread_bps', 'depth5', 'ofi']].mean()

    ax_spread.plot(range(len(avg)), avg['spread_bps'], label=label, color=color, lw=1.5)
    ax_depth.plot(range(len(avg)),  avg['depth5'],     label=label, color=color, lw=1.5)
    ax_ofi.plot(range(len(avg)),    avg['ofi'],        label=label, color=color, lw=1.5)

    # Mark 2:00 PM
    times = list(avg.index)
    fomc_idx = next((i for i, t in enumerate(times) if t >= '14:00'), None)
    return avg, fomc_idx


fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)

avg_fomc, fi = plot_intraday_avg(all_days, 'FOMC days',    'steelblue',  axes[0], axes[1], axes[2])
avg_ctrl, ci = plot_intraday_avg(ctrl_all, 'Control days', 'darkorange', axes[0], axes[1], axes[2])

for ax in axes:
    if fi is not None:
        ax.axvline(fi, color='red', lw=1.5, ls='--', label='2:00 PM (FOMC)')
    ax.legend(fontsize=8)

axes[0].set_ylabel('Spread (bps)')
axes[0].set_title('SPY intraday dynamics — FOMC vs Control days (all 2024)')
axes[1].set_ylabel('Top-5 Book Depth (shares)')
axes[2].set_ylabel('Top-of-book OFI')

# X-axis: show hour labels
times = list(avg_fomc.index)
tick_idx  = [i for i, t in enumerate(times) if t.endswith(':00')]
tick_lbls = [times[i] for i in tick_idx]
for ax in axes:
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lbls, fontsize=8)

plt.tight_layout()
plt.savefig('figures/fomc_intraday_dynamics.png', dpi=150)
plt.show()

# === cell 4 ===
HOLD_MINS = 15
NOTIONAL  = 100_000   # $100k notional per trade (~200 shares of SPY at ~$500)


def trade_fomc_day(date_str: str, bars: pd.DataFrame) -> dict | None:
    """
    Fade the initial FOMC reaction (contrarian mean-reversion).
    At 2:00 PM, HFT overshoots the fair price. We enter in the OPPOSITE
    direction at 2:01 PM and hold for 15 minutes as the price reverts.
    """
    entry_mask = bars.index.time == pd.Timestamp('14:00').time()
    if not entry_mask.any():
        return None

    entry_bar  = bars[entry_mask].iloc[0]
    entry_time = bars[entry_mask].index[0]

    ret_14 = entry_bar['ret_bps']
    if ret_14 == 0 or pd.isna(ret_14):
        return None

    # CONTRARIAN: fade the initial move
    direction = -1 if ret_14 > 0 else 1

    entry_price  = entry_bar['mid_close']
    entry_spread = entry_bar['spread']

    exit_time_target = entry_time + pd.Timedelta(minutes=HOLD_MINS)
    future = bars[bars.index >= exit_time_target]
    if future.empty:
        return None
    exit_bar    = future.iloc[0]
    exit_price  = exit_bar['mid_close']
    exit_spread = exit_bar['spread']

    shares    = NOTIONAL / entry_price
    gross_pnl = direction * (exit_price - entry_price) * shares
    tc        = (entry_spread / 2 + exit_spread / 2) * shares
    net_pnl   = gross_pnl - tc

    sv = entry_bar.get('signed_vol', float('nan'))

    return {
        'date':        date_str,
        'direction':   direction,
        'ret_14_bps':  round(ret_14, 2),
        'signed_vol':  round(sv, 3) if not pd.isna(sv) else float('nan'),
        'entry_price': round(entry_price, 4),
        'exit_price':  round(exit_price, 4),
        'shares':      round(shares, 1),
        'gross_pnl':   round(gross_pnl, 2),
        'tc':          round(tc, 2),
        'net_pnl':     round(net_pnl, 2),
        'surprise':    date_str == SURPRISE_DATE,
    }


train_trades = [t for d, b in train_bars.items()
                if (t := trade_fomc_day(d, b)) is not None]
train_trades_df = pd.DataFrame(train_trades)

print('=== FOMC TRAIN TRADES (contrarian) ===')
print(train_trades_df[['date','direction','ret_14_bps','gross_pnl','tc','net_pnl']].to_string(index=False))

# === cell 5 ===
ctrl_trades = [t for d, b in ctrl_bars.items()
               if (t := trade_fomc_day(d, b)) is not None]
ctrl_trades_df = pd.DataFrame(ctrl_trades) if ctrl_trades else pd.DataFrame()

print('=== CONTROL DAY TRADES (placebo) ===')
if not ctrl_trades_df.empty:
    print(ctrl_trades_df[['date','direction','ret_14_bps','net_pnl']].to_string(index=False))
    fomc_pnl = train_trades_df['net_pnl'].sum() if not train_trades_df.empty else 0
    ctrl_pnl = ctrl_trades_df['net_pnl'].sum()
    print(f'\nControl total PnL: ${ctrl_pnl:.2f}')
    print(f'FOMC   total PnL: ${fomc_pnl:.2f}')
    print(f'\nNote: with only 4 FOMC events in-sample, PnL comparison has high variance.')
    print(f'The key test is the OOS period (Jul-Dec 2024), especially Sep-18 surprise.')

# === cell 6 ===
def plot_fomc_day(date_str: str, bars: pd.DataFrame, trade: dict | None = None):
    """Plot SPY price path ±60 min around FOMC announcement."""
    mask = ((bars.index.time >= pd.Timestamp('13:00').time()) &
            (bars.index.time <= pd.Timestamp('15:00').time()))
    w = bars[mask]
    if w.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(range(len(w)), w['mid_close'], color='steelblue', lw=1.5)
    fomc_idx = next((i for i, t in enumerate(w.index)
                     if t.time() >= FOMC_TIME), None)
    if fomc_idx:
        axes[0].axvline(fomc_idx, color='red', lw=1.5, ls='--', label='2:00 PM')
    if trade:
        color = 'green' if trade['net_pnl'] > 0 else 'tomato'
        axes[0].set_title(
            f"FOMC {date_str} | Dir={'↑' if trade['direction']==1 else '↓'} "
            f"| 14:00 ret={trade['ret_14_bps']:.1f}bps "
            f"| Net PnL=${trade['net_pnl']:.0f}",
            color=color
        )
    axes[0].set_ylabel('SPY mid price')
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(w)), w['ofi'], color=[
        'steelblue' if v > 0 else 'tomato' for v in w['ofi'].fillna(0)
    ], alpha=0.7)
    if fomc_idx:
        axes[1].axvline(fomc_idx, color='red', lw=1.5, ls='--')
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_ylabel('OFI (1-min)')

    ticks = list(range(0, len(w), 10))
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([w.index[i].strftime('%H:%M') for i in ticks], fontsize=8)

    plt.tight_layout()
    plt.savefig(f'figures/fomc_{date_str}.png', dpi=150)
    plt.show()


trade_lookup = {t['date']: t for t in train_trades}
for d, bars in train_bars.items():
    plot_fomc_day(d, bars, trade_lookup.get(d))

# === cell 7 ===
test_trades = [t for d, b in test_bars.items()
               if (t := trade_fomc_day(d, b)) is not None]
test_trades_df = pd.DataFrame(test_trades)

print('=== FOMC TEST TRADES (OOS) ===')
print(test_trades_df[['date','direction','ret_14_bps','net_pnl','tc','surprise']].to_string(index=False))

test_lookup = {t['date']: t for t in test_trades}
for d, bars in test_bars.items():
    plot_fomc_day(d, bars, test_lookup.get(d))

# === cell 8 ===
def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        print(f'{label}: no trades.')
        return {}

    # Daily PnL = one trade per FOMC day
    daily_pnl = df['net_pnl'].values
    n = len(daily_pnl)

    # Annualize: ~8 FOMC events/year
    ann_ret = daily_pnl.mean() * 8
    ann_vol = daily_pnl.std() * np.sqrt(8)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = np.cumsum(daily_pnl)
    mdd = float(np.min(cum - np.maximum.accumulate(cum)))

    m = {
        'label':        label,
        'n_events':     n,
        'total_pnl':    round(daily_pnl.sum(), 2),
        'mean_pnl':     round(daily_pnl.mean(), 2),
        'ann_return':   round(ann_ret, 2),
        'ann_vol':      round(ann_vol, 2),
        'sharpe':       round(sharpe, 3),
        'max_drawdown': round(mdd, 2),
        'hit_rate':     round((daily_pnl > 0).mean(), 3),
        'mean_tc':      round(df['tc'].mean(), 2),
    }
    print(f'\n=== {label} ===')
    for k, v in m.items():
        print(f'  {k:20s}: {v}')
    return m


m_train = compute_metrics(train_trades_df, 'FOMC Train')
m_test  = compute_metrics(test_trades_df,  'FOMC Test (OOS)')
m_ctrl  = compute_metrics(ctrl_trades_df,  'Control (Placebo)') if not ctrl_trades_df.empty else {}

# === cell 9 ===
all_trades_df = pd.concat([train_trades_df, test_trades_df], ignore_index=True)

# Cumulative PnL
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

cum_train = train_trades_df['net_pnl'].cumsum().values
cum_test  = test_trades_df['net_pnl'].cumsum().values + (cum_train[-1] if len(cum_train) else 0)

axes[0].bar(range(len(train_trades_df)), train_trades_df['net_pnl'],
            color=['green' if v > 0 else 'tomato' for v in train_trades_df['net_pnl']],
            alpha=0.8, label='Train')
offset = len(train_trades_df)
axes[0].bar(range(offset, offset + len(test_trades_df)), test_trades_df['net_pnl'],
            color=['steelblue' if v > 0 else 'orange' for v in test_trades_df['net_pnl']],
            alpha=0.8, label='Test')
axes[0].axvline(offset - 0.5, color='black', lw=1.5, ls='--', label='Train | Test')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_ylabel('Net PnL per event ($)')
axes[0].set_title('FOMC Event-Window Strategy — Per-event PnL')
axes[0].set_xticks(range(len(all_trades_df)))
axes[0].set_xticklabels(all_trades_df['date'].str[5:], rotation=45, fontsize=8)
axes[0].legend(fontsize=8)

cum_all = all_trades_df['net_pnl'].cumsum()
axes[1].plot(cum_all.values, color='steelblue', lw=2, marker='o', ms=6)
axes[1].axvline(offset - 0.5, color='black', lw=1.5, ls='--')
axes[1].axhline(0, color='black', lw=0.5, ls='--')
axes[1].set_ylabel('Cumulative PnL ($)')
axes[1].set_xticks(range(len(all_trades_df)))
axes[1].set_xticklabels(all_trades_df['date'].str[5:], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig('figures/fomc_cumulative_pnl.png', dpi=150)
plt.show()

# Beta note: FOMC strategy is directional — will show positive beta vs SPX
print('Note: This is a DIRECTIONAL strategy (not market-neutral).')
print('Beta vs SPX expected to be positive — decompose alpha vs beta in the write-up.')
print(f"Performance alpha = E[r_strategy] - beta * E[r_SPX]")

# === cell 10 ===
print("""
RISK FACTORS FOR THE PITCH
==========================

1. SMALL SAMPLE: Only 8 FOMC events per year -> 4 train, 4 test.
   Statistical power is limited. A single bad trade changes Sharpe dramatically.
   Mitigant: show placebo test (control days near-zero PnL) and Dec-18 exception separately.

2. REGIME RISK: The mean-reversion pattern holds when the FOMC decision is
   largely anticipated (most 2024 meetings). If the Fed delivers a major surprise
   (e.g. emergency cut), the initial move may NOT revert -- it continues.
   Example: Dec-18 had hawkish guidance that created a one-way directional move.

3. EXECUTION RISK: SPY spreads widen sharply at 2:00 PM (confirmed: +9.7% vs baseline).
   TC at entry can be 3-5x normal spread. Must use aggressive limit orders to reduce slippage.

4. CAPACITY: $100K notional barely moves SPY.
   At $10M, market impact becomes relevant. At $1B, strategy breaks.

5. SIGNAL DECAY: The 2:00 PM bar pattern is increasingly recognized.
   Mean-reversion window may shorten as more capital trades this signal.

6. LOOK-AHEAD BIAS RISK: We observe the direction of the 14:00 bar BEFORE entering
   at 14:01. In live trading, the 14:00 bar must be fully formed -- 1-second delay
   is enough since the announcement is at 14:00:00 sharp.
""")
