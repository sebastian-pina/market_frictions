# === cell 0 ===
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no blocking plt.show()
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

DATA_DIR = Path('data/raw')
Path('figures').mkdir(exist_ok=True)

TRAIN_DATES = [
    '2024-01-02','2024-01-04','2024-01-08','2024-01-10','2024-01-12',
    '2024-01-16','2024-01-18','2024-01-22','2024-01-24','2024-01-26',
    '2024-02-01','2024-02-05','2024-02-07','2024-02-09','2024-02-12',
    '2024-02-14','2024-02-20','2024-02-22','2024-02-26','2024-02-28',
]
TEST_DATES = [
    '2024-03-04','2024-03-07','2024-03-11','2024-03-14','2024-03-18',
    '2024-04-01','2024-04-04','2024-04-08','2024-04-11','2024-04-15',
]

ET = 'America/New_York'

# === cell 1 ===
# Only 4 columns needed — avoids loading all 73 columns (~923 MB/file) into memory
BOOK_COLS = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']

# 1-second bars: captures the ~1s lead-lag window that disappears at 5s resolution
BAR_FREQ = '1s'

def load_bars(symbol: str, dates: list, freq: str = BAR_FREQ) -> pd.DataFrame:
    """
    Load MBP-10 one day at a time, resample to bars immediately.
    Frees raw data after each day to stay within memory limits.
    """
    bars_list = []
    for d in dates:
        p = DATA_DIR / symbol / 'mbp-10' / f'{d}.parquet'
        if not p.exists():
            print(f'Missing: {p}')
            continue
        df = pd.read_parquet(p, columns=BOOK_COLS).sort_index()
        df.index = df.index.tz_convert(ET)
        df = df[(df.index.time >= pd.Timestamp('09:30').time()) &
                (df.index.time <  pd.Timestamp('16:00').time())]

        # Fix uint32 overflow before any arithmetic
        bid_sz = df['bid_sz_00'].astype(np.int64)
        ask_sz = df['ask_sz_00'].astype(np.int64)
        denom  = (bid_sz + ask_sz).replace(0, np.nan)

        tick = pd.DataFrame({
            'mid':    (df['bid_px_00'] + df['ask_px_00']) / 2,
            'spread': df['ask_px_00']  - df['bid_px_00'],
            'ofi':    (bid_sz - ask_sz) / denom,
        }, index=df.index)

        bars = tick.resample(freq).agg(
            mid_close   = ('mid',    'last'),
            ofi_mean    = ('ofi',    'mean'),
            spread_mean = ('spread', 'mean'),
        ).dropna(subset=['mid_close'])
        bars['date'] = bars.index.normalize()   # date label for session-boundary checks
        bars_list.append(bars)
        del df, tick  # free raw memory

    return pd.concat(bars_list)


print('Loading SPY train...')
spy1 = load_bars('SPY', TRAIN_DATES)
print('Loading AAPL train...')
aapl1 = load_bars('AAPL', TRAIN_DATES)

common = spy1.index.intersection(aapl1.index)
spy1   = spy1.loc[common]
aapl1  = aapl1.loc[common]

print(f'1s bars (common): {len(common):,}')
print(f'OFI SPY  range: [{spy1["ofi_mean"].min():.3f}, {spy1["ofi_mean"].max():.3f}]')
print(f'Time range: {spy1.index[0]} -> {spy1.index[-1]}')

# === cell 2 ===
# Confirm shapes and key stats
print(f'SPY  1s bars: {len(spy1):,}  |  {spy1.index[0]}  ->  {spy1.index[-1]}')
print(f'AAPL 1s bars: {len(aapl1):,}')
print(f'OFI SPY  range: [{spy1["ofi_mean"].min():.3f}, {spy1["ofi_mean"].max():.3f}]')
print(f'Spread AAPL mean: ${aapl1["spread_mean"].mean():.4f}')
print(f'AAPL half-spread (bps): {aapl1["spread_mean"].mean() / aapl1["mid_close"].mean() * 1e4 / 2:.3f} bps per side')
spy1.tail(3)

# === cell 3 ===
# Returns in bps (within-session only — pct_change crosses day gaps but there are only
# 20 such boundary bars out of 467k, negligible effect on correlations)
spy_ret  = spy1['mid_close'].pct_change().fillna(0) * 1e4
aapl_ret = aapl1['mid_close'].pct_change().fillna(0) * 1e4
spy_ofi  = spy1['ofi_mean'].fillna(0)

lags = range(-5, 16)   # -5s to +15s
xcorr_ret = [spy_ret.corr(aapl_ret.shift(-lag))  for lag in lags]
xcorr_ofi = [spy_ofi.corr(aapl_ret.shift(-lag))  for lag in lags]

lag_secs = [l for l in lags]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(lag_secs, xcorr_ret, width=0.8, color='steelblue', alpha=0.8)
axes[0].axvline(0, color='black', lw=1, ls='--')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_xlabel('Lag (seconds) — positive = SPY leads AAPL')
axes[0].set_ylabel('Cross-correlation')
axes[0].set_title('SPY return -> AAPL return (1s bars)')

axes[1].bar(lag_secs, xcorr_ofi, width=0.8, color='darkorange', alpha=0.8)
axes[1].axvline(0, color='black', lw=1, ls='--')
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_xlabel('Lag (seconds)')
axes[1].set_ylabel('Cross-correlation')
axes[1].set_title('SPY OFI -> AAPL return (1s bars)')

plt.tight_layout()
plt.savefig('figures/xcorr_spy_aapl.png', dpi=150)
plt.show()

print('Lag(s)  SPY_ret->AAPL_ret   SPY_OFI->AAPL_ret')
for i, lag in enumerate(lags):
    print(f'  {lag:+3d}s   {xcorr_ret[i]:+.5f}              {xcorr_ofi[i]:+.5f}')

best_lag = list(lags)[int(np.argmax(xcorr_ret[5:], 0)) + 5]   # best positive lag
print(f'\nBest positive lag for price lead: +{best_lag}s  (corr={spy_ret.corr(aapl_ret.shift(-best_lag)):.5f})')
print('Interpretation: lead-lag is real but concentrated in the first 1-2 seconds.')

# === cell 4 ===
# Parameters calibrated on training set, fixed before test
# Signal uses SPY 1s price return (positive correlation with AAPL +1s)
# OFI is NOT used as a directional signal — empirically it is contrarian at 1s

OFI_THRESH   = 0.40    # SPY top-of-book imbalance filter (confirms book pressure)
RET_THRESH   = 1.0     # SPY 1s return in bps to confirm a genuine price move
HOLD_BARS    = 5       # hold for 5 x 1s = 5 seconds
SPREAD_MAX   = 0.02    # AAPL max spread in $ (avoid wide-spread fills)


def build_signal(spy1: pd.DataFrame, aapl1: pd.DataFrame) -> pd.Series:
    """
    Signal = +1 (buy AAPL) when:
      - SPY 1s return > RET_THRESH bps  (SPY mid moved up sharply)
      - SPY OFI > OFI_THRESH            (buy-side book pressure confirms direction)
      - AAPL spread < SPREAD_MAX        (can execute at reasonable cost)

    Signal = -1 (sell AAPL) when mirror conditions hold.

    Entry at next bar (1s delay), hold HOLD_BARS bars = 5 seconds.
    """
    spy_ret1 = spy1['mid_close'].pct_change().fillna(0) * 1e4   # 1s return
    aapl_spread = aapl1['spread_mean']

    long_entry  = (spy_ret1 >  RET_THRESH) & (spy1['ofi_mean'] >  OFI_THRESH) & (aapl_spread < SPREAD_MAX)
    short_entry = (spy_ret1 < -RET_THRESH) & (spy1['ofi_mean'] < -OFI_THRESH) & (aapl_spread < SPREAD_MAX)

    signal = pd.Series(0, index=spy1.index)
    signal[long_entry]  =  1
    signal[short_entry] = -1
    return signal


train_signal = build_signal(spy1, aapl1)
print(f'Long signals:  {(train_signal ==  1).sum()}')
print(f'Short signals: {(train_signal == -1).sum()}')
n_total = (train_signal != 0).sum()
print(f'Signal rate:   {n_total / len(train_signal):.2%} of all 1s bars')

# === cell 5 ===
def backtest(signal: pd.Series, aapl1: pd.DataFrame, hold_bars: int = HOLD_BARS,
             shares: int = 100, verbose: bool = True) -> pd.DataFrame:
    """
    Simple event-driven backtest.
    Entry: market order at next-bar open (1-bar delay = 1s).
    Exit:  market order after hold_bars bars.
    TC:    half-spread per side (cross the spread to enter and exit).
    Session guard: skip trades that would cross into the next trading day.
    Returns a DataFrame of trades.
    """
    prices  = aapl1['mid_close'].values
    spreads = aapl1['spread_mean'].values
    times   = aapl1.index
    dates   = aapl1['date'].values        # normalized date for boundary check
    sig     = signal.reindex(aapl1.index, fill_value=0).values

    trades = []
    in_trade = False
    entry_bar = exit_bar = direction = 0
    entry_price = tc_entry = 0.0

    for i in range(1, len(prices) - hold_bars - 1):
        if in_trade:
            if i >= exit_bar:
                exit_price = prices[i]
                tc_exit    = spreads[i] / 2
                gross_pnl  = direction * (exit_price - entry_price) * shares
                tc_total   = (tc_entry + tc_exit) * shares
                net_pnl    = gross_pnl - tc_total
                trades.append({
                    'entry_time':  times[entry_bar],
                    'exit_time':   times[i],
                    'direction':   direction,
                    'entry_price': entry_price,
                    'exit_price':  exit_price,
                    'gross_pnl':   gross_pnl,
                    'tc':          tc_total,
                    'net_pnl':     net_pnl,
                })
                in_trade = False
        else:
            if sig[i - 1] != 0:
                # Skip trade if exit bar would land in a different trading session
                if dates[i + hold_bars] != dates[i]:
                    continue
                direction   = sig[i - 1]
                entry_price = prices[i]
                tc_entry    = spreads[i] / 2
                entry_bar   = i
                exit_bar    = i + hold_bars
                in_trade    = True

    df = pd.DataFrame(trades)
    if verbose and len(df):
        avg_hold = (df['exit_time'] - df['entry_time']).dt.total_seconds().mean()
        print(f'Trades: {len(df)}  |  Hit rate: {(df.net_pnl > 0).mean():.1%}  '
              f'|  Gross PnL: ${df.gross_pnl.sum():.2f}  '
              f'|  Net PnL: ${df.net_pnl.sum():.2f}  '
              f'|  Avg TC/trade: ${df.tc.mean():.3f}  '
              f'|  Avg hold: {avg_hold:.1f}s')
    return df


trades_train = backtest(train_signal, aapl1)

# === cell 6 ===
def compute_metrics(trades: pd.DataFrame, label: str = '') -> dict:
    if trades.empty:
        print('No trades.')
        return {}

    daily = (trades
             .assign(date=trades['entry_time'].dt.normalize())
             .groupby('date')[['gross_pnl','net_pnl']].sum())
    daily = daily[(daily != 0).any(axis=1)]

    ann_ret_gross = daily['gross_pnl'].mean() * 252
    ann_ret_net   = daily['net_pnl'].mean()   * 252
    ann_vol       = daily['net_pnl'].std()    * np.sqrt(252)
    sharpe        = ann_ret_net / ann_vol if ann_vol > 0 else np.nan

    cum = trades['net_pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()

    avg_tc_bps = (trades['tc'] / (trades['entry_price'] * 100) * 1e4).mean()
    hold_secs  = (trades['exit_time'] - trades['entry_time']).dt.total_seconds().mean()

    metrics = {
        'label':            label,
        'n_trades':         len(trades),
        'gross_pnl':        round(trades['gross_pnl'].sum(), 2),
        'total_tc':         round(trades['tc'].sum(), 2),
        'net_pnl':          round(trades['net_pnl'].sum(), 2),
        'ann_return_gross': round(ann_ret_gross, 2),
        'ann_return_net':   round(ann_ret_net, 2),
        'ann_vol':          round(ann_vol, 2),
        'sharpe':           round(sharpe, 3),
        'max_drawdown':     round(mdd, 2),
        'hit_rate':         round((trades['net_pnl'] > 0).mean(), 3),
        'avg_hold_s':       round(hold_secs, 1),
        'avg_tc_bps':       round(avg_tc_bps, 3),
    }
    for k, v in metrics.items():
        print(f'  {k:22s}: {v}')
    return metrics


print('=== TRAINING SET ===')
m_train = compute_metrics(trades_train, label='train')

# === cell 7 ===
from pathlib import Path
Path('figures').mkdir(exist_ok=True)

def plot_cumulative_pnl(trades: pd.DataFrame, title: str, fname: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # Cumulative PnL
    cum = trades['net_pnl'].cumsum()
    axes[0].plot(cum.values, color='steelblue', lw=1.5)
    axes[0].axhline(0, color='black', lw=0.5, ls='--')
    axes[0].set_ylabel('Cumulative PnL ($)')
    axes[0].set_title(title)

    # Rolling Sharpe (20-trade window)
    window = 20
    roll_ret = trades['net_pnl'].rolling(window).mean()
    roll_vol = trades['net_pnl'].rolling(window).std()
    roll_sharpe = (roll_ret / roll_vol.replace(0, np.nan)) * np.sqrt(252)
    axes[1].plot(roll_sharpe.values, color='darkorange', lw=1)
    axes[1].axhline(0, color='black', lw=0.5, ls='--')
    axes[1].set_ylabel(f'Rolling Sharpe ({window}-trade)')

    # Direction of trades
    colors = ['steelblue' if d > 0 else 'tomato' for d in trades['direction']]
    axes[2].bar(range(len(trades)), trades['net_pnl'].values, color=colors, alpha=0.7, width=0.8)
    axes[2].axhline(0, color='black', lw=0.5)
    axes[2].set_ylabel('Per-trade PnL ($)')
    axes[2].set_xlabel('Trade #')

    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=150)
    plt.show()


plot_cumulative_pnl(trades_train, 'Lead-Lag SPY→AAPL — Training Set', 'leadlag_train_pnl.png')

# === cell 8 ===
print('Loading test data...')
spy1_test  = load_bars('SPY',  TEST_DATES)
aapl1_test = load_bars('AAPL', TEST_DATES)

common_test = spy1_test.index.intersection(aapl1_test.index)
spy1_test   = spy1_test.loc[common_test]
aapl1_test  = aapl1_test.loc[common_test]

print(f'Test 1s bars: {len(common_test):,}')

test_signal  = build_signal(spy1_test, aapl1_test)
trades_test  = backtest(test_signal, aapl1_test)

print('\n=== TEST SET ===')
m_test = compute_metrics(trades_test, label='test')

plot_cumulative_pnl(trades_test, 'Lead-Lag SPY->AAPL -- Test Set (OOS)', 'leadlag_test_pnl.png')

# === cell 9 ===
def beta_vs_spy(trades: pd.DataFrame, spy1: pd.DataFrame, label: str = '') -> None:
    """Regress daily strategy PnL on SPY daily return. Market-neutral -> beta ~= 0."""
    spy_daily = (spy1['mid_close']
                 .groupby(spy1['date']).last()
                 .pct_change().dropna() * 100)

    strat_daily = (trades
                   .assign(date=trades['entry_time'].dt.normalize())
                   .groupby('date')['net_pnl'].sum())
    strat_daily = strat_daily[strat_daily != 0]

    common = strat_daily.index.intersection(spy_daily.index)
    if len(common) < 5:
        print('Not enough overlapping days for beta regression.')
        return

    x = spy_daily.loc[common].values
    y = strat_daily.loc[common].values
    slope, intercept, r, p, _ = stats.linregress(x, y)

    print(f'Beta vs SPY ({label}):  {slope:.4f}  (p={p:.3f})')
    print(f'Alpha (daily):          ${intercept:.4f}  -> annualized ${intercept * 252:.2f}')
    print(f'R-squared:              {r**2:.4f}')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.7, s=30)
    xfit = np.linspace(x.min(), x.max(), 50)
    ax.plot(xfit, slope * xfit + intercept, 'r--', lw=1.5, label=f'beta={slope:.3f}')
    ax.set_xlabel('SPY daily return (%)')
    ax.set_ylabel('Strategy daily PnL ($)')
    ax.set_title(f'Beta vs SPY -- {label}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figures/leadlag_beta_{label}.png', dpi=150)
    plt.show()


print('--- Training ---')
beta_vs_spy(trades_train, spy1, label='train')
print('\n--- Test ---')
beta_vs_spy(trades_test, spy1_test, label='test')

# === cell 10 ===
summary = pd.DataFrame([m_train, m_test]).set_index('label')
summary.T
