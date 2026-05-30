import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('data/raw')
Path('figures').mkdir(exist_ok=True)
ET = 'America/New_York'

FOMC_TRAIN    = ['2024-01-31','2024-03-20','2024-05-01','2024-06-12']
FOMC_TEST     = ['2024-07-31','2024-09-18','2024-11-07','2024-12-18']
CONTROL       = ['2024-01-24','2024-03-13','2024-04-24','2024-06-05',
                  '2024-07-24','2024-09-11','2024-10-30','2024-12-11']
SURPRISE_DATE = '2024-09-18'
HOLD_MINS     = 15
NOTIONAL      = 100_000
BOOK_COLS     = ['bid_px_00','ask_px_00','bid_sz_00','ask_sz_00']


def load_day_bars(symbol, date_str):
    p = DATA_DIR / symbol / 'mbp-10' / f'{date_str}.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p, columns=BOOK_COLS).sort_index()
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
    bars['date']    = date_str
    del df, tick
    return bars


def trade_day(symbol, date_str, bars, fade=True):
    mask = bars.index.time == pd.Timestamp('14:00').time()
    if not mask.any():
        return None
    entry_bar  = bars[mask].iloc[0]
    entry_time = bars[mask].index[0]
    ret_14 = entry_bar['ret_bps']
    if ret_14 == 0 or pd.isna(ret_14):
        return None
    # fade=True: contrarian (enter opposite); fade=False: momentum (enter same)
    initial_dir = 1 if ret_14 > 0 else -1
    direction   = -initial_dir if fade else initial_dir

    entry_price  = entry_bar['mid_close']
    entry_spread = entry_bar['spread']
    future = bars[bars.index >= entry_time + pd.Timedelta(minutes=HOLD_MINS)]
    if future.empty:
        return None
    exit_bar    = future.iloc[0]
    exit_price  = exit_bar['mid_close']
    exit_spread = exit_bar['spread']
    shares    = NOTIONAL / entry_price
    gross_pnl = direction * (exit_price - entry_price) * shares
    tc        = (entry_spread / 2 + exit_spread / 2) * shares
    return {
        'date':        date_str,
        'symbol':      symbol,
        'direction':   direction,
        'ret_14_bps':  round(ret_14, 2),
        'entry_price': round(entry_price, 4),
        'exit_price':  round(exit_price, 4),
        'gross_pnl':   round(gross_pnl, 2),
        'tc':          round(tc, 2),
        'net_pnl':     round(gross_pnl - tc, 2),
        'surprise':    date_str == SURPRISE_DATE,
    }


def run_symbol(symbol, fade=True):
    mode = 'contrarian (fade)' if fade else 'momentum (follow)'
    print(f'\n{"="*55}')
    print(f'  {symbol}  |  {mode}')
    print(f'{"="*55}')

    def load_set(dates):
        out = {}
        for d in dates:
            try:
                out[d] = load_day_bars(symbol, d)
            except FileNotFoundError:
                print(f'  Missing {d}')
        return out

    train_bars = load_set(FOMC_TRAIN)
    test_bars  = load_set(FOMC_TEST)
    ctrl_bars  = load_set(CONTROL)

    train_t = [t for d, b in train_bars.items() if (t := trade_day(symbol, d, b, fade)) is not None]
    test_t  = [t for d, b in test_bars.items()  if (t := trade_day(symbol, d, b, fade)) is not None]
    ctrl_t  = [t for d, b in ctrl_bars.items()  if (t := trade_day(symbol, d, b, fade)) is not None]

    results = {}
    for trades, lbl in [(train_t, 'Train'), (test_t, 'Test (OOS)'), (ctrl_t, 'Control')]:
        if not trades:
            print(f'  {lbl}: no trades')
            continue
        df = pd.DataFrame(trades)
        sharpe = df['net_pnl'].mean() / df['net_pnl'].std() * np.sqrt(8) if df['net_pnl'].std() > 0 else float('nan')
        print(f'\n  [{lbl}]  N={len(df)}  PnL=${df["net_pnl"].sum():.0f}  '
              f'Hit={( df["net_pnl"] > 0).mean():.0%}  '
              f'Sharpe={sharpe:.2f}  AvgTC=${df["tc"].mean():.1f}')
        print(df[['date', 'ret_14_bps', 'direction', 'gross_pnl', 'tc', 'net_pnl']].to_string(index=False))
        results[lbl] = df
    return results


# ── Pre-FOMC spread widening ─────────────────────────────────────────────────
print('\n=== Pre-FOMC spread widening vs baseline (11:00-13:00) ===')
for sym in ['XLF', 'XLU', 'SPY']:
    ratios = []
    for d in FOMC_TRAIN + FOMC_TEST:
        try:
            b    = load_day_bars(sym, d)
            base = b[(b.index.time >= pd.Timestamp('11:00').time()) &
                     (b.index.time <  pd.Timestamp('13:00').time())]['spread'].mean()
            pre  = b[(b.index.time >= pd.Timestamp('13:30').time()) &
                     (b.index.time <  pd.Timestamp('14:00').time())]['spread'].mean()
            if base > 0:
                ratios.append(pre / base)
        except FileNotFoundError:
            pass
    if ratios:
        print(f'  {sym}: spread ratio pre/baseline = {np.mean(ratios):.3f}  '
              f'(+{(np.mean(ratios)-1)*100:.1f}%)')

# ── Individual symbol strategies ─────────────────────────────────────────────
xlf_res = run_symbol('XLF', fade=True)
xlu_res = run_symbol('XLU', fade=True)

# ── Pair trade: XLF vs XLU ───────────────────────────────────────────────────
print('\n' + '='*55)
print('  PAIR TRADE: XLF - XLU  (rate-sensitive sector spread)')
print('='*55)
print('  Hypothesis: rate CUT -> XLU rallies (bond proxy), XLF mixed')
print('  Pair: long XLU + short XLF when 14:00 shows cut signal (SPY up)')
print('  Both with contrarian fade -> net = XLF_fade + XLU_fade PnL')

for lbl in ['Train', 'Test (OOS)', 'Control']:
    xlf_df = xlf_res.get(lbl)
    xlu_df = xlu_res.get(lbl)
    if xlf_df is None or xlu_df is None:
        continue
    merged = pd.merge(
        xlf_df[['date','net_pnl']].rename(columns={'net_pnl':'xlf_pnl'}),
        xlu_df[['date','net_pnl']].rename(columns={'net_pnl':'xlu_pnl'}),
        on='date', how='inner'
    )
    merged['pair_pnl'] = merged['xlf_pnl'] + merged['xlu_pnl']
    sharpe = merged['pair_pnl'].mean() / merged['pair_pnl'].std() * np.sqrt(8) if merged['pair_pnl'].std() > 0 else float('nan')
    print(f'\n  [{lbl}]  PnL=${merged["pair_pnl"].sum():.0f}  '
          f'Hit={(merged["pair_pnl"]>0).mean():.0%}  Sharpe={sharpe:.2f}')
    print(merged[['date','xlf_pnl','xlu_pnl','pair_pnl']].to_string(index=False))

print('\nDone.')
