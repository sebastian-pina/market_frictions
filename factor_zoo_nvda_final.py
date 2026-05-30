"""
Factor Zoo â€” Microstructure L2 Strategy  (ProjectPlan-compliant, final)
=========================================================================
USAGE: set SYMBOL below, then run.

Data splits
  Train : Jan-Feb 2024  (18 days)
  Val   : Mar-Apr 2024  (7 days)
  OOS   : Oct 2024      (17 days)

11 features from MBP-10 only (trade-schema features 13-15 dropped â€”
unavailable in OOS parquets):
  ofi_1, ofi_5, ofi_30, ofi_multilevel,
  book_imbalance, depth_ratio, bid_slope, ask_slope,
  microprice_drift, spread_bps, realized_vol

ProjectPlan compliance
  âœ“ Execution : BUY fills at ask_px_00, SELL at bid_px_00; walk book if
                order exceeds top-of-book depth
  âœ“ Stop-loss : early exit when unrealized loss > 1.5Ã— spread per share
  âœ“ Signal flip: early exit if model sign reverses past threshold
  âœ“ Hit rate  : % of individual TRADES with positive net PnL
  âœ“ Rolling Sharpe: 21-day window
  âœ“ Label winsorization at 1st / 99th pct of training labels
  âœ“ Inventory  : position in shares reconstructed from trade log
  âœ“ Capacity   : $10M / $100M / $1B (Almgren-Chriss impact model)
  âœ“ All 5 required plots + summary metrics CSV
"""

import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• CHANGE THIS â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
SYMBOL = 'NVDA'    # 'AAPL' | 'NVDA' | 'SPY'
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

DATA_DIR   = Path('data/raw')
FIG_DIR    = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

ET        = 'America/New_York'
HOLD_S    = 60        # 60-second hold period
NOTIONAL  = 50_000    # $50K per trade
OPEN_BUF  = 5 * 60    # skip first 5 min (300 bars)
CLOSE_BUF = 60        # skip last 60 bars (no label)
LEVELS    = 5         # depth levels for OFI / slopes
INV_CAP   = 500       # absolute inventory cap (shares)
KILL_PCT  = 0.02      # daily kill switch: -2% of notional
SL_MULT   = 1.5       # stop-loss multiplier (Ã— spread per share)
ROLL_WIN  = 21        # rolling Sharpe window (trading days)
WINSOR_LO = 0.01      # label winsorisation lower tail
WINSOR_HI = 0.99      # label winsorisation upper tail

TRAIN_DATES = [
    '2024-01-02','2024-01-04','2024-01-08','2024-01-10','2024-01-12',
    '2024-01-16','2024-01-18','2024-01-22','2024-01-24','2024-01-26',
    '2024-02-05','2024-02-07','2024-02-09','2024-02-12','2024-02-14',
    '2024-02-20','2024-02-22','2024-02-26',
]
VAL_DATES = [
    '2024-02-28','2024-03-04','2024-03-07',
    '2024-03-11','2024-03-14','2024-03-18','2024-04-01',
]
OCT_DATES = [
    '2024-10-01','2024-10-02','2024-10-03','2024-10-04',
    '2024-10-07','2024-10-08','2024-10-09','2024-10-10',
    '2024-10-11','2024-10-14','2024-10-15','2024-10-16',
    '2024-10-17','2024-10-18','2024-10-21',
    '2024-10-22','2024-10-23',
]
ALL_DATES = TRAIN_DATES + VAL_DATES + OCT_DATES

FEATURE_NAMES = [
    'ofi_1','ofi_5','ofi_30','ofi_multilevel',
    'book_imbalance','depth_ratio','bid_slope','ask_slope',
    'microprice_drift','spread_bps','realized_vol',
]

BOOK_COLS = (
    [f'bid_px_{i:02d}' for i in range(10)] +
    [f'ask_px_{i:02d}' for i in range(10)] +
    [f'bid_sz_{i:02d}' for i in range(10)] +
    [f'ask_sz_{i:02d}' for i in range(10)]
)

_FEAT_CACHE: dict = {}   # key = f'{SYMBOL}_{date}'
_SPY_CACHE:  dict = {}   # key = date_str â†’ daily return


# â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_day(date_str, symbol=None):
    sym  = symbol or SYMBOL
    path = DATA_DIR / sym / 'mbp-10' / f'{date_str}.parquet'
    mbp  = pd.read_parquet(path, engine='fastparquet', columns=BOOK_COLS).sort_index()
    if mbp.index.tzinfo is None:
        mbp.index = mbp.index.tz_localize('UTC').tz_convert(ET)
    else:
        mbp.index = mbp.index.tz_convert(ET)
    mbp = mbp.between_time('09:30', '16:00')
    sz_cols = [c for c in mbp.columns if '_sz_' in c]
    mbp[sz_cols] = mbp[sz_cols].fillna(0).astype(np.int64)
    return mbp


# â”€â”€ Feature engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _book_slope(prefix, b):
    qs  = np.stack([b[f'{prefix}_{i:02d}'].values for i in range(LEVELS)],
                   axis=1).astype(float)
    idx = np.arange(LEVELS, dtype=float)
    return (LEVELS * (qs @ idx) - idx.sum() * qs.sum(axis=1)) / 50.0


def compute_features(mbp):
    """1-second bars: 11 features + microprice + spread_raw + all book cols."""
    b    = mbp.resample('1s').last().dropna(subset=['bid_px_00'])

    bid0   = b['bid_sz_00'].values.astype(np.int64)
    ask0   = b['ask_sz_00'].values.astype(np.int64)
    denom0 = np.where(bid0 + ask0 > 0, bid0 + ask0, np.nan)

    # OFI features
    ofi_raw = (bid0 - ask0) / denom0
    ofi_s   = pd.Series(ofi_raw, index=b.index)
    ofi_5   = ofi_s.rolling(5,  min_periods=1).mean().values
    ofi_30  = ofi_s.rolling(30, min_periods=1).mean().values

    # Multi-level OFI (5 levels)
    ml_ofi = np.zeros(len(b))
    for i in range(LEVELS):
        bi = b[f'bid_sz_{i:02d}'].values.astype(np.int64)
        ai = b[f'ask_sz_{i:02d}'].values.astype(np.int64)
        d  = np.where(bi + ai > 0, bi + ai, np.nan)
        ml_ofi += np.where(np.isfinite(d), (bi - ai) / d, 0.0)
    ml_ofi /= LEVELS

    # Book imbalance
    book_imbal = bid0 / denom0

    # Depth ratio
    top3 = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) +
               b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3))
    deep = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) +
               b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3, 10))
    depth_ratio = top3 / np.where(deep > 0, deep, np.nan)

    # Book slopes
    bid_slope = _book_slope('bid_sz', b)
    ask_slope = _book_slope('ask_sz', b)

    # Microprice
    micro_num  = ask0 * b['bid_px_00'].values + bid0 * b['ask_px_00'].values
    microprice = micro_num / denom0
    micro_s    = pd.Series(microprice, index=b.index)

    # Microprice drift (10s finite difference, bps/s)
    micro_drift = micro_s.diff(10) / (10 * micro_s.shift(10)) * 1e4

    # Spread in bps
    spread_raw = b['ask_px_00'].values - b['bid_px_00'].values
    spread_bps = np.where(microprice > 0, spread_raw / microprice * 1e4, np.nan)

    # Realized vol (60s rolling std of bps returns)
    bps_ret      = micro_s.pct_change() * 1e4
    realized_vol = bps_ret.rolling(60, min_periods=10).std()

    feat = pd.DataFrame({
        'ofi_1':           ofi_raw,
        'ofi_5':           ofi_5,
        'ofi_30':          ofi_30,
        'ofi_multilevel':  ml_ofi,
        'book_imbalance':  book_imbal,
        'depth_ratio':     depth_ratio,
        'bid_slope':       bid_slope,
        'ask_slope':       ask_slope,
        'microprice_drift': micro_drift.values,
        'spread_bps':      spread_bps,
        'realized_vol':    realized_vol.values,
        'microprice':      microprice,
        'spread_raw':      spread_raw,
    }, index=b.index)

    # Attach resampled book px/sz columns (used by walk-book execution)
    for col in BOOK_COLS:
        feat[col] = b[col].values

    return feat


def make_labels(feat_df, hold_s=HOLD_S):
    mp = feat_df['microprice']
    return (mp.shift(-hold_s) / mp - 1) * 1e4


# â”€â”€ Dataset builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_dataset(dates, tag='building', winsor_bounds=None):
    print(f'\n[{tag}] Loading {len(dates)} days...')
    Xs, ys = [], []
    for d in dates:
        key = f'{SYMBOL}_{d}'
        try:
            if key not in _FEAT_CACHE:
                mbp = load_day(d)
                _FEAT_CACHE[key] = compute_features(mbp)
            feat = _FEAT_CACHE[key]
            lbl  = make_labels(feat)
            n    = len(feat)
            sl   = slice(OPEN_BUF, n - CLOSE_BUF)
            X    = feat.iloc[sl][FEATURE_NAMES].copy()
            y    = lbl.iloc[sl]
            mask = X.notna().all(axis=1) & y.notna()
            Xs.append(X[mask])
            ys.append(y[mask])
            print(f'  {d}: {mask.sum():,} bars')
        except Exception as e:
            print(f'  {d}: ERROR â€“ {e}')
    if not Xs:
        raise RuntimeError('No data loaded.')
    X_all = pd.concat(Xs)
    y_all = pd.concat(ys)
    # Winsorise using pre-computed bounds (or compute if not provided)
    if winsor_bounds is None:
        lo = y_all.quantile(WINSOR_LO)
        hi = y_all.quantile(WINSOR_HI)
        winsor_bounds = (lo, hi)
    y_all = y_all.clip(*winsor_bounds)
    print(f'  Total: {len(X_all):,} bars | y mean={y_all.mean():.3f} '
          f'std={y_all.std():.3f} bps | winsor [{winsor_bounds[0]:.2f}, '
          f'{winsor_bounds[1]:.2f}]')
    return X_all, y_all, winsor_bounds


# â”€â”€ Model training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_ols(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    model  = Ridge(alpha=1.0)
    model.fit(scaler.transform(X_train), y_train.values)
    return model, scaler


def train_lgbm(X_train, y_train, X_val=None, y_val=None):
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=200, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0,
        n_jobs=2, random_state=42, verbose=-1,
    )
    fit_kw = {}
    if X_val is not None:
        fit_kw['eval_set'] = [(X_val.values, y_val.values)]
        fit_kw['callbacks'] = [lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)]
    model.fit(X_train.values, y_train.values, **fit_kw)
    return model


def predict(model, X, scaler=None):
    Xv = X[FEATURE_NAMES].values
    if scaler is not None:
        Xv = scaler.transform(Xv)
    return model.predict(Xv)


def ic_stats(y_true, y_pred):
    v = np.isfinite(y_true) & np.isfinite(y_pred)
    if v.sum() < 10:
        return np.nan, np.nan
    p = np.corrcoef(y_true[v], y_pred[v])[0, 1]
    s, _ = stats.spearmanr(y_true[v], y_pred[v])
    return float(p), float(s)


# â”€â”€ Ablation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def ablation(X_train, y_train):
    print('\n[Ablation] Drop-one IC on training set (OLS)...')
    sc = StandardScaler().fit(X_train)
    base_pred = Ridge(alpha=1.0).fit(
        sc.transform(X_train), y_train
    ).predict(sc.transform(X_train))
    base_ic, _ = ic_stats(y_train.values, base_pred)

    drops = {}
    for feat in FEATURE_NAMES:
        cols = [f for f in FEATURE_NAMES if f != feat]
        sc2  = StandardScaler().fit(X_train[cols])
        pred = Ridge(alpha=1.0).fit(
            sc2.transform(X_train[cols]), y_train
        ).predict(sc2.transform(X_train[cols]))
        ic_drop, _ = ic_stats(y_train.values, pred)
        drops[feat] = base_ic - ic_drop
        print(f'  -{feat:<22}  base={base_ic:.4f}  drop={drops[feat]:+.5f}')
    return drops, base_ic


# â”€â”€ Walk-book execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _walk_fill(bid_px_arr, ask_px_arr, bid_sz_arr, ask_sz_arr, i, direction, shares):
    """
    Volume-weighted avg fill price for 'shares' at bar index i.
    direction=+1 â†’ buy (walk ask levels), direction=-1 â†’ sell (walk bid levels).
    Returns fill_price (float) or np.nan if book is empty.
    """
    remaining  = int(shares)
    total_cost = 0.0
    last_px    = np.nan

    for level in range(10):
        if direction == 1:
            px = ask_px_arr[i, level]
            sz = int(ask_sz_arr[i, level])
        else:
            px = bid_px_arr[i, level]
            sz = int(bid_sz_arr[i, level])
        if not (np.isfinite(px) and px > 0):
            break
        last_px = px
        fill = min(remaining, sz)
        total_cost += fill * px
        remaining  -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        if np.isfinite(last_px):
            total_cost += remaining * last_px  # fill remainder at deepest level
        else:
            return np.nan

    return total_cost / shares


# â”€â”€ Backtest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def backtest(dates, model, scaler, threshold, model_type, tag=''):
    """
    Simulate the 60s hold strategy on 'dates'.
    Returns (daily_df, trades_df).

    Execution:
      â€¢ Entry: walk-book fill at ask (long) or bid (short)
      â€¢ Exit: 60s later at bid (long) or ask (short)
      â€¢ Early exits: stop-loss (loss > 1.5Ã— spread/share) or signal flip

    PnL:
      gross = direction Ã— (mid_exit âˆ’ mid_entry) Ã— shares
      TC    = |fill_entry âˆ’ mid_entry| + |fill_exit âˆ’ mid_exit|   (per share Ã— shares)
      net   = gross âˆ’ TC
    """
    trades    = []
    daily_pnl = {}

    for d in dates:
        key = f'{SYMBOL}_{d}'
        try:
            if key not in _FEAT_CACHE:
                mbp = load_day(d)
                _FEAT_CACHE[key] = compute_features(mbp)
            feat = _FEAT_CACHE[key]
            n    = len(feat)
            bars = feat.iloc[OPEN_BUF : n - CLOSE_BUF].copy()
            bars = bars[bars[FEATURE_NAMES].notna().all(axis=1)]
            if len(bars) == 0:
                continue

            preds   = predict(model, bars, scaler if model_type == 'ols' else None)
            bar_idx = bars.index
            nb      = len(bars)

            # Pre-extract book arrays for fast walk-book access
            bid_px = np.stack([bars[f'bid_px_{l:02d}'].values for l in range(10)], 1)
            ask_px = np.stack([bars[f'ask_px_{l:02d}'].values for l in range(10)], 1)
            bid_sz = np.stack([bars[f'bid_sz_{l:02d}'].values for l in range(10)], 1)
            ask_sz = np.stack([bars[f'ask_sz_{l:02d}'].values for l in range(10)], 1)
            micro  = bars['microprice'].values
            spr    = bars['spread_raw'].values

            day_net      = 0.0
            day_gross    = 0.0
            killed       = False
            next_entry_i = 0
            position     = 0  # current shares held

            for i in range(nb):
                if killed or i < next_entry_i:
                    continue
                pred = preds[i]
                if abs(pred) < threshold:
                    continue
                mp = micro[i]
                if not (np.isfinite(mp) and mp > 0):
                    continue

                direction = 1 if pred > 0 else -1
                shares    = max(1, int(NOTIONAL / mp))

                # Inventory cap check
                if abs(position + direction * shares) > INV_CAP:
                    continue

                # Walk-book entry fill
                entry_fill = _walk_fill(bid_px, ask_px, bid_sz, ask_sz,
                                        i, direction, shares)
                if not (np.isfinite(entry_fill) and entry_fill > 0):
                    continue

                sl_per_share = SL_MULT * spr[i]  # stop-loss threshold per share

                # Nominal 60s exit index
                exit_ts  = bar_idx[i] + pd.Timedelta(seconds=HOLD_S)
                exit_idx = bar_idx.searchsorted(exit_ts)
                if exit_idx >= nb:
                    continue

                # Scan for early exit (stop-loss or signal flip) within hold window
                actual_exit  = exit_idx
                exit_reason  = 'timeout'
                for j in range(i + 1, min(exit_idx + 1, nb)):
                    cur_mp = micro[j]
                    if not np.isfinite(cur_mp):
                        continue
                    # Stop-loss measured from entry MIDPRICE (not fill price).
                    # entry_fill is already above micro[i] by half-spread â€” comparing
                    # fill to current mid would trigger on virtually every trade.
                    loss_ps = direction * (micro[i] - cur_mp)  # adverse mid move per share
                    if loss_ps > sl_per_share:
                        actual_exit = j
                        exit_reason = 'stoploss'
                        break
                    if abs(preds[j]) >= threshold and preds[j] * direction < 0:
                        actual_exit = j
                        exit_reason = 'flip'
                        break

                exit_mp   = micro[actual_exit]
                exit_fill = _walk_fill(bid_px, ask_px, bid_sz, ask_sz,
                                       actual_exit, -direction, shares)
                if not (np.isfinite(exit_mp) and np.isfinite(exit_fill) and exit_fill > 0):
                    continue

                # PnL
                gross = direction * (exit_mp - mp) * shares
                tc    = (abs(entry_fill - mp) + abs(exit_fill - exit_mp)) * shares
                net   = gross - tc

                day_gross += gross
                day_net   += net
                position   = 0   # flat after exit
                next_entry_i = actual_exit + 1

                trades.append({
                    'date'       : d,
                    'ts'         : str(bar_idx[i]),
                    'exit_ts'    : str(bar_idx[actual_exit]),
                    'exit_reason': exit_reason,
                    'direction'  : direction,
                    'pred_bps'   : round(float(pred), 3),
                    'entry_mp'   : round(float(mp), 4),
                    'exit_mp'    : round(float(exit_mp), 4),
                    'entry_fill' : round(float(entry_fill), 4),
                    'exit_fill'  : round(float(exit_fill), 4),
                    'shares'     : int(shares),
                    'gross'      : round(gross, 2),
                    'tc'         : round(tc, 2),
                    'net'        : round(net, 2),
                })

                if day_net < -NOTIONAL * KILL_PCT:
                    killed = True

            daily_pnl[d] = {'gross': day_gross, 'net': day_net}

        except Exception as e:
            print(f'  {d}: backtest error â€“ {e}')

    if not daily_pnl:
        return pd.DataFrame(), pd.DataFrame()

    daily_df  = pd.DataFrame(daily_pnl).T
    daily_df.index = pd.to_datetime(daily_df.index)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    _print_bt_stats(daily_df, trades_df, tag)
    return daily_df, trades_df


def _print_bt_stats(daily_df, trades_df, tag):
    if daily_df.empty:
        print(f'  [{tag}] No trades.')
        return
    pnl  = daily_df['net']
    ann  = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    sh   = ann / vol if vol > 0 else 0
    cum  = pnl.cumsum()
    mdd  = (cum - cum.cummax()).min()
    n_tr = len(trades_df) if not trades_df.empty else 0
    hit  = (trades_df['net'] > 0).mean() if n_tr > 0 else np.nan
    avg_tc = trades_df['tc'].mean() if n_tr > 0 else 0
    print(f'\n  [{tag}]  Days={len(pnl)}  Trades={n_tr}')
    print(f'  Ann.Ret=${ann:,.0f}  Ann.Vol=${vol:,.0f}  Sharpe={sh:.2f}')
    print(f'  MaxDD=${mdd:,.0f}  HitRate(trades)={hit:.1%}  AvgTC=${avg_tc:.2f}')


# â”€â”€ SPY benchmark (market beta) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _spy_daily_ret(date_str):
    if date_str in _SPY_CACHE:
        return _SPY_CACHE[date_str]
    bench_sym = 'AAPL' if SYMBOL == 'SPY' else 'SPY'
    try:
        path = DATA_DIR / bench_sym / 'mbp-10' / f'{date_str}.parquet'
        cols = ['bid_px_00','ask_px_00','bid_sz_00','ask_sz_00']
        raw  = pd.read_parquet(path, engine='fastparquet', columns=cols).sort_index()
        if raw.index.tzinfo is None:
            raw.index = raw.index.tz_localize('UTC').tz_convert(ET)
        else:
            raw.index = raw.index.tz_convert(ET)
        raw  = raw.between_time('09:30','16:00')
        b0   = raw['bid_sz_00'].fillna(0).astype(np.int64)
        a0   = raw['ask_sz_00'].fillna(0).astype(np.int64)
        den  = np.where(b0 + a0 > 0, b0 + a0, np.nan)
        mp   = pd.Series(
            (a0 * raw['bid_px_00'].values + b0 * raw['ask_px_00'].values) / den,
            index=raw.index
        ).resample('1s').last().dropna()
        if len(mp) < 10:
            _SPY_CACHE[date_str] = np.nan
        else:
            o  = mp.between_time('09:35','09:40').mean()
            c  = mp.between_time('15:55','16:00').mean()
            _SPY_CACHE[date_str] = (c - o) / o if (np.isfinite(o) and o > 0) else np.nan
    except Exception:
        _SPY_CACHE[date_str] = np.nan
    return _SPY_CACHE[date_str]


# â”€â”€ Performance metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_metrics(daily_df, trades_df, dates):
    if daily_df.empty:
        return {}
    pnl  = daily_df['net']
    ann  = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    sh   = ann / vol if vol > 0 else 0
    cum  = pnl.cumsum()
    mdd  = (cum - cum.cummax()).min()

    n_tr     = len(trades_df) if not trades_df.empty else 0
    hit      = float((trades_df['net'] > 0).mean()) if n_tr > 0 else np.nan
    avg_tc   = float(trades_df['tc'].mean()) if n_tr > 0 else 0
    turnover = n_tr * NOTIONAL / (len(pnl) * NOTIONAL) if len(pnl) > 0 else 0
    avg_hold = HOLD_S  # seconds

    spy_rets   = np.array([_spy_daily_ret(d) for d in dates])
    strat_rets = pnl.values / NOTIONAL
    valid      = np.isfinite(spy_rets) & np.isfinite(strat_rets)
    beta, alpha_ann = np.nan, np.nan
    if valid.sum() >= 5:
        slp, icpt, *_ = stats.linregress(spy_rets[valid], strat_rets[valid])
        beta      = slp
        alpha_ann = icpt * 252

    return {
        'sharpe'   : round(sh, 3),
        'ann_ret'  : round(ann, 0),
        'ann_vol'  : round(vol, 0),
        'max_dd'   : round(mdd, 0),
        'hit_rate' : round(hit, 3) if np.isfinite(hit) else np.nan,
        'n_trades' : n_tr,
        'avg_tc'   : round(avg_tc, 2),
        'turnover' : round(turnover, 3),
        'avg_hold_s': avg_hold,
        'beta_spy' : round(beta, 3) if np.isfinite(beta) else np.nan,
        'alpha_ann': round(alpha_ann, 4) if np.isfinite(alpha_ann) else np.nan,
        'total_pnl': round(pnl.sum(), 0),
    }


# â”€â”€ Capacity analysis (Almgren-Chriss) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def capacity_analysis(trades_df):
    """
    Almgren-Chriss impact model:
      temp_impact  = Î· Ã— Ïƒ Ã— (Q / V) Ã— âˆš(T/Ï„)
      perm_impact  = Î³ Ã— Ïƒ Ã— (Q / V)

    where:
      Ïƒ = daily volatility of stock (approx from realized_vol)
      V = daily volume ($)
      Q = strategy notional per trade Ã— n_trades_per_day
      Î· â‰ˆ 0.142  (Almgren et al. 2005, AAPL proxy)
      Î³ â‰ˆ 0.071  (permanent impact coefficient)
      T = 1 day, Ï„ = 1 trade per signal

    We report impact cost in bps for $10M / $100M / $1B AUM.
    """
    print('\n--- Capacity Analysis (Almgren-Chriss) ---')
    # Typical AAPL-like parameters
    eta   = 0.142   # temporary impact coefficient
    gamma = 0.071   # permanent impact coefficient
    sigma_daily = 0.012     # ~1.2% daily vol (AAPL/NVDA/SPY proxy)
    V_daily     = 5e9       # $5B daily dollar volume (AAPL ~$5B, NVDA ~$10B, SPY ~$30B)
    T   = 1.0       # 1 trading day
    tau = 1.0 / 23  # 23 trades per day â‰ˆ typical strategy frequency

    header = f"{'AUM':>10}  {'Trades/day':>12}  {'Q/day ($)':>12}  " \
             f"{'Temp impact (bps)':>18}  {'Perm impact (bps)':>18}  {'Total impact (bps)':>20}"
    print(header)
    print('-' * len(header))

    if not trades_df.empty:
        n_trades_per_day = len(trades_df) / max(1, trades_df['date'].nunique())
    else:
        n_trades_per_day = 10

    for aum in [10e6, 100e6, 1e9]:
        scale = aum / NOTIONAL  # scale factor vs base $50K notional
        Q_day = aum * (n_trades_per_day / 23)  # notional traded per day (scaled)
        q     = Q_day / V_daily  # participation rate

        temp = eta * sigma_daily * q * np.sqrt(T / tau) * 1e4
        perm = gamma * sigma_daily * q * 1e4
        tot  = temp + perm

        label = f'${aum/1e6:.0f}M' if aum < 1e9 else '$1B'
        print(f"  {label:>8}  {n_trades_per_day*scale:>12.1f}  "
              f"{Q_day:>12,.0f}  {temp:>18.1f}  {perm:>18.1f}  {tot:>20.1f}")
    print()


# â”€â”€ Inventory reconstruction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_inventory(trades_df, all_dates):
    """
    Return (timestamps, positions) arrays at 1-min resolution.
    Position = number of shares currently held (positive=long, negative=short).
    """
    if trades_df.empty:
        return np.array([]), np.array([])

    # Build full intraday timeline at 1-min resolution
    segments = []
    for d in all_dates:
        start = pd.Timestamp(f'{d} 09:30', tz=ET)
        end   = pd.Timestamp(f'{d} 16:00', tz=ET)
        segments.append(pd.date_range(start, end, freq='1min'))
    full_idx = segments[0].append(segments[1:])
    pos = pd.Series(0.0, index=full_idx)

    for _, tr in trades_df.iterrows():
        ts_e = pd.Timestamp(tr['ts'])
        ts_x = pd.Timestamp(tr['exit_ts'])
        if ts_e.tzinfo is None:
            ts_e = ts_e.tz_localize(ET)
        if ts_x.tzinfo is None:
            ts_x = ts_x.tz_localize(ET)
        mask = (pos.index >= ts_e) & (pos.index <= ts_x)
        pos[mask] += tr['direction'] * tr['shares']

    return pos


# â”€â”€ Formatting helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fmt_k(x, pos=None):
    return f'${x/1000:.0f}k' if abs(x) >= 1000 else f'${x:.0f}'


# â”€â”€ Plot 1: Cumulative PnL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_cumulative_pnl(results, sym):
    periods = [
        ('Train\n(Jan-Feb 2024)', TRAIN_DATES),
        ('Val\n(Mar-Apr 2024)',   VAL_DATES),
        ('OOS\n(Oct 2024)',       OCT_DATES),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
    fig.suptitle(f'Cumulative PnL â€” {sym} Microstructure Strategy (60s hold)',
                 fontsize=12, y=1.01)

    bar_colors = ['#90CAF9', '#A5D6A7']
    line_cols  = ['#1565C0', '#2E7D32']

    for row, (model_name, (daily_df, _)) in enumerate(results):
        for col, (label, dates) in enumerate(periods):
            ax  = axes[row, col]
            sub = daily_df[daily_df.index.strftime('%Y-%m-%d').isin(dates)] \
                  if not daily_df.empty else pd.DataFrame()
            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='grey')
            else:
                cg = sub['gross'].cumsum().values
                cn = sub['net'].cumsum().values
                xs = range(len(sub))
                ax.bar(xs, cg, color=bar_colors[row], alpha=0.4, label='Gross')
                ax.plot(xs, cn, color=line_cols[row], linewidth=1.8, label='Net')
                ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
                ax.set_title(f'{label}\nNet ${sub["net"].sum():,.0f}', fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f'{model_name}\nCum. PnL ($)', fontsize=9)

    handles = [
        plt.Rectangle((0,0),1,1, color=bar_colors[0], alpha=0.4),
        plt.Line2D([0],[0], color=line_cols[0], linewidth=2),
    ]
    fig.legend(handles, ['Gross','Net'], loc='lower center', ncol=2,
               bbox_to_anchor=(0.5,-0.03))
    plt.tight_layout()
    fn = FIG_DIR / f'fz_{sym.lower()}_pnl.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fn.name}')


# â”€â”€ Plot 2: Rolling Sharpe (21-day) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_rolling_sharpe(results, sym):
    fig, ax = plt.subplots(figsize=(13, 4))
    colors  = {'OLS': '#1565C0', 'LightGBM': '#2E7D32'}
    styles  = {'OLS': '-',       'LightGBM': '--'}

    for model_name, (daily_df, _) in results:
        if daily_df.empty:
            continue
        pnl  = daily_df['net']
        roll = (pnl.rolling(ROLL_WIN).mean() * 252 /
                (pnl.rolling(ROLL_WIN).std() * np.sqrt(252) + 1e-9))
        key  = 'OLS' if 'OLS' in model_name else 'LightGBM'
        ax.plot(roll.values, label=model_name,
                color=colors[key], linestyle=styles[key], linewidth=1.6)

    n_tr = len(TRAIN_DATES)
    n_va = len(VAL_DATES)
    ax.axvline(n_tr,        color='grey',      linestyle=':', linewidth=1)
    ax.axvline(n_tr + n_va, color='darkorange', linestyle=':', linewidth=1.5)
    yl = ax.get_ylim()
    yt = yl[0] + (yl[1]-yl[0]) * 0.87
    ax.text(n_tr / 2,          yt, 'Train',       ha='center', fontsize=8, color='grey')
    ax.text(n_tr + n_va/2,     yt, 'Val',         ha='center', fontsize=8, color='grey')
    ax.text(n_tr + n_va + len(OCT_DATES)/2, yt,
            'OOS Oct-2024', ha='center', fontsize=8, color='darkorange')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'{ROLL_WIN}-day Rolling Sharpe (annualised) â€” {sym}', fontsize=11)
    ax.set_xlabel('Trading day')
    ax.set_ylabel('Rolling Sharpe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fn = FIG_DIR / f'fz_{sym.lower()}_rolling_sharpe.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fn.name}')


# â”€â”€ Plot 3: Inventory (position in shares) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_inventory(results, sym):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    period_colors = {'Train': '#1565C0', 'Val': '#6A1B9A', 'OOS': '#BF360C'}

    for ax, (model_name, (daily_df, trades_df)) in zip(axes, results):
        pos = build_inventory(trades_df, ALL_DATES)
        if len(pos) == 0:
            ax.text(0.5, 0.5, 'no trades', ha='center', va='center',
                    transform=ax.transAxes, color='grey')
            ax.set_ylabel(f'{model_name}\nPosition (shares)')
            continue

        # Downsample to 5-min for readability
        pos5 = pos.resample('5min').last().fillna(0)
        ax.fill_between(range(len(pos5)), pos5.values,
                        where=pos5.values > 0, color='#1565C0', alpha=0.5, label='Long')
        ax.fill_between(range(len(pos5)), pos5.values,
                        where=pos5.values < 0, color='#C62828', alpha=0.5, label='Short')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_ylabel(f'{model_name}\nPosition (shares)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f'Intraday Position in Shares â€” {sym} (5-min resolution)', fontsize=11)
    axes[1].set_xlabel('5-min bar (all dates)')
    plt.tight_layout()
    fn = FIG_DIR / f'fz_{sym.lower()}_inventory.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fn.name}')


# â”€â”€ Plot 4: Feature importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_importance(ols_model, lgbm_model, ablation_drops, sym):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # OLS coefficients
    ax   = axes[0]
    coef = pd.Series(ols_model.coef_, index=FEATURE_NAMES).sort_values()
    ax.barh(coef.index, coef.values,
            color=['#C62828' if v < 0 else '#1B5E20' for v in coef.values])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('OLS (Ridge)\nStandardised Coeff.', fontsize=10)
    ax.set_xlabel('Coefficient')
    ax.grid(True, axis='x', alpha=0.3)

    # LightGBM importance
    ax  = axes[1]
    imp = pd.Series(lgbm_model.feature_importances_,
                    index=FEATURE_NAMES).sort_values()
    ax.barh(imp.index, imp.values, color='#1565C0')
    ax.set_title('LightGBM\nFeature Importance (Gain)', fontsize=10)
    ax.set_xlabel('Importance')
    ax.grid(True, axis='x', alpha=0.3)

    # Ablation IC drop
    ax  = axes[2]
    ab  = pd.Series(ablation_drops).sort_values()
    ax.barh(ab.index, ab.values,
            color=['#4CAF50' if v >= 0 else '#F44336' for v in ab.values])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Ablation (OLS)\nIC Drop on Training', fontsize=10)
    ax.set_xlabel('IC drop (positive = adds value)')
    ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle(f'Feature Importance â€” {sym} Microstructure Factor Zoo', fontsize=12)
    plt.tight_layout()
    fn = FIG_DIR / f'fz_{sym.lower()}_importance.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fn.name}')


# â”€â”€ Plot 5: Train vs Test IC comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_ic_comparison(ic_table, sym, break_even_ic=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    models  = list(ic_table.keys())
    periods = ['Train', 'Val', 'OOS Oct-2024']
    period_colors = {'Train': '#1565C0', 'Val': '#6A1B9A', 'OOS Oct-2024': '#BF360C'}

    x = np.arange(len(models))
    w = 0.25
    for j, period in enumerate(periods):
        vals = [ic_table[m].get(period, np.nan) for m in models]
        ax.bar(x + (j - 1) * w, vals, w,
               label=period, color=period_colors[period])

    if break_even_ic is not None:
        ax.axhline(break_even_ic, color='red', linestyle='--', linewidth=1.5,
                   label=f'Break-even IC = {break_even_ic:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Pearson IC')
    ax.set_title(f'IC by Model and Period â€” {sym}\n'
                 f'(Break-even IC = spread/Ïƒy; OOS IC must exceed this to profit)',
                 fontsize=10)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fn = FIG_DIR / f'fz_{sym.lower()}_ic.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fn.name}')


# â”€â”€ Summary CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_metrics_csv(metrics_table, ic_table, sym):
    rows = []
    for model in metrics_table:
        for period in metrics_table[model]:
            m  = metrics_table[model][period]
            ic = ic_table.get(model, {}).get(period, np.nan)
            rows.append({'Model': model, 'Period': period, 'IC': round(ic, 4), **m})
    df  = pd.DataFrame(rows)
    fn  = f'factor_zoo_{sym.lower()}_final_metrics.csv'
    df.to_csv(fn, index=False)
    print(f'\nMetrics saved to {fn}')
    print(df.to_string(index=False))
    return df


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def filter_by_dates(daily_df, dates):
    if daily_df.empty:
        return pd.DataFrame(columns=['gross','net'])
    mask = daily_df.index.strftime('%Y-%m-%d').isin(dates)
    return daily_df[mask]


def filter_trades(trades_df, dates):
    if trades_df.empty:
        return pd.DataFrame()
    return trades_df[trades_df['date'].isin(dates)]


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == '__main__':
    print('=' * 65)
    print(f'FACTOR ZOO â€” {SYMBOL}  MBP-10  |  11 book features  |  60s hold')
    print('=' * 65)

    # â”€â”€ 1. Build datasets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_train, y_train, w_bounds = build_dataset(
        TRAIN_DATES, f'TRAIN  (Jan-Feb 2024) [{SYMBOL}]'
    )
    X_val, y_val, _ = build_dataset(
        VAL_DATES, f'VAL    (Mar-Apr 2024) [{SYMBOL}]',
        winsor_bounds=w_bounds
    )
    X_oct, y_oct, _ = build_dataset(
        OCT_DATES, f'OOS    (Oct 2024)     [{SYMBOL}]',
        winsor_bounds=w_bounds
    )

    # â”€â”€ 2. Train OLS (Ridge) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\n=== OLS (Ridge) [{SYMBOL}] ===')
    ols, scaler = train_ols(X_train, y_train)

    ols_pred_tr  = predict(ols, X_train, scaler)
    ols_pred_val = predict(ols, X_val,   scaler)
    ols_pred_oct = predict(ols, X_oct,   scaler)

    ols_ic_tr,  _ = ic_stats(y_train.values, ols_pred_tr)
    ols_ic_val, _ = ic_stats(y_val.values,   ols_pred_val)
    ols_ic_oct, _ = ic_stats(y_oct.values,   ols_pred_oct)
    print(f'  Train IC: {ols_ic_tr:.4f}')
    print(f'  Val   IC: {ols_ic_val:.4f}')
    print(f'  OOS   IC: {ols_ic_oct:.4f}')

    # â”€â”€ 3. Train LightGBM (early-stop on val) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\n=== LightGBM [{SYMBOL}] ===')
    lgbm = train_lgbm(X_train, y_train, X_val, y_val)
    print(f'  Best iteration: {lgbm.best_iteration_}')

    lgbm_pred_tr  = predict(lgbm, X_train)
    lgbm_pred_val = predict(lgbm, X_val)
    lgbm_pred_oct = predict(lgbm, X_oct)

    lgbm_ic_tr,  _ = ic_stats(y_train.values, lgbm_pred_tr)
    lgbm_ic_val, _ = ic_stats(y_val.values,   lgbm_pred_val)
    lgbm_ic_oct, _ = ic_stats(y_oct.values,   lgbm_pred_oct)
    print(f'  Train IC: {lgbm_ic_tr:.4f}')
    print(f'  Val   IC: {lgbm_ic_val:.4f}')
    print(f'  OOS   IC: {lgbm_ic_oct:.4f}')

    # â”€â”€ 4. Ablation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    drops, base_ic = ablation(X_train, y_train)

    # â”€â”€ 5. Break-even IC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # IC_be = full_round_trip_spread_bps / sigma_y(60s labels)
    # sigma_y is std of 60s forward returns (from training labels), NOT 1s vol
    print('\n=== Break-even IC (train statistics) ===')
    train_spreads = []
    for d in TRAIN_DATES:
        key = f'{SYMBOL}_{d}'
        if key in _FEAT_CACHE:
            spd = _FEAT_CACHE[key]['spread_bps'].dropna()
            if len(spd) > 10:
                train_spreads.append(spd.median())
    avg_spread  = np.median(train_spreads) if train_spreads else 0.5
    sigma_y     = float(y_train.std())   # std of 60s forward return in bps (correct sigma_y)
    be_ic       = avg_spread / sigma_y if sigma_y > 0 else np.nan
    print(f'  Median spread_bps (round-trip): {avg_spread:.3f}')
    print(f'  sigma_y of 60s labels (bps)   : {sigma_y:.3f}')
    print(f'  Break-even IC                 : {be_ic:.4f}')
    print(f'  OLS OOS IC / BE_IC            : {ols_ic_oct / be_ic:.3f}x')

    # â”€â”€ 6. Threshold search on val set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n=== Threshold search (OLS, val set) ===')
    best_thresh, best_ic_th = 0.3, -np.inf
    for th in np.arange(0.1, 2.0, 0.1):
        mask = np.abs(ols_pred_val) > th
        if mask.sum() < 100:
            break
        ic_th, _ = ic_stats(y_val.values[mask], ols_pred_val[mask])
        if ic_th > best_ic_th:
            best_ic_th, best_thresh = ic_th, float(th)
    print(f'  Best threshold: {best_thresh:.2f} bps  (val IC = {best_ic_th:.4f})')

    # â”€â”€ 7. Backtest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    BT_DATES = TRAIN_DATES + VAL_DATES + OCT_DATES
    print(f'\n=== Backtest: OLS [{SYMBOL}] ===')
    ols_daily, ols_trades = backtest(
        BT_DATES, ols, scaler, best_thresh, 'ols', f'OLS {SYMBOL}'
    )
    print(f'\n=== Backtest: LightGBM [{SYMBOL}] ===')
    lgbm_daily, lgbm_trades = backtest(
        BT_DATES, lgbm, None, best_thresh, 'lgbm', f'LightGBM {SYMBOL}'
    )

    # â”€â”€ 8. Metrics by period â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics_table = {}
    for mname, (dd, td) in [
        ('OLS',      (ols_daily,  ols_trades)),
        ('LightGBM', (lgbm_daily, lgbm_trades)),
    ]:
        metrics_table[mname] = {}
        for pname, dates in [
            ('Train',       TRAIN_DATES),
            ('Val',         VAL_DATES),
            ('OOS Oct-2024', OCT_DATES),
        ]:
            sub_d = filter_by_dates(dd, dates)
            sub_t = filter_trades(td, dates)
            metrics_table[mname][pname] = compute_metrics(sub_d, sub_t, dates)

    ic_table = {
        'OLS':      {'Train': ols_ic_tr,  'Val': ols_ic_val,  'OOS Oct-2024': ols_ic_oct},
        'LightGBM': {'Train': lgbm_ic_tr, 'Val': lgbm_ic_val, 'OOS Oct-2024': lgbm_ic_oct},
    }

    save_metrics_csv(metrics_table, ic_table, SYMBOL)

    # â”€â”€ 9. Capacity analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    capacity_analysis(ols_trades)

    # â”€â”€ 10. Exit-reason breakdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not ols_trades.empty and 'exit_reason' in ols_trades.columns:
        er = ols_trades['exit_reason'].value_counts()
        print('\n=== OLS Exit reasons ===')
        print(er.to_string())

    # â”€â”€ 11. Figures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n=== Generating figures ===')
    plot_cumulative_pnl([
        ('OLS',      (ols_daily,  ols_trades)),
        ('LightGBM', (lgbm_daily, lgbm_trades)),
    ], SYMBOL)

    plot_rolling_sharpe([
        ('OLS (Ridge)', (ols_daily,  ols_trades)),
        ('LightGBM',    (lgbm_daily, lgbm_trades)),
    ], SYMBOL)

    plot_inventory([
        ('OLS',      (ols_daily, ols_trades)),
        ('LightGBM', (lgbm_daily, lgbm_trades)),
    ], SYMBOL)

    plot_importance(ols, lgbm, drops, SYMBOL)

    plot_ic_comparison(ic_table, SYMBOL, break_even_ic=float(be_ic))

    print(f'\n=== Done â€” {SYMBOL} ===')
    print(f'  Figures : figures/fz_{SYMBOL.lower()}_*.png')
    print(f'  Metrics : factor_zoo_{SYMBOL.lower()}_final_metrics.csv')


