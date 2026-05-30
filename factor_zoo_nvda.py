"""
Factor Zoo â€“ AAPL L2 Microstructure (12 book features)
=======================================================
Train: Jan-Feb 2024 (18 days)
Val:   Mar-Apr 2024 (7 days)   [within-sample OOS check]
OOS:   Oct 2024    (15 days)   [official test â€” Databento ITCH MBP-10]

Features 1-12 (book-only, consistent across all periods):
  ofi_1, ofi_5, ofi_30, ofi_multilevel,
  book_imbalance, depth_ratio, bid_slope, ask_slope,
  microprice, microprice_drift, spread, realized_vol

Trade features (13-15) dropped: Oct 2024 parquets contain book columns only.
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

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR = Path('data/raw')
FIG_DIR  = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

ET = 'America/New_York'
SYMBOL = 'NVDA'  # ← parametric symbol

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

HOLD_S    = 60       # hold 60 seconds
NOTIONAL  = 50_000   # $50K per trade
OPEN_BUF  = 5 * 60   # skip first 5 minutes = 300 bars
CLOSE_BUF = 60       # skip last 60 seconds (no label available)
LEVELS    = 5
KILL_PCT  = 0.02     # daily kill-switch: -2% of notional

FEATURE_NAMES = [
    'ofi_1','ofi_5','ofi_30','ofi_multilevel',
    'book_imbalance','depth_ratio','bid_slope','ask_slope',
    'microprice_drift','spread_bps','realized_vol',
]
# microprice excluded: non-stationary (price level ~$185 in train, ~$220 in Oct 2024)
# spread_bps = spread/microprice*1e4 replaces raw spread (stationary ratio)

BOOK_COLS = (
    [f'bid_px_{i:02d}' for i in range(10)] +
    [f'ask_px_{i:02d}' for i in range(10)] +
    [f'bid_sz_{i:02d}' for i in range(10)] +
    [f'ask_sz_{i:02d}' for i in range(10)]
)

# In-memory feature cache: avoids reloading parquets in backtest
_FEAT_CACHE: dict = {}


# â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_day(date_str, symbol='AAPL'):
    path = DATA_DIR / symbol / 'mbp-10' / f'{date_str}.parquet'
    mbp = pd.read_parquet(path, engine='fastparquet', columns=BOOK_COLS).sort_index()
    if mbp.index.tzinfo is None:
        mbp.index = mbp.index.tz_localize('UTC').tz_convert(ET)
    else:
        mbp.index = mbp.index.tz_convert(ET)
    mbp = mbp.between_time('09:30', '16:00')
    sz_cols = [c for c in mbp.columns if '_sz_' in c]
    mbp[sz_cols] = mbp[sz_cols].fillna(0).astype(np.int64)
    return mbp


# â”€â”€ Feature engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _book_slope(prefix, b):
    """OLS slope of queue depth across 5 levels (vectorized)."""
    qs = np.stack([b[f'{prefix}_{i:02d}'].values for i in range(LEVELS)], axis=1).astype(float)
    # For x=[0,1,2,3,4]: OLS slope = (5*sum(i*qi) - 10*sum(qi)) / 50
    idx = np.arange(LEVELS, dtype=float)
    weighted = qs @ idx
    total    = qs.sum(axis=1)
    return (LEVELS * weighted - idx.sum() * total) / 50.0


def compute_features(mbp):
    """Returns 1-second bars with 12 book features + microprice column."""
    b = mbp.resample('1s').last().dropna(subset=['bid_px_00'])

    bid0   = b['bid_sz_00'].values.astype(np.int64)
    ask0   = b['ask_sz_00'].values.astype(np.int64)
    denom0 = np.where(bid0 + ask0 > 0, bid0 + ask0, np.nan)

    # 1. OFI instantaneous
    ofi_raw = (bid0 - ask0) / denom0
    ofi_s   = pd.Series(ofi_raw, index=b.index)
    # 2-3. Rolling OFI
    ofi_5  = ofi_s.rolling(5,  min_periods=1).mean().values
    ofi_30 = ofi_s.rolling(30, min_periods=1).mean().values

    # 4. Multi-level OFI (5 levels, equal weight)
    ml_ofi = np.zeros(len(b))
    for i in range(LEVELS):
        bi = b[f'bid_sz_{i:02d}'].values.astype(np.int64)
        ai = b[f'ask_sz_{i:02d}'].values.astype(np.int64)
        d  = np.where(bi + ai > 0, bi + ai, np.nan)
        ml_ofi += np.where(np.isfinite(d), (bi - ai) / d, 0.0)
    ml_ofi /= LEVELS

    # 5. Book imbalance
    book_imbal = bid0 / denom0

    # 6. Depth ratio: top-3 vs levels 3-9 (both sides)
    top3 = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) +
               b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3))
    deep = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) +
               b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3, 10))
    depth_ratio = top3 / np.where(deep > 0, deep, np.nan)

    # 7-8. Book slopes
    bid_slope = _book_slope('bid_sz', b)
    ask_slope = _book_slope('ask_sz', b)

    # 9. Microprice
    micro_num  = ask0 * b['bid_px_00'].values + bid0 * b['ask_px_00'].values
    microprice = micro_num / denom0
    micro_s    = pd.Series(microprice, index=b.index)

    # 10. Microprice drift: 10s finite difference in bps/s (fast; polyfit equiv)
    micro_drift = micro_s.diff(10) / (10 * micro_s.shift(10)) * 1e4

    # 11. Spread in bps (stationary: 0.01/190*1e4 â‰ˆ 0.53 bps regardless of price era)
    spread_raw = b['ask_px_00'].values - b['bid_px_00'].values
    spread_bps = np.where(microprice > 0, spread_raw / microprice * 1e4, np.nan)

    # 12. Realized vol: 60s rolling std of bps returns
    bps_ret      = micro_s.pct_change() * 1e4
    realized_vol = bps_ret.rolling(60, min_periods=10).std()

    return pd.DataFrame({
        'ofi_1':            ofi_raw,
        'ofi_5':            ofi_5,
        'ofi_30':           ofi_30,
        'ofi_multilevel':   ml_ofi,
        'book_imbalance':   book_imbal,
        'depth_ratio':      depth_ratio,
        'bid_slope':        bid_slope,
        'ask_slope':        ask_slope,
        'microprice':       microprice,       # kept for label & backtest only
        'microprice_drift': micro_drift.values,
        'spread_raw':       spread_raw,       # kept for backtest TC calculation
        'spread_bps':       spread_bps,       # feature: normalized spread
        'realized_vol':     realized_vol.values,
    }, index=b.index)


def make_labels(feat_df, hold_s=HOLD_S):
    """60-second forward microprice return in bps."""
    mp = feat_df['microprice']
    return (mp.shift(-hold_s) / mp - 1) * 1e4


# â”€â”€ Dataset builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_dataset(dates, tag='building'):
    print(f'\n[{tag}] Loading {len(dates)} days...')
    Xs, ys = [], []
    for d in dates:
        try:
            if f'{SYMBOL}_{d}' not in _FEAT_CACHE:
                mbp = load_day(d, SYMBOL)
                _FEAT_CACHE[f'{SYMBOL}_{d}'] = compute_features(mbp)
                del mbp
            feat = _FEAT_CACHE[f'{SYMBOL}_{d}']
            lbl  = make_labels(feat)
            n    = len(feat)
            sl   = slice(OPEN_BUF, n - CLOSE_BUF)
            extra = [c for c in ['microprice', 'spread_raw'] if c not in FEATURE_NAMES]
            X    = feat.iloc[sl][FEATURE_NAMES + extra].copy()
            y    = lbl.iloc[sl]
            mask = X[FEATURE_NAMES].notna().all(axis=1) & y.notna()
            Xs.append(X[mask]);  ys.append(y[mask])
            print(f'  {d}: {mask.sum():,} bars')
        except Exception as e:
            print(f'  {d}: ERROR - {e}')
    if not Xs:
        raise RuntimeError('No data loaded.')
    X_all = pd.concat(Xs)
    y_all = pd.concat(ys)
    print(f'  Total: {len(X_all):,} bars | y mean={y_all.mean():.3f} std={y_all.std():.3f} bps')
    return X_all, y_all


# â”€â”€ Model training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_ols(X_train, y_train):
    scaler = StandardScaler().fit(X_train[FEATURE_NAMES])
    Xs = scaler.transform(X_train[FEATURE_NAMES])
    model = Ridge(alpha=1.0)
    model.fit(Xs, y_train.values)
    return model, scaler


def train_lgbm(X_train, y_train, X_val=None, y_val=None):
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=4, num_leaves=15, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        n_jobs=2, random_state=42, verbose=-1
    )
    fit_kw = {}
    if X_val is not None:
        fit_kw['eval_set'] = [(X_val[FEATURE_NAMES].values, y_val.values)]
        fit_kw['callbacks'] = [lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)]
    model.fit(X_train[FEATURE_NAMES].values, y_train.values, **fit_kw)
    return model


def predict(model, X, scaler=None):
    Xf = X[FEATURE_NAMES].values
    if scaler is not None:
        Xf = scaler.transform(Xf)
    return model.predict(Xf)


def ic_stats(y_true, y_pred):
    """Returns (pearson_ic, spearman_ic)."""
    v = np.isfinite(y_true) & np.isfinite(y_pred)
    if v.sum() < 10:
        return np.nan, np.nan
    p = np.corrcoef(y_true[v], y_pred[v])[0, 1]
    s, _ = stats.spearmanr(y_true[v], y_pred[v])
    return float(p), float(s)


# â”€â”€ Ablation (drop-one IC on training set) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def ablation(X_train, y_train):
    print('\n[Ablation] Drop-one IC on training set (OLS)...')
    sc_full = StandardScaler().fit(X_train[FEATURE_NAMES])
    base_pred = Ridge(alpha=1.0).fit(
        sc_full.transform(X_train[FEATURE_NAMES]), y_train
    ).predict(sc_full.transform(X_train[FEATURE_NAMES]))
    base_ic, _ = ic_stats(y_train.values, base_pred)

    drops = {}
    for feat in FEATURE_NAMES:
        cols = [f for f in FEATURE_NAMES if f != feat]
        sc   = StandardScaler().fit(X_train[cols])
        pred = Ridge(alpha=1.0).fit(
            sc.transform(X_train[cols]), y_train
        ).predict(sc.transform(X_train[cols]))
        ic_drop, _ = ic_stats(y_train.values, pred)
        drops[feat] = base_ic - ic_drop
        print(f'  -{feat:<25} base_ic={base_ic:.4f}  ic={ic_drop:.4f}  drop={drops[feat]:+.5f}')
    return drops, base_ic


# â”€â”€ Backtest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def backtest(dates, model, scaler, threshold, model_type, tag=''):
    """
    Simulates 60s hold HFT strategy on the given dates.
    Returns (daily_pnl: Series, trades_df: DataFrame).
    """
    trades    = []
    daily_pnl = {}

    for d in dates:
        try:
            if f'{SYMBOL}_{d}' not in _FEAT_CACHE:
                mbp = load_day(d, SYMBOL)
                _FEAT_CACHE[f'{SYMBOL}_{d}'] = compute_features(mbp)
                del mbp
            feat = _FEAT_CACHE[f'{SYMBOL}_{d}']
            n    = len(feat)
            bars = feat.iloc[OPEN_BUF : n - CLOSE_BUF].copy()
            bars = bars[bars[FEATURE_NAMES].notna().all(axis=1)]
            if len(bars) == 0:
                continue

            preds    = predict(model, bars, scaler if model_type == 'ols' else None)
            bar_idx  = bars.index  # DatetimeIndex for binary search

            day_net   = 0.0
            day_gross = 0.0
            killed    = False
            next_entry_i = 0  # enforce one trade at a time (hold 60s before re-entering)

            for i, (ts, pred) in enumerate(zip(bar_idx, preds)):
                if killed or i < next_entry_i:
                    continue
                if abs(pred) < threshold:
                    continue

                mp     = bars['microprice'].iat[i]
                spread = bars['spread_raw'].iat[i]
                if not (np.isfinite(mp) and np.isfinite(spread) and mp > 0 and spread > 0):
                    continue

                direction = 1 if pred > 0 else -1
                shares    = max(1, int(NOTIONAL / mp))

                # Fast binary search for exit bar (O(log n) vs O(n))
                exit_ts  = ts + pd.Timedelta(seconds=HOLD_S)
                exit_idx = bar_idx.searchsorted(exit_ts)
                if exit_idx >= len(bars):
                    continue
                exit_mp  = bars['microprice'].iat[exit_idx]
                exit_spd = bars['spread_raw'].iat[exit_idx]
                if not (np.isfinite(exit_mp) and np.isfinite(exit_spd)):
                    continue

                gross = direction * (exit_mp - mp) * shares
                tc    = (spread / 2 + exit_spd / 2) * shares
                net   = gross - tc

                day_gross += gross
                day_net   += net
                next_entry_i = exit_idx  # wait until after exit before next trade

                trades.append({
                    'date': d, 'ts': str(ts), 'direction': direction,
                    'pred_bps': round(float(pred), 3),
                    'entry_mp': round(float(mp), 4),
                    'exit_mp':  round(float(exit_mp), 4),
                    'shares': int(shares), 'gross': round(gross, 2),
                    'tc': round(tc, 2), 'net': round(net, 2),
                })

                if day_net < -NOTIONAL * KILL_PCT:
                    killed = True

            daily_pnl[d] = {'gross': day_gross, 'net': day_net}
            del feat

        except Exception as e:
            print(f'  {d}: backtest error - {e}')

    if not daily_pnl:
        return pd.DataFrame(), pd.DataFrame()

    daily_df  = pd.DataFrame(daily_pnl).T
    daily_df.index = pd.to_datetime(daily_df.index)
    trades_df = pd.DataFrame(trades)

    _print_backtest_stats(daily_df, trades_df, tag)
    return daily_df, trades_df


def _print_backtest_stats(daily_df, trades_df, tag):
    if daily_df.empty:
        print(f'  [{tag}] No trades.')
        return
    pnl  = daily_df['net']
    ann  = pnl.mean() * 252
    vol  = pnl.std()  * np.sqrt(252)
    sh   = ann / vol if vol > 0 else 0
    cum  = pnl.cumsum()
    mdd  = (cum - cum.cummax()).min()
    hit  = (pnl > 0).mean()
    n_tr = len(trades_df) if not trades_df.empty else 0
    avg_tc = trades_df['tc'].mean() if not trades_df.empty else 0
    print(f'\n  [{tag}]  Days={len(pnl)}  Trades={n_tr}')
    print(f'  Ann.Return=${ann:,.0f}  Ann.Vol=${vol:,.0f}  Sharpe={sh:.2f}')
    print(f'  MaxDD=${mdd:,.0f}  HitRate={hit:.1%}  AvgTC=${avg_tc:.2f}')


# â”€â”€ SPY daily return (for beta calculation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def spy_daily_ret(date_str):
    spy_key = f'SPY_{date_str}'
    try:
        if spy_key not in _FEAT_CACHE:
            path = DATA_DIR / 'SPY' / 'mbp-10' / f'{date_str}.parquet'
            cols = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
            raw = pd.read_parquet(path, engine='fastparquet', columns=cols).sort_index()
            if raw.index.tzinfo is None:
                raw.index = raw.index.tz_localize('UTC').tz_convert(ET)
            else:
                raw.index = raw.index.tz_convert(ET)
            raw = raw.between_time('09:30', '16:00')
            b0  = raw['bid_sz_00'].fillna(0).astype(np.int64)
            a0  = raw['ask_sz_00'].fillna(0).astype(np.int64)
            den = np.where(b0 + a0 > 0, b0 + a0, np.nan)
            mp  = (a0 * raw['bid_px_00'].values + b0 * raw['ask_px_00'].values) / den
            _FEAT_CACHE[spy_key] = pd.Series(mp, index=raw.index).resample('1s').last().dropna()
        mp = _FEAT_CACHE[spy_key]
        if len(mp) < 10:
            return np.nan
        open_mp  = mp.between_time('09:35', '09:40').mean()
        close_mp = mp.between_time('15:55', '16:00').mean()
        if not (np.isfinite(open_mp) and np.isfinite(close_mp) and open_mp > 0):
            return np.nan
        return (close_mp - open_mp) / open_mp
    except Exception:
        return np.nan


# â”€â”€ Performance metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_metrics(daily_df, trades_df, dates, notional=NOTIONAL):
    if daily_df.empty:
        return {}
    pnl  = daily_df['net']
    ann  = pnl.mean() * 252
    vol  = pnl.std()  * np.sqrt(252)
    sh   = ann / vol if vol > 0 else 0
    cum  = pnl.cumsum()
    mdd  = (cum - cum.cummax()).min()
    hit  = (pnl > 0).mean()

    n_tr     = len(trades_df) if not trades_df.empty else 0
    turnover = n_tr * notional / len(pnl) / notional if len(pnl) > 0 else 0
    avg_hold = HOLD_S

    # CAPM vs SPY
    spy_rets = np.array([spy_daily_ret(d) for d in dates])
    strat_rets = pnl.values / notional
    valid = np.isfinite(spy_rets) & np.isfinite(strat_rets)
    beta, alpha = np.nan, np.nan
    if valid.sum() >= 5:
        slope, intercept, *_ = stats.linregress(spy_rets[valid], strat_rets[valid])
        beta  = slope
        alpha = intercept * 252  # annualized

    return {
        'sharpe':    round(sh, 3),
        'ann_ret':   round(ann, 0),
        'ann_vol':   round(vol, 0),
        'max_dd':    round(mdd, 0),
        'hit_rate':  round(hit, 3),
        'n_trades':  n_tr,
        'turnover':  round(turnover, 3),
        'avg_hold_s': avg_hold,
        'beta_spy':  round(beta, 3) if np.isfinite(beta)  else np.nan,
        'alpha_ann': round(alpha, 4) if np.isfinite(alpha) else np.nan,
        'total_pnl': round(pnl.sum(), 0),
    }


# â”€â”€ Plotting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _fmt_k(x, pos=None):
    return f'${x/1000:.0f}k' if abs(x) >= 1000 else f'${x:.0f}'


def plot_cumulative_pnl(results, filename='factor_zoo_nvda_pnl.png'):
    """
    results: list of (label, daily_df, colour)
    Plots cumulative gross & net PnL.
    """
    periods = [
        ('Training period\n(Jan-Feb 2024)', TRAIN_DATES),
        ('Val. period\n(Mar-Apr 2024)',      VAL_DATES),
        ('OOS Test\n(Oct 2024)',             OCT_DATES),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
    fig.suptitle('Cumulative PnL â€” AAPL Microstructure Strategy (60s hold)', fontsize=12, y=1.01)

    row_labels = ['OLS (Ridge)', 'LightGBM']
    colors_gross = ['#90CAF9', '#A5D6A7']
    colors_net   = ['#1565C0', '#2E7D32']

    for row, (model_name, (daily_df, trades_df)) in enumerate(results):
        for col, (period_label, period_dates) in enumerate(periods):
            ax = axes[row, col]

            # Filter daily_df to this period
            mask = daily_df.index.strftime('%Y-%m-%d').isin(period_dates)
            sub  = daily_df[mask]

            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='grey')
            else:
                cum_gross = sub['gross'].cumsum().values
                cum_net   = sub['net'].cumsum().values
                xs        = range(len(sub))
                ax.bar(xs, cum_gross, color=colors_gross[row], alpha=0.4, label='Gross')
                ax.plot(xs, cum_net, color=colors_net[row], linewidth=1.8, label='Net')
                ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
                total = sub['net'].sum()
                ax.set_title(f'{period_label}\nNet total: ${total:,.0f}', fontsize=9)

            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f'{model_name}\nCum. PnL ($)', fontsize=9)
            if row == 0:
                ax.set_xlabel('Trading day')

    handles = [
        plt.Rectangle((0,0),1,1, color=colors_gross[0], alpha=0.4),
        plt.Line2D([0],[0], color=colors_net[0], linewidth=2),
    ]
    fig.legend(handles, ['Gross', 'Net'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_rolling_sharpe(results, window=5, filename='factor_zoo_nvda_rolling_sharpe.png'):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {'OLS': '#1565C0', 'LightGBM': '#2E7D32'}
    styles = {'OLS': '-', 'LightGBM': '--'}

    for model_name, (daily_df, _) in results:
        if daily_df.empty:
            continue
        pnl = daily_df['net']
        roll_sh = (pnl.rolling(window).mean() * 252 /
                   (pnl.rolling(window).std() * np.sqrt(252) + 1e-9))
        key = 'OLS' if 'OLS' in model_name else 'LightGBM'
        ax.plot(roll_sh.values, label=model_name,
                color=colors[key], linestyle=styles[key], linewidth=1.6)

    # Mark period boundaries
    n_train = len(TRAIN_DATES)
    n_val   = len(VAL_DATES)
    ax.axvline(n_train, color='grey', linestyle=':', linewidth=1)
    ax.axvline(n_train + n_val, color='darkorange', linestyle=':', linewidth=1.5)
    ylims = ax.get_ylim()
    y_label = ylims[0] + (ylims[1] - ylims[0]) * 0.85
    ax.text(n_train / 2, y_label, 'Train', ha='center', fontsize=8, color='grey')
    ax.text(n_train + n_val / 2, y_label, 'Val', ha='center', fontsize=8, color='grey')
    ax.text(n_train + n_val + len(OCT_DATES) / 2, y_label,
            'OOS Oct 2024', ha='center', fontsize=8, color='darkorange')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'{window}-day Rolling Sharpe (annualized)', fontsize=11)
    ax.set_xlabel('Trading day')
    ax.set_ylabel('Rolling Sharpe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_importance(ols_model, ols_scaler, lgbm_model, ablation_drops,
                    filename='factor_zoo_nvda_importance.png'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # â”€â”€ OLS coefficients â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax = axes[0]
    coef = pd.Series(ols_model.coef_, index=FEATURE_NAMES).sort_values()
    colors = ['#C62828' if v < 0 else '#1B5E20' for v in coef.values]
    ax.barh(coef.index, coef.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('OLS (Ridge)\nStandardized Coefficients', fontsize=10)
    ax.set_xlabel('Coefficient')
    ax.grid(True, axis='x', alpha=0.3)

    # â”€â”€ LightGBM feature importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax = axes[1]
    imp = pd.Series(lgbm_model.feature_importances_,
                    index=FEATURE_NAMES).sort_values()
    ax.barh(imp.index, imp.values, color='#1565C0')
    ax.set_title('LightGBM\nFeature Importance (Gain)', fontsize=10)
    ax.set_xlabel('Importance')
    ax.grid(True, axis='x', alpha=0.3)

    # â”€â”€ Ablation IC drop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax = axes[2]
    ab = pd.Series(ablation_drops).sort_values()
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in ab.values]
    ax.barh(ab.index, ab.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Ablation Study (OLS)\nIC Drop When Feature Removed', fontsize=10)
    ax.set_xlabel('IC drop (positive = feature adds value)')
    ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Feature Importance â€” AAPL Microstructure Factor Zoo', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_ic_comparison(ic_table, filename='factor_zoo_nvda_ic_comparison.png'):
    """Bar chart: IC by model Ã— period."""
    fig, ax = plt.subplots(figsize=(8, 4))
    df = pd.DataFrame(ic_table).T   # models as rows, periods as columns
    x  = np.arange(len(df))
    w  = 0.25
    cols = list(df.columns)
    period_colors = {'Train': '#1565C0', 'Val': '#6A1B9A', 'OOS Oct-2024': '#BF360C'}

    for i, period in enumerate(cols):
        offset = (i - len(cols)/2 + 0.5) * w
        ax.bar(x + offset, df[period].values, w,
               label=period, color=period_colors.get(period, f'C{i}'))

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Pearson IC')
    ax.set_title('Information Coefficient by Model and Period', fontsize=11)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_inventory(daily_df_ols, daily_df_lgbm, filename='factor_zoo_nvda_inventory.png'):
    """Daily PnL as bar chart (proxy for inventory/activity)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (label, daily_df) in zip(axes, [
        ('OLS (Ridge)', daily_df_ols),
        ('LightGBM', daily_df_lgbm)
    ]):
        if daily_df.empty:
            continue
        pnl = daily_df['net']
        colors = ['#2E7D32' if v >= 0 else '#C62828' for v in pnl.values]
        ax.bar(range(len(pnl)), pnl.values, color=colors)
        ax.axhline(0, color='black', linewidth=0.6)
        # Mark period boundaries
        n_tr = len(TRAIN_DATES)
        n_va = len(VAL_DATES)
        ax.axvline(n_tr - 0.5, color='grey', linestyle=':', linewidth=1)
        ax.axvline(n_tr + n_va - 0.5, color='darkorange', linestyle=':', linewidth=1.5)
        ax.set_ylabel(f'{label}\nNet PnL ($)', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Daily Net PnL by Model\n(Train | Val | OOS Oct 2024)', fontsize=11)
    axes[1].set_xlabel('Trading day')
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


# â”€â”€ Summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_summary_table(metrics_table, ic_table, filename='factor_zoo_nvda_metrics.csv'):
    rows = []
    for model in metrics_table:
        for period in metrics_table[model]:
            m = metrics_table[model][period]
            ic = ic_table.get(model, {}).get(period, np.nan)
            rows.append({'Model': model, 'Period': period, 'IC': ic, **m})
    df = pd.DataFrame(rows).set_index(['Model', 'Period'])
    df.to_csv(filename)
    print(f'\nSummary saved to {filename}')
    print(df.to_string())
    return df


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == '__main__':
    import sys

    # â”€â”€ 1. Build datasets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('=' * 60)
    print('FACTOR ZOO â€” AAPL MBP-10 (11 Book Features)')
    print('=' * 60)

    X_train, y_train = build_dataset(TRAIN_DATES, 'TRAIN  (Jan-Feb 2024)')
    X_val,   y_val   = build_dataset(VAL_DATES,   'VAL    (Mar-Apr 2024)')
    X_oct,   y_oct   = build_dataset(OCT_DATES,   'OOS    (Oct 2024)')

    # â”€â”€ 2. Train OLS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n=== OLS (Ridge) ===')
    ols, scaler = train_ols(X_train, y_train)

    ols_pred_tr  = predict(ols, X_train, scaler)
    ols_pred_val = predict(ols, X_val,   scaler)
    ols_pred_oct = predict(ols, X_oct,   scaler)

    ols_ic_tr,  ols_ic_tr_s  = ic_stats(y_train.values, ols_pred_tr)
    ols_ic_val, ols_ic_val_s = ic_stats(y_val.values,   ols_pred_val)
    ols_ic_oct, ols_ic_oct_s = ic_stats(y_oct.values,   ols_pred_oct)
    print(f'  Train IC: {ols_ic_tr:.4f} (Spearman {ols_ic_tr_s:.4f})')
    print(f'  Val   IC: {ols_ic_val:.4f} (Spearman {ols_ic_val_s:.4f})')
    print(f'  OOS   IC: {ols_ic_oct:.4f} (Spearman {ols_ic_oct_s:.4f})')

    # â”€â”€ 3. Train LightGBM (early-stop on val) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n=== LightGBM (early-stop on val) ===')
    lgbm = train_lgbm(X_train, y_train, X_val, y_val)
    print(f'  Best iteration: {lgbm.best_iteration_}')

    lgbm_pred_tr  = predict(lgbm, X_train)
    lgbm_pred_val = predict(lgbm, X_val)
    lgbm_pred_oct = predict(lgbm, X_oct)

    lgbm_ic_tr,  lgbm_ic_tr_s  = ic_stats(y_train.values, lgbm_pred_tr)
    lgbm_ic_val, lgbm_ic_val_s = ic_stats(y_val.values,   lgbm_pred_val)
    lgbm_ic_oct, lgbm_ic_oct_s = ic_stats(y_oct.values,   lgbm_pred_oct)
    print(f'  Train IC: {lgbm_ic_tr:.4f} (Spearman {lgbm_ic_tr_s:.4f})')
    print(f'  Val   IC: {lgbm_ic_val:.4f} (Spearman {lgbm_ic_val_s:.4f})')
    print(f'  OOS   IC: {lgbm_ic_oct:.4f} (Spearman {lgbm_ic_oct_s:.4f})')

    # â”€â”€ 4. Ablation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    drops, base_ic = ablation(X_train, y_train)

    # â”€â”€ 5. Threshold optimization on val set (OLS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n=== Threshold search (OLS, val set) ===')
    ols_val_preds = predict(ols, X_val, scaler)
    best_thresh, best_ic = 0.3, -np.inf
    for th in np.arange(0.1, 2.0, 0.1):
        mask = np.abs(ols_val_preds) > th
        if mask.sum() < 200:
            break
        ic_th, _ = ic_stats(y_val.values[mask], ols_val_preds[mask])
        if ic_th > best_ic:
            best_ic, best_thresh = ic_th, th
    print(f'  Best threshold: {best_thresh:.2f} bps (val IC={best_ic:.4f})')

    # â”€â”€ 6. Backtest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ALL_BT_DATES = TRAIN_DATES + VAL_DATES + OCT_DATES

    print('\n=== Backtest: OLS ===')
    ols_daily, ols_trades = backtest(
        ALL_BT_DATES, ols, scaler, best_thresh, 'ols', 'OLS (all periods)'
    )

    print('\n=== Backtest: LightGBM ===')
    lgbm_daily, lgbm_trades = backtest(
        ALL_BT_DATES, lgbm, None, best_thresh, 'lgbm', 'LightGBM (all periods)'
    )

    # â”€â”€ 7. Metrics by period â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def filter_by_dates(daily_df, dates):
        if daily_df.empty:
            return pd.DataFrame(columns=['gross','net'])
        mask = daily_df.index.strftime('%Y-%m-%d').isin(dates)
        return daily_df[mask]

    def filter_trades_by_dates(trades_df, dates):
        if trades_df.empty:
            return pd.DataFrame()
        return trades_df[trades_df['date'].isin(dates)]

    metrics_table = {}
    for model_name, (daily_df, trades_df) in [
        ('OLS', (ols_daily, ols_trades)),
        ('LightGBM', (lgbm_daily, lgbm_trades)),
    ]:
        metrics_table[model_name] = {}
        for period_name, dates in [
            ('Train', TRAIN_DATES), ('Val', VAL_DATES), ('OOS Oct-2024', OCT_DATES)
        ]:
            sub_d = filter_by_dates(daily_df, dates)
            sub_t = filter_trades_by_dates(trades_df, dates)
            metrics_table[model_name][period_name] = compute_metrics(
                sub_d, sub_t, dates
            )

    ic_table = {
        'OLS': {
            'Train': ols_ic_tr, 'Val': ols_ic_val, 'OOS Oct-2024': ols_ic_oct
        },
        'LightGBM': {
            'Train': lgbm_ic_tr, 'Val': lgbm_ic_val, 'OOS Oct-2024': lgbm_ic_oct
        },
    }

    summary_df = save_summary_table(metrics_table, ic_table)

    # â”€â”€ 8. Figures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n=== Generating figures ===')
    plot_cumulative_pnl([
        ('OLS', (ols_daily,  ols_trades)),
        ('LightGBM', (lgbm_daily, lgbm_trades)),
    ])

    plot_rolling_sharpe([
        ('OLS (Ridge)', (ols_daily,  ols_trades)),
        ('LightGBM',    (lgbm_daily, lgbm_trades)),
    ])

    plot_importance(ols, scaler, lgbm, drops)

    plot_ic_comparison(ic_table)

    plot_inventory(ols_daily, lgbm_daily)

    print('\n=== Done. ===')
    print(f'Figures saved to {FIG_DIR}/')
    print(f'Metrics saved to factor_zoo_nvda_metrics.csv')

