"""
Factor Zoo – Multi-Stock: AAPL, NVDA, SPY (Oct 2024 OOS)
=========================================================
Runs the full Factor Zoo pipeline for each symbol independently,
then produces a cross-stock summary table and comparison plots.

Train: Jan-Feb 2024 (18 days)
Val:   Mar-Apr 2024 (7 days)
OOS:   Oct 2024    (17 days)
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

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path('data/raw')
FIG_DIR   = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

ET = 'America/New_York'

SYMBOLS = ['AAPL', 'NVDA', 'SPY']

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

HOLD_S    = 60
NOTIONAL  = 50_000
OPEN_BUF  = 5 * 60
CLOSE_BUF = 60
LEVELS    = 5
KILL_PCT  = 0.02

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

# Per-symbol feature cache: keys are "{SYMBOL}_{date}"
_FEAT_CACHE: dict = {}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_day(date_str, symbol):
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


# ── Feature engineering ───────────────────────────────────────────────────────
def _book_slope(prefix, b):
    qs  = np.stack([b[f'{prefix}_{i:02d}'].values for i in range(LEVELS)], axis=1).astype(float)
    idx = np.arange(LEVELS, dtype=float)
    weighted = qs @ idx
    total    = qs.sum(axis=1)
    return (LEVELS * weighted - idx.sum() * total) / 50.0


def compute_features(mbp):
    b = mbp.resample('1s').last().dropna(subset=['bid_px_00'])

    bid0   = b['bid_sz_00'].values.astype(np.int64)
    ask0   = b['ask_sz_00'].values.astype(np.int64)
    denom0 = np.where(bid0 + ask0 > 0, bid0 + ask0, np.nan)

    ofi_raw = (bid0 - ask0) / denom0
    ofi_s   = pd.Series(ofi_raw, index=b.index)
    ofi_5   = ofi_s.rolling(5,  min_periods=1).mean().values
    ofi_30  = ofi_s.rolling(30, min_periods=1).mean().values

    ml_ofi = np.zeros(len(b))
    for i in range(LEVELS):
        bi = b[f'bid_sz_{i:02d}'].values.astype(np.int64)
        ai = b[f'ask_sz_{i:02d}'].values.astype(np.int64)
        d  = np.where(bi + ai > 0, bi + ai, np.nan)
        ml_ofi += np.where(np.isfinite(d), (bi - ai) / d, 0.0)
    ml_ofi /= LEVELS

    book_imbal = bid0 / denom0

    top3 = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) +
               b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3))
    deep = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) +
               b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3, 10))
    depth_ratio = top3 / np.where(deep > 0, deep, np.nan)

    bid_slope = _book_slope('bid_sz', b)
    ask_slope = _book_slope('ask_sz', b)

    micro_num  = ask0 * b['bid_px_00'].values + bid0 * b['ask_px_00'].values
    microprice = micro_num / denom0
    micro_s    = pd.Series(microprice, index=b.index)

    micro_drift = micro_s.diff(10) / (10 * micro_s.shift(10)) * 1e4

    spread_raw = b['ask_px_00'].values - b['bid_px_00'].values
    spread_bps = np.where(microprice > 0, spread_raw / microprice * 1e4, np.nan)

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
        'microprice':       microprice,
        'microprice_drift': micro_drift.values,
        'spread_raw':       spread_raw,
        'spread_bps':       spread_bps,
        'realized_vol':     realized_vol.values,
    }, index=b.index)


def make_labels(feat_df, hold_s=HOLD_S):
    mp = feat_df['microprice']
    return (mp.shift(-hold_s) / mp - 1) * 1e4


# ── Dataset builder ───────────────────────────────────────────────────────────
def build_dataset(dates, symbol, tag='building'):
    print(f'\n[{tag}] Loading {len(dates)} days...')
    Xs, ys = [], []
    available = []
    for d in dates:
        cache_key = f'{symbol}_{d}'
        try:
            if cache_key not in _FEAT_CACHE:
                mbp = load_day(d, symbol)
                _FEAT_CACHE[cache_key] = compute_features(mbp)
                del mbp
            feat = _FEAT_CACHE[cache_key]
            lbl  = make_labels(feat)
            n    = len(feat)
            sl   = slice(OPEN_BUF, n - CLOSE_BUF)
            extra = [c for c in ['microprice', 'spread_raw'] if c not in FEATURE_NAMES]
            X    = feat.iloc[sl][FEATURE_NAMES + extra].copy()
            y    = lbl.iloc[sl]
            mask = X[FEATURE_NAMES].notna().all(axis=1) & y.notna()
            Xs.append(X[mask]);  ys.append(y[mask])
            available.append(d)
            print(f'  {d}: {mask.sum():,} bars')
        except Exception as e:
            print(f'  {d}: SKIP - {e}')
    if not Xs:
        raise RuntimeError(f'No data loaded for {symbol}.')
    X_all = pd.concat(Xs)
    y_all = pd.concat(ys)
    print(f'  Total: {len(X_all):,} bars | y mean={y_all.mean():.3f} std={y_all.std():.3f} bps')
    return X_all, y_all, available


# ── Model training ────────────────────────────────────────────────────────────
def train_ols(X_train, y_train):
    scaler = StandardScaler().fit(X_train[FEATURE_NAMES])
    model  = Ridge(alpha=1.0)
    model.fit(scaler.transform(X_train[FEATURE_NAMES]), y_train.values)
    return model, scaler


def train_lgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=4, num_leaves=15, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        n_jobs=2, random_state=42, verbose=-1
    )
    model.fit(
        X_train[FEATURE_NAMES].values, y_train.values,
        eval_set=[(X_val[FEATURE_NAMES].values, y_val.values)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )
    return model


def predict(model, X, scaler=None):
    Xf = X[FEATURE_NAMES].values
    if scaler is not None:
        Xf = scaler.transform(Xf)
    return model.predict(Xf)


def ic_stats(y_true, y_pred):
    v = np.isfinite(y_true) & np.isfinite(y_pred)
    if v.sum() < 10:
        return np.nan, np.nan
    p = np.corrcoef(y_true[v], y_pred[v])[0, 1]
    s, _ = stats.spearmanr(y_true[v], y_pred[v])
    return float(p), float(s)


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(dates, symbol, model, scaler, threshold, model_type, tag=''):
    trades    = []
    daily_pnl = {}

    for d in dates:
        cache_key = f'{symbol}_{d}'
        try:
            if cache_key not in _FEAT_CACHE:
                mbp = load_day(d, symbol)
                _FEAT_CACHE[cache_key] = compute_features(mbp)
                del mbp
            feat = _FEAT_CACHE[cache_key]
            n    = len(feat)
            bars = feat.iloc[OPEN_BUF : n - CLOSE_BUF].copy()
            bars = bars[bars[FEATURE_NAMES].notna().all(axis=1)]
            if len(bars) == 0:
                continue

            preds   = predict(model, bars, scaler if model_type == 'ols' else None)
            bar_idx = bars.index

            day_net      = 0.0
            day_gross    = 0.0
            killed       = False
            next_entry_i = 0

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
                next_entry_i = exit_idx

                trades.append({
                    'date': d, 'direction': direction,
                    'pred_bps': round(float(pred), 3),
                    'entry_mp': round(float(mp), 4),
                    'shares': int(shares),
                    'gross': round(gross, 2), 'tc': round(tc, 2), 'net': round(net, 2),
                })

                if day_net < -NOTIONAL * KILL_PCT:
                    killed = True

            daily_pnl[d] = {'gross': day_gross, 'net': day_net}

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


# ── Benchmark daily return (for beta calculation) ─────────────────────────────
def benchmark_daily_ret(date_str, bench_symbol):
    """Returns daily microprice return for bench_symbol on date_str."""
    key = f'{bench_symbol}_bench_{date_str}'
    try:
        if key not in _FEAT_CACHE:
            path = DATA_DIR / bench_symbol / 'mbp-10' / f'{date_str}.parquet'
            cols = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
            raw  = pd.read_parquet(path, engine='fastparquet', columns=cols).sort_index()
            if raw.index.tzinfo is None:
                raw.index = raw.index.tz_localize('UTC').tz_convert(ET)
            else:
                raw.index = raw.index.tz_convert(ET)
            raw  = raw.between_time('09:30', '16:00')
            b0   = raw['bid_sz_00'].fillna(0).astype(np.int64)
            a0   = raw['ask_sz_00'].fillna(0).astype(np.int64)
            den  = np.where(b0 + a0 > 0, b0 + a0, np.nan)
            mp   = (a0 * raw['bid_px_00'].values + b0 * raw['ask_px_00'].values) / den
            _FEAT_CACHE[key] = pd.Series(mp, index=raw.index).resample('1s').last().dropna()
        mp = _FEAT_CACHE[key]
        if len(mp) < 10:
            return np.nan
        open_mp  = mp.between_time('09:35', '09:40').mean()
        close_mp = mp.between_time('15:55', '16:00').mean()
        if not (np.isfinite(open_mp) and np.isfinite(close_mp) and open_mp > 0):
            return np.nan
        return (close_mp - open_mp) / open_mp
    except Exception:
        return np.nan


# ── Performance metrics ───────────────────────────────────────────────────────
def compute_metrics(daily_df, trades_df, dates, symbol, notional=NOTIONAL):
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

    # Use SPY as benchmark for AAPL/NVDA; use AAPL for SPY itself
    bench = 'AAPL' if symbol == 'SPY' else 'SPY'
    bench_rets  = np.array([benchmark_daily_ret(d, bench) for d in dates])
    strat_rets  = pnl.values / notional
    valid = np.isfinite(bench_rets) & np.isfinite(strat_rets)
    beta, alpha = np.nan, np.nan
    if valid.sum() >= 5:
        slope, intercept, *_ = stats.linregress(bench_rets[valid], strat_rets[valid])
        beta  = slope
        alpha = intercept * 252

    # Spread & sigma_y stats for break-even IC calculation
    spread_bps_mean = np.nan
    sigma_y = np.nan
    if not trades_df.empty and 'entry_mp' in trades_df.columns:
        pass  # will compute from features below

    return {
        'sharpe':    round(sh, 3),
        'ann_ret':   round(ann, 0),
        'ann_vol':   round(vol, 0),
        'max_dd':    round(mdd, 0),
        'hit_rate':  round(hit, 3),
        'n_trades':  n_tr,
        'turnover':  round(turnover, 3),
        'avg_hold_s': HOLD_S,
        'beta':      round(beta,  3) if np.isfinite(beta)  else np.nan,
        'alpha_ann': round(alpha, 4) if np.isfinite(alpha) else np.nan,
        'total_pnl': round(pnl.sum(), 0),
    }


def filter_by_dates(daily_df, dates):
    if daily_df.empty:
        return pd.DataFrame(columns=['gross', 'net'])
    mask = daily_df.index.strftime('%Y-%m-%d').isin(dates)
    return daily_df[mask]


def filter_trades_by_dates(trades_df, dates):
    if trades_df.empty:
        return pd.DataFrame()
    return trades_df[trades_df['date'].isin(dates)]


# ── Per-symbol pipeline ───────────────────────────────────────────────────────
def run_symbol(symbol):
    print(f'\n{"=" * 60}')
    print(f'FACTOR ZOO — {symbol} MBP-10 (11 Book Features)')
    print(f'{"=" * 60}')

    # Check which Oct dates actually exist for this symbol
    available_oct = [d for d in OCT_DATES
                     if (DATA_DIR / symbol / 'mbp-10' / f'{d}.parquet').exists()]
    available_train = [d for d in TRAIN_DATES
                       if (DATA_DIR / symbol / 'mbp-10' / f'{d}.parquet').exists()]
    available_val = [d for d in VAL_DATES
                     if (DATA_DIR / symbol / 'mbp-10' / f'{d}.parquet').exists()]

    print(f'  Available: Train={len(available_train)}, Val={len(available_val)}, OOS={len(available_oct)}')

    X_train, y_train, _ = build_dataset(available_train, symbol, f'TRAIN  ({symbol})')
    X_val,   y_val,   _ = build_dataset(available_val,   symbol, f'VAL    ({symbol})')
    X_oct,   y_oct,   _ = build_dataset(available_oct,   symbol, f'OOS    ({symbol})')

    # ── OLS ──────────────────────────────────────────────────────────────────
    print(f'\n=== OLS (Ridge) — {symbol} ===')
    ols, scaler = train_ols(X_train, y_train)

    ols_ic_tr,  ols_ic_tr_s  = ic_stats(y_train.values, predict(ols, X_train, scaler))
    ols_ic_val, ols_ic_val_s = ic_stats(y_val.values,   predict(ols, X_val,   scaler))
    ols_ic_oct, ols_ic_oct_s = ic_stats(y_oct.values,   predict(ols, X_oct,   scaler))
    print(f'  Train IC: {ols_ic_tr:.4f} (Spearman {ols_ic_tr_s:.4f})')
    print(f'  Val   IC: {ols_ic_val:.4f} (Spearman {ols_ic_val_s:.4f})')
    print(f'  OOS   IC: {ols_ic_oct:.4f} (Spearman {ols_ic_oct_s:.4f})')

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print(f'\n=== LightGBM — {symbol} ===')
    lgbm = train_lgbm(X_train, y_train, X_val, y_val)
    print(f'  Best iteration: {lgbm.best_iteration_}')

    lgbm_ic_tr,  lgbm_ic_tr_s  = ic_stats(y_train.values, predict(lgbm, X_train))
    lgbm_ic_val, lgbm_ic_val_s = ic_stats(y_val.values,   predict(lgbm, X_val))
    lgbm_ic_oct, lgbm_ic_oct_s = ic_stats(y_oct.values,   predict(lgbm, X_oct))
    print(f'  Train IC: {lgbm_ic_tr:.4f} (Spearman {lgbm_ic_tr_s:.4f})')
    print(f'  Val   IC: {lgbm_ic_val:.4f} (Spearman {lgbm_ic_val_s:.4f})')
    print(f'  OOS   IC: {lgbm_ic_oct:.4f} (Spearman {lgbm_ic_oct_s:.4f})')

    # ── Threshold optimization ────────────────────────────────────────────────
    ols_val_preds = predict(ols, X_val, scaler)
    best_thresh, best_val_ic = 0.3, -np.inf
    for th in np.arange(0.1, 2.0, 0.1):
        mask = np.abs(ols_val_preds) > th
        if mask.sum() < 200:
            break
        ic_th, _ = ic_stats(y_val.values[mask], ols_val_preds[mask])
        if ic_th > best_val_ic:
            best_val_ic, best_thresh = ic_th, th
    print(f'\n  Best threshold: {best_thresh:.2f} bps (val IC={best_val_ic:.4f})')

    # ── Backtest ──────────────────────────────────────────────────────────────
    all_bt_dates = available_train + available_val + available_oct
    print(f'\n=== Backtest OLS — {symbol} ===')
    ols_daily, ols_trades = backtest(
        all_bt_dates, symbol, ols, scaler, best_thresh, 'ols', f'OLS {symbol} (all)'
    )
    print(f'\n=== Backtest LightGBM — {symbol} ===')
    lgbm_daily, lgbm_trades = backtest(
        all_bt_dates, symbol, lgbm, None, best_thresh, 'lgbm', f'LGBM {symbol} (all)'
    )

    # ── Per-period metrics ────────────────────────────────────────────────────
    metrics_table = {}
    ic_table = {
        'OLS':      {'Train': ols_ic_tr,  'Val': ols_ic_val,  'OOS': ols_ic_oct},
        'LightGBM': {'Train': lgbm_ic_tr, 'Val': lgbm_ic_val, 'OOS': lgbm_ic_oct},
    }
    period_map = [('Train', available_train), ('Val', available_val), ('OOS', available_oct)]

    for model_name, (daily_df, trades_df) in [
        ('OLS', (ols_daily, ols_trades)),
        ('LightGBM', (lgbm_daily, lgbm_trades)),
    ]:
        metrics_table[model_name] = {}
        for period_name, dates in period_map:
            sub_d = filter_by_dates(daily_df, dates)
            sub_t = filter_trades_by_dates(trades_df, dates)
            metrics_table[model_name][period_name] = compute_metrics(
                sub_d, sub_t, dates, symbol
            )

    # ── Microstructure stats (spread & sigma_y for break-even IC) ─────────────
    all_feat = pd.concat([
        _FEAT_CACHE[f'{symbol}_{d}']
        for d in available_train
        if f'{symbol}_{d}' in _FEAT_CACHE
    ])
    sigma_y   = float(y_train.std())
    spread_bps_mean = float(np.nanmedian(all_feat['spread_bps'].values))
    ic_breakeven = spread_bps_mean / sigma_y if sigma_y > 0 else np.nan
    price_mean   = float(np.nanmedian(all_feat['microprice'].values))
    print(f'\n  [{symbol}] Microstructure stats (train):')
    print(f'    Median price:    ${price_mean:.2f}')
    print(f'    Median spread:   {spread_bps_mean:.3f} bps')
    print(f'    sigma_y:         {sigma_y:.3f} bps')
    print(f'    IC break-even:   {ic_breakeven:.4f}  (need IC ≥ {ic_breakeven:.3f} to profit)')
    print(f'    OLS OOS IC:      {ols_ic_oct:.4f}  (ratio: {ols_ic_oct/ic_breakeven:.2f}x break-even)')

    return {
        'symbol': symbol,
        'ols': ols, 'scaler': scaler, 'lgbm': lgbm,
        'ols_daily': ols_daily, 'ols_trades': ols_trades,
        'lgbm_daily': lgbm_daily, 'lgbm_trades': lgbm_trades,
        'ic_table': ic_table,
        'metrics_table': metrics_table,
        'microstructure': {
            'price': price_mean,
            'spread_bps': spread_bps_mean,
            'sigma_y': sigma_y,
            'ic_breakeven': ic_breakeven,
        },
        'available': {'train': available_train, 'val': available_val, 'oct': available_oct},
    }


# ── Cross-stock summary table ─────────────────────────────────────────────────
def save_cross_stock_summary(results_list):
    rows = []
    for r in results_list:
        sym = r['symbol']
        ms  = r['microstructure']
        for model in ['OLS', 'LightGBM']:
            for period in ['Train', 'Val', 'OOS']:
                m  = r['metrics_table'][model].get(period, {})
                ic = r['ic_table'][model].get(period, np.nan)
                rows.append({
                    'Symbol': sym, 'Model': model, 'Period': period,
                    'IC': round(ic, 5) if np.isfinite(ic) else np.nan,
                    'Sharpe': m.get('sharpe', np.nan),
                    'Ann.Ret($)': m.get('ann_ret', np.nan),
                    'Ann.Vol($)': m.get('ann_vol', np.nan),
                    'MaxDD($)': m.get('max_dd', np.nan),
                    'HitRate': m.get('hit_rate', np.nan),
                    'Trades': m.get('n_trades', np.nan),
                    'Spread(bps)': round(ms['spread_bps'], 3),
                    'Sigma_y(bps)': round(ms['sigma_y'], 3),
                    'IC_breakeven': round(ms['ic_breakeven'], 4),
                })

    df = pd.DataFrame(rows).set_index(['Symbol', 'Model', 'Period'])
    df.to_csv('factor_zoo_multi_metrics.csv')
    print('\n\n' + '=' * 70)
    print('CROSS-STOCK SUMMARY')
    print('=' * 70)

    # OOS-only compact table
    oos_rows = [r for r in rows if r['Period'] == 'OOS']
    oos_df = pd.DataFrame(oos_rows)[
        ['Symbol', 'Model', 'IC', 'Sharpe', 'Ann.Ret($)', 'HitRate',
         'Spread(bps)', 'Sigma_y(bps)', 'IC_breakeven']
    ].set_index(['Symbol', 'Model'])
    print('\n--- OOS Oct 2024 Results ---')
    print(oos_df.to_string())

    # Microstructure comparison
    print('\n--- Microstructure (break-even IC vs observed) ---')
    for r in results_list:
        sym = r['symbol']
        ms  = r['microstructure']
        ols_oos_ic  = r['ic_table']['OLS']['OOS']
        lgbm_oos_ic = r['ic_table']['LightGBM']['OOS']
        ratio = ols_oos_ic / ms['ic_breakeven'] if ms['ic_breakeven'] > 0 else np.nan
        viable = 'VIABLE' if ratio >= 1 else f'need {1/ratio:.1f}x more IC'
        print(f'  {sym:4s}  price=${ms["price"]:6.1f}  spread={ms["spread_bps"]:.3f}bps  '
              f'σ_y={ms["sigma_y"]:.2f}bps  '
              f'IC_be={ms["ic_breakeven"]:.4f}  '
              f'OLS_OOS_IC={ols_oos_ic:.4f}  → {viable}')

    print('\nSaved to factor_zoo_multi_metrics.csv')
    return df


# ── Comparison plots ──────────────────────────────────────────────────────────
def plot_ic_cross_stock(results_list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    symbols = [r['symbol'] for r in results_list]
    periods = ['Train', 'Val', 'OOS']
    colors  = {'Train': '#1565C0', 'Val': '#6A1B9A', 'OOS': '#BF360C'}
    x = np.arange(len(symbols))
    w = 0.22

    for ax, model in zip(axes, ['OLS', 'LightGBM']):
        for j, period in enumerate(periods):
            ics = [r['ic_table'][model].get(period, np.nan) for r in results_list]
            offset = (j - 1) * w
            ax.bar(x + offset, ics, w, label=period, color=colors[period])
        ax.set_xticks(x)
        ax.set_xticklabels(symbols)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
        ax.set_ylabel('Pearson IC')
        ax.set_title(f'{model} — IC by Symbol and Period')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Information Coefficient: AAPL vs NVDA vs SPY', fontsize=12)
    plt.tight_layout()
    fname = 'factor_zoo_multi_ic.png'
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_sharpe_cross_stock(results_list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    symbols = [r['symbol'] for r in results_list]
    periods = ['Train', 'Val', 'OOS']
    colors  = {'Train': '#1565C0', 'Val': '#6A1B9A', 'OOS': '#BF360C'}
    x = np.arange(len(symbols))
    w = 0.22

    for ax, model in zip(axes, ['OLS', 'LightGBM']):
        for j, period in enumerate(periods):
            sharpes = [r['metrics_table'][model].get(period, {}).get('sharpe', np.nan)
                       for r in results_list]
            offset = (j - 1) * w
            ax.bar(x + offset, sharpes, w, label=period, color=colors[period])
        ax.set_xticks(x)
        ax.set_xticklabels(symbols)
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'{model} — Sharpe by Symbol and Period')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Net Sharpe Ratio: AAPL vs NVDA vs SPY', fontsize=12)
    plt.tight_layout()
    fname = 'factor_zoo_multi_sharpe.png'
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_breakeven_comparison(results_list):
    """Bar chart: break-even IC vs observed IC (OLS Train/Val/OOS) per symbol."""
    fig, ax = plt.subplots(figsize=(10, 5))
    symbols = [r['symbol'] for r in results_list]
    x = np.arange(len(symbols))
    w = 0.2

    ic_be    = [r['microstructure']['ic_breakeven'] for r in results_list]
    ols_tr   = [r['ic_table']['OLS']['Train'] for r in results_list]
    ols_val  = [r['ic_table']['OLS']['Val']   for r in results_list]
    ols_oos  = [r['ic_table']['OLS']['OOS']   for r in results_list]

    ax.bar(x - 1.5*w, ic_be,   w, label='Break-even IC', color='#B71C1C', alpha=0.85)
    ax.bar(x - 0.5*w, ols_tr,  w, label='OLS Train IC',  color='#1565C0', alpha=0.85)
    ax.bar(x + 0.5*w, ols_val, w, label='OLS Val IC',    color='#6A1B9A', alpha=0.85)
    ax.bar(x + 1.5*w, ols_oos, w, label='OLS OOS IC',    color='#BF360C', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=11)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('IC')
    ax.set_title('Break-even IC vs Observed IC by Symbol\n'
                 '(bar must exceed break-even to be profitable)', fontsize=11)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fname = 'factor_zoo_multi_breakeven.png'
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_oos_pnl_cross_stock(results_list):
    """Cumulative OOS net PnL for all symbols, OLS model."""
    fig, axes = plt.subplots(1, len(results_list), figsize=(14, 4), sharey=False)
    if len(results_list) == 1:
        axes = [axes]

    for ax, r in zip(axes, results_list):
        sym = r['symbol']
        daily = r['ols_daily']
        if daily.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue
        oct_d = filter_by_dates(daily, r['available']['oct'])
        if oct_d.empty:
            continue
        cum_net   = oct_d['net'].cumsum().values
        cum_gross = oct_d['gross'].cumsum().values
        xs = range(len(oct_d))
        ax.fill_between(xs, 0, cum_gross, alpha=0.3, color='#90CAF9', label='Gross')
        ax.plot(xs, cum_net, color='#1565C0', linewidth=2, label='Net')
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        total = oct_d['net'].sum()
        ax.set_title(f'{sym}\nOOS total: ${total:,.0f}', fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, p: f'${x/1000:.0f}k' if abs(x) >= 1000 else f'${x:.0f}'
        ))
        ax.set_xlabel('Oct 2024 trading day')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('OOS Cumulative PnL (OLS) — Oct 2024', fontsize=12)
    plt.tight_layout()
    fname = 'factor_zoo_multi_oos_pnl.png'
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_results = []
    for sym in SYMBOLS:
        res = run_symbol(sym)
        all_results.append(res)

    print('\n\n=== Generating cross-stock figures ===')
    plot_ic_cross_stock(all_results)
    plot_sharpe_cross_stock(all_results)
    plot_breakeven_comparison(all_results)
    plot_oos_pnl_cross_stock(all_results)

    save_cross_stock_summary(all_results)

    print('\n=== Done. ===')
    print('Figures: factor_zoo_multi_ic.png, factor_zoo_multi_sharpe.png,')
    print('         factor_zoo_multi_breakeven.png, factor_zoo_multi_oos_pnl.png')
    print('Metrics: factor_zoo_multi_metrics.csv')
