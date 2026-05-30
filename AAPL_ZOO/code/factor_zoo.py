import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

DATA_DIR = Path('data/raw')
Path('figures').mkdir(exist_ok=True)
ET = 'America/New_York'

# ── Date split (no Feb-1 earnings, no FOMC days Jan-31 / Mar-20) ─────────────
TRAIN_DATES = [
    '2024-01-02','2024-01-04','2024-01-08','2024-01-10','2024-01-12',
    '2024-01-16','2024-01-18','2024-01-22','2024-01-24','2024-01-26',
    '2024-02-05','2024-02-07','2024-02-09','2024-02-12','2024-02-14',
    '2024-02-20','2024-02-22','2024-02-26',
]
TEST_DATES = [
    '2024-02-28','2024-03-04','2024-03-07',
    '2024-03-11','2024-03-14','2024-03-18','2024-04-01',
]

HOLD_S    = 60       # hold 60 seconds
NOTIONAL  = 50_000   # $50K per trade
OPEN_BUF  = 5 * 60   # skip first 5 minutes (300 bars)
CLOSE_BUF = 60       # skip last 60 seconds (no label)

LEVELS = 5   # levels used for multi-level features
BOOK_COLS = (
    [f'bid_px_{i:02d}' for i in range(LEVELS)] +
    [f'ask_px_{i:02d}' for i in range(LEVELS)] +
    [f'bid_sz_{i:02d}' for i in range(LEVELS)] +
    [f'ask_sz_{i:02d}' for i in range(LEVELS)]
)
FEATURE_NAMES = [
    'ofi_1','ofi_5','ofi_30','ofi_multilevel',
    'book_imbalance','depth_ratio','bid_slope','ask_slope',
    'microprice','microprice_drift',
    'spread','realized_vol',
    'trade_arrival','signed_trade_imbal','queue_exhaustion',
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_day(date_str):
    mbp_path = DATA_DIR / 'AAPL' / 'mbp-10' / f'{date_str}.parquet'
    trd_path = DATA_DIR / 'AAPL' / 'trades'  / f'{date_str}.parquet'
    all_book_cols = (
        [f'bid_px_{i:02d}' for i in range(10)] +
        [f'ask_px_{i:02d}' for i in range(10)] +
        [f'bid_sz_{i:02d}' for i in range(10)] +
        [f'ask_sz_{i:02d}' for i in range(10)]
    )
    mbp = pd.read_parquet(mbp_path, engine='fastparquet', columns=all_book_cols).sort_index()
    mbp.index = mbp.index.tz_convert(ET)
    mbp = mbp.between_time('09:30','16:00')
    # cast sizes to int64
    sz_cols = [c for c in mbp.columns if '_sz_' in c]
    mbp[sz_cols] = mbp[sz_cols].astype(np.int64)

    trd = pd.read_parquet(trd_path, engine='fastparquet', columns=['price','size','side']).sort_index()
    trd.index = trd.index.tz_convert(ET)
    trd = trd.between_time('09:30','16:00')
    trd['size'] = trd['size'].astype(np.int64)
    return mbp, trd


# ── Feature engineering ───────────────────────────────────────────────────────
def _book_slope(sz_cols_prefix, df):
    """Linear slope of qty across 5 levels (vectorized, no loop)."""
    # levels 0..4, weights from OLS formula: slope = (5*Σi*qi - 10*Σqi) / 50
    qs = np.stack([df[f'{sz_cols_prefix}_{i:02d}'].values for i in range(LEVELS)], axis=1).astype(float)
    weights = np.array([0, 1, 2, 3, 4], dtype=float)
    weighted = qs @ weights          # Σ i*qi
    total    = qs.sum(axis=1)        # Σ qi
    return (5 * weighted - 10 * total) / 50.0


def compute_features(mbp, trd):
    """Returns 1-second bar DataFrame with 15 features + microprice (for labels)."""
    # ── 1s resampling of book ─────────────────────────────────────────────────
    # Use last snapshot per second (most recent book state)
    b = mbp.resample('1s').last().dropna(subset=['bid_px_00'])

    bid0 = b['bid_sz_00'].values.astype(np.int64)
    ask0 = b['ask_sz_00'].values.astype(np.int64)
    denom0 = np.where(bid0 + ask0 > 0, bid0 + ask0, np.nan)

    # 1. OFI at best level (instantaneous)
    ofi_raw = (bid0 - ask0) / denom0

    # 2-3. OFI lags
    ofi_s = pd.Series(ofi_raw, index=b.index)
    ofi_5  = ofi_s.rolling(5,  min_periods=1).mean().values
    ofi_30 = ofi_s.rolling(30, min_periods=1).mean().values

    # 4. Multi-level OFI (levels 0-4, equal weight)
    ml_ofi = np.zeros(len(b))
    for i in range(LEVELS):
        bi = b[f'bid_sz_{i:02d}'].values.astype(np.int64)
        ai = b[f'ask_sz_{i:02d}'].values.astype(np.int64)
        d  = np.where(bi + ai > 0, bi + ai, np.nan)
        ml_ofi += (bi - ai) / d
    ml_ofi /= LEVELS

    # 5. Book imbalance
    book_imbal = bid0 / np.where(bid0 + ask0 > 0, bid0 + ask0, np.nan)

    # 6. Depth ratio: top-3 vs levels 4-9 (both sides combined)
    top3_bid  = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) for i in range(3))
    top3_ask  = sum(b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3))
    deep_bid  = sum(b[f'bid_sz_{i:02d}'].values.astype(np.int64) for i in range(3, 10))
    deep_ask  = sum(b[f'ask_sz_{i:02d}'].values.astype(np.int64) for i in range(3, 10))
    top3  = top3_bid + top3_ask
    deep  = deep_bid + deep_ask
    depth_ratio = top3 / np.where(deep > 0, deep, np.nan)

    # 7-8. Book slopes
    bid_slope = _book_slope('bid_sz', b)
    ask_slope = _book_slope('ask_sz', b)

    # 9. Microprice
    micro_num = (b['ask_sz_00'].values.astype(np.int64) * b['bid_px_00'].values +
                 b['bid_sz_00'].values.astype(np.int64) * b['ask_px_00'].values)
    micro_den = np.where(bid0 + ask0 > 0, bid0 + ask0, np.nan)
    microprice = micro_num / micro_den

    # 10. Microprice drift (10s finite difference, bps/s)
    micro_s = pd.Series(microprice, index=b.index)
    micro_ret = micro_s.pct_change() * 1e4  # tick returns in bps
    micro_drift = micro_s.rolling(10, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] * 1e4 if len(x) > 1 else np.nan, raw=True
    )

    # 11. Spread
    spread = b['ask_px_00'].values - b['bid_px_00'].values

    # 12. Realized microprice vol (60s rolling std of bps returns)
    realized_vol = micro_ret.rolling(60, min_periods=10).std().values

    # ── Trade-based features ─────────────────────────────────────────────────
    buy_vol  = trd[trd['side'] == 'A']['size'].resample('1s').sum()
    sell_vol = trd[trd['side'] == 'B']['size'].resample('1s').sum()
    trade_ct = trd['size'].resample('1s').count()

    buy_vol  = buy_vol.reindex(b.index,  fill_value=0)
    sell_vol = sell_vol.reindex(b.index, fill_value=0)
    trade_ct = trade_ct.reindex(b.index, fill_value=0)

    # 13. Trade arrival rate (30s rolling count of trades)
    trade_arrival = trade_ct.rolling(30, min_periods=1).mean().values

    # 14. Signed trade imbalance (30s rolling)
    buy_30  = buy_vol.rolling(30,  min_periods=1).sum()
    sell_30 = sell_vol.rolling(30, min_periods=1).sum()
    total_30 = buy_30 + sell_30
    signed_imbal = ((buy_30 - sell_30) / total_30.replace(0, np.nan)).values

    # 15. Queue exhaustion (rolling 5s: trades at best / starting queue)
    trd_at_bid = trd[trd['price'] == b['bid_px_00'].reindex(trd.index, method='ffill')]['size']
    trd_at_ask = trd[trd['price'] == b['ask_px_00'].reindex(trd.index, method='ffill')]['size']
    vol_at_best = (trd_at_bid.resample('1s').sum().reindex(b.index, fill_value=0) +
                   trd_at_ask.resample('1s').sum().reindex(b.index, fill_value=0))
    queue_size  = (b['bid_sz_00'] + b['ask_sz_00']).rolling(5, min_periods=1).mean()
    queue_exhaust = (vol_at_best.rolling(5, min_periods=1).sum() /
                     queue_size.replace(0, np.nan)).values

    feat = pd.DataFrame({
        'ofi_1':           ofi_raw,
        'ofi_5':           ofi_5,
        'ofi_30':          ofi_30,
        'ofi_multilevel':  ml_ofi,
        'book_imbalance':  book_imbal,
        'depth_ratio':     depth_ratio,
        'bid_slope':       bid_slope,
        'ask_slope':       ask_slope,
        'microprice':      microprice,
        'microprice_drift':micro_drift.values,
        'spread':          spread,
        'realized_vol':    realized_vol,
        'trade_arrival':   trade_arrival,
        'signed_trade_imbal': signed_imbal,
        'queue_exhaustion':queue_exhaust,
    }, index=b.index)

    return feat


def make_labels(feat_df, hold_s=HOLD_S):
    """1-minute forward microprice return in bps."""
    micro = feat_df['microprice']
    fwd   = micro.shift(-hold_s)
    label = (fwd / micro - 1) * 1e4
    return label


def build_dataset(dates, label='building'):
    print(f'\n[{label}] Loading {len(dates)} days...')
    feats, labels = [], []
    for d in dates:
        try:
            mbp, trd = load_day(d)
            feat = compute_features(mbp, trd)
            lbl  = make_labels(feat)
            # drop opening buffer, last HOLD_S bars (no label), and NaNs
            n = len(feat)
            valid = feat.iloc[OPEN_BUF : n - CLOSE_BUF].copy()
            valid_lbl = lbl.iloc[OPEN_BUF : n - CLOSE_BUF]
            mask = valid[FEATURE_NAMES].notna().all(axis=1) & valid_lbl.notna()
            feats.append(valid[mask])
            labels.append(valid_lbl[mask])
            print(f'  {d}: {mask.sum():,} bars')
            del mbp, trd
        except Exception as e:
            print(f'  {d}: ERROR {e}')
    X = pd.concat(feats)[FEATURE_NAMES]
    y = pd.concat(labels)
    print(f'  Total: {len(X):,} bars  |  label mean={y.mean():.3f} bps  std={y.std():.3f} bps')
    return X, y


# ── Model training ────────────────────────────────────────────────────────────
def train_ols(X_train, y_train, scaler):
    Xs = scaler.transform(X_train)
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(Xs, y_train)
    return model


def train_lgbm(X_train, y_train):
    params = dict(
        objective='regression', metric='rmse',
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, n_jobs=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    return model


def eval_model(model, X, y, scaler=None, label=''):
    Xp = scaler.transform(X) if scaler else X
    pred = model.predict(Xp)
    r2  = r2_score(y, pred)
    ic  = np.corrcoef(pred, y)[0, 1]
    print(f'  [{label}]  R²={r2:.4f}  IC={ic:.4f}')
    return pred


# ── Ablation study ────────────────────────────────────────────────────────────
def ablation(X_train, y_train, scaler, model_type='lgbm'):
    base_pred = eval_model(
        train_lgbm(X_train, y_train) if model_type == 'lgbm' else train_ols(X_train, y_train, scaler),
        X_train, y_train, None if model_type == 'lgbm' else scaler, 'baseline'
    )
    base_ic = abs(np.corrcoef(base_pred, y_train)[0, 1])
    drops = {}
    for feat in FEATURE_NAMES:
        X_drop = X_train.drop(columns=[feat])
        if model_type == 'lgbm':
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                  num_leaves=31, verbose=-1, n_jobs=-1)
            m.fit(X_drop, y_train)
            pred = m.predict(X_drop)
        else:
            sc2 = StandardScaler().fit(X_drop)
            m = Ridge(alpha=1.0).fit(sc2.transform(X_drop), y_train)
            pred = m.predict(sc2.transform(X_drop))
        ic_drop = abs(np.corrcoef(pred, y_train)[0, 1])
        drops[feat] = base_ic - ic_drop
    return drops


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(dates, model, scaler, threshold, model_type='lgbm', label=''):
    trades = []
    daily_pnl = []

    for d in dates:
        try:
            mbp, trd = load_day(d)
            feat = compute_features(mbp, trd)
            n = len(feat)
            valid_idx = slice(OPEN_BUF, n - HOLD_S)
            X = feat.iloc[valid_idx][FEATURE_NAMES].dropna()
            if X.empty:
                continue
            Xp = scaler.transform(X) if scaler else X
            preds = model.predict(Xp)

            day_net = 0.0
            inventory = 0
            daily_kill = False

            for i, (ts, pred) in enumerate(zip(X.index, preds)):
                if daily_kill:
                    break
                if abs(pred) < threshold:
                    continue

                direction = 1 if pred > 0 else -1

                # entry price
                row = feat.loc[ts]
                entry_mid    = row['microprice']
                entry_spread = row['spread']
                if pd.isna(entry_mid) or pd.isna(entry_spread) or entry_mid <= 0:
                    continue

                shares = int(NOTIONAL / entry_mid)
                if shares == 0:
                    continue

                # inventory cap
                if abs(inventory + direction * shares) > 500:
                    continue

                # find exit bar
                exit_ts = ts + pd.Timedelta(seconds=HOLD_S)
                future = feat[feat.index >= exit_ts]
                if future.empty:
                    continue
                exit_row    = future.iloc[0]
                exit_mid    = exit_row['microprice']
                exit_spread = exit_row['spread']
                if pd.isna(exit_mid) or pd.isna(exit_spread):
                    continue

                # stop-loss: if unrealized loss > 1.5 × spread, exit at current price
                stop_loss_thresh = 1.5 * entry_spread
                gross = direction * (exit_mid - entry_mid) * shares
                tc    = (entry_spread / 2 + exit_spread / 2) * shares
                net   = gross - tc

                # apply stop-loss (simplified check at exit)
                if direction * (exit_mid - entry_mid) < -stop_loss_thresh:
                    sl_price = entry_mid - direction * stop_loss_thresh
                    gross = direction * (sl_price - entry_mid) * shares
                    net   = gross - tc

                inventory += direction * shares
                day_net   += net

                trades.append({
                    'date': d, 'ts': ts, 'direction': direction,
                    'pred_bps': round(pred, 3),
                    'entry_mid': round(entry_mid, 4),
                    'exit_mid':  round(exit_mid, 4),
                    'shares': shares, 'gross': round(gross, 2),
                    'tc': round(tc, 2), 'net': round(net, 2),
                })

                # daily kill switch at -2% capital
                if day_net < -NOTIONAL * 0.02:
                    daily_kill = True

                inventory = 0  # flat after each trade (hold until exit)

            daily_pnl.append({'date': d, 'pnl': day_net})
            del mbp, trd

        except Exception as e:
            print(f'  {d}: backtest error {e}')

    trades_df = pd.DataFrame(trades)
    daily_df  = pd.DataFrame(daily_pnl)
    if trades_df.empty:
        print(f'  [{label}] No trades generated')
        return trades_df, daily_df

    ann  = daily_df['pnl'].mean() * 252
    vol  = daily_df['pnl'].std() * np.sqrt(252)
    sh   = ann / vol if vol > 0 else np.nan
    hit  = (trades_df['net'] > 0).mean()
    mdd  = (daily_df['pnl'].cumsum() - daily_df['pnl'].cumsum().cummax()).min()
    avg_hold = HOLD_S
    avg_tc   = trades_df['tc'].mean()

    print(f'\n  [{label}]')
    print(f'  Trades={len(trades_df)}  Days={len(daily_df)}  Hit={hit:.1%}')
    print(f'  Ann.Return=${ann:,.0f}  Ann.Vol=${vol:,.0f}  Sharpe={sh:.2f}')
    print(f'  MaxDD=${mdd:,.0f}  AvgTC=${avg_tc:.2f}  AvgHold={avg_hold}s')
    return trades_df, daily_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── 1. Build datasets ──────────────────────────────────────────────────────
    X_train, y_train = build_dataset(TRAIN_DATES, 'TRAIN')
    X_test,  y_test  = build_dataset(TEST_DATES,  'TEST')

    # ── 2. Normalize ───────────────────────────────────────────────────────────
    scaler = StandardScaler().fit(X_train)

    # ── 3. Train models ────────────────────────────────────────────────────────
    print('\n=== OLS (Ridge) ===')
    ols = train_ols(X_train, y_train, scaler)
    ols_train_pred = eval_model(ols, X_train, y_train, scaler, 'train')
    ols_test_pred  = eval_model(ols, X_test,  y_test,  scaler, 'test')

    print('\n=== LightGBM ===')
    lgbm = train_lgbm(X_train, y_train)
    lgbm_train_pred = eval_model(lgbm, X_train, y_train, label='train')
    lgbm_test_pred  = eval_model(lgbm, X_test,  y_test,  label='test')

    # ── 4. Ablation study ──────────────────────────────────────────────────────
    print('\n=== Ablation (LightGBM, IC drop when feature removed) ===')
    drops = ablation(X_train, y_train, scaler, 'lgbm')
    for feat, drop in sorted(drops.items(), key=lambda x: -x[1]):
        bar = '#' * int(max(drop, 0) * 5000)
        print(f'  {feat:<25} {drop:+.5f}  {bar}')

    # ── 5. Feature importance plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ablation
    ax = axes[0]
    feats_sorted = sorted(drops.items(), key=lambda x: x[1])
    names = [f[0] for f in feats_sorted]
    vals  = [f[1] for f in feats_sorted]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Ablation: IC drop when feature removed\n(positive = feature adds value)', fontsize=10)
    ax.set_xlabel('IC drop')

    # LightGBM native importance
    ax2 = axes[1]
    imp = pd.Series(lgbm.feature_importances_, index=FEATURE_NAMES).sort_values()
    ax2.barh(imp.index, imp.values, color='#2196F3')
    ax2.set_title('LightGBM feature importance (gain)', fontsize=10)
    ax2.set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig('figures/factor_zoo_importance.png', dpi=150)
    plt.close()

    # ── 6. Threshold optimization on train set ─────────────────────────────────
    print('\n=== Threshold optimization (LightGBM, train set) ===')
    # Use train predictions to find best threshold
    lgbm_train_preds = lgbm.predict(X_train)
    best_thresh, best_ic = 0.5, -np.inf
    for th in np.arange(0.5, 5.0, 0.25):
        mask = np.abs(lgbm_train_preds) > th
        if mask.sum() < 100:
            break
        ic = np.corrcoef(lgbm_train_preds[mask], y_train.values[mask])[0, 1]
        print(f'  threshold={th:.2f}  n={mask.sum():,}  IC={ic:.4f}')
        if ic > best_ic:
            best_ic, best_thresh = ic, th
    print(f'  --> Best threshold: {best_thresh:.2f} bps (IC={best_ic:.4f})')

    # ── 7. Backtest ────────────────────────────────────────────────────────────
    print('\n=== Backtest: OLS ===')
    ols_tr, ols_daily_tr = backtest(TRAIN_DATES, ols, scaler, best_thresh, 'ols', 'OLS Train')
    ols_te, ols_daily_te = backtest(TEST_DATES,  ols, scaler, best_thresh, 'ols', 'OLS Test')

    print('\n=== Backtest: LightGBM ===')
    lgbm_tr, lgbm_daily_tr = backtest(TRAIN_DATES, lgbm, None, best_thresh, 'lgbm', 'LightGBM Train')
    lgbm_te, lgbm_daily_te = backtest(TEST_DATES,  lgbm, None, best_thresh, 'lgbm', 'LightGBM Test')

    # ── 8. PnL plots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, daily, title in [
        (axes[0,0], ols_daily_tr,  'OLS — Train'),
        (axes[0,1], ols_daily_te,  'OLS — Test (OOS)'),
        (axes[1,0], lgbm_daily_tr, 'LightGBM — Train'),
        (axes[1,1], lgbm_daily_te, 'LightGBM — Test (OOS)'),
    ]:
        if daily.empty:
            ax.set_title(title + ' (no trades)')
            continue
        cum = daily['pnl'].cumsum()
        ax.plot(cum.values, color='#2196F3', linewidth=1.5)
        ax.fill_between(range(len(cum)), cum.values, 0,
                        where=cum.values >= 0, color='#4CAF50', alpha=0.3)
        ax.fill_between(range(len(cum)), cum.values, 0,
                        where=cum.values < 0, color='#F44336', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'{title}  (Total=${daily["pnl"].sum():,.0f})', fontsize=10)
        ax.set_xlabel('Day')
        ax.set_ylabel('Cumulative Net PnL ($)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/factor_zoo_pnl.png', dpi=150)
    plt.close()

    print('\nFigures saved: figures/factor_zoo_importance.png, figures/factor_zoo_pnl.png')
    print('\nDone.')
