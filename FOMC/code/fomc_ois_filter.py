"""
FOMC Rate-Path Surprise Filter — Two-Track Framework
======================================================
SEP days (Mar/Jun/Sep/Dec): extrapolate an implied 1Y and 2Y rate from the dot-plot
    using the expectations hypothesis with MULTI-YEAR SEP projections. The dot-plot
    publishes year-end rate projections for the current year + next 2-3 years.
    We integrate the full implied path over a 24-month horizon.

    Non-December meetings (month m = 3, 6, 9; M_rem = 12-m remaining months in year):
        Phase 1 (M_rem months): linear ramp r0 → r_YE1  (current year remaining)
        Phase 2 (12 months):    linear ramp r_YE1 → r_YE2  (entire next calendar year)
        Phase 3 (m months):     flat at r_YE2  (first m months of year+2)

        r_1Y = [M_rem*(r0+r_YE1)/2 + m*(r_YE1+(r_YE2-r_YE1)*m/24)] / 12
        r_2Y = [M_rem*(r0+r_YE1)/2 + 12*(r_YE1+r_YE2)/2 + m*r_YE2] / 24

    December meetings (m=12; r_YE1=next-year-end, r_YE2=year-after-next-end):
        r_1Y = (r0 + r_YE1) / 2
        r_2Y = (r0 + 2*r_YE1 + r_YE2) / 4

    z_2Y = (r_2Y - ACMRNY02_prev)*100 / max(σ_30d(ACMRNY02), 8 bps)
         ACMRNY02 = ACM risk-neutral 2Y yield (DGS2 minus term premium).
         Dots are pure-expectations; comparison must strip the term premium.

    Filter threshold: |z_2Y| <= 1.0.  z_1Y also computed for comparison.

Non-SEP days: always trade (no filter).
    The rate decision alone is the only new information; it is almost always fully priced.
"""
import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import io
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

# ── SEP dot-plot: multi-year FFR median projections at each quarterly meeting ──
# Source: FOMC Summary of Economic Projections (released same time as statement)
# Format: (r_YE1, r_YE2)
#   Non-December: r_YE1 = current-year-end, r_YE2 = next-year-end
#   December:     r_YE1 = next-year-end,    r_YE2 = year-after-next-end
SEP_MULTIYEAR = {
    '2021-03-17': (0.125, 0.125),   # 2021 YE=0.125, 2022 YE=0.125 (dots flat at ZLB)
    '2021-06-16': (0.125, 0.125),   # 2021 YE=0.125, 2022 YE=0.125
    '2021-09-22': (0.125, 0.250),   # 2021 YE=0.125, 2022 YE=0.250 (first hike split)
    '2021-12-15': (0.875, 1.625),   # 2022 YE=0.875, 2023 YE=1.625 (December)
    '2022-03-16': (1.875, 2.750),   # 2022 YE=1.875, 2023 YE=2.750
    '2022-06-15': (3.375, 3.750),   # 2022 YE=3.375, 2023 YE=3.750
    '2022-09-21': (4.375, 4.625),   # 2022 YE=4.375, 2023 YE=4.625
    '2022-12-14': (5.125, 4.125),   # 2023 YE=5.125, 2024 YE=4.125 (December)
    '2023-03-22': (5.125, 4.250),   # 2023 YE=5.125, 2024 YE=4.250
    '2023-06-14': (5.625, 4.625),   # 2023 YE=5.625, 2024 YE=4.625
    '2023-09-20': (5.625, 5.125),   # 2023 YE=5.625, 2024 YE=5.125
    '2023-12-13': (4.625, 3.625),   # 2024 YE=4.625, 2025 YE=3.625 (December)
    '2024-03-20': (4.625, 3.875),   # 2024 YE=4.625, 2025 YE=3.875
    '2024-06-12': (5.125, 4.125),   # 2024 YE=5.125, 2025 YE=4.125
    '2024-09-18': (4.375, 3.375),   # 2024 YE=4.375, 2025 YE=3.375
    '2024-12-18': (3.875, 3.375),   # 2025 YE=3.875, 2026 YE=3.375 (December)
    '2025-03-19': (3.875, 3.625),   # 2025 YE=3.875, 2026 YE=3.625
    '2025-06-18': (3.875, 3.625),   # 2025 YE=3.875, 2026 YE=3.625
}
SEP_DATA  = {k: v[0] for k, v in SEP_MULTIYEAR.items()}  # r_YE1 backward compat
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
    df = pd.read_parquet(p, engine='pyarrow', columns=BOOK_COLS).sort_index()
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


# ── 1a. Fetch FRED rate series ────────────────────────────────────────────────
FRED_DGS1_URL   = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1'
FRED_DGS2_URL   = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2'
FRED_DGS1MO_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO'
FRED_EFFR_URL   = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=EFFR'

def fetch_fred(url, col, label):
    print(f'Fetching {label} from FRED...')
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            s = pd.read_csv(r, index_col='observation_date', parse_dates=True,
                            na_values=['.'])[col].dropna()
        s = s.loc['2020-01-01':]
        diff     = s.diff() * 100
        roll_std = diff.rolling(30, min_periods=10).std()
        print(f'  Loaded {len(s)} obs ({s.index[0].date()} – {s.index[-1].date()})')
        return s, roll_std
    except Exception as e:
        print(f'  {label} fetch failed: {e}')
        return None, None

dgs1,   roll_std_dgs1   = fetch_fred(FRED_DGS1_URL,   'DGS1',   'DGS1  (1-Year Treasury)')
dgs2,   roll_std_dgs2   = fetch_fred(FRED_DGS2_URL,   'DGS2',   'DGS2  (2-Year Treasury)')
dgs1mo, roll_std_dgs1mo = fetch_fred(FRED_DGS1MO_URL, 'DGS1MO', 'DGS1MO (1-Month Treasury)')
effr,   _               = fetch_fred(FRED_EFFR_URL,   'EFFR',   'EFFR  (Effective FFR)')

# ── 1b. ACM risk-neutral 2Y yield (NY Fed) — pure-expectations 2Y rate
# ACMRNY02 = DGS2 - ACMTP02; dots are pure-expectations, so comparison must match.
# Data is monthly; reindexed to daily DGS2 grid with ffill (prior month-end is observable).
NYFED_ACM_URL = ('https://www.newyorkfed.org/medialibrary/media/research/'
                 'data_indicators/ACMTermPremium.xls')
dgs2_rn          = None
roll_std_dgs2_rn = None
print('Fetching ACM risk-neutral 2Y yield (NY Fed)...')
try:
    with urllib.request.urlopen(NYFED_ACM_URL, timeout=30) as r:
        raw = r.read()
    acm_df  = pd.read_excel(io.BytesIO(raw), index_col=0, parse_dates=True)
    acmrny2 = acm_df['ACMRNY02'].dropna()
    acmrny2 = acmrny2.loc['2020-01-01':]
    print(f'  Loaded {len(acmrny2)} obs ({acmrny2.index[0].date()} – {acmrny2.index[-1].date()})')
    if dgs2 is not None:
        # Monthly ACM → daily grid; ffill carries prior month-end (observable at trade time)
        dgs2_rn          = acmrny2.reindex(dgs2.index, method='ffill')
        dgs2_rn_diff     = dgs2_rn.diff() * 100
        roll_std_dgs2_rn = dgs2_rn_diff.rolling(30, min_periods=10).std()
except Exception as e:
    print(f'  ACM fetch failed: {e}')


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
        # SEP track: extrapolate implied 1Y and 2Y rates from dot-plot using
        # multi-year SEP projections and the expectations hypothesis.
        # Both inputs available at 14:01: SEP released at 14:00; EFFR/DGS from prior close.
        r_YE1, r_YE2 = SEP_MULTIYEAR[fomc_str]
        sep_median = r_YE1   # primary year-end projection (backward compat label)
        m          = fomc_ts.month     # 3, 6, 9, or 12

        prev_dgs2_idx = dgs2.index[dgs2.index < fomc_ts]   if dgs2   is not None else pd.DatetimeIndex([])
        prev_dgs1_idx = dgs1.index[dgs1.index < fomc_ts]   if dgs1   is not None else pd.DatetimeIndex([])
        prev_effr_idx = effr.index[effr.index < fomc_ts]   if effr   is not None else pd.DatetimeIndex([])

        r_0       = effr[prev_effr_idx[-1]]   if len(prev_effr_idx) > 0 else sep_median
        dgs2_prev = dgs2[prev_dgs2_idx[-1]]   if len(prev_dgs2_idx) > 0 else np.nan
        dgs1_prev = dgs1[prev_dgs1_idx[-1]]   if len(prev_dgs1_idx) > 0 else np.nan

        # ACM-adjusted 2Y (risk-neutral): dots = pure expectations, so strip term premium
        dgs2_rn_prev = dgs2_rn[prev_dgs2_idx[-1]] if (dgs2_rn is not None and len(prev_dgs2_idx) > 0) else np.nan

        # Multi-year implied rates (expectations hypothesis)
        if m == 12:
            # December: Phase 1 = 12-month ramp r0→r_YE1; Phase 2 = 12-month ramp r_YE1→r_YE2
            r_1Y = (r_0 + r_YE1) / 2
            r_2Y = (r_0 + 2 * r_YE1 + r_YE2) / 4
        else:
            # Non-December: Phase 1 (m_rem months r0→r_YE1), Phase 2 (12 months r_YE1→r_YE2),
            #               Phase 3 (m months flat at r_YE2). Total = m_rem+12+m = 24 months.
            m_rem = 12 - m
            phase2_avg_1Y = r_YE1 + (r_YE2 - r_YE1) * m / 24   # avg over first m months of ramp
            r_1Y = (m_rem * (r_0 + r_YE1) / 2 + m * phase2_avg_1Y) / 12
            r_2Y = (m_rem * (r_0 + r_YE1) / 2 + 12 * (r_YE1 + r_YE2) / 2 + m * r_YE2) / 24

        # Guidance gaps in bps
        gap_1Y = (r_1Y - dgs1_prev)   * 100 if not np.isnan(dgs1_prev)   else np.nan
        gap_2Y = (r_2Y - dgs2_rn_prev) * 100 if not np.isnan(dgs2_rn_prev) else np.nan

        # Rolling vol (floored), using prior-day value for real-time discipline
        def floored_std(roll_std_s, idx):
            if roll_std_s is None or len(idx) == 0: return SEP_FLOOR_STD
            v = roll_std_s.get(idx[-1], np.nan)
            return max(v, SEP_FLOOR_STD) if not np.isnan(v) else SEP_FLOOR_STD

        vol_1Y = floored_std(roll_std_dgs1,    prev_dgs1_idx)
        vol_2Y = floored_std(roll_std_dgs2_rn, prev_dgs2_idx)  # vol of risk-neutral 2Y

        z_1Y = gap_1Y / vol_1Y if not np.isnan(gap_1Y) else np.nan
        z_2Y = gap_2Y / vol_2Y if not np.isnan(gap_2Y) else np.nan

        fomc_dgs2 = dgs2.get(fomc_ts, np.nan) if dgs2 is not None else np.nan  # informational

        rec.update({
            'sep_median':    round(r_YE1, 3),
            'r_YE2':         round(r_YE2, 3),
            'effr_prev':     round(r_0, 3),
            'dgs1_prev':     round(dgs1_prev, 3)    if not np.isnan(dgs1_prev)    else np.nan,
            'dgs2_prev':     round(dgs2_prev, 3)    if not np.isnan(dgs2_prev)    else np.nan,
            'dgs2_rn_prev':  round(dgs2_rn_prev, 3) if not np.isnan(dgs2_rn_prev) else np.nan,
            'dgs2_fomc':     round(fomc_dgs2, 3)    if not np.isnan(fomc_dgs2)    else np.nan,
            'r_1Y_impl':     round(r_1Y, 3),
            'r_2Y_impl':     round(r_2Y, 3),
            'gap_1Y_bps':    round(gap_1Y, 1)       if not np.isnan(gap_1Y)       else np.nan,
            'gap_2Y_bps':    round(gap_2Y, 1)       if not np.isnan(gap_2Y)       else np.nan,
            'vol_1Y_bps':    round(vol_1Y, 2),
            'vol_2Y_bps':    round(vol_2Y, 2),
            'z_1Y':          round(z_1Y, 2)          if not np.isnan(z_1Y)        else np.nan,
            'z_2Y':          round(z_2Y, 2)          if not np.isnan(z_2Y)        else np.nan,
            # primary z_score for backward compat — use z_1Y (user preference)
            'surprise_bps':  round(gap_1Y, 1)       if not np.isnan(gap_1Y)       else np.nan,
            'roll_std_bps':  round(vol_1Y, 2),
            'z_score':       round(z_1Y, 2)          if not np.isnan(z_1Y)        else np.nan,
        })
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
for c in ['net_pnl','gross_pnl','tc','ret_14',
          'sep_median','r_YE2','effr_prev','dgs1_prev','dgs2_prev','dgs2_rn_prev','dgs2_fomc',
          'r_1Y_impl','r_2Y_impl','gap_1Y_bps','gap_2Y_bps',
          'vol_1Y_bps','vol_2Y_bps','z_1Y','z_2Y',
          'dgs1mo_prev','implied_move_bps','actual_move_bps',
          'surprise_bps','roll_std_bps','z_score']:
    if c not in df.columns:
        df[c] = np.nan


# ── 3. Classify events ────────────────────────────────────────────────────────
Z_SMALL        = 1.0
Z_LARGE        = 1.5
THRESH_SMALL   = 5.0
THRESH_MEDIUM  = 12.0

def classify_z(z_val):
    if pd.isna(z_val): return 'unknown'
    if abs(z_val) <= Z_SMALL: return 'small'
    if abs(z_val) <  Z_LARGE: return 'medium'
    return 'large'

def classify(row):
    z = row.get('z_score', np.nan)
    if not pd.isna(z): return classify_z(z)
    s = row.get('surprise_bps', np.nan)
    if pd.isna(s): return 'unknown'
    return 'small' if abs(s) <= THRESH_SMALL else ('medium' if abs(s) <= THRESH_MEDIUM else 'large')

df['signal_class'] = df.apply(classify, axis=1)

# Independent signal columns for 1Y and 2Y filters (SEP only; non-SEP always 'small')
df['signal_1Y'] = df['z_1Y'].apply(classify_z)
df['signal_2Y'] = df['z_2Y'].apply(classify_z)
df.loc[df['surprise_type'] == 'target', ['signal_1Y', 'signal_2Y']] = 'small'  # always trade


# ── 4. Print full event table ─────────────────────────────────────────────────
SEP_df = df[df['surprise_type'] == 'SEP']
print('\n' + '='*145)
print(f'{"Date":<12} {"Regime":<14} {"EFFR":>6} {"r_YE1":>6} {"r_YE2":>6} {"r1Y":>6} {"r2Y":>6} '
      f'{"ACMRN":>6} {"gap2Y":>7} {"vol2Y":>6} '
      f'{"z_2Y":>7} {"sig2Y":<8} {"Net PnL":>9}')
print('-'*145)
for _, r in SEP_df.iterrows():
    def fmt(v, fmt_str='+.1f'): return f'{v:{fmt_str}}' if not pd.isna(v) else '  N/A'
    pnl_str = f'${r["net_pnl"]:+,.0f}' if not pd.isna(r.get('net_pnl', np.nan)) else '    N/A'
    print(f'{str(r["date"].date()):<12} {r["regime"]:<14} '
          f'{fmt(r.get("effr_prev"), ".3f"):>6} {fmt(r.get("sep_median"), ".3f"):>6} '
          f'{fmt(r.get("r_YE2"), ".3f"):>6} '
          f'{fmt(r.get("r_1Y_impl"), ".3f"):>6} {fmt(r.get("r_2Y_impl"), ".3f"):>6} '
          f'{fmt(r.get("dgs2_rn_prev"), ".3f"):>6} '
          f'{fmt(r.get("gap_2Y_bps")):>7} {fmt(r.get("vol_2Y_bps"), ".1f"):>6} '
          f'{fmt(r.get("z_2Y"), "+.2f"):>7} {str(r.get("signal_2Y","?")):8} {pnl_str:>9}')

print('\nNon-SEP events (always traded):')
non_sep_df = df[df['surprise_type'] == 'target']
print(f'{"Date":<12} {"Regime":<18} {"Net PnL":>9}')
for _, r in non_sep_df.iterrows():
    pnl_str = f'${r["net_pnl"]:+,.0f}' if not pd.isna(r.get('net_pnl', np.nan)) else '    N/A'
    print(f'{str(r["date"].date()):<12} {r["regime"]:<18} {pnl_str:>9}')


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
    print(f'  {label:<52}  N={len(pnl):2d}  Total=${total:+,.0f}  '
          f'Sharpe={sharpe:.2f}  p={p:.3f}  Hit={hit:.0%}')

non_sep_all = df[df['surprise_type'] == 'target']

print('\n' + '='*90)
print('FILTERED BACKTEST COMPARISON')
print('='*90)
summarize(df,             'All 40 events (unfiltered)')
summarize(non_sep_all,    'Non-SEP (22 events, always trade)  ** clean result')

print('\n--- z_1Y filter: r_1Y_impl vs DGS1 ---------------------------------------------------')
sep_1Y_small = df[(df['surprise_type']=='SEP') & (df['signal_1Y']=='small')]
sep_1Y_excl  = df[(df['surprise_type']=='SEP') & (df['signal_1Y']!='small')]
tt_1Y = pd.concat([sep_1Y_small, non_sep_all]).sort_values('date')
summarize(tt_1Y,          'Two-track (SEP |z_1Y|<=1.0 + all non-SEP)')
summarize(sep_1Y_small,   '  SEP traded (z_1Y small)')
summarize(sep_1Y_excl,    '  SEP excluded (z_1Y medium/large)')

print('\n--- z_2Y filter: r_2Y_impl vs DGS2 ---------------------------------------------------')
sep_2Y_small = df[(df['surprise_type']=='SEP') & (df['signal_2Y']=='small')]
sep_2Y_excl  = df[(df['surprise_type']=='SEP') & (df['signal_2Y']!='small')]
tt_2Y = pd.concat([sep_2Y_small, non_sep_all]).sort_values('date')
summarize(tt_2Y,          'Two-track (SEP |z_2Y|<=1.0 + all non-SEP)')
summarize(sep_2Y_small,   '  SEP traded (z_2Y small)')
summarize(sep_2Y_excl,    '  SEP excluded (z_2Y medium/large)')

print('\n--- SEP events: side-by-side z_1Y vs z_2Y -----------------------------------------')
print(f'  {"Date":<12} {"z_1Y":>7} {"sig1Y":<8} {"z_2Y":>7} {"sig2Y":<8} {"Net PnL":>9}')
for _, r in df[df['surprise_type']=='SEP'].iterrows():
    z1 = f'{r["z_1Y"]:+.2f}' if not pd.isna(r.get('z_1Y',np.nan)) else '   N/A'
    z2 = f'{r["z_2Y"]:+.2f}' if not pd.isna(r.get('z_2Y',np.nan)) else '   N/A'
    pnl = f'${r["net_pnl"]:+,.0f}' if not pd.isna(r.get('net_pnl',np.nan)) else '    N/A'
    print(f'  {str(r["date"].date()):<12} {z1:>7} {str(r["signal_1Y"]):<8} '
          f'{z2:>7} {str(r["signal_2Y"]):<8} {pnl:>9}')


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
             'effr_prev','sep_median','r_YE2',
             'dgs1_prev','dgs2_prev','dgs2_rn_prev','dgs2_fomc',
             'r_1Y_impl','r_2Y_impl',
             'gap_1Y_bps','vol_1Y_bps','z_1Y','signal_1Y',
             'gap_2Y_bps','vol_2Y_bps','z_2Y','signal_2Y',
             'dgs1mo_prev','implied_move_bps','actual_move_bps',
             'surprise_bps','roll_std_bps','z_score','signal_class',
             'ret_14','gross_pnl','tc','net_pnl']
out_cols = [c for c in col_order if c in df.columns]
df[out_cols].to_csv('FOMC/fomc_surprise_table.csv', index=False)
print('Table saved: FOMC/fomc_surprise_table.csv')
