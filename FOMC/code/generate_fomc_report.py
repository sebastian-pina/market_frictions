"""
Generate all figures for the FOMC full report.
Run from FinalProject/ as: python FOMC/code/generate_fomc_report.py
Figures saved to FOMC/figures/.
"""
import matplotlib
matplotlib.use('Agg')
import io, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

FIG_DIR = Path('FOMC/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('FOMC/fomc_surprise_table.csv', parse_dates=['date'])

# ── Fetch ACM data (NY Fed) for time-series figures ───────────────────────────
NYFED_ACM_URL = ('https://www.newyorkfed.org/medialibrary/media/research/'
                 'data_indicators/ACMTermPremium.xls')
acm_rny2 = acm_tp2 = None
print('Fetching ACM data from NY Fed...')
try:
    with urllib.request.urlopen(NYFED_ACM_URL, timeout=30) as r:
        raw = r.read()
    acm_df   = pd.read_excel(io.BytesIO(raw), index_col=0, parse_dates=True)
    acm_rny2 = acm_df['ACMRNY02'].dropna().loc['2020-06-01':]
    acm_tp2  = acm_df['ACMTP02'].dropna().loc['2020-06-01':]
    print(f'  ACM loaded ({len(acm_rny2)} obs, {acm_rny2.index[0].date()} to {acm_rny2.index[-1].date()})')
except Exception as e:
    print(f'  ACM fetch failed: {e}')

# Also fetch DGS2 for ACM decomposition figure
FRED_DGS2_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2'
dgs2_series = None
print('Fetching DGS2 from FRED...')
try:
    with urllib.request.urlopen(FRED_DGS2_URL, timeout=20) as r:
        dgs2_series = pd.read_csv(r, index_col='observation_date',
                                  parse_dates=True, na_values=['.'])['DGS2'].dropna()
    dgs2_series = dgs2_series.loc['2020-06-01':]
    print(f'  DGS2 loaded ({len(dgs2_series)} obs)')
except Exception as e:
    print(f'  DGS2 fetch failed: {e}')

# ── Hardcoded summary stats from the pipeline ─────────────────────────────────
SPREAD_RATIOS = {
    2021: {'fomc': 1.097, 'control': 0.985},
    2022: {'fomc': 1.306, 'control': 1.062},
    2023: {'fomc': 1.177, 'control': 0.972},
    2024: {'fomc': 1.216, 'control': 1.002},
    2025: {'fomc': 1.272, 'control': 1.009},
}
AVG_TC = {2021: 5.0, 2022: 15.2, 2023: 8.5, 2024: 9.0, 2025: 8.9}

YEAR_PNL = {
    2021: {'fomc': 835,  'control': 106},
    2022: {'fomc': -787, 'control': 179},
    2023: {'fomc': 290,  'control': -339},
    2024: {'fomc': 252,  'control': -223},
    2025: {'fomc': 977,  'control': 554},
}

REGIME_COLOR = {
    'hold': '#1565C0', 'taper': '#0D47A1',
    'hike+25': '#BF360C', 'hike+50': '#C62828', 'hike+75': '#B71C1C',
    'cut-50': '#2E7D32', 'cut-25': '#388E3C', 'cut-25+hawk': '#827717',
    '?': '#546E7A',
}
REGIME_GROUP = {
    'hold': 'Hold/Cut', 'taper': 'Hold/Cut',
    'hike+25': 'Hiking', 'hike+50': 'Hiking', 'hike+75': 'Hiking',
    'cut-50': 'Hold/Cut', 'cut-25': 'Hold/Cut', 'cut-25+hawk': 'Hold/Cut',
    '?': 'Hold/Cut',
}

SEP_DATES = df[df['surprise_type'] == 'SEP']['date'].dt.strftime('%Y-%m-%d').tolist()


# ── Fig 1: Spread ratio by year (FOMC vs Control) ────────────────────────────
def fig_spread_ratio():
    years = list(SPREAD_RATIOS.keys())
    fomc_r = [SPREAD_RATIOS[y]['fomc'] for y in years]
    ctrl_r = [SPREAD_RATIOS[y]['control'] for y in years]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(years))
    w = 0.35
    bars1 = ax.bar(x - w/2, fomc_r, w, label='FOMC days',    color='#C62828', alpha=0.85)
    bars2 = ax.bar(x + w/2, ctrl_r, w, label='Control days', color='#1565C0', alpha=0.85)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='Baseline (ratio = 1)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_ylabel('Pre-announcement / Baseline Spread Ratio', fontsize=11)
    ax.set_title('Pre-FOMC Spread Widening: FOMC Days vs. Control Days\n'
                 '(1:30–2:00 PM spread relative to 11 AM–1 PM baseline)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0.9, 1.45)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, yr in zip(bars1, years):
        pct = (SPREAD_RATIOS[yr]['fomc'] - 1) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'+{pct:.1f}%', ha='center', va='bottom', fontsize=8, color='#C62828')
    for bar, yr in zip(bars2, years):
        pct = (SPREAD_RATIOS[yr]['control'] - 1) * 100
        lbl = f'{pct:+.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                lbl, ha='center', va='bottom', fontsize=7, color='#555')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_spread_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_spread_ratio.png')


# ── Fig 2: Scatter ret_14 vs Net PnL ─────────────────────────────────────────
def fig_ret_pnl_scatter():
    plot_df = df.dropna(subset=['ret_14', 'net_pnl']).copy()
    plot_df['group'] = plot_df['regime'].map(REGIME_GROUP).fillna('Hold/Cut')
    plot_df['track'] = np.where(plot_df['surprise_type'] == 'SEP', 'SEP', 'Non-SEP')

    fig, ax = plt.subplots(figsize=(9, 6))
    group_colors = {'Hiking': '#C62828', 'Hold/Cut': '#1565C0'}
    markers = {'SEP': 'o', 'Non-SEP': '^'}

    for group, gdf in plot_df.groupby('group'):
        for track, tdf in gdf.groupby('track'):
            ax.scatter(tdf['ret_14'], tdf['net_pnl'],
                       color=group_colors[group], alpha=0.85, s=80,
                       marker=markers[track], label=f'{group} / {track}', zorder=3)

    for _, row in plot_df.iterrows():
        if abs(row['net_pnl']) > 200 or abs(row['ret_14']) > 20:
            ax.annotate(row['date'].strftime('%b-%y'),
                        (row['ret_14'], row['net_pnl']),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=7, color='#555')

    for group, gdf in plot_df.groupby('group'):
        if len(gdf) >= 3:
            slope, intercept, r, p, _ = stats.linregress(gdf['ret_14'], gdf['net_pnl'])
            x_line = np.linspace(gdf['ret_14'].min(), gdf['ret_14'].max(), 50)
            ax.plot(x_line, intercept + slope * x_line,
                    color=group_colors[group], linestyle='--', alpha=0.5, linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_xlabel('2:00 PM 1-min Return (bps)  [negative = market fell on announcement]', fontsize=10)
    ax.set_ylabel('Net Strategy PnL ($)', fontsize=10)
    ax.set_title('Initial Bar Return vs. Strategy PnL\n'
                 '(Circle = SEP event, Triangle = Non-SEP; Hold/Cut blue, Hiking red)', fontsize=11)
    ax.legend(fontsize=9, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_ret_pnl_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_ret_pnl_scatter.png')


# ── Fig 3: FOMC vs Control PnL by year ───────────────────────────────────────
def fig_fomc_vs_control():
    years = list(YEAR_PNL.keys())
    fomc_pnl = [YEAR_PNL[y]['fomc']    for y in years]
    ctrl_pnl = [YEAR_PNL[y]['control'] for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    w = 0.38
    fomc_colors = ['#2E7D32' if v >= 0 else '#C62828' for v in fomc_pnl]
    ctrl_colors = ['#81C784' if v >= 0 else '#EF9A9A' for v in ctrl_pnl]
    regime_labels = ['ZLB Hold', 'Hiking', 'Transition', 'Cutting', 'Post-cut']

    ax.bar(x - w/2, fomc_pnl, w, color=fomc_colors, label='FOMC strategy', alpha=0.9)
    ax.bar(x + w/2, ctrl_pnl, w, color=ctrl_colors, label='Control (placebo)', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{y}\n({regime_labels[i]})'
                        for i, y in enumerate(years)], fontsize=10)
    ax.set_ylabel('Annual Net PnL ($)', fontsize=11)
    ax.set_title('FOMC Strategy vs. Matched Control Days (Placebo Test)\n'
                 'Edge should be FOMC-specific — control days should show no pattern', fontsize=11)

    for bar in ax.patches:
        h = bar.get_height()
        if abs(h) > 20:   # skip tiny bars to avoid clutter
            ax.text(bar.get_x() + bar.get_width()/2, h + (15 if h >= 0 else -45),
                    f'${h:+,.0f}', ha='center', va='bottom', fontsize=8)

    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_vs_control.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_vs_control.png')


# ── Fig 4: ACM Term Premium Decomposition ─────────────────────────────────────
def fig_acm_decomposition():
    """
    Show DGS2 = risk-neutral yield + term premium over time.
    Left: DGS2 vs ACMRNY02 with TP shaded.
    Right: ACMTP02 term premium over time with zero line.
    Marks SEP meeting dates.
    """
    if acm_rny2 is None or dgs2_series is None:
        print('Skipping fig_acm_decomposition (ACM or DGS2 data not available)')
        return

    # Align ACM (monthly) to DGS2 (daily) via forward-fill
    acm_rny2_daily = acm_rny2.reindex(dgs2_series.index, method='ffill')
    acm_tp2_daily  = acm_tp2.reindex(dgs2_series.index, method='ffill')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: DGS2 vs risk-neutral 2Y ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(dgs2_series.index, dgs2_series.values,
            color='#1565C0', linewidth=1.8, label='DGS2 (raw 2-Year Treasury)', zorder=3)
    ax.plot(acm_rny2_daily.index, acm_rny2_daily.values,
            color='#E65100', linewidth=1.8, linestyle='--',
            label='ACMRNY02 (risk-neutral 2Y, no term premium)', zorder=3)

    # Shade the term premium gap
    ax.fill_between(dgs2_series.index,
                    dgs2_series.values, acm_rny2_daily.reindex(dgs2_series.index).values,
                    where=(dgs2_series.values > acm_rny2_daily.reindex(dgs2_series.index).values),
                    alpha=0.15, color='#1565C0', label='Term premium (positive)')
    ax.fill_between(dgs2_series.index,
                    dgs2_series.values, acm_rny2_daily.reindex(dgs2_series.index).values,
                    where=(dgs2_series.values < acm_rny2_daily.reindex(dgs2_series.index).values),
                    alpha=0.15, color='#E65100', label='Term premium (negative / QE suppressed)')

    # Mark SEP meeting dates
    for sd in SEP_DATES:
        sd_ts = pd.Timestamp(sd)
        if dgs2_series.index[0] <= sd_ts <= dgs2_series.index[-1]:
            ax.axvline(sd_ts, color='grey', linewidth=0.5, alpha=0.4, linestyle=':')

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title('DGS2 = Risk-Neutral Rate + Term Premium\n'
                 'Dot-plot (pure expectations) must be compared to risk-neutral, not raw DGS2',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

    # ── Right: Term premium over time ─────────────────────────────────────────
    ax2 = axes[1]
    tp_vals = acm_tp2_daily.reindex(dgs2_series.index).values
    ax2.fill_between(dgs2_series.index, 0, tp_vals,
                     where=(tp_vals >= 0), alpha=0.5, color='#1565C0', label='Positive TP')
    ax2.fill_between(dgs2_series.index, 0, tp_vals,
                     where=(tp_vals < 0), alpha=0.5, color='#E65100', label='Negative TP (QE era)')
    ax2.plot(dgs2_series.index, tp_vals, color='#333', linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=0.8)

    # Mark SEP dates
    for sd in SEP_DATES:
        sd_ts = pd.Timestamp(sd)
        if dgs2_series.index[0] <= sd_ts <= dgs2_series.index[-1]:
            ax2.axvline(sd_ts, color='grey', linewidth=0.5, alpha=0.4, linestyle=':')

    # Annotate key periods
    ax2.text(pd.Timestamp('2021-06-01'), -0.45, 'ZLB / QE era\n(TP < 0: yields\nsuppressed below\nexpectations)',
             fontsize=7.5, color='#E65100', ha='center')
    ax2.text(pd.Timestamp('2022-09-01'),  0.4, 'Hiking cycle\n(TP > 0:\nuncertainty premium)',
             fontsize=7.5, color='#1565C0', ha='center')

    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('2-Year Term Premium (%)', fontsize=10)
    ax2.set_title('Adrian-Crump-Moench 2Y Term Premium (ACMTP02)\n'
                  'Monthly, from NY Fed — dots marked with grey lines are SEP meetings',
                  fontsize=10)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.25)

    plt.suptitle('Why We Strip the Term Premium: DGS2 vs. Pure-Expectations Rate (ACM Model)',
                 fontsize=12, y=1.01, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_acm_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_acm_decomposition.png')


# ── Fig 5: SEP dots implied rate vs ACM risk-neutral rate ─────────────────────
def fig_sep_vs_ois():
    """
    Left: At each SEP meeting, show DGS2_prev, dgs2_rn_prev (ACM), and dots-implied 2Y rate.
          The gap between the implied rate and dgs2_rn_prev drives the z_2Y filter.
    Right: gap_2Y_bps (implied rate minus ACMRNY02) in bps, colored by signal_2Y.
    """
    sep_df = df[df['surprise_type'] == 'SEP'].dropna(subset=['dgs2_prev']).copy()
    sep_df = sep_df.sort_values('date').reset_index(drop=True)

    sig_color = {'small': '#2E7D32', 'medium': '#F57F17', 'large': '#C62828', 'unknown': '#9E9E9E'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: three-way comparison at each SEP date ───────────────────────────
    ax = axes[0]
    x = np.arange(len(sep_df))
    w = 0.25

    ax.bar(x - w, sep_df['dgs2_prev'], w,
           color='#90CAF9', alpha=0.9, label='DGS2 prior day (raw 2Y yield)')
    ax.bar(x,     sep_df['dgs2_rn_prev'], w,
           color='#1565C0', alpha=0.9, label='ACMRNY02 prior day (risk-neutral 2Y)')
    ax.bar(x + w, sep_df['r_2Y_impl'], w,
           color='#E65100', alpha=0.9, label='Dots-implied 2Y rate (multi-year EH)')

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%b\n%y') for d in sep_df['date']], fontsize=7)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title('SEP Meetings: Raw DGS2 vs. Risk-Neutral 2Y vs. Dots-Implied Rate\n'
                 'Guidance gap = implied rate minus ACMRNY02 (apples-to-apples)',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # ── Right: guidance gap in bps, colored by signal ─────────────────────────
    ax2 = axes[1]
    gap_vals = sep_df['gap_2Y_bps'].values
    z2_vals  = sep_df['z_2Y'].values
    sigs     = sep_df['signal_2Y'].values
    colors   = [sig_color.get(s, '#9E9E9E') for s in sigs]

    ax2.bar(x, gap_vals, color=colors, alpha=0.88, edgecolor='white', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%b\n%y') for d in sep_df['date']], fontsize=7)
    ax2.set_ylabel('Implied 2Y Rate minus ACMRNY02 (bps)\n[+ = dots more hawkish than market]',
                   fontsize=9)
    ax2.set_title('Guidance Gap: Dots-Implied Rate minus Risk-Neutral Market Rate\n'
                  'Green = traded (|z|≤1.0), orange = medium skip, red = large skip',
                  fontsize=10)

    # Annotate only traded (small) events with z and PnL; show z only for others
    y_range = max(abs(gap_vals.max()), abs(gap_vals.min())) + 10
    for i, (gap, z, s, pnl) in enumerate(zip(gap_vals, z2_vals, sigs, sep_df['net_pnl'].values)):
        if not np.isnan(z):
            if s == 'small':
                # Full annotation for traded events
                y_z = gap + (8 if gap >= 0 else -8)
                va  = 'bottom' if gap >= 0 else 'top'
                ax2.text(i, y_z, f'z={z:+.2f}', ha='center', va=va, fontsize=7,
                         color='#1B5E20', fontweight='bold')
                if not np.isnan(pnl):
                    y_p = gap - (8 if gap >= 0 else -8)
                    va2 = 'top' if gap >= 0 else 'bottom'
                    ax2.text(i, y_p, f'${pnl:+.0f}', ha='center', va=va2, fontsize=7,
                             color='#2E7D32' if pnl > 0 else '#C62828', fontweight='bold')
            else:
                # Compact z-score only for skipped events (alternating height to avoid overlap)
                offset = 6 if i % 2 == 0 else 14
                y_z = gap + (offset if gap >= 0 else -offset)
                va  = 'bottom' if gap >= 0 else 'top'
                ax2.text(i, y_z, f'z={z:+.1f}', ha='center', va=va, fontsize=5.5, color='#555')

    patches = [
        mpatches.Patch(color='#2E7D32', alpha=0.88, label='Small — |z|≤1.0 (traded)'),
        mpatches.Patch(color='#F57F17', alpha=0.88, label='Medium — 1.0<|z|<1.5 (skip)'),
        mpatches.Patch(color='#C62828', alpha=0.88, label='Large — |z|≥1.5 (skip)'),
    ]
    ax2.legend(handles=patches, fontsize=9, loc='lower right')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('SEP Guidance Gap: Fed Dot-Plot vs. ACM Risk-Neutral Market Rate (2021–2025)',
                 fontsize=12, y=1.01, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_sep_vs_ois.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_sep_vs_ois.png')


# ── Fig 6: z_2Y distribution (ACM-adjusted) ───────────────────────────────────
def fig_zscore_dist():
    sep_df = df[df['surprise_type'] == 'SEP'].dropna(subset=['z_2Y']).copy()

    Z_SMALL = 1.0
    Z_LARGE = 1.5
    colors_map = {'small': '#2E7D32', 'medium': '#F57F17', 'large': '#C62828', 'unknown': '#9E9E9E'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of z_2Y by classification
    ax = axes[0]
    for cls, cdf in sep_df.groupby('signal_2Y'):
        ax.hist(cdf['z_2Y'], bins=8, color=colors_map.get(cls, 'grey'),
                alpha=0.75, label=f'{cls} (n={len(cdf)})', edgecolor='white')
    ax.axvline(-Z_LARGE, color='#C62828', linestyle='--', linewidth=1.5)
    ax.axvline(+Z_LARGE, color='#C62828', linestyle='--', linewidth=1.5, label=f'|z|={Z_LARGE}')
    ax.axvline(-Z_SMALL, color='#F57F17', linestyle=':', linewidth=1.5)
    ax.axvline(+Z_SMALL, color='#F57F17', linestyle=':', linewidth=1.5, label=f'|z|={Z_SMALL} (trade threshold)')
    ax.axvline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xlabel('Guidance gap z-score  ( z = gap / rolling vol )', fontsize=10)
    ax.set_ylabel('Count (SEP meetings)', fontsize=10)
    ax.set_title('Distribution of ACM-Adjusted Guidance Surprises\n'
                 '(18 SEP meetings; green = traded, orange = medium skip, red = large skip)',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Right: z_2Y by year with PnL encoded as marker size
    ax2 = axes[1]
    year_colors = {2021:'#1565C0', 2022:'#C62828', 2023:'#6A1B9A',
                   2024:'#2E7D32', 2025:'#E65100'}
    for yr, ydf in sep_df.groupby('year'):
        sc = ax2.scatter([yr]*len(ydf), ydf['z_2Y'],
                         color=year_colors.get(yr, 'grey'),
                         s=70 + ydf['net_pnl'].abs().fillna(50),
                         alpha=0.85, zorder=3, label=str(yr))
        # Annotate the traded (small) events
        for _, row in ydf.iterrows():
            if row.get('signal_2Y') == 'small':
                ax2.annotate(f"traded\n${row['net_pnl']:+.0f}",
                             (yr, row['z_2Y']),
                             textcoords='offset points', xytext=(12, 0),
                             fontsize=7, color='#2E7D32',
                             arrowprops=dict(arrowstyle='-', color='#2E7D32', lw=0.8))

    ax2.axhline(Z_SMALL,  color='#F57F17', linestyle=':', linewidth=1.2)
    ax2.axhline(-Z_SMALL, color='#F57F17', linestyle=':', linewidth=1.2, label='|z|=1.0 threshold')
    ax2.axhline(Z_LARGE,  color='#C62828', linestyle='--', linewidth=1.2)
    ax2.axhline(-Z_LARGE, color='#C62828', linestyle='--', linewidth=1.2, label='|z|=1.5')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('ACM Guidance Gap z-score (z_2Y)', fontsize=10)
    ax2.set_title('Guidance Surprise Magnitude by Year\n'
                  '(Marker size ~ |PnL|; labeled = traded events with outcome)',
                  fontsize=10)
    ax2.legend(fontsize=9, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('ACM-Adjusted Guidance Surprise Filter — SEP Events Only', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_zscore_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_zscore_analysis.png')


# ── Fig 7: Regime-conditioned cumulative PnL ─────────────────────────────────
def fig_regime_pnl():
    plot_df = df.dropna(subset=['net_pnl']).copy()
    plot_df = plot_df.sort_values('date').reset_index(drop=True)
    plot_df['group'] = plot_df['regime'].map(REGIME_GROUP).fillna('Hold/Cut')

    hc   = plot_df[plot_df['group'] == 'Hold/Cut'].reset_index(drop=True)
    hk   = plot_df[plot_df['group'] == 'Hiking'].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(range(len(plot_df)), plot_df['net_pnl'].cumsum(),
            color='grey', linewidth=1.5, linestyle='--', label='All events (N=40)', alpha=0.7)
    if not hc.empty:
        ax.plot(range(len(hc)), hc['net_pnl'].cumsum(),
                color='#1565C0', linewidth=2.2, label=f'Hold/Cut (N={len(hc)})')
    if not hk.empty:
        ax.plot(range(len(hk)), hk['net_pnl'].cumsum(),
                color='#C62828', linewidth=2.2, label=f'Hiking 2022 (N={len(hk)})')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_xlabel('Event number', fontsize=10)
    ax.set_ylabel('Cumulative Net PnL ($)', fontsize=10)
    ax.set_title('Regime-Conditioned Cumulative PnL\n'
                 'Strategy works in hold/cut regimes; fails exclusively in 2022 hiking cycle',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    yr_pnl = plot_df.groupby('year')['net_pnl'].sum()
    bar_colors = ['#2E7D32' if v >= 0 else '#C62828' for v in yr_pnl.values]
    ax2.bar(yr_pnl.index, yr_pnl.values, color=bar_colors, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8)
    for yr, pnl in yr_pnl.items():
        ax2.text(yr, pnl + (20 if pnl >= 0 else -65),
                 f'${pnl:+,.0f}', ha='center', va='bottom', fontsize=9)
    regime_labels = {2021:'ZLB Hold', 2022:'Hiking', 2023:'Transition', 2024:'Cutting', 2025:'Post-cut'}
    ax2.set_xticks(list(yr_pnl.index))
    ax2.set_xticklabels([f'{y}\n({regime_labels.get(y,"")})'
                         for y in yr_pnl.index], fontsize=9)
    ax2.set_ylabel('Annual Net PnL ($)', fontsize=10)
    ax2.set_title('Annual PnL by Year\n(All 40 traded events)', fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('FOMC Contrarian Strategy: Regime Analysis', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_regime_pnl.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_regime_pnl.png')


# ── Fig 8: TC progression + spread trend ──────────────────────────────────────
def fig_tc_trend():
    years  = list(AVG_TC.keys())
    tcs    = list(AVG_TC.values())
    fomc_r = [SPREAD_RATIOS[y]['fomc'] for y in years]
    ctrl_r = [SPREAD_RATIOS[y]['control'] for y in years]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.bar(years, tcs, color='#90CAF9', alpha=0.9, width=0.5, label='Avg TC per trade ($)')
    ax2.plot(years, fomc_r, color='#C62828', linewidth=2.2, marker='o',
             markersize=7, label='FOMC spread ratio (pre-ann / baseline)')
    ax2.plot(years, ctrl_r, color='#90CAF9', linewidth=1.8, marker='s', markersize=5,
             linestyle='--', label='Control spread ratio')
    ax2.axhline(1.0, color='grey', linewidth=0.6, linestyle=':')

    # Annotate TC bars
    for yr, tc in zip(years, tcs):
        ax1.text(yr, tc + 0.15, f'${tc:.1f}', ha='center', va='bottom', fontsize=8.5)
    # Annotate FOMC spread ratio
    for yr, ratio in zip(years, fomc_r):
        ax2.text(yr, ratio + 0.003, f'{ratio:.3f}', ha='center', va='bottom', fontsize=8,
                 color='#C62828')

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Average TC per Trade ($)', fontsize=11, color='#1565C0')
    ax2.set_ylabel('Pre-announcement / Baseline Spread Ratio', fontsize=11, color='#C62828')
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax2.tick_params(axis='y', labelcolor='#C62828')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax1.set_title('Transaction Cost and Spread Widening Trend (2021–2025)\n'
                  'Higher spread widening creates larger overshoots AND higher TC (net effect: profitable in hold/cut)',
                  fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_tc_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_tc_trend.png')


# ── Fig 9: Two-track ACM filter breakdown ────────────────────────────────────
def fig_two_track():
    """
    Show the z_2Y two-track strategy decomposition.
    Left: Cumulative PnL — unfiltered, z_2Y two-track, non-SEP only, SEP excluded.
    Right: Annual PnL by track using signal_2Y classification.
    """
    plot_df = df.dropna(subset=['net_pnl', 'surprise_type']).sort_values('date').reset_index(drop=True)

    non_sep   = plot_df[plot_df['surprise_type'] == 'target'].reset_index(drop=True)
    sep_small = plot_df[(plot_df['surprise_type'] == 'SEP') &
                        (plot_df['signal_2Y'] == 'small')].reset_index(drop=True)
    sep_excl  = plot_df[(plot_df['surprise_type'] == 'SEP') &
                        (plot_df['signal_2Y'] != 'small')].reset_index(drop=True)
    two_track = pd.concat([non_sep, sep_small]).sort_values('date').reset_index(drop=True)

    # Compute stats
    def sharpe_p(series):
        s = series.dropna()
        if len(s) < 3: return np.nan, np.nan
        sh = s.mean() / s.std() * np.sqrt(len(s))
        _, p = stats.ttest_1samp(s, 0)
        return sh, p

    sh_tt, p_tt = sharpe_p(two_track['net_pnl'])
    sh_ns, p_ns = sharpe_p(non_sep['net_pnl'])
    sh_sm, p_sm = sharpe_p(sep_small['net_pnl'])
    hit_tt = (two_track['net_pnl'] > 0).mean()
    hit_ns = (non_sep['net_pnl'] > 0).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Cumulative PnL per track ────────────────────────────────────────
    ax = axes[0]
    ax.plot(range(len(plot_df)), plot_df['net_pnl'].cumsum(),
            color='grey', linewidth=1.5, linestyle='--', alpha=0.55,
            label=f'Unfiltered — all 40 events  (${plot_df["net_pnl"].sum():+,.0f})')
    ax.plot(range(len(two_track)), two_track['net_pnl'].cumsum(),
            color='#1565C0', linewidth=2.8,
            label=f'Two-track ACM filter (N={len(two_track)})  (${two_track["net_pnl"].sum():+,.0f})')
    ax.plot(range(len(non_sep)), non_sep['net_pnl'].cumsum(),
            color='#42A5F5', linewidth=2.0, linestyle='-.',
            label=f'Non-SEP only (N={len(non_sep)})  (${non_sep["net_pnl"].sum():+,.0f})')
    ax.plot(range(len(sep_excl)), sep_excl['net_pnl'].cumsum(),
            color='#EF9A9A', linewidth=1.6, linestyle=':',
            label=f'SEP excluded |z|>1.0 (N={len(sep_excl)})  (${sep_excl["net_pnl"].sum():+,.0f})')

    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xlabel('Event count', fontsize=10)
    ax.set_ylabel('Cumulative Net PnL ($)', fontsize=10)
    ax.set_title(f'Two-Track Cumulative PnL\n(ACM z-filter selects {len(two_track)} of 40 events)',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add stats annotation
    ax.text(0.02, 0.97,
            f'Two-track (N={len(two_track)}):  Sharpe={sh_tt:.2f}, p={p_tt:.3f}, Hit={hit_tt:.0%}\n'
            f'Non-SEP only (N={len(non_sep)}):  Sharpe={sh_ns:.2f}, p={p_ns:.3f}, Hit={hit_ns:.0%}\n'
            f'SEP traded (N={len(sep_small)}):    Sharpe={sh_sm:.2f}, p={p_sm:.3f}, Hit={(sep_small["net_pnl"]>0).mean():.0%}',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85))

    # ── Right: Annual PnL by track ────────────────────────────────────────────
    ax2 = axes[1]
    years = sorted(plot_df['year'].unique())
    x = np.arange(len(years))
    w = 0.25

    nonsep_yr  = non_sep.groupby('year')['net_pnl'].sum().reindex(years, fill_value=0)
    sepsm_yr   = sep_small.groupby('year')['net_pnl'].sum().reindex(years, fill_value=0)
    sepex_yr   = sep_excl.groupby('year')['net_pnl'].sum().reindex(years, fill_value=0)

    ax2.bar(x - w, nonsep_yr.values,  w, color='#1565C0', alpha=0.85,
            label=f'Non-SEP always trade (N={len(non_sep)})')
    ax2.bar(x,     sepsm_yr.values,   w, color='#42A5F5', alpha=0.85,
            label=f'SEP |z| ≤ 1.0 traded (N={len(sep_small)})')
    ax2.bar(x + w, sepex_yr.values,   w, color='#EF9A9A', alpha=0.65,
            label=f'SEP |z| > 1.0 skipped (N={len(sep_excl)})')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in years], fontsize=11)
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Annual Net PnL ($)', fontsize=10)
    ax2.set_title('Annual PnL by Track (ACM Guidance-Gap Filter)\n'
                  'Non-SEP events are the primary PnL engine in every year',
                  fontsize=10)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Two-Track FOMC Filter: ACM-Adjusted Guidance Gap Classification',
                 fontsize=12, y=1.01, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_two_track.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_two_track.png')


# ── Fig 10: Rolling Sharpe + Cumulative PnL (extended) ───────────────────────
def fig_extended_pnl():
    """Cumulative PnL full history + rolling Sharpe (trailing 8 events)."""
    plot_df = df.dropna(subset=['net_pnl']).sort_values('date').reset_index(drop=True)

    # Two-track subset
    sep_small = plot_df[(plot_df['surprise_type'] == 'SEP') &
                        (plot_df['signal_2Y'] == 'small')]
    non_sep   = plot_df[plot_df['surprise_type'] == 'target']
    two_track = pd.concat([non_sep, sep_small]).sort_values('date').reset_index(drop=True)

    # Rolling Sharpe on the two-track (trailing 8 events)
    ROLL_W = 8
    pnl_tt = two_track['net_pnl'].values
    roll_sh = []
    for i in range(len(pnl_tt)):
        window = pnl_tt[max(0, i - ROLL_W + 1):i + 1]
        if len(window) >= 3 and window.std() > 0:
            roll_sh.append(window.mean() / window.std() * np.sqrt(len(window)))
        else:
            roll_sh.append(np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Top: cumulative PnL
    ax = axes[0]
    cum_all = plot_df['net_pnl'].cumsum()
    cum_tt  = two_track['net_pnl'].cumsum()

    ax.fill_between(range(len(cum_all)), 0, cum_all.values,
                    where=(cum_all.values >= 0), alpha=0.2, color='#1565C0')
    ax.fill_between(range(len(cum_all)), 0, cum_all.values,
                    where=(cum_all.values < 0), alpha=0.2, color='#C62828')
    ax.plot(range(len(cum_all)), cum_all.values,
            color='grey', linewidth=1.5, linestyle='--', alpha=0.7,
            label=f'All 40 events unfiltered (${cum_all.iloc[-1]:+,.0f})')
    ax.plot(range(len(cum_tt)), cum_tt.values,
            color='#1565C0', linewidth=2.5,
            label=f'Two-track ACM filter, N={len(two_track)} (${cum_tt.iloc[-1]:+,.0f})')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')

    # Add year labels on x-axis
    for yr in [2021, 2022, 2023, 2024, 2025]:
        yr_mask = plot_df['year'] == yr
        if yr_mask.any():
            first_idx = plot_df[yr_mask].index[0]
            ax.axvline(first_idx, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.text(first_idx + 0.3, ax.get_ylim()[0] if ax.get_ylim()[0] > -500 else -800,
                    str(yr), fontsize=8, color='grey')

    ax.set_ylabel('Cumulative Net PnL ($)', fontsize=10)
    ax.set_title('Full Cumulative PnL — All 40 Events and Two-Track ACM Strategy (2021–2025)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Bottom: rolling Sharpe
    ax2 = axes[1]
    ax2.plot(range(len(roll_sh)), roll_sh, color='#1565C0', linewidth=2.0,
             label=f'Rolling {ROLL_W}-event Sharpe (two-track)')
    ax2.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax2.axhline(2.0, color='#2E7D32', linewidth=0.8, linestyle='--', alpha=0.7, label='Sharpe=2.0')
    ax2.fill_between(range(len(roll_sh)), 0, roll_sh,
                     where=[v >= 0 if not np.isnan(v) else False for v in roll_sh],
                     alpha=0.2, color='#1565C0')
    ax2.fill_between(range(len(roll_sh)), 0, roll_sh,
                     where=[v < 0 if not np.isnan(v) else False for v in roll_sh],
                     alpha=0.2, color='#C62828')
    ax2.set_xlabel('Event number (two-track)', fontsize=10)
    ax2.set_ylabel(f'Rolling {ROLL_W}-event Sharpe', fontsize=10)
    ax2.set_title(f'Rolling {ROLL_W}-event Sharpe Ratio — Two-Track Strategy\n'
                  '(2022 shows negative; hold/cut years consistently positive)',
                  fontsize=10)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_extended_pnl.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_extended_pnl.png')


if __name__ == '__main__':
    print('Generating FOMC report figures...\n')
    fig_spread_ratio()
    fig_ret_pnl_scatter()
    fig_fomc_vs_control()
    fig_acm_decomposition()
    fig_sep_vs_ois()
    fig_zscore_dist()
    fig_regime_pnl()
    fig_tc_trend()
    fig_two_track()
    fig_extended_pnl()
    print('\nAll figures saved to FOMC/figures/')
