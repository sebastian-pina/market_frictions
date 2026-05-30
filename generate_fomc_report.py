"""
Generate all figures and summary data for FOMC presentation report.
Reads fomc_surprise_table.csv and produces 6 publication-ready figures.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv('fomc_surprise_table.csv', parse_dates=['date'])

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

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='No widening (ratio=1)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_ylabel('Pre-FOMC / Baseline Spread Ratio', fontsize=11)
    ax.set_title('Pre-FOMC Spread Widening: FOMC Days vs. Control Days\n'
                 '(Ratio of 1:30–2:00 PM spread to 11 AM–1 PM baseline)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.45)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate FOMC bars with % widening
    for bar, yr in zip(bars1, years):
        pct = (SPREAD_RATIOS[yr]['fomc'] - 1) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'+{pct:.1f}%', ha='center', va='bottom', fontsize=8, color='#C62828')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_spread_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_spread_ratio.png')


# ── Fig 2: Scatter ret_14 vs Net PnL (colored by regime group) ───────────────
def fig_ret_pnl_scatter():
    plot_df = df.dropna(subset=['ret_14', 'net_pnl']).copy()
    plot_df['group'] = plot_df['regime'].map(REGIME_GROUP).fillna('Hold/Cut')

    fig, ax = plt.subplots(figsize=(9, 6))
    group_colors = {'Hiking': '#C62828', 'Hold/Cut': '#1565C0'}

    for group, gdf in plot_df.groupby('group'):
        ax.scatter(gdf['ret_14'], gdf['net_pnl'],
                   color=group_colors[group], alpha=0.85, s=80,
                   label=group, zorder=3)
        # Label notable events
        for _, row in gdf.iterrows():
            if abs(row['net_pnl']) > 200 or abs(row['ret_14']) > 25:
                ax.annotate(str(row['date'].date()),
                            (row['ret_14'], row['net_pnl']),
                            textcoords='offset points', xytext=(5, 5),
                            fontsize=7, color='grey')

    # Fit lines per group
    for group, gdf in plot_df.groupby('group'):
        if len(gdf) >= 3:
            slope, intercept, r, p, _ = stats.linregress(gdf['ret_14'], gdf['net_pnl'])
            x_line = np.linspace(gdf['ret_14'].min(), gdf['ret_14'].max(), 50)
            ax.plot(x_line, intercept + slope * x_line,
                    color=group_colors[group], linestyle='--', alpha=0.5, linewidth=1.5)

    # Shade quadrants
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.7, linestyle=':')

    ax.set_xlabel('2:00 PM 1-min Return (bps)  [neg = market fell on announcement]', fontsize=10)
    ax.set_ylabel('Net Strategy PnL ($)', fontsize=10)
    ax.set_title('FOMC Event: Initial Bar Return vs. Strategy PnL\n'
                 '(Contrarian: fade the 2:00 PM move, hold to 2:15 PM)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_ret_pnl_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_ret_pnl_scatter.png')


# ── Fig 3: FOMC vs Control PnL by year (paired bars) ─────────────────────────
def fig_fomc_vs_control():
    years = list(YEAR_PNL.keys())
    fomc_pnl = [YEAR_PNL[y]['fomc']    for y in years]
    ctrl_pnl = [YEAR_PNL[y]['control'] for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    w = 0.38
    fomc_colors = ['#2E7D32' if v >= 0 else '#C62828' for v in fomc_pnl]
    ctrl_colors = ['#81C784' if v >= 0 else '#EF9A9A' for v in ctrl_pnl]

    ax.bar(x - w/2, fomc_pnl, w, color=fomc_colors, label='FOMC strategy', alpha=0.9)
    ax.bar(x + w/2, ctrl_pnl, w, color=ctrl_colors, label='Control (placebo)', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{y}\n({["Hold","Hike","Tr.","Cut","Post"][i]})'
                        for i, y in enumerate(years)], fontsize=10)
    ax.set_ylabel('Annual PnL ($)', fontsize=11)
    ax.set_title('Strategy PnL: FOMC Days vs. Matched Control Days\n'
                 '(Placebo test — edge should be FOMC-specific)', fontsize=11)

    # Value labels
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (15 if h >= 0 else -40),
                f'${h:+,.0f}', ha='center', va='bottom', fontsize=8)

    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_vs_control.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_vs_control.png')


# ── Fig 4: z-score distribution + classification ─────────────────────────────
def fig_zscore_dist():
    plot_df = df.dropna(subset=['z_score']).copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of z-scores colored by class
    ax = axes[0]
    colors_map = {'small': '#1565C0', 'medium': '#F57F17', 'large': '#C62828'}
    for cls, cdf in plot_df.groupby('signal_class'):
        ax.hist(cdf['z_score'], bins=12, color=colors_map.get(cls, 'grey'),
                alpha=0.7, label=f'{cls} (n={len(cdf)})', edgecolor='white')
    ax.axvline(-1.5, color='#C62828', linestyle='--', linewidth=1.5)
    ax.axvline(+1.5, color='#C62828', linestyle='--', linewidth=1.5, label='|z|=1.5 cutoff')
    ax.axvline(-1.0, color='#F57F17', linestyle=':', linewidth=1.5)
    ax.axvline(+1.0, color='#F57F17', linestyle=':', linewidth=1.5, label='|z|=1.0 cutoff')
    ax.set_xlabel('DGS2 Surprise (z-score)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of FOMC Surprises\n(DGS2 z-score, 2021–2025)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: z-score by year scatter
    ax = axes[1]
    year_colors = {2021:'#1565C0',2022:'#C62828',2023:'#6A1B9A',2024:'#2E7D32',2025:'#E65100'}
    for yr, ydf in plot_df.groupby('year'):
        ax.scatter([yr]*len(ydf), ydf['z_score'],
                   color=year_colors.get(yr,'grey'), alpha=0.8, s=60, zorder=3,
                   label=str(yr))
    ax.axhline(1.5,  color='#C62828', linestyle='--', linewidth=1.2)
    ax.axhline(-1.5, color='#C62828', linestyle='--', linewidth=1.2, label='|z|=1.5')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Year')
    ax.set_ylabel('DGS2 Surprise (z-score)')
    ax.set_title('FOMC Surprise Magnitude by Year\n(negative = dovish surprise)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('OIS Rate-Path Surprise Filter Analysis', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_zscore_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_zscore_analysis.png')


# ── Fig 5: Regime-conditioned cumulative PnL ─────────────────────────────────
def fig_regime_pnl():
    plot_df = df.dropna(subset=['net_pnl']).copy()
    plot_df = plot_df.sort_values('date').reset_index(drop=True)
    plot_df['group'] = plot_df['regime'].map(REGIME_GROUP).fillna('Hold/Cut')

    hc  = plot_df[plot_df['group'] == 'Hold/Cut'].copy().reset_index(drop=True)
    hk  = plot_df[plot_df['group'] == 'Hiking'].copy().reset_index(drop=True)
    all_ = plot_df.copy().reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: cumulative PnL by regime
    ax = axes[0]
    if not all_.empty:
        ax.plot(range(len(all_)), all_['net_pnl'].cumsum(),
                color='grey', linewidth=1.5, linestyle='--', label='All events', alpha=0.7)
    if not hc.empty:
        ax.plot(range(len(hc)), hc['net_pnl'].cumsum(),
                color='#1565C0', linewidth=2.2, label=f'Hold/Cut (n={len(hc)})')
    if not hk.empty:
        ax.plot(range(len(hk)), hk['net_pnl'].cumsum(),
                color='#90CAF9', linewidth=2.2, label=f'Hiking (n={len(hk)})')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_xlabel('Event number')
    ax.set_ylabel('Cumulative Net PnL ($)')
    ax.set_title('Regime-Conditioned Cumulative PnL\n(contrarian FOMC fade strategy)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: bar chart by year
    ax = axes[1]
    yr_pnl = plot_df.groupby('year')['net_pnl'].sum()
    bar_colors = ['#2E7D32' if v >= 0 else '#C62828' for v in yr_pnl.values]
    ax.bar(yr_pnl.index, yr_pnl.values, color=bar_colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    for yr, pnl in yr_pnl.items():
        ax.text(yr, pnl + (20 if pnl >= 0 else -60),
                f'${pnl:+,.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Net PnL ($)')
    ax.set_title('Annual PnL by Year\n(FOMC-day strategy, traded events only)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('FOMC Contrarian Strategy: Regime Analysis', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_regime_pnl.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_regime_pnl.png')


# ── Fig 6: TC cost progression + spread widening trend ───────────────────────
def fig_tc_trend():
    years  = list(AVG_TC.keys())
    tcs    = list(AVG_TC.values())
    fomc_r = [SPREAD_RATIOS[y]['fomc'] for y in years]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    color_tc   = '#1565C0'
    color_sprd = '#1565C0'

    ax1.bar(years, tcs, color='#90CAF9', alpha=0.9, width=0.5, label='Avg TC per trade')
    ax2.plot(years, fomc_r, color='#1565C0', linewidth=2.2, marker='o',
             markersize=7, label='FOMC spread ratio')
    ax2.plot(years, [SPREAD_RATIOS[y]['control'] for y in years],
             color='#90CAF9', linewidth=1.8, marker='s', markersize=5,
             linestyle='--', label='Control spread ratio')
    ax2.axhline(1.0, color='grey', linewidth=0.6, linestyle=':')

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Average TC per Trade ($)', color='#1565C0', fontsize=11)
    ax2.set_ylabel('Pre-FOMC / Baseline Spread Ratio', color='#1565C0', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax2.tick_params(axis='y', labelcolor='#1565C0')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax1.set_title('Transaction Cost Progression and Spread Widening (2021–2025)\n'
                  'Higher spread widening → higher TC → higher gross PnL needed to profit',
                  fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_tc_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_tc_trend.png')


# ── Fig 7: SEP dot-plot median vs OIS/DGS2 market path ───────────────────────
def fig_sep_vs_ois():
    """Compare Fed SEP median end-year projections vs DGS2 market path.

    SEP surprise = SEP_projected_year_end - DGS2_day_before_meeting
    Positive  = Fed projected HIGHER rates than market priced (hawkish surprise)
    Negative  = Fed projected LOWER rates than market priced (dovish surprise)
    """

    # (date, current_ff_rate_midpoint, sep_median_current_or_next_year_end, year_label)
    # December SEP projects *next* calendar year; all others project current year.
    SEP_ROWS = [
        ('2021-03-17', 0.125, 0.125, '2021'),
        ('2021-06-16', 0.125, 0.125, '2021'),
        ('2021-09-22', 0.125, 0.125, '2021'),
        ('2021-12-15', 0.125, 0.875, '2022'),   # Dec → projects 2022
        ('2022-03-16', 0.375, 1.875, '2022'),
        ('2022-06-15', 1.625, 3.375, '2022'),
        ('2022-09-21', 3.125, 4.375, '2022'),
        ('2022-12-14', 4.375, 5.125, '2023'),   # Dec → projects 2023
        ('2023-03-22', 4.875, 5.125, '2023'),
        ('2023-06-14', 5.125, 5.625, '2023'),
        ('2023-09-20', 5.375, 5.625, '2023'),
        ('2023-12-13', 5.375, 4.625, '2024'),   # Dec → projects 2024
        ('2024-03-20', 5.375, 4.625, '2024'),
        ('2024-06-12', 5.375, 5.125, '2024'),
        ('2024-09-18', 4.875, 4.375, '2024'),
        ('2024-12-18', 4.375, 3.875, '2025'),   # Dec → projects 2025
        ('2025-03-19', 4.375, 3.875, '2025'),
        ('2025-06-18', 4.375, 3.875, '2025'),
    ]

    sep_df = pd.DataFrame(SEP_ROWS, columns=['date', 'ff_rate', 'sep_median', 'proj_year'])
    sep_df['date'] = pd.to_datetime(sep_df['date'])

    # Merge with surprise table to get DGS2 at each SEP meeting
    merged = sep_df.merge(
        df[['date', 'dgs2_prev', 'dgs2_fomc', 'net_pnl', 'regime']],
        on='date', how='left'
    )
    merged['surprise_ffr_minus_dgs2'] = merged['sep_median'] - merged['dgs2_prev']

    # ── Panel layout ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: time-series — DGS2 path vs SEP step-function ───────────────────
    ax = axes[0]

    # DGS2 from all FOMC meetings (proxy for market OIS)
    fomc_dgs2 = df.dropna(subset=['dgs2_fomc']).sort_values('date')
    ax.plot(fomc_dgs2['date'], fomc_dgs2['dgs2_fomc'],
            color='#1565C0', linewidth=1.8, marker='o', markersize=4,
            label='DGS2 (market OIS proxy)', zorder=3)

    # SEP median step function
    sep_sorted = merged.sort_values('date').reset_index(drop=True)
    ax.step(sep_sorted['date'], sep_sorted['sep_median'],
            color='#90CAF9', linewidth=2.0, linestyle='--', where='post',
            label='SEP median projection (year-end FF rate)', zorder=4)
    ax.scatter(sep_sorted['date'], sep_sorted['sep_median'],
               color='#90CAF9', s=60, zorder=5)

    # Shade gap
    for i in range(len(sep_sorted) - 1):
        d0, d1 = sep_sorted.loc[i, 'date'], sep_sorted.loc[i+1, 'date']
        s_val   = sep_sorted.loc[i, 'sep_median']
        # Approx DGS2 in this window: use start-of-window DGS2
        dgs2_in_window = fomc_dgs2[(fomc_dgs2['date'] >= d0) & (fomc_dgs2['date'] < d1)]
        if dgs2_in_window.empty:
            continue
        for _, row in dgs2_in_window.iterrows():
            if s_val > row['dgs2_fomc']:
                ax.fill_between([row['date'], row['date']],
                                row['dgs2_fomc'], s_val,
                                alpha=0.15, color='#90CAF9')
            else:
                ax.fill_between([row['date'], row['date']],
                                s_val, row['dgs2_fomc'],
                                alpha=0.15, color='#1565C0')

    # Annotate major surprise events
    big_surprises = merged[merged['surprise_ffr_minus_dgs2'].abs() >= 0.5].copy()
    for _, row in big_surprises.iterrows():
        if pd.notna(row['sep_median']):
            label = f"{row['date'].strftime('%b-%y')}\n({row['surprise_ffr_minus_dgs2']:+.2f}%)"
            ax.annotate(label, (row['date'], row['sep_median']),
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=7, color='grey', arrowprops=dict(arrowstyle='-', color='grey', lw=0.8))

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title('OIS/DGS2 Market Path vs. Fed SEP Median Projection\n'
                 '(Gap = market vs. Fed forward guidance divergence)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: bar chart — SEP surprise (SEP_median - DGS2_prev) ─────────────
    ax = axes[1]
    merged_clean = merged.dropna(subset=['dgs2_prev']).copy()
    merged_clean['surprise_ffr_minus_dgs2'] = (
        merged_clean['sep_median'] - merged_clean['dgs2_prev']
    )
    colors = ['#C62828' if v > 0 else '#1565C0' for v in merged_clean['surprise_ffr_minus_dgs2']]
    ax.bar(range(len(merged_clean)), merged_clean['surprise_ffr_minus_dgs2'],
           color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(merged_clean)))
    ax.set_xticklabels(
        [d.strftime('%b\n%y') for d in merged_clean['date']],
        fontsize=7, rotation=0
    )
    ax.set_ylabel('SEP Median - DGS2 (percentage points)', fontsize=10)
    ax.set_title('SEP Forward Guidance Surprise vs. Market Pricing\n'
                 '(Red = Fed more hawkish than DGS2 priced; Blue = more dovish)', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Add PnL annotation for traded SEP events
    for i, (_, row) in enumerate(merged_clean.iterrows()):
        if pd.notna(row.get('net_pnl')) and row.get('net_pnl') != 0:
            y_off = row['surprise_ffr_minus_dgs2'] + (0.05 if row['surprise_ffr_minus_dgs2'] >= 0 else -0.12)
            ax.text(i, y_off, f"${row['net_pnl']:+.0f}",
                    ha='center', va='bottom', fontsize=6.5, color='#2E7D32' if row['net_pnl'] > 0 else '#B71C1C')

    # Add legend patches
    hawk_patch = mpatches.Patch(color='#C62828', alpha=0.85, label='Hawkish surprise (Fed > mkt)')
    dove_patch  = mpatches.Patch(color='#1565C0', alpha=0.85, label='Dovish surprise (Fed < mkt)')
    ax.legend(handles=[hawk_patch, dove_patch], fontsize=9)

    plt.suptitle('Fed SEP Guidance vs. OIS Market Path: Surprise Decomposition (2021-2025)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_sep_vs_ois.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_sep_vs_ois.png')

    # Print surprise table
    print('\nSEP Surprise Table (SEP_median - DGS2_prev):')
    print(merged_clean[['date', 'ff_rate', 'sep_median', 'dgs2_prev',
                         'surprise_ffr_minus_dgs2', 'net_pnl']].to_string(index=False))


# ── Fig 8: Two-track strategy breakdown ──────────────────────────────────────
def fig_two_track():
    """Show the SEP vs non-SEP event breakdown and two-track portfolio performance."""
    plot_df = df.dropna(subset=['net_pnl','surprise_type']).copy()
    plot_df = plot_df.sort_values('date').reset_index(drop=True)

    # Define tracks
    non_sep    = plot_df[plot_df['surprise_type'] == 'target'].copy()
    sep_small  = plot_df[(plot_df['surprise_type'] == 'SEP') &
                          (plot_df['signal_class'] == 'small')].copy()
    sep_excl   = plot_df[(plot_df['surprise_type'] == 'SEP') &
                          (plot_df['signal_class'].isin(['medium','large']))].copy()
    two_track  = pd.concat([non_sep, sep_small]).sort_values('date').reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Cumulative PnL for each track ───────────────────────────────────
    ax = axes[0]
    all_sorted = plot_df.copy()
    ax.plot(range(len(all_sorted)), all_sorted['net_pnl'].cumsum(),
            color='grey', linewidth=1.5, linestyle='--', alpha=0.6, label='Unfiltered (40 events)')
    ax.plot(range(len(two_track)), two_track['net_pnl'].cumsum(),
            color='#1565C0', linewidth=2.5, label=f'Two-track (29 events): SEP|z|≤1.0 + all non-SEP')
    ax.plot(range(len(non_sep)), non_sep['net_pnl'].cumsum(),
            color='#42A5F5', linewidth=2.0, linestyle='-.', label=f'Non-SEP only (22 events)')
    ax.plot(range(len(sep_excl)), sep_excl['net_pnl'].cumsum(),
            color='#90CAF9', linewidth=1.5, linestyle=':', label=f'Excluded SEP events (11 events)')

    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xlabel('Event count', fontsize=10)
    ax.set_ylabel('Cumulative Net PnL ($)', fontsize=10)
    ax.set_title('Two-Track Cumulative PnL\n(non-SEP always-trade vs SEP z-score filtered)',
                 fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Annotate final values
    for data, label, color in [
        (two_track, f'+${two_track["net_pnl"].sum():,.0f}', '#1565C0'),
        (non_sep,   f'+${non_sep["net_pnl"].sum():,.0f}',   '#42A5F5'),
        (sep_excl,  f'${sep_excl["net_pnl"].sum():,.0f}',   '#90CAF9'),
    ]:
        n = len(data)
        val = data['net_pnl'].cumsum().iloc[-1] if not data.empty else 0
        ax.annotate(label, xy=(n-1, val), xytext=(5, 0), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold')

    # ── Right: Annual PnL decomposition by track ──────────────────────────────
    ax = axes[1]
    years = sorted(plot_df['year'].unique())
    x = np.arange(len(years))
    w = 0.25

    nonsep_yr  = non_sep.groupby('year')['net_pnl'].sum().reindex(years, fill_value=0)
    sepsm_yr   = sep_small.groupby('year')['net_pnl'].sum().reindex(years, fill_value=0)
    sepexcl_yr = sep_excl.groupby('year')['net_pnl'].sum().reindex(years, fill_value=0)

    ax.bar(x - w, nonsep_yr.values,  w, color='#1565C0', alpha=0.85, label='Non-SEP (traded)')
    ax.bar(x,     sepsm_yr.values,   w, color='#42A5F5', alpha=0.85, label='SEP small |z|≤1.0 (traded)')
    ax.bar(x + w, sepexcl_yr.values, w, color='#90CAF9', alpha=0.6,  label='SEP excl. |z|>1.0 (skipped)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Annual Net PnL ($)', fontsize=10)
    ax.set_title('Annual PnL by Track\n(Non-SEP events are the primary PnL driver)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Summary stats annotation
    sharpe_tt = (two_track['net_pnl'].mean() / two_track['net_pnl'].std() *
                 np.sqrt(len(two_track))) if len(two_track) > 1 else 0
    _, p_tt = stats.ttest_1samp(two_track['net_pnl'], 0)
    sharpe_ns = (non_sep['net_pnl'].mean() / non_sep['net_pnl'].std() *
                 np.sqrt(len(non_sep))) if len(non_sep) > 1 else 0
    _, p_ns = stats.ttest_1samp(non_sep['net_pnl'], 0)
    ax.text(0.02, 0.97,
            f'Two-track (29 events): Sharpe={sharpe_tt:.2f}, p={p_tt:.3f}\n'
            f'Non-SEP (22 events):   Sharpe={sharpe_ns:.2f}, p={p_ns:.3f}',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle('Two-Track FOMC Filter: SEP vs. Non-SEP Event Decomposition', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fomc_two_track.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved fomc_two_track.png')


if __name__ == '__main__':
    print('Generating FOMC report figures...')
    fig_spread_ratio()
    fig_ret_pnl_scatter()
    fig_fomc_vs_control()
    fig_zscore_dist()
    fig_regime_pnl()
    fig_tc_trend()
    fig_sep_vs_ois()
    fig_two_track()
    print('\nDone. Saved to figures/')
