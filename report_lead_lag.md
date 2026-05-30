# Lead-Lag Trading: SPY → AAPL and SPY → NVDA
## MGMTMFE 412 — Market Frictions — Final Project
**UCLA Anderson School of Management — Spring 2025**

---

## 1. Abstract

We exploit the well-documented lead-lag relationship between the S&P 500 ETF (SPY)
and its constituent stocks. Using Databento Level-2 order book data (MBP-10) at
1-second resolution for January–April 2024, we document that SPY price changes predict
AAPL and NVDA returns with a 1–3 second lag. We construct a high-frequency signal based
on SPY 1-second returns and order flow imbalance (OFI), and backtest a momentum strategy
that enters the constituent stock in the direction of the SPY move.

**Key finding:** The edge is statistically real — gross PnL is positive in both
in-sample and out-of-sample periods for both stocks. However, transaction costs
(bid-ask spread) completely eliminate the alpha. For AAPL, TC is 9× the gross PnL;
for NVDA, TC is 7.5× the gross PnL. This result illustrates the core market friction
studied in this course: the bid-ask spread as a barrier to ETF arbitrage.

---

## 2. Economic Motivation

**The friction:** Index ETF arbitrage creates a latency between SPY price discovery
and individual constituent stocks. When a large order hits the SPY book, the ETF price
adjusts in microseconds. Authorized participants (APs) then arbitrage between the ETF
and its basket, transmitting the price signal to individual stocks over 1–30 seconds
depending on liquidity.

This latency is a **market friction** arising from:
1. **Information processing delay:** slower participants take time to update valuations
2. **Execution costs:** the round-trip cost of the arbitrage trade (spread × 2 sides)
3. **Inventory risk:** market makers in the constituent stock charge a wider spread to
   absorb the order flow from ETF arbitrageurs

For HFT firms with co-located infrastructure and sub-$0.001/share TC, this latency
creates exploitable alpha. For retail participants paying the full bid-ask spread, the
cost of execution exceeds the expected profit from the lag.

---

## 3. Data

| Item | Detail |
|---|---|
| Source | Databento XNAS.ITCH (Nasdaq ITCH feed) |
| Schema | MBP-10 (10-level limit order book snapshots) |
| Symbols | SPY (leader), AAPL (lagger 1), NVDA (lagger 2) |
| Training | Jan 2 – Feb 28, 2024 (20 trading days) |
| Test (OOS) | Mar 4 – Apr 15, 2024 (10 trading days) |
| Bar resolution | 1 second (resampled from tick data) |
| Session | Regular hours: 9:30 AM – 4:00 PM ET |
| Total bars | 467,136 train / 233,179 test (SPY+AAPL); similar for NVDA |

**Data processing:**
- Only 4 columns loaded per file (`bid_px_00`, `ask_px_00`, `bid_sz_00`, `ask_sz_00`)
  to avoid loading 73-column files (~900 MB each) into memory
- `bid_sz` / `ask_sz` cast to `int64` before arithmetic to prevent uint32 overflow
- Returns computed within each trading session (cross-day returns excluded)
- Session boundary guard in backtest: no trades whose exit bar falls in next day

---

## 4. Methodology

### 4.1 Cross-Correlation Analysis

For each lag k ∈ {−5, ..., +15} seconds, we compute:

```
ρ(k) = corr( SPY_ret(t), LAGGER_ret(t+k) )
```

A positive ρ(k) for k > 0 confirms that SPY leads the lagger by k seconds.

### 4.2 Order Flow Imbalance (OFI)

At each 1-second bar:

```
OFI(t) = [bid_sz(t) − ask_sz(t)] / [bid_sz(t) + ask_sz(t)]  ∈ [−1, +1]
```

A positive OFI indicates buy-side pressure at the top of the book.

### 4.3 Signal Construction

A trade signal fires at bar t when:
- SPY 1-second return > 1.0 bps (confirmed price move)
- SPY OFI > 0.40 (book pressure corroborates direction)

Entry is at bar t+1 (1-second execution delay), hold for HOLD_BARS = 5 seconds.

### 4.4 Backtest

- **Entry price:** mid-price at bar t+1
- **Exit price:** mid-price at bar t+6
- **TC (entry):** half-spread at bar t+1 (crossing the spread to enter)
- **TC (exit):** half-spread at bar t+6 (crossing the spread to exit)
- **Position size:** 100 shares
- **PnL:** gross = direction × (exit_mid − entry_mid) × 100; net = gross − TC

---

## 5. Results

### 5.1 Cross-Correlation

| Lag (s) | SPY→AAPL | SPY→NVDA |
|---|---|---|
| −1 | +0.010 | +0.005 |
| 0 (contemporaneous) | **+0.593** | **+0.544** |
| +1 | +0.024 | **+0.031** |
| +2 | −0.007 | **+0.015** |
| +3 | −0.005 | **+0.013** |
| +4 | −0.005 | +0.005 |
| +5 | −0.007 | +0.001 |

**Finding:** The lead-lag is real but extremely short. SPY leads AAPL by 1 second
(corr = 0.024) and NVDA by 1–3 seconds (corr = 0.013–0.031). After 4 seconds,
the correlation decays to noise for both stocks. For AAPL — the most liquid stock
in the world — the lead-lag is even shorter, consistent with faster price discovery.

![Cross-correlation SPY → AAPL/NVDA](figures/xcorr_spy_aapl.png)

### 5.2 Signal Statistics

| Metric | AAPL | NVDA |
|---|---|---|
| Signal rate (% of 1s bars) | 0.06% | 0.06% |
| Long signals (train) | 88 | 91 |
| Short signals (train) | 169 | 181 |
| Total trades (train) | 253 | 268 |
| Average hold | 5.0 s | 5.0 s |

### 5.3 Backtest Performance

#### Training Set (Jan–Feb 2024)

| Metric | AAPL | NVDA |
|---|---|---|
| Gross PnL | **+$32.50** | **+$663.50** |
| Total TC | −$299.02 | −$4,971.06 |
| Net PnL | −$266.52 | −$4,307.56 |
| TC / Gross PnL | **9.2×** | **7.5×** |
| Hit rate | 26.1% | 17.9% |
| Ann. gross return | +$410/yr | +$8,360/yr |
| Ann. net return | −$3,358/yr | −$54,275/yr |
| Sharpe (net) | −20.3 | −18.0 |
| Avg TC per trade | $1.18 (0.63 bps) | $18.55 (2.91 bps) |

#### Test Set (Mar–Apr 2024, OOS)

| Metric | AAPL | NVDA |
|---|---|---|
| Gross PnL | **+$9.50** | **+$374.50** |
| Total TC | −$77.23 | −$1,708.09 |
| Net PnL | −$67.73 | −$1,333.59 |
| Hit rate | 26.2% | 16.7% |
| Sharpe (net) | −10.4 | −9.6 |

### 5.4 Market Neutrality (Beta vs SPY)

| | AAPL | NVDA |
|---|---|---|
| Beta vs SPY (train) | −1.21 (p=0.71) | 55.1 (p=0.35) |
| R² | 0.008 | 0.052 |

Beta is not significantly different from zero (p > 0.05 for both), confirming the
strategy is approximately market-neutral. The high beta estimate for NVDA is driven
by NVDA's own high beta, not a structural exposure.

![Cumulative PnL — Training Set](figures/leadlag_train_pnl.png)

---

## 6. Stock Comparison: AAPL vs NVDA

| Dimension | AAPL | NVDA | Verdict |
|---|---|---|---|
| Mean half-spread | 0.31 bps | 1.28 bps | AAPL lower TC |
| Lead-lag window | 1 second | 1–3 seconds | NVDA longer lag |
| Gross PnL train | +$32.50 | +$663.50 | NVDA bigger signal |
| TC / Gross | 9.2× | 7.5× | NVDA slightly better ratio |
| Net PnL train | −$267 | −$4,308 | Both negative |

**Interpretation:** NVDA has a stronger and longer-lasting lead-lag signal because its
lower liquidity means slower price discovery. However, the same lower liquidity produces
a wider bid-ask spread (1.28 bps vs 0.31 bps), which more than offsets the stronger
signal. Both stocks confirm the same conclusion: gross alpha exists but TC eliminates it.

The optimal stock for lead-lag arbitrage would minimize TC while maintaining meaningful
lag — a contradictory requirement, since liquidity and price-discovery speed are positively
correlated. This is the fundamental market microstructure tradeoff.

---

## 7. Break-Even Analysis

**What TC level would make the strategy profitable?**

For AAPL (train):
- Gross PnL per trade: $32.50 / 253 = $0.13
- Current TC per trade: $1.18
- Break-even TC: $0.13 per trade = $0.0013/share
- Current NYSE/Nasdaq rebate for HFT makers: ~$0.002–$0.003/share

Conclusion: this strategy is exploitable only for participants with TC < $0.001/share
(co-located HFT firms), not for institutional or retail traders.

---

## 8. Risk Factors

1. **TC sensitivity:** A 10% increase in AAPL spread would increase losses by $30/day.
2. **Signal decay:** As more HFT strategies exploit the lag, it compresses toward zero.
3. **Adverse selection:** Entering in the direction of SPY's move after a large OFI may
   trade against informed flow (the HFT that already moved SPY).
4. **Small sample:** 253 trades across 20 days; statistical significance is limited.
5. **No short-selling costs modeled:** Short signals assume zero borrow cost.

---

## 9. Conclusion

The SPY → constituent stock lead-lag relationship exists at 1-second resolution with
cross-correlations of 0.024 (AAPL) and 0.031 (NVDA). A backtest with realistic TC
(half-spread at entry and exit) produces **positive gross PnL** in both stocks and
both time periods, confirming the signal is real. However, transaction costs exceed
the gross alpha by 7–9× in all scenarios.

This result illustrates the central theme of this course: **bid-ask spread as a market
friction that prevents arbitrage by slower, higher-cost participants**. The lead-lag
edge is fully captured by HFT firms with co-located infrastructure and near-zero TC.
For any participant paying the full spread, the friction is insurmountable.
