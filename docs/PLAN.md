# Buy a Dip? - Research Plan (pre-registered, v2)

## 0. Hypothesis as stated
> 当价格接近 200/120/60/50/20/5 日均线时买入，是一个好的时机，因为价格大概率围绕一个范围上下波动。

### Correction-first reframing
| Problem with the original framing | Consequence | Fix in this repo |
|---|---|---|
| "Price oscillates around the MA" is a **level mean-reversion** claim. Equity returns show weak reversal at 1 week to 1 month and **momentum** at 3-12 months (Jegadeesh 1990; Jegadeesh & Titman 1993; Moskowitz, Ooi & Pedersen 2012) | Dips to short and long MAs should behave differently | Grid by MA length x horizon |
| "Good time to buy" needs a benchmark | NVDA/LLY/TSLA rose on almost any entry day | Two "not timing the market" benchmarks (section 2) |
| Ticker list chosen in hindsight | Survivorship and selection bias inflate any long rule | Same-ticker benchmark removes own drift. The SPY benchmark does **not** (see 2.2) |
| A fixed % band means different things for different tickers | +/-5% is 97% of days for a 5-day MA and about one day's move for TSLA | **ATR-only bands**: +/-0.5 and +/-1 ATR14 |
| A close-based signal needs the close before you can trade | Next-day fill adds an overnight gap | **Open-based signal, fill at the same open** |

## 1. Event definition (all inputs known at the open of day t)
- `MA_{t-1}`, `ATR_{t-1}` are computed from closes through yesterday (split-adjusted, not dividend-adjusted: the chart traders see).
- `dist_t = open_t / MA_{t-1} - 1`, `width_t = k * ATR_{t-1} / MA_{t-1}`, with `k` in {0.5, 1.0}.
- **from_above (the dip):** `|dist_t| <= width_t` and yesterday's close was above the zone.
- **from_below (control):** `|dist_t| <= width_t` and yesterday's close was below the zone.
- 5-bar cooldown per ticker/spec.
- Grid: 6 MAs x 2 bands x 2 directions x 6 horizons = **144 pooled cells** per benchmark, x 12 tickers = 1,728 per-ticker cells.

## 2. Outcome and benchmarks
- `R_h = adj_open[t+h] / adj_open[t] - 1`: buy at the signal open, sell at the open h trading days later, total-return prices.
- Horizons: 5 / 10 / 20 (short), 60 / 120 (medium), 250 (long).

### 2.1 `own` (primary): "would buying any day have done as well?"
`B_own[t]` = mean `R_h` over **every day** in a centered 252-bar window of the same ticker.
`excess_own = R_h - B_own`. This isolates timing. The stock's trend over that year cancels out.

### 2.2 `spy` (secondary): "did it beat just owning SPY?"
`B_spy[t]` = SPY's `R_h` over the identical dates.
`excess_spy = R_h - B_spy` contains the stock's alpha/beta **and** the hindsight selection of winners. For NVDA, any entry day beats SPY. Read it only next to `own`. SPY itself is excluded.

## 3. Inference
| Issue | Method |
|---|---|
| Overlapping windows; all names dipping together (2000-02, 2008, 2020, 2022) | Circular time-shift placebo over all admissible shifts (FFT). It preserves event count, clustering and overlap |
| 144 x 2 pooled + 1,728 x 2 per-ticker tests | Benjamini-Hochberg FDR within each benchmark, significance at q < 0.10 |
| Regime | SPY above/below its 200-day SMA at yesterday's close. Periods: 2000-02, 2003-07, 2008-09, 2010-19, 2020-26 |

## 3b. Regimes (v3)
| Label | Known when? | Definition | Use |
|---|---|---|---|
| `spy_trend` | at the open | SPY close_{t-1} above / below its 200-day SMA | filter |
| `spy_dd` | at the open | SPY drawdown from its running high: <10%, 10-20%, >=20% | filter |
| `stock_trend` | at the open | ticker close_{t-1} above / below its own 200-day SMA | filter |
| `phase` | hindsight | for each SPY close drawdown >= 20%: decline = peak->trough, recovery = trough->old peak, bull = rest | descriptive |
| `episode` | hindsight | the same windows, named: dotcom, gfc, bear_2018, covid, bear_2022 | descriptive (small n) |

- Regime placebo: excess is masked to days in the regime, so the null is "random days **in the same regime**".
- Filtered strategies: random entries are drawn only from days when the filter is on, so the filter gets no credit for regime timing alone.
- H6' : a dip edge that exists only in an ex-post phase is **not tradable**.

## 4. Strategy test
- Buy at the dip open, sell at the open h days later. A new dip extends the hold. 5 bps per side, cash at 0% when flat.
- Compared with: buy & hold, and **random entries with the same number of trades and the same hold** (100 draws), reported as a percentile.
- Portfolio: equal weight across the tickers listed that day.

## 5. Sub-hypotheses and pass criteria
| ID | Statement | Pass |
|---|---|---|
| H1 | A dip beats `own` at horizon h | excess_own > 0, q < 0.10, and larger than the 10 bps round trip |
| H4 | Sign pattern by MA length is consistent across both ATR bands | same sign in both bands |
| H5 | from_below underperforms ("resistance") | excess_own < 0, q < 0.10 |
| H6 | Effect is regime-robust | same sign in both SPY regimes and >= 4 of 5 periods |
| H7 | Strategy beats random entry on Sharpe net of costs | >= 60% of ticker-specs above the random median, and a top-decile share well above 10% |

## 6. Known limitations
- 12 hand-picked survivors. A confirmatory test needs a point-in-time universe that includes delisted names.
- Later listings (GOOGL 2004, TSLA 2010) only cover part of the sample.
- The open is one snapshot. Intraday touches that never print at the open are not captured.
- At 250 days, even 27 years give only about 25 independent windows per ticker.
