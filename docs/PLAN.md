# Buy a Dip? - Research Plan (pre-registered)

## 0. Hypothesis as stated
> 当价格触及（或接近）200/120/60/50/20/5 日均线时买入，是一个好的时机，因为价格大概率围绕一个范围上下波动。

### Correction-first reframing
| Problem with the original framing | Consequence | Fix in this repo |
|---|---|---|
| "Price oscillates around the MA" is a **level mean-reversion** claim; equity returns show weak reversal at 1w-1m, **momentum** at 3-12m (Jegadeesh 1990; Jegadeesh & Titman 1993; Moskowitz, Ooi & Pedersen 2012) | Short-MA and long-MA touches should behave differently; pooling them hides the sign flip | Grid by MA length x horizon; report sign pattern (H4) |
| "Good time to buy" has no benchmark | NVDA/LLY/TSLA go up on almost any entry day 2020+ | **Excess vs random entry in the same ticker-year** |
| Ticker list chosen in hindsight | Survivorship + selection bias inflate any long rule | Excess metric removes own drift; result is illustrative, not confirmatory (H8 = future work on point-in-time S&P 500) |
| 1-day results are noise | Next-day edge is dominated by bid/ask and gaps | Horizons 5/10/20 (short), 60/120 (medium), 250 (long) |
| An exact "touch" is fragile | Depends on intraday prints, rarely exact | Proximity bands: fixed +/-1/2/3/5% and volatility-scaled +/-0.5/1 ATR |
| A fixed 5% band is not comparable across MAs/tickers | SPY is within 5% of its 5-day MA ~97% of days -> "signal" = always-on | Report `% of days in zone` per definition; ATR bands as the scale-free alternative |

## 1. Event definition
- `dist_t = close_t / MA_t - 1` on split-adjusted (not dividend-adjusted) closes = what a chart shows.
- Zone: `|dist_t| <= w_t` with `w_t = band%` or `k * ATR14_t / MA_t`.
- **from_above (the hypothesis):** inside the zone today, above the zone yesterday.
- **from_below (control):** inside the zone today, below the zone yesterday ("rally into resistance").
- De-clustering: 5-bar cooldown per ticker/spec.
- Grid: 6 MAs x 6 bands x 2 directions x 6 horizons = **432 pooled cells**, x12 tickers = 5,184 per-ticker cells.

## 2. Outcome
- Entry next open `t+1`, exit close `t+h`, total-return prices.
- `excess = R_h - mean(R_h over all days of same ticker & calendar year)`.
- Path risk: MAE (worst low) / MFE (best high) inside the holding window.

## 3. Inference
| Issue | Method |
|---|---|
| Overlapping h-day windows, cross-sectional clustering (all 12 names dip in Mar-2020/2022) | **Circular time-shift placebo**: shift the full event calendar by s bars (same s for all tickers), all admissible s via FFT. Null preserves event count, clustering and overlap. |
| 432 / 5,184 tests | Benjamini-Hochberg FDR; significance marker at q < 0.10 |
| Regime | Split by SPY above/below its 200-day SMA; year-by-year table |

## 4. Strategy test (H7)
- Long next open after a from_above event, hold h days, extend on new events, 5 bps per side, cash at 0% when flat.
- Benchmarks: buy & hold; **random entries with identical #trades and h** (100 draws) -> percentile.
- Equal-weight 12-sleeve portfolio vs equal-weight buy & hold.

## 5. Sub-hypotheses and pass criteria
| ID | H1 statement | Pass |
|---|---|---|
| H1 | from_above beats random entry at horizon h | mean excess > 0 with BH q < 0.10 and > round-trip cost |
| H4 | Sign pattern by MA length: short MA ~0/+ at 5-10d, long MA ~0 or + at 60-250d (momentum) | pattern consistent across bands |
| H5 | from_below underperforms ("resistance") | mean excess < 0, q < 0.10 (prior: rejected) |
| H6 | Effect is regime-dependent | sign holds in both SPY regimes and >= 4 of 7 years |
| H7 | Strategy beats random entry and B&H on Sharpe net of costs | >= 60% of ticker-specs above random median and top-decile share >> 10% |

## 6. Known limitations (not fixed here)
- 2020-2026 is one macro path (crash, V, 2022 bear, AI rally). Re-run with `buyadip fetch --data-start 2005-01-01 && buyadip run --start 2010-01-01` for robustness.
- 12 hand-picked survivors; a point-in-time universe with delisted names is required for a confirmatory test.
- No intraday-touch variant (Low_t <= MA) - close-band only, entry at next open (no limit-fill illusion).
- Long horizons (120/250d) have only ~6-12 non-overlapping windows per ticker in this sample; treat as descriptive.
