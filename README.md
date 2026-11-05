# Buy a Dip?

**Question:** If a stock pulls back to (or near) its 5 / 20 / 50 / 60 / 120 / 200-day moving average, is that a better-than-random time to buy?

**Universe:** SPY, QQQ, TSLA, GOOGL, NVDA, AMD, ASML, MSFT, ISRG, URI, WM, LLY. **Sample:** 2020-01-01 to the latest close (MAs warmed up on data from 2015).

> Not investment advice. This is a research harness. The ticker list was chosen in hindsight, so every result here is illustrative, not confirmatory.

## Results (run of 2026-11-05; sample 2020-01-02 to 2026-11-04)

Full tables and heatmaps are in [`results/REPORT.md`](results/REPORT.md). GitHub Actions regenerates them on every code push and every Saturday, so the numbers below can drift.

**Verdict: no general dip-buying edge.** Buying a dip to an MA did no better than buying a random day of the same stock in the same year. Only 1 of 216 dip cells is significantly positive after FDR. 3 are significantly negative.

**Median excess return, dip from above** (median across the 6 band definitions, in %):

| MA \ hold (days) | 5 | 10 | 20 | 60 | 120 | 250 |
|---|---|---|---|---|---|---|
| 5 | -0.1 | +0.1 | -0.1 | -1.0 | **-2.0** | -0.7 |
| 20 | 0.0 | -0.3 | -0.6 | -1.1 | -1.6 | -1.2 |
| 50 | 0.0 | -0.4 | -1.0 | -1.3 | -0.7 | -0.6 |
| 60 | -0.1 | -0.4 | -0.8 | -0.7 | 0.0 | +0.3 |
| 120 | +0.1 | -0.2 | -0.4 | +1.0 | +3.6 | +6.0 |
| 200 | +0.2 | -0.1 | +0.8 | +3.5 | +6.0 | +3.6 |

| Finding | Evidence |
|---|---|
| **Short and medium MAs (5–60 day):** dips do *worse* than a random entry over 20–120 days | 5-day MA at 60 and 120 days: -2.2% to -2.5% excess, significant (q = 0.09). Only 19% of dip cells are positive at 10–20 days |
| **Long MAs (120 / 200 day):** about +3.5% to +6% at 60–250 days, but fragile | Only one cell survives FDR (120-day MA, ±0.5 ATR, 250 days: +8.2%, q = 0.09). In 10% of those trades the price first fell another 44% or more (worst-case drawdown during the hold). The gain comes from 2020–21; in 2022 it was -5% to -9% |
| **"Resistance" is not supported** | Rallying up to the MA from below did as well as or better than the dip at 60–250 days. That is momentum, not a ceiling |
| **A fixed 5% band is not a signal on short MAs** | The close is within 5% of the 5-day MA on 96% of days, and of the 20-day MA on 70% |
| **Tradable version loses to buy & hold** | Equal-weight portfolio, 5–20 day holds: Sharpe 0.57–0.65 and CAGR 3–7%, against 1.26 and 35% for buy & hold. It caught up only at 250-day holds, when it was invested 95% of the time |
| **No timing skill vs random entries** | Compared with random entries using the same number of trades and the same hold, the dip strategy beat the median random entry in 35–55% of ticker × spec cases (chance = 50%). It landed in the top 10% in 3–12% (chance = 10%) |

**How to read the long-MA result.** Over 2020–2026, big drawdowns in these 12 names (Mar-2020, late 2022, Apr-2025) were followed by recoveries. That is one macro path, applied to stocks picked *after* they won, and the effect reversed in 2022. Re-test it on longer history and a universe that includes delisted names before relying on it (see `docs/PLAN.md` §6).

## Design in one table

| Element | Choice | Why |
|---|---|---|
| Signal price | Split-adjusted close, no dividend adjustment | The MA is the one traders see on a chart |
| "Near the MA" | Fixed ±1 / 2 / 3 / 5 % band **and** ±0.5 / 1 ATR band | An exact touch is fragile. A fixed % band means something different on WM than on TSLA, and on the 5-day MA than on the 200-day |
| Dip | Inside the band today, **above** the band yesterday (5-bar cooldown) | Separates a pullback from just sitting near the line |
| Control | Inside the band today, **below** the band yesterday | The "resistance" side |
| Entry / exit | Next open → close h days later, total-return prices | No same-bar fill, no look-ahead |
| Horizons | 5, 10, 20 (short) · 60, 120 (medium) · 250 (long) days | A next-day result alone is noise |
| Benchmark | **Excess** = return minus the mean return of a random entry day in the same ticker and calendar year | Removes "NVDA went up anyway" |
| Inference | Circular time-shift placebo (FFT, all shifts) + Benjamini-Hochberg FDR over every cell | Accounts for overlapping windows, for days when all 12 names dip together, and for 432 / 5,184 tests |
| Strategy | Buy the dip, hold h days, 5 bps per side, versus buy & hold **and** versus random entries with the same number of trades and the same h | Separates timing skill from simply being out of the market more |

Full pre-registered plan, sub-hypotheses H1–H7 and pass criteria: [`docs/PLAN.md`](docs/PLAN.md).

## Prior evidence (summary of the literature review)

| Source | Finding | Implication |
|---|---|---|
| Brock, Lakonishok & LeBaron (1992); Sullivan, Timmermann & White (1999); Bajgrowicz & Scaillet (2012) | MA rules worked on the DJIA before 1987, then decayed. The gains vanish after data-snooping control and small transaction costs | Expect a small edge or none |
| Han, Yang & Zhou (2013); Faber (2007) | The MA edge that survives is **trend / risk control**: be long above the MA. It cuts drawdowns more than it raises returns | Buying a touch from above happens exactly when that trend signal is weakest |
| Jegadeesh (1990); Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012) | Reversal over 1 week to 1 month, momentum over 3–12 months | Short-MA dips: maybe a small bounce. Long-MA dips: momentum argues against a bounce |
| Marwood (S&P 500 stocks, 2000–2017); The Long Trend (50-day touch, 11.6k events) | 200-day pullback ≈ benchmark. 50-day touch: about +0.3% excess over 5 days, fading to negative by 60 days | Edge is small, short-lived and depends on how the touch is defined |
| Osler (2000, 2003); Kavajecz & Odders-White (2004) | Support/resistance "works" mainly because orders cluster at those levels, and price often accelerates once a level breaks | Expect two-sided outcomes: watch MAE, not only the mean |

## Run it

```bash
pip install -e ".[dev]"
pytest -q                       # no-look-ahead, alignment, placebo, accounting tests
buyadip fetch                   # Yahoo -> data/*.parquet (retries, stooq fallback)
buyadip run                     # full grid -> results/
buyadip run --start 2022-01-01 --tickers SPY QQQ --ma-type ema   # variants
```

## Layout

```
src/buyadip/
  config.py     pre-registered grid (MAs, bands, horizons, costs)
  data.py       download, validation (split-jump guard, OHLC sanity), wide panel
  signals.py    SMA/EMA, ATR, zone entry detection, cooldown
  events.py     forward returns, MAE/MFE, ticker-year baseline, excess
  stats.py      FFT circular-shift placebo, BH-FDR
  backtest.py   strategy vs buy & hold vs equal-trade random entries
  analysis.py   orchestration -> tables
  report.py     CSVs, heatmaps, REPORT.md
tests/          pytest
.github/workflows/run.yml   fetch + simulate + commit results
```

## Known limitations

- 2020–2026 is a single macro path: crash, V-shaped recovery, 2022 bear market, AI rally. At 120 and 250 days there are only about 6–12 independent windows per ticker.
- Survivor bias. A confirmatory test needs a point-in-time S&P 500 universe that includes delisted names.
- Signals use daily closes only. There is no intraday-touch or limit-order fill model.
