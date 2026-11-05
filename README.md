# Buy a Dip?

**Question:** If a stock pulls back to (or near) its 5 / 20 / 50 / 60 / 120 / 200-day moving average, is that a better-than-random time to buy?

**Universe:** SPY, QQQ, TSLA, GOOGL, NVDA, AMD, ASML, MSFT, ISRG, URI, WM, LLY. **Sample:** 2020-01-01 to the latest close (MAs warmed up on data from 2015).

> Not investment advice. This is a research harness. The ticker list was chosen in hindsight, so every result here is illustrative, not confirmatory.

## Results

See [`results/REPORT.md`](results/REPORT.md). GitHub Actions regenerates it on every push to `main` and every Saturday.

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
