# Buy a Dip?

**Question:** If a stock pulls back to (or near) its 5 / 20 / 50 / 60 / 120 / 200-day moving average, is that a better-than-random time to buy?

**Universe:** SPY, QQQ, TSLA, GOOGL, NVDA, AMD, ASML, MSFT, ISRG, URI, WM, LLY. **Sample:** 2000-01-01 to 2026-10-31 (6,748 bars). GOOGL joins in 2004, TSLA in 2010 and ISRG in mid-2000. MAs are warmed up on data from 1998.

> Not investment advice. The ticker list was chosen in hindsight, so results are illustrative, not confirmatory.

## Design

| Element | Choice | Why |
|---|---|---|
| "Near the MA" | \|open_t - MA_{t-1}\| <= k x ATR14_{t-1}, with k = 0.5 or 1.0 | ATR = the stock's normal daily range, so "near" means the same for WM and TSLA. Fixed % bands were dropped |
| Dip | Today's open is inside the zone and yesterday's close was above it | A pullback, not just sitting near the line |
| Control | Today's open is inside the zone and yesterday's close was below it | The "resistance" side |
| Timing | Signal and fill both at **today's open**. MA and ATR use closes through yesterday | Executable before the bell, with no look-ahead and no overnight gap after the signal |
| Exit | Open h trading days later. h = 5, 10, 20 / 60, 120 / 250 | Short / medium / long holds |
| Benchmark `own` (primary) | Mean h-day return of buying the **same ticker on any day** within ±126 bars | "Not timing the market." Removes the stock's own trend |
| Benchmark `spy` | SPY over the identical dates | Beat the market? It also picks up the hindsight selection of winners |
| Inference | Circular time-shift placebo + Benjamini-Hochberg FDR | Handles overlapping holds, days when all names dip together, and the 144 x 2 tests |
| Strategy | Buy at the dip open and hold h days, 5 bps per side. Compared with buy & hold **and** with random entries using the same number of trades and the same hold | Separates timing skill from simply holding less |

Plan, sub-hypotheses and pass criteria: [`docs/PLAN.md`](docs/PLAN.md). Full tables and heatmaps: [`results/REPORT.md`](results/REPORT.md).

## Results (v2 run, 2000-01 to 2026-10, 12 tickers pooled)

**Excess vs `own` for a dip from above (±1 ATR, %, bold = q < 0.10):**

| MA \ hold | 5d | 10d | 20d | 60d | 120d | 250d |
|---|---|---|---|---|---|---|
| 5 | **-0.4** | -0.2 | -0.2 | **-1.7** | **-3.1** | **-2.9** |
| 20 | -0.1 | -0.3 | **-0.7** | -0.9 | **-1.8** | -1.0 |
| 50 | -0.1 | -0.1 | **-0.9** | -1.1 | -0.9 | -1.3 |
| 60 | -0.3 | **-0.6** | **-0.9** | -0.9 | -0.4 | +0.3 |
| 120 | +0.1 | +0.1 | -0.1 | +0.9 | **+2.2** | +1.7 |
| 200 | +0.2 | +0.2 | +0.2 | **+2.2** | **+3.4** | +2.6 |

| Finding | Evidence |
|---|---|
| **Dips to the 5–60 day MAs are worse than buying any day** | 10 of 144 dip cells are significantly negative; all are on the 5–60 day MAs. Example: 5-day MA, 120-day hold, -3.1% |
| **Dips to the 120/200-day MA are positive at 60–120 day holds, but only in some periods** | 5 cells are significant (+2% to +3.4%), and the sign holds whether SPY was above or below its 200-day MA. By period: 2000-02 -5% to -9%, 2003-07 +2% to +5%, 2008-09 -4% to +2%, 2010-19 about 0 to +3%, 2020-26 +9% to +13%. That passes only 2–4 of the 5 periods, short of the 4-of-5 rule set in advance |
| **It depends on which ticker** | 200-day MA, 120-day hold: NVDA +17.7%, AMD +12.3%, URI +6.9%, but TSLA -11.8% and LLY -1.2%. In 10% of these trades the price first fell another 32–38% before the hold ended |
| **Rallying up to the MA from below beats the dip** | 27 cells are significantly positive, on every MA length, at 60–250 days (e.g. 20-day MA, 120-day hold: +3.0%; 120-day MA, 250-day hold: +4.5%). The MA acts as a breakout point (momentum), not as resistance |
| **The SPY benchmark says nothing here** | Every cell is +6% to +20% vs SPY at 120–250 days, and none is significant. These 12 names beat SPY from *any* entry day, which is the hindsight selection |
| **Tradable version: no timing skill** | The dip strategy never beat buy & hold on Sharpe (best 0.90 vs 0.92). Against random entries with the same trades and hold, it beat the median 40–51% of the time (chance = 50%) and landed in the top 10% 5–10% of the time (chance = 10%) |
| **Lower drawdown only comes from holding less** | 200-day MA, 60-day hold: max drawdown -26% vs -62% for buy & hold, but CAGR 7.2% vs 23.2% at 32% exposure. Random entries with the same exposure do as well |

**Bottom line.** The idea that "price oscillates around the MA" fails for short MAs: dips there keep falling, which is short-term momentum. The long-MA dip effect exists in the pooled data, but it comes from a few names (NVDA, AMD, URI) and from 2020–26, and it reversed in 2000–02. As a timing rule it adds nothing over random entries. The more robust pattern in this data is the opposite of the hypothesis: buying when price rises up to its MA from below.

## Prior evidence (literature review)

| Source | Finding |
|---|---|
| Brock, Lakonishok & LeBaron (1992); Sullivan, Timmermann & White (1999); Bajgrowicz & Scaillet (2012) | MA rules worked on the DJIA before 1987, then decayed, and the gain disappears after costs and data-snooping control |
| Han, Yang & Zhou (2013); Faber (2007) | The MA edge that survives is trend / risk control (stay long above the MA), not buying the dip |
| Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012) | Momentum at 3–12 months, consistent with "rally from below" doing better than "dip from above" here |
| Marwood; The Long Trend (practitioner backtests) | 200-day pullbacks roughly match the benchmark; 50-day touches show a small 5-day edge that fades |

## Run it

```bash
pip install -e ".[dev]"
pytest -q                       # no-look-ahead, alignment, placebo, accounting tests
buyadip fetch                   # Yahoo -> data/*.parquet (retries, stooq fallback)
buyadip run                     # full grid, 2000-01-01..2026-10-31 -> results/
buyadip run --start 2010-01-01 --end 2019-12-31 --ma-type ema   # variants
```

## Layout

```
src/buyadip/
  config.py     pre-registered grid (MAs, bands, horizons, costs)
  data.py       download, validation (split-jump guard, OHLC sanity), wide panel
  signals.py    SMA/EMA, ATR, open-in-zone detection, cooldown
  events.py     open-to-open returns, MAE/MFE, own + SPY benchmarks
  stats.py      FFT circular-shift placebo, BH-FDR
  backtest.py   strategy vs buy & hold vs equal-trade random entries
  analysis.py   orchestration -> tables
  report.py     CSVs, heatmaps, REPORT.md
tests/          pytest
.github/workflows/run.yml   fetch + simulate + commit results (on push / manual run)
```

## Known limitations

- Survivor bias. A confirmatory test needs a point-in-time S&P 500 universe that includes delisted names.
- The open is a single snapshot. Intraday touches that never print at the open are not captured.
- At 250-day holds there are only about 25 independent windows per ticker, even over 27 years.
