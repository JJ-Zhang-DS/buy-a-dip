"""Tradable version of the signal (hypothesis H7).

Rule per ticker: on an event at close t, buy at open t+1, hold h trading days, sell at the
close of t+h. A new event while already long extends the exit to (new t)+h (no pyramiding).
Flat = cash at 0% (deliberately conservative for the strategy; T-bills would help it a bit).

Benchmarks:
  * buy & hold of the same ticker over the same window
  * random-entry strategy with the SAME number of entries and the SAME holding period.
    This separates "timing skill" from "being invested less of the time".
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def positions(event_rows: np.ndarray, h: int, T: int) -> np.ndarray:
    """Bool (T,) in-market mask from event row indices."""
    d = np.zeros(T + 2, dtype=np.int64)
    start = event_rows + 1
    stop = np.minimum(event_rows + h + 1, T)  # exclusive
    keep = start < T
    np.add.at(d, start[keep], 1)
    np.add.at(d, stop[keep], -1)
    return np.cumsum(d)[:T] > 0


def strategy_returns(pos: np.ndarray, o: np.ndarray, c: np.ndarray, cost: float) -> np.ndarray:
    """Daily returns. o, c are total-return adjusted open/close (T,)."""
    prev_pos = np.r_[False, pos[:-1]]
    next_pos = np.r_[pos[1:], False]
    prev_c = np.r_[np.nan, c[:-1]]
    r = np.zeros_like(c)
    entry = pos & ~prev_pos
    hold = pos & prev_pos
    r[entry] = c[entry] / o[entry] - 1.0 - cost
    r[hold] = c[hold] / prev_c[hold] - 1.0
    r[pos & ~next_pos] -= cost
    return r


def metrics(r: np.ndarray, pos: np.ndarray | None = None) -> dict:
    r = np.nan_to_num(r)
    T = r.size
    eq = np.cumprod(1.0 + r)
    years = T / TRADING_DAYS
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    mdd = (eq / peak - 1.0).min()
    out = dict(cagr=cagr, vol=sd * np.sqrt(TRADING_DAYS), sharpe=sharpe, max_dd=mdd,
               total_return=eq[-1] - 1.0)
    if pos is not None:
        out["exposure"] = pos.mean()
        out["ret_per_exposure"] = (r.mean() * TRADING_DAYS / pos.mean()) if pos.mean() > 0 else np.nan
        out["n_trades"] = int((pos & ~np.r_[False, pos[:-1]]).sum())
    return out


def run_ticker(event_rows: np.ndarray, h: int, o: np.ndarray, c: np.ndarray,
               cost: float, n_random: int, rng: np.random.Generator
               ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Strategy vs buy&hold vs random entries for one ticker/spec."""
    T = c.size
    pos = positions(event_rows, h, T)
    strat_r = strategy_returns(pos, o, c, cost)
    strat = metrics(strat_r, pos)

    bh_r = np.r_[0.0, c[1:] / c[:-1] - 1.0]
    bh = metrics(bh_r)

    k = event_rows.size
    rand_sh, rand_cagr, rand_exp = [], [], []
    if k > 0:
        candidates = np.arange(T - 1)
        for _ in range(n_random):
            rows = np.sort(rng.choice(candidates, size=min(k, candidates.size), replace=False))
            p = positions(rows, h, T)
            m = metrics(strategy_returns(p, o, c, cost), p)
            rand_sh.append(m["sharpe"])
            rand_cagr.append(m["cagr"])
            rand_exp.append(m["exposure"])
    rand_sh = np.array(rand_sh, dtype=float)
    out = {f"strat_{k_}": v for k_, v in strat.items()}
    out.update({f"bh_{k_}": v for k_, v in bh.items()})
    if rand_sh.size:
        out["rand_sharpe_median"] = float(np.nanmedian(rand_sh))
        out["rand_cagr_median"] = float(np.nanmedian(rand_cagr))
        out["rand_exposure_median"] = float(np.nanmedian(rand_exp))
        # Percentile of the real strategy's Sharpe inside the random-entry distribution
        out["sharpe_pctile_vs_random"] = float(np.mean(rand_sh < strat["sharpe"])) \
            if np.isfinite(strat["sharpe"]) else np.nan
    return out, strat_r, bh_r
