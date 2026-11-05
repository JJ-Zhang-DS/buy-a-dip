"""Tradable version of the signal.

Rule per ticker: signal at the open of day t -> buy at that open, hold h trading days,
sell at the open of t+h. A new event while already long extends the exit (no pyramiding).
Flat = cash at 0% (conservative for the strategy; T-bills would help it a little).

Benchmarks:
  * buy & hold of the same ticker over the same window
  * random-entry strategy with the SAME number of entries and the SAME holding period
    -> separates timing skill from simply being in the market less of the time.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def positions(event_rows: np.ndarray, h: int, T: int) -> np.ndarray:
    """Bool (T,): True on days whose CLOSE is held (entry day t .. t+h-1)."""
    d = np.zeros(T + 1, dtype=np.int64)
    start = event_rows[event_rows < T]
    np.add.at(d, start, 1)
    np.add.at(d, np.minimum(start + h, T), -1)
    return np.cumsum(d)[:T] > 0


def strategy_returns(pos: np.ndarray, o: np.ndarray, c: np.ndarray, cost: float) -> np.ndarray:
    """Daily returns from total-return adjusted open/close arrays (T,).

    entry day : close/open - 1 - cost            (bought at the open)
    held day  : close/prev_close - 1
    exit day  : open/prev_close - 1 - cost       (sold at the open; first day not held)
    """
    prev_pos = np.r_[False, pos[:-1]]
    prev_c = np.r_[np.nan, c[:-1]]
    r = np.zeros_like(c)
    entry = pos & ~prev_pos
    hold = pos & prev_pos
    exit_ = ~pos & prev_pos
    r[entry] = c[entry] / o[entry] - 1.0 - cost
    r[hold] = c[hold] / prev_c[hold] - 1.0
    r[exit_] = o[exit_] / prev_c[exit_] - 1.0 - cost
    return r


def metrics(r: np.ndarray, pos: np.ndarray | None = None) -> dict:
    r = np.nan_to_num(r)
    T = r.size
    eq = np.cumprod(1.0 + r)
    years = T / TRADING_DAYS
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    mdd = (eq / np.maximum.accumulate(eq) - 1.0).min()
    out = dict(cagr=cagr, vol=sd * np.sqrt(TRADING_DAYS), sharpe=sharpe, max_dd=mdd,
               total_return=eq[-1] - 1.0)
    if pos is not None:
        out["exposure"] = pos.mean()
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
    rand_sh, rand_cagr = [], []
    if k > 0:
        for _ in range(n_random):
            rows = np.sort(rng.choice(T - 1, size=min(k, T - 1), replace=False))
            p = positions(rows, h, T)
            m = metrics(strategy_returns(p, o, c, cost), p)
            rand_sh.append(m["sharpe"])
            rand_cagr.append(m["cagr"])
    rand_sh = np.array(rand_sh, dtype=float)
    out = {f"strat_{k_}": v for k_, v in strat.items()}
    out.update({f"bh_{k_}": v for k_, v in bh.items()})
    if rand_sh.size:
        out["rand_sharpe_median"] = float(np.nanmedian(rand_sh))
        out["rand_cagr_median"] = float(np.nanmedian(rand_cagr))
        out["sharpe_pctile_vs_random"] = float(np.mean(rand_sh < strat["sharpe"])) \
            if np.isfinite(strat["sharpe"]) else np.nan
    return out, strat_r, bh_r
