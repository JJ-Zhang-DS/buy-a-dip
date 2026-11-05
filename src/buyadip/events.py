"""Forward-return matrices for the event study (open-to-open).

For a signal at the open of day t and horizon h:
    R_h[t]   = adj_open[t+h] / adj_open[t] - 1      buy at today's open, sell at the open h days later
    MAE_h[t] = min(adj_low[t..t+h-1])  / adj_open[t] - 1
    MFE_h[t] = max(adj_high[t..t+h-1]) / adj_open[t] - 1

Two "not timing the market" benchmarks:
    own : B_own[t] = mean of R_h over ALL days in a centered `baseline_window` bars of the
          SAME ticker  -> "would buying any day around then have done as well?"
    spy : B_spy[t] = SPY's R_h over the identical dates -> "did it beat just owning SPY?"
          (mixes in the stock's own drift/beta; read together with `own`)
Excess = R_h - benchmark. Benchmarks are ex-post on purpose: they are yardsticks, not signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel


@dataclass
class Forward:
    h: int
    ret: pd.DataFrame
    base_own: pd.DataFrame
    base_spy: pd.DataFrame
    mae: pd.DataFrame
    mfe: pd.DataFrame
    hit_base: pd.DataFrame  # share of positive R_h in the same window (any-day hit rate)

    @property
    def excess_own(self) -> pd.DataFrame:
        return self.ret - self.base_own

    @property
    def excess_spy(self) -> pd.DataFrame:
        return self.ret - self.base_spy


def forward_returns(panel: Panel, h: int, sample: pd.Series, baseline_window: int,
                    market: str) -> Forward:
    entry = panel.adj_open
    ret = panel.adj_open.shift(-h) / entry - 1.0
    lo = panel.adj_low.rolling(h, min_periods=h).min().shift(-(h - 1))
    hi = panel.adj_high.rolling(h, min_periods=h).max().shift(-(h - 1))
    mae, mfe = lo / entry - 1.0, hi / entry - 1.0

    minp = baseline_window // 2
    roll = dict(window=baseline_window, center=True, min_periods=minp)
    base_own = ret.rolling(**roll).mean()
    pos = (ret > 0).astype(float).where(ret.notna())
    hit_base = pos.rolling(**roll).mean()
    base_spy = pd.DataFrame({c: ret[market] for c in ret.columns}, index=ret.index)
    base_spy[market] = np.nan  # SPY vs itself is identically 0: exclude

    s = sample.values
    return Forward(h=h, ret=ret[s], base_own=base_own[s], base_spy=base_spy[s],
                   mae=mae[s], mfe=mfe[s], hit_base=hit_base[s])


def sample_mask(index: pd.DatetimeIndex, start: str, end: str | None) -> pd.Series:
    m = index >= pd.Timestamp(start)
    if end is not None:
        m &= index <= pd.Timestamp(end)
    return pd.Series(m, index=index)
