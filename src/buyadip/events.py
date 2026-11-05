"""Forward-return matrices for the event study.

For a signal on day t (close) and horizon h:
    entry  = adj_open[t+1]                     (next open; no same-bar fill)
    exit   = adj_close[t+h]                    (close h trading days after the signal)
    R_h[t] = exit / entry - 1
    MAE_h  = min(adj_low[t+1..t+h])  / entry - 1   (max adverse excursion, <= 0 mostly)
    MFE_h  = max(adj_high[t+1..t+h]) / entry - 1   (max favourable excursion)

Excess return removes the ticker's own drift in that calendar year:
    X_h[t] = R_h[t] - mean(R_h over ALL sample days of the same ticker and year)
This is what makes "NVDA went up 10x" irrelevant: a signal only scores if entries on signal
days beat entries on random days of the same stock in the same year.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel


@dataclass
class Forward:
    h: int
    ret: pd.DataFrame      # R_h
    excess: pd.DataFrame   # X_h
    base: pd.DataFrame     # ticker-year baseline mean of R_h
    mae: pd.DataFrame
    mfe: pd.DataFrame


def forward_returns(panel: Panel, h: int, sample: pd.Series) -> Forward:
    entry = panel.adj_open.shift(-1)
    exit_ = panel.adj_close.shift(-h)
    ret = exit_ / entry - 1.0
    lo = panel.adj_low.rolling(h, min_periods=h).min().shift(-h)
    hi = panel.adj_high.rolling(h, min_periods=h).max().shift(-h)
    mae = lo / entry - 1.0
    mfe = hi / entry - 1.0

    ret = ret[sample.values]
    mae, mfe = mae[sample.values], mfe[sample.values]
    year = ret.index.year
    base = ret.groupby(year).transform("mean")
    excess = ret - base
    return Forward(h=h, ret=ret, excess=excess, base=base, mae=mae, mfe=mfe)


def sample_mask(index: pd.DatetimeIndex, start: str, end: str | None) -> pd.Series:
    m = index >= pd.Timestamp(start)
    if end is not None:
        m &= index <= pd.Timestamp(end)
    return pd.Series(m, index=index)
