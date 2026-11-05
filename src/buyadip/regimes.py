"""Bull / bear market regimes and named crisis episodes.

Two kinds of labels, deliberately kept apart:

EX-ANTE (known at the open of day t -> usable as a trading filter)
  spy_trend : SPY close_{t-1} above / below its 200-day SMA
  spy_dd    : SPY drawdown from its running high at close_{t-1}:
              "<10%" normal, "10-20%" correction, ">=20%" bear territory
  stock_trend (per ticker): the ticker's close_{t-1} above / below its own 200-day SMA

EX-POST (uses the future -> descriptive only, never a filter)
  phase   : for every SPY drawdown of >= 20% (close basis)
              decline  = peak -> trough
              recovery = trough -> first close back at the old peak
              bull     = everything else
  episode : the same windows, named (dotcom, gfc, covid, 2022 ...)
  You only know a bear market was a bear market, and where its bottom was, afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPISODE_NAMES = {2000: "dotcom", 2007: "gfc", 2020: "covid", 2022: "bear_2022"}


@dataclass
class Episode:
    name: str
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovered: pd.Timestamp | None  # None = not back at the peak by end of data
    depth: float


def bear_episodes(close: pd.Series, threshold: float = 0.20) -> list[Episode]:
    """Peak-to-trough drawdowns of at least `threshold`, peak to full recovery."""
    c = close.dropna()
    vals, idx = c.to_numpy(), c.index
    out: list[Episode] = []
    i, n = 0, len(vals)
    peak_i = 0
    while i < n:
        if vals[i] >= vals[peak_i]:
            peak_i = i
            i += 1
            continue
        if vals[i] / vals[peak_i] - 1 <= -threshold:
            # inside a bear market: find recovery (first close >= peak) and the trough before it
            j = i
            while j < n and vals[j] < vals[peak_i]:
                j += 1
            seg = slice(peak_i, j)
            trough_i = peak_i + int(np.argmin(vals[seg]))
            rec = idx[j] if j < n else None
            yr = idx[peak_i].year
            out.append(Episode(EPISODE_NAMES.get(yr, f"bear_{yr}"), idx[peak_i], idx[trough_i], rec,
                               float(vals[trough_i] / vals[peak_i] - 1)))
            if j >= n:
                break
            i, peak_i = j, j
            continue
        i += 1
    return out


def market_regimes(spy_close: pd.Series, ma_window: int = 200, threshold: float = 0.20) -> tuple[pd.DataFrame, list[Episode]]:
    c = spy_close
    m = c.rolling(ma_window, min_periods=ma_window).mean()
    trend = np.where(m.isna(), None, np.where(c > m, "spy_above_200", "spy_below_200"))
    dd = c / c.cummax() - 1
    dd_lab = np.select([dd > -0.10, dd > -0.20], ["dd_lt10", "dd_10_20"], "dd_ge20")
    ex_ante = pd.DataFrame({"spy_trend": trend, "spy_dd": dd_lab}, index=c.index).shift(1)

    eps = bear_episodes(c, threshold)
    phase = pd.Series("bull", index=c.index, dtype=object)
    episode = pd.Series("none", index=c.index, dtype=object)
    for e in eps:
        end = e.recovered if e.recovered is not None else c.index[-1]
        phase[(c.index >= e.peak) & (c.index <= e.trough)] = "decline"
        phase[(c.index > e.trough) & (c.index < end)] = "recovery"
        episode[(c.index >= e.peak) & (c.index <= e.trough)] = f"{e.name}_decline"
        episode[(c.index > e.trough) & (c.index < end)] = f"{e.name}_recovery"
    out = ex_ante.assign(phase=phase, episode=episode)
    return out, eps


def stock_trend(close: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """'stock_above_200' / 'stock_below_200' known at today's open (uses close_{t-1})."""
    m = close.rolling(window, min_periods=window).mean()
    lab = np.where(m.isna(), None, np.where(close > m, "stock_above_200", "stock_below_200"))
    return pd.DataFrame(lab, index=close.index, columns=close.columns).shift(1)
