"""Moving averages and "open near the MA" event detection.

Timing convention (look-ahead guard):
  * MA_{t-1} and ATR_{t-1} use closes through YESTERDAY, so they are known before the bell.
  * The signal compares TODAY's open with them, and the trade is filled at that same open.
  * Nothing in this module reads any price after open_t.

    dist_t  = open_t / MA_{t-1} - 1
    width_t = k * ATR_{t-1} / MA_{t-1}
    from_above: |dist_t| <= width_t  and  close_{t-1} / MA_{t-1} - 1 >  width_t
    from_below: |dist_t| <= width_t  and  close_{t-1} / MA_{t-1} - 1 < -width_t
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(close: pd.DataFrame, window: int, kind: str = "sma") -> pd.DataFrame:
    if kind == "sma":
        return close.rolling(window, min_periods=window).mean()
    if kind == "ema":
        return close.ewm(span=window, adjust=False, min_periods=window).mean()
    raise ValueError(kind)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder ATR on split-adjusted prices (typical daily range in $)."""
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def apply_cooldown(events: np.ndarray, cooldown: int) -> np.ndarray:
    """Keep an event only if no kept event occurred in the previous `cooldown` bars."""
    if cooldown <= 0:
        return events
    out = np.zeros_like(events, dtype=bool)
    for j in range(events.shape[1]):
        last = -10 ** 9
        for i in np.flatnonzero(events[:, j]):
            if i - last > cooldown:
                out[i, j] = True
                last = i
    return out


def open_distance(open_: pd.DataFrame, close: pd.DataFrame, ma: pd.DataFrame,
                  atr_df: pd.DataFrame, k: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (dist_open_t, dist_prev_close, width_t), all relative to MA_{t-1}."""
    ma_prev = ma.shift(1)
    width = k * atr_df.shift(1) / ma_prev
    dist_open = open_ / ma_prev - 1.0
    dist_prev = close.shift(1) / ma_prev - 1.0
    return dist_open, dist_prev, width


def detect_events(open_: pd.DataFrame, close: pd.DataFrame, ma: pd.DataFrame,
                  atr_df: pd.DataFrame, k: float, direction: str, cooldown: int) -> pd.DataFrame:
    """First open inside the +/- k*ATR zone after yesterday's close was outside it.

    A gap straight through the zone (above yesterday -> below today) is NOT an event:
    the price never opened "near" the MA.
    """
    dist_open, dist_prev, width = open_distance(open_, close, ma, atr_df, k)
    inside = dist_open.abs() <= width
    if direction == "from_above":
        came_from = dist_prev > width
    elif direction == "from_below":
        came_from = dist_prev < -width
    else:
        raise ValueError(direction)
    raw = (inside & came_from & width.notna()).to_numpy()
    return pd.DataFrame(apply_cooldown(raw, cooldown), index=close.index, columns=close.columns)


def regime_above_ma(close: pd.Series, window: int) -> pd.Series:
    """True when the market closed above its own SMA YESTERDAY (known at today's open)."""
    m = close.rolling(window, min_periods=window).mean()
    return (close > m).where(m.notna()).shift(1)
