"""Moving averages and "near-the-MA" event detection.

Timing convention (look-ahead guard):
  * Everything in this module uses bars up to and including day t (the signal day close).
  * The trade is entered at the OPEN of t+1 (see events.py). Nothing here reads t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Band


def moving_average(close: pd.DataFrame, window: int, kind: str = "sma") -> pd.DataFrame:
    if kind == "sma":
        return close.rolling(window, min_periods=window).mean()
    if kind == "ema":
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        return ema
    raise ValueError(kind)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder ATR on split-adjusted prices."""
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def band_width(band: Band, ma: pd.DataFrame, atr_df: pd.DataFrame) -> pd.DataFrame:
    """Half-width of the zone, expressed as a fraction of the MA."""
    if band.kind == "pct":
        return pd.DataFrame(band.width, index=ma.index, columns=ma.columns)
    return band.width * atr_df / ma


def apply_cooldown(events: np.ndarray, cooldown: int) -> np.ndarray:
    """Keep an event only if no kept event occurred in the previous `cooldown` bars.

    events: bool array (T, N). Sequential by construction (greedy de-clustering).
    """
    if cooldown <= 0:
        return events
    out = np.zeros_like(events, dtype=bool)
    T, N = events.shape
    for j in range(N):
        last = -10 ** 9
        idx = np.flatnonzero(events[:, j])
        for i in idx:
            if i - last > cooldown:
                out[i, j] = True
                last = i
    return out


def detect_events(
    close: pd.DataFrame,
    ma: pd.DataFrame,
    width: pd.DataFrame,
    direction: str,
    cooldown: int,
) -> pd.DataFrame:
    """First day the close enters the +/-width zone around the MA.

    from_above: yesterday the close was ABOVE the zone, today it is inside it  -> "dip to the MA"
    from_below: yesterday the close was BELOW the zone, today it is inside it  -> "rally to the MA"

    A day that gaps straight through the zone (e.g. +6% above to -6% below a 5% band) is not an
    event: the price never closed "near" the MA. That mirrors how a discretionary trader would
    see it on a daily chart, and it is a known source of selection (see docs/PLAN.md).
    """
    dist = close / ma - 1.0
    inside = dist.abs() <= width
    if direction == "from_above":
        came_from = dist.shift(1) > width.shift(1)
    elif direction == "from_below":
        came_from = dist.shift(1) < -width.shift(1)
    else:
        raise ValueError(direction)
    raw = (inside & came_from & ma.notna() & ma.shift(1).notna()).to_numpy()
    kept = apply_cooldown(raw, cooldown)
    return pd.DataFrame(kept, index=close.index, columns=close.columns)


def regime_above_ma(close: pd.Series, window: int) -> pd.Series:
    """True when the regime ticker closes above its own SMA (known at t)."""
    m = close.rolling(window, min_periods=window).mean()
    return (close > m).where(m.notna())
