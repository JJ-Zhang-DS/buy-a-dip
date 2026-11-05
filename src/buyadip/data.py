"""Price data: download, cache, validate, and pivot into a wide panel.

Two price series per ticker, on purpose:
  * close/open/high/low  : split-adjusted, NOT dividend-adjusted. This is the chart that
                           traders look at, so the MA signal is computed on it.
  * adj_*                : total-return (split + dividend adjusted). All P&L uses these,
                           otherwise dividend payers (SPY, MSFT, WM, LLY) are understated.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED = ["open", "high", "low", "close", "adj_close", "volume"]


def _download_yfinance(ticker: str, start: str, retries: int = 4) -> pd.DataFrame:
    import yfinance as yf  # imported lazily: tests do not need network deps

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker, start=start, auto_adjust=False, actions=False,
                progress=False, threads=False, multi_level_index=False,
            )
            if df is None or df.empty:
                raise RuntimeError(f"empty frame for {ticker}")
            df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
            return df
        except Exception as err:  # yfinance raises a zoo of exception types
            last_err = err
            wait = 5 * 2 ** attempt
            log.warning("yfinance %s attempt %d failed: %s (retry in %ss)", ticker, attempt + 1, err, wait)
            time.sleep(wait)
    raise RuntimeError(f"yfinance download failed for {ticker}") from last_err


def _download_stooq(ticker: str, start: str) -> pd.DataFrame:
    """Fallback source. Stooq data is split+dividend adjusted, so adj_close == close."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
    if df.empty or "Close" not in df:
        raise RuntimeError(f"stooq returned nothing for {ticker}")
    df = df.rename(columns=str.lower)
    df["adj_close"] = df["close"]
    log.warning("%s: using stooq fallback (dividend-adjusted closes; signal != chart MA)", ticker)
    return df.loc[start:]


def validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    df = df[REQUIRED].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close", "adj_close"])
    if (df[["open", "high", "low", "close", "adj_close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive prices")
    # High/low must bracket open/close (tolerance for vendor rounding)
    tol = 1e-6 * df["close"]
    if ((df["high"] + tol < df[["open", "close"]].max(axis=1)) |
            (df["low"] - tol > df[["open", "close"]].min(axis=1))).mean() > 0.01:
        raise ValueError(f"{ticker}: >1% of bars have inconsistent OHLC")
    # A >75% one-day move in a split-adjusted close almost always means a missed split
    # (largest genuine moves here: AMD +52% in 2016, ~-50% single days in 2000-02).
    jumps = df["close"].pct_change().abs()
    if (jumps > 0.75).any():
        raise ValueError(f"{ticker}: suspicious jump on {jumps.idxmax().date()} (unadjusted split?)")
    return df


def drop_partial_session(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Yahoo returns today's still-forming bar during market hours; a signal on it would use
    a close that does not exist yet. Drop it unless the US session has closed."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 30):
        return df.iloc[:-1]
    return df


def fetch(tickers: list[str] | tuple[str, ...], start: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        try:
            raw = _download_yfinance(t, start)
        except RuntimeError:
            raw = _download_stooq(t, start)
        df = drop_partial_session(validate(raw, t))
        df.to_parquet(data_dir / f"{t}.parquet")
        log.info("%s: %d bars %s -> %s", t, len(df), df.index[0].date(), df.index[-1].date())
        time.sleep(1.0)  # be polite; Yahoo rate-limits shared CI IPs


@dataclass
class Panel:
    """Wide (date x ticker) frames aligned on a common trading calendar."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj_open: pd.DataFrame
    adj_high: pd.DataFrame
    adj_low: pd.DataFrame
    adj_close: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index  # type: ignore[return-value]


def load_panel(tickers: list[str] | tuple[str, ...], data_dir: Path) -> Panel:
    frames = {}
    for t in tickers:
        path = data_dir / f"{t}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `buyadip fetch` first")
        frames[t] = validate(pd.read_parquet(path), t)
    return panel_from_frames(frames)


def panel_from_frames(frames: dict[str, pd.DataFrame]) -> Panel:
    def wide(col: str) -> pd.DataFrame:
        return pd.DataFrame({t: f[col] for t, f in frames.items()}).sort_index()

    o, h, l, c, ac = (wide(k) for k in ["open", "high", "low", "close", "adj_close"])
    # Total-return factor. Applying it to O/H/L gives dividend-adjusted intraday prices.
    f = ac / c
    if not np.isfinite(f.stack()).all():
        raise ValueError("non-finite adjustment factor")
    # Each ticker keeps its own listing history; NaN where a ticker did not trade.
    return Panel(open=o, high=h, low=l, close=c,
                 adj_open=o * f, adj_high=h * f, adj_low=l * f, adj_close=ac)
