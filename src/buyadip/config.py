"""Experiment configuration.

Everything that defines a "test" lives here so the grid is pre-registered in code.
Each (MA, band, direction, horizon) cell counts toward the multiple-testing burden,
so adding cells is not free: it widens the BH-FDR correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TICKERS = [
    "SPY", "QQQ", "TSLA", "GOOGL", "NVDA", "AMD",
    "ASML", "MSFT", "ISRG", "URI", "WM", "LLY",
]


@dataclass(frozen=True)
class Config:
    tickers: tuple[str, ...] = tuple(DEFAULT_TICKERS)
    # Download history from data_start so the 200-day MA is warm by sample_start.
    # Tickers listed later (GOOGL 2004, TSLA 2010) simply enter the sample when their
    # own MA is warm.
    data_start: str = "1998-01-01"
    sample_start: str = "2000-01-01"
    sample_end: str | None = "2026-10-31"

    ma_windows: tuple[int, ...] = (5, 20, 50, 60, 120, 200)
    ma_type: str = "sma"  # "sma" | "ema"
    # Zone half-width in ATR units: |open - MA| <= k * ATR14. ATR scales with each
    # ticker's own volatility, so "near" means the same thing for WM and TSLA.
    atr_bands: tuple[float, ...] = (0.5, 1.0)
    directions: tuple[str, ...] = ("from_above", "from_below")
    # short: 5/10/20d, medium: 60/120d, long: 250d (~1 trading year)
    horizons: tuple[int, ...] = (5, 10, 20, 60, 120, 250)
    # Minimum bars between two events on the same ticker/spec (de-clustering).
    cooldown: int = 5
    atr_window: int = 14

    # "Not timing the market" benchmark: mean h-day return of entering on ANY day in a
    # centered window of this many bars around the signal (same ticker).
    baseline_window: int = 252
    market_ticker: str = "SPY"  # second benchmark: SPY over the identical h-day window

    # Placebo test: circular time shifts of the event calendar. Shifts closer than
    # horizon + placebo_min_gap bars to zero are excluded (they overlap the real events).
    placebo_min_gap: int = 20

    # Strategy simulation
    cost_bps_per_side: float = 5.0
    n_random_strategies: int = 100
    seed: int = 20201

    regime_ma: int = 200

    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        if self.ma_type not in {"sma", "ema"}:
            raise ValueError(f"ma_type must be sma|ema, got {self.ma_type!r}")
        bad = set(self.directions) - {"from_above", "from_below"}
        if bad:
            raise ValueError(f"unknown directions: {bad}")
        if min(self.horizons) < 1 or min(self.ma_windows) < 2:
            raise ValueError("horizons must be >=1 and MA windows >=2")
        if not self.atr_bands or min(self.atr_bands) <= 0:
            raise ValueError("atr_bands must be non-empty and > 0")
        if self.market_ticker not in self.tickers:
            raise ValueError("market_ticker must be part of tickers")
        if self.baseline_window < 20:
            raise ValueError("baseline_window too short")


def band_label(k: float) -> str:
    return f"atr{k:g}"
