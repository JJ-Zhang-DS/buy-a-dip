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
class Band:
    """Proximity band around the MA.

    kind="pct": |close/MA - 1| <= width            (fixed %, e.g. 0.05 = +/-5%)
    kind="atr": |close - MA|   <= width * ATR14     (volatility-scaled)
    """
    kind: str
    width: float

    @property
    def label(self) -> str:
        if self.kind == "pct":
            return f"pct{self.width * 100:g}"
        return f"atr{self.width:g}"

    def __post_init__(self) -> None:
        if self.kind not in {"pct", "atr"}:
            raise ValueError(f"Band.kind must be 'pct' or 'atr', got {self.kind!r}")
        if not (self.width > 0):
            raise ValueError(f"Band.width must be > 0, got {self.width}")


@dataclass(frozen=True)
class Config:
    tickers: tuple[str, ...] = tuple(DEFAULT_TICKERS)
    # History is downloaded from data_start so a 200-day MA is fully warmed up
    # by sample_start. Events are only counted inside [sample_start, sample_end].
    data_start: str = "2015-01-01"
    sample_start: str = "2020-01-01"
    sample_end: str | None = None  # None = last available bar

    ma_windows: tuple[int, ...] = (5, 20, 50, 60, 120, 200)
    ma_type: str = "sma"  # "sma" | "ema"
    bands: tuple[Band, ...] = (
        Band("pct", 0.01), Band("pct", 0.02), Band("pct", 0.03), Band("pct", 0.05),
        Band("atr", 0.5), Band("atr", 1.0),
    )
    directions: tuple[str, ...] = ("from_above", "from_below")
    # short: 5/10/20d, medium: 60/120d, long: 250d (~1 trading year)
    horizons: tuple[int, ...] = (5, 10, 20, 60, 120, 250)
    # Minimum bars between two events on the same ticker/spec (de-clustering).
    cooldown: int = 5
    atr_window: int = 14

    # Placebo test: circular time shifts of the event calendar. Shifts closer than
    # horizon + placebo_min_gap bars to zero are excluded (they overlap the real events).
    placebo_min_gap: int = 20

    # Strategy simulation
    cost_bps_per_side: float = 5.0
    n_random_strategies: int = 100
    seed: int = 20201

    regime_ticker: str = "SPY"
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
        if self.regime_ticker not in self.tickers:
            raise ValueError("regime_ticker must be part of tickers")
