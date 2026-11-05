import numpy as np
import pandas as pd
import pytest

from buyadip import backtest as bt
from buyadip.config import Config
from buyadip.data import panel_from_frames, validate
from buyadip.events import forward_returns, sample_mask
from buyadip.signals import apply_cooldown, atr, detect_events, moving_average
from buyadip.stats import _circ_xcorr, bh_fdr, shift_placebo


def _frame(close: np.ndarray, start="2019-01-01", div_factor=1.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=close.size)
    c = pd.Series(close, idx)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01, "low": c * 0.99,
                         "close": c, "adj_close": c * div_factor, "volume": 1e6}, index=idx)


def _random_walk(n=900, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))


def _events(p, w=20, k=1.0, d="from_above", cd=5):
    ma = moving_average(p.close, w)
    return detect_events(p.open, p.close, ma, atr(p.high, p.low, p.close), k, d, cd)


def test_no_lookahead_in_signals():
    """Signal at day t may use open_t but nothing after it (not close_t, high_t, low_t)."""
    a = _frame(_random_walk())
    cut = 600
    b = a.copy()
    # perturb everything after the open of day `cut`
    for col in ("high", "low", "close", "adj_close"):
        b.iloc[cut:, b.columns.get_loc(col)] *= 1.07
    b.iloc[cut + 1:, b.columns.get_loc("open")] *= 1.07
    b["high"] = b[["high", "open", "close"]].max(axis=1)
    b["low"] = b[["low", "open", "close"]].min(axis=1)
    ea = _events(panel_from_frames({"X": a}))
    eb = _events(panel_from_frames({"X": b}))
    pd.testing.assert_frame_equal(ea.iloc[: cut + 1], eb.iloc[: cut + 1])


def _fw(p, h):
    m = sample_mask(p.index, str(p.index[0].date()), None)
    return forward_returns(p, h, m, 60, "SPY")


def test_forward_return_alignment():
    """Entry = open_t, exit = open_{t+h}."""
    close = np.arange(1, 401, dtype=float) + 100
    p = panel_from_frames({"SPY": _frame(close), "X": _frame(close * 2)})
    f = _fw(p, 5)
    t = 10
    assert f.ret["X"].iloc[t] == pytest.approx(p.adj_open["X"].iloc[t + 5] / p.adj_open["X"].iloc[t] - 1)
    assert f.ret["X"].iloc[-5:].isna().all()
    # SPY benchmark uses SPY's return over the same dates; SPY vs itself is excluded
    assert f.base_spy["X"].iloc[t] == pytest.approx(f.ret["SPY"].iloc[t])
    assert f.base_spy["SPY"].isna().all()
    # same-ticker benchmark is a centered rolling mean of all days
    assert f.base_own["X"].iloc[200] == pytest.approx(f.ret["X"].iloc[170:230].mean())


def test_total_return_used_for_pnl():
    close = np.full(300, 100.0)
    adj = np.linspace(90, 100, 300)  # dividends accrue -> adj close rises
    fr = _frame(close)
    fr["adj_close"] = adj
    p = panel_from_frames({"SPY": fr})
    assert (_fw(p, 20).ret["SPY"].dropna() > 0).all()


def test_detect_from_above_simple():
    # flat at 100 with 1% daily range (ATR ~2); day 30 closes at 110 (far above MA),
    # day 31 opens at 101 (back within 1 ATR of yesterday's MA) -> dip event on day 31
    n = 45
    idx = pd.bdate_range("2019-01-01", periods=n)
    c = np.full(n, 100.0); c[30] = 110.0
    o = np.full(n, 100.0); o[30] = 100.0; o[31] = 101.0
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                       "close": c, "adj_close": c, "volume": 1e6}, index=idx)
    ev = _events(panel_from_frames({"X": df}), w=5, k=1.0, cd=0)["X"]
    assert ev.iloc[31] and not ev.iloc[30]
    assert ev.sum() == 1


def test_cooldown():
    e = np.zeros((20, 1), bool)
    e[[1, 2, 3, 9, 10], 0] = True
    kept = apply_cooldown(e, 5)[:, 0]
    assert np.flatnonzero(kept).tolist() == [1, 9]


def test_circ_xcorr_matches_bruteforce():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 3))
    e = (rng.random((64, 3)) < 0.2).astype(float)
    c = _circ_xcorr(x, e)
    for s in [0, 1, 7, 33, 63]:
        brute = sum((x[:, j] * np.roll(e[:, j], s)).sum() for j in range(3))
        assert c[s] == pytest.approx(brute)


def test_placebo_detects_planted_edge_and_not_noise():
    rng = np.random.default_rng(2)
    T, N = 1500, 6
    x = rng.normal(0, 0.05, (T, N))
    e = rng.random((T, N)) < 0.03
    noise = shift_placebo(x, e, 30)
    x2 = x.copy()
    x2[e] += 0.03  # planted +3% edge on event days
    planted = shift_placebo(x2, e, 30)
    assert planted["p"] < 0.01 and planted["z"] > 3
    assert noise["p"] > 0.01


def test_bh_fdr():
    q = bh_fdr(np.array([0.01, 0.04, 0.03, np.nan, 0.5]))
    assert np.isnan(q[3])
    assert q[0] == pytest.approx(0.04)
    assert np.all(q[np.isfinite(q)] <= 1)


def test_backtest_accounting():
    c = 100 * 1.01 ** np.arange(50)  # +1%/day
    o = c / 1.005                   # open half way
    pos = bt.positions(np.array([10]), 3, c.size)
    assert np.flatnonzero(pos).tolist() == [10, 11, 12]  # closes held
    r = bt.strategy_returns(pos, o, c, cost=0.0)
    assert r[10] == pytest.approx(0.005)            # open -> close on entry day
    assert r[11] == pytest.approx(0.01)
    assert r[13] == pytest.approx(o[13] / c[12] - 1)  # sold at the open of t+h
    # compounded strategy return == open_t -> open_{t+h}
    assert np.prod(1 + r) - 1 == pytest.approx(o[13] / o[10] - 1)
    r_cost = bt.strategy_returns(pos, o, c, cost=0.001)
    assert r_cost.sum() == pytest.approx(r.sum() - 0.002)


def test_overlapping_events_extend_not_double():
    pos = bt.positions(np.array([5, 7]), 4, 30)
    assert np.flatnonzero(pos).tolist() == list(range(5, 11))


def test_validate_rejects_unadjusted_split():
    close = np.r_[np.full(50, 400.0), np.full(50, 80.0)]
    with pytest.raises(ValueError, match="split"):
        validate(_frame(close), "X")


def test_drop_partial_session():
    from buyadip.data import drop_partial_session
    df = _frame(np.full(5, 100.0), start="2026-11-02")  # Mon..Fri
    intraday = pd.Timestamp("2026-11-06 12:00", tz="America/New_York")
    after = pd.Timestamp("2026-11-06 17:00", tz="America/New_York")
    assert len(drop_partial_session(df, intraday)) == 4
    assert len(drop_partial_session(df, after)) == 5


def test_config_fails_fast():
    with pytest.raises(ValueError):
        Config(atr_bands=(-0.5,))
    with pytest.raises(ValueError):
        Config(ma_type="wma")


def test_panel_with_late_listing():
    """Tickers with shorter histories (TSLA 2010) leave NaN before listing, not an error."""
    a = _frame(_random_walk(400, 1), start="2019-01-01")
    b = _frame(_random_walk(200, 2), start="2019-08-01")
    p = panel_from_frames({"SPY": a, "LATE": b})
    assert p.close["LATE"].isna().sum() > 0
    m = sample_mask(p.index, "2019-01-01", None)
    f = forward_returns(p, 5, m, 60, "SPY")
    assert f.ret["LATE"].notna().sum() > 100
