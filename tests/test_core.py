import numpy as np
import pandas as pd
import pytest

from buyadip import backtest as bt
from buyadip.config import Band, Config
from buyadip.data import panel_from_frames, validate
from buyadip.events import forward_returns, sample_mask
from buyadip.signals import apply_cooldown, atr, band_width, detect_events, moving_average
from buyadip.stats import _circ_xcorr, bh_fdr, shift_placebo


def _frame(close: np.ndarray, start="2019-01-01", div_factor=1.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=close.size)
    c = pd.Series(close, idx)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01, "low": c * 0.99,
                         "close": c, "adj_close": c * div_factor, "volume": 1e6}, index=idx)


def _random_walk(n=900, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))


def test_no_lookahead_in_signals():
    """Changing prices AFTER day t must not change any signal at or before t."""
    a = _random_walk()
    b = a.copy()
    cut = 600
    b[cut + 1:] *= np.linspace(0.5, 2.0, b.size - cut - 1)
    out = []
    for arr in (a, b):
        p = panel_from_frames({"X": _frame(arr)})
        ma = moving_average(p.close, 20)
        w = band_width(Band("atr", 1.0), ma, atr(p.high, p.low, p.close))
        out.append(detect_events(p.close, ma, w, "from_above", 5).iloc[: cut + 1])
    pd.testing.assert_frame_equal(out[0], out[1])


def test_forward_return_alignment():
    """Entry = next open, exit = close at t+h."""
    close = np.arange(1, 401, dtype=float) + 100
    p = panel_from_frames({"X": _frame(close)})
    m = sample_mask(p.index, str(p.index[0].date()), None)
    f = forward_returns(p, 5, m)
    t = 10
    entry = p.adj_open["X"].iloc[t + 1]
    exit_ = p.adj_close["X"].iloc[t + 5]
    assert f.ret["X"].iloc[t] == pytest.approx(exit_ / entry - 1)
    assert f.ret["X"].iloc[-5:].isna().all()  # no future bars -> undefined


def test_total_return_used_for_pnl():
    close = np.full(300, 100.0)
    adj = np.linspace(90, 100, 300)  # dividends accrue -> adj close rises
    fr = _frame(close)
    fr["adj_close"] = adj
    p = panel_from_frames({"X": fr})
    m = sample_mask(p.index, str(p.index[0].date()), None)
    assert (forward_returns(p, 20, m).ret["X"].dropna() > 0).all()


def test_detect_from_above_simple():
    # 30 flat days at 100 (MA=100), spike to 110, then back to 101 -> one dip event
    close = np.r_[np.full(30, 100.0), [110.0], [101.0], np.full(10, 101.0)]
    p = panel_from_frames({"X": _frame(close)})
    ma = moving_average(p.close, 5)
    w = band_width(Band("pct", 0.03), ma, ma)
    ev = detect_events(p.close, ma, w, "from_above", 0)["X"]
    assert ev.sum() >= 1
    assert ev.iloc[31]  # first day back inside the band
    assert not ev.iloc[30]


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
    pos = bt.positions(np.array([9]), 3, c.size)
    assert np.flatnonzero(pos).tolist() == [10, 11, 12]
    r = bt.strategy_returns(pos, o, c, cost=0.0)
    assert r[10] == pytest.approx(0.005)
    assert r[11] == pytest.approx(0.01)
    assert r[9] == 0 and r[13] == 0
    r_cost = bt.strategy_returns(pos, o, c, cost=0.001)
    assert r_cost.sum() == pytest.approx(r.sum() - 0.002)


def test_overlapping_events_extend_not_double():
    pos = bt.positions(np.array([5, 7]), 4, 30)
    assert np.flatnonzero(pos).tolist() == list(range(6, 12))


def test_validate_rejects_unadjusted_split():
    close = np.r_[np.full(50, 400.0), np.full(50, 100.0)]
    with pytest.raises(ValueError, match="split"):
        validate(_frame(close), "X")


def test_config_fails_fast():
    with pytest.raises(ValueError):
        Band("pct", -0.1)
    with pytest.raises(ValueError):
        Config(ma_type="wma")
