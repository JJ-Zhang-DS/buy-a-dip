"""Orchestrates the full grid: signals -> event study -> placebo -> FDR -> backtest."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import Config
from .data import Panel
from .events import forward_returns, sample_mask
from .signals import atr, band_width, detect_events, moving_average, regime_above_ma
from .stats import bh_fdr, shift_placebo

log = logging.getLogger(__name__)


def _cell_stats(mask: np.ndarray, fw: dict) -> dict:
    n = int(mask.sum())
    if n == 0:
        return dict(n=0)
    x = fw["X"][mask]
    r = fw["R"][mask]
    return dict(
        n=n,
        mean_ret=float(r.mean()),
        mean_base=float(fw["B"][mask].mean()),
        mean_excess=float(x.mean()),
        median_excess=float(np.median(x)),
        hit_rate=float((r > 0).mean()),
        hit_rate_base=float(fw["BH"][mask].mean()),
        mae_mean=float(fw["MAE"][mask].mean()),
        mae_p10=float(np.quantile(fw["MAE"][mask], 0.10)),
        mfe_mean=float(fw["MFE"][mask].mean()),
    )


def run(panel: Panel, cfg: Config) -> dict[str, pd.DataFrame]:
    idx = panel.index
    smask = sample_mask(idx, cfg.sample_start, cfg.sample_end)
    if smask.sum() < 250:
        raise ValueError("sample shorter than one year")
    s_idx = idx[smask.values]
    years = s_idx.year.to_numpy()
    tickers = panel.tickers
    T, N = smask.sum(), len(tickers)
    log.info("sample %s -> %s : %d bars x %d tickers", s_idx[0].date(), s_idx[-1].date(), T, N)

    regime = regime_above_ma(panel.close[cfg.regime_ticker], cfg.regime_ma)[smask.values].to_numpy()

    # Forward-return matrices per horizon (sample rows only)
    fws = {}
    for h in cfg.horizons:
        f = forward_returns(panel, h, smask)
        R = f.ret.to_numpy()
        base_hit = (f.ret > 0).where(f.ret.notna()).groupby(f.ret.index.year).transform("mean")
        fws[h] = dict(R=R, X=f.excess.to_numpy(), B=f.base.to_numpy(), BH=base_hit.to_numpy(),
                      MAE=f.mae.to_numpy(), MFE=f.mfe.to_numpy(), valid=np.isfinite(R))

    atr_df = atr(panel.high, panel.low, panel.close, cfg.atr_window)
    o_np = panel.adj_open[smask.values].to_numpy()
    c_np = panel.adj_close[smask.values].to_numpy()
    cost = cfg.cost_bps_per_side / 1e4
    rng = np.random.default_rng(cfg.seed)

    pooled, by_ticker, by_year, freq, bt_rows, port_rows = [], [], [], [], [], []

    for w in cfg.ma_windows:
        ma = moving_average(panel.close, w, cfg.ma_type)
        dist = (panel.close / ma - 1.0)[smask.values]
        for band in cfg.bands:
            width = band_width(band, ma, atr_df)
            inside = (dist.abs() <= width[smask.values]).to_numpy()
            ev_sets = {}
            for direction in cfg.directions:
                E = detect_events(panel.close, ma, width, direction, cfg.cooldown)[smask.values].to_numpy()
                ev_sets[direction] = E
            yrs = T / bt.TRADING_DAYS
            for j, t in enumerate(tickers):
                freq.append(dict(ma=w, band=band.label, ticker=t,
                                 pct_days_in_zone=float(np.nanmean(inside[:, j])),
                                 **{f"events_per_year_{d}": float(E[:, j].sum() / yrs) for d, E in ev_sets.items()}))

            for direction, E in ev_sets.items():
                key = dict(ma=w, band=band.label, direction=direction)
                for h in cfg.horizons:
                    fw = fws[h]
                    m = E & fw["valid"]
                    row = dict(**key, h=h, **_cell_stats(m, fw))
                    pl = shift_placebo(fw["X"], E, h + cfg.placebo_min_gap)
                    row.update(n_tickers=int(m.any(axis=0).sum()), n_dates=int(m.any(axis=1).sum()),
                               z=pl["z"], p=pl["p"], null_sd=pl["null_sd"])
                    for name, rmask in (("up", regime == 1), ("down", regime == 0)):
                        mm = m & rmask[:, None]
                        row[f"n_regime_{name}"] = int(mm.sum())
                        row[f"excess_regime_{name}"] = float(fw["X"][mm].mean()) if mm.any() else np.nan
                    pooled.append(row)

                    for y in np.unique(years):
                        my = m & (years == y)[:, None]
                        if my.any():
                            by_year.append(dict(**key, h=h, year=int(y), n=int(my.sum()),
                                                mean_excess=float(fw["X"][my].mean()),
                                                mean_ret=float(fw["R"][my].mean())))

                    for j, t in enumerate(tickers):
                        mj = m[:, [j]]
                        rowt = dict(**key, h=h, ticker=t, **_cell_stats(mj, {k: v[:, [j]] for k, v in fw.items()}))
                        plj = shift_placebo(fw["X"][:, [j]], E[:, [j]], h + cfg.placebo_min_gap)
                        rowt.update(z=plj["z"], p=plj["p"])
                        by_ticker.append(rowt)

                    # Strategy simulation: only the "buy the dip" direction is a long strategy.
                    if direction != "from_above":
                        continue
                    strat_cols, bh_cols = [], []
                    for j, t in enumerate(tickers):
                        ok = np.isfinite(c_np[:, j]) & np.isfinite(o_np[:, j])
                        if ok.sum() < 250:
                            continue
                        rows_ = np.flatnonzero(E[ok, j])
                        res, sr, br = bt.run_ticker(rows_, h, o_np[ok, j], c_np[ok, j], cost,
                                                    cfg.n_random_strategies, rng)
                        bt_rows.append(dict(**key, h=h, ticker=t, **res))
                        sr_full = np.zeros(T); sr_full[ok] = sr
                        br_full = np.zeros(T); br_full[ok] = br
                        strat_cols.append(sr_full); bh_cols.append(br_full)
                    # Equal-weight portfolio: 1/N of capital per ticker sleeve, sleeve idle = cash.
                    if strat_cols:
                        ps = np.mean(strat_cols, axis=0)
                        pb = np.mean(bh_cols, axis=0)
                        port_rows.append(dict(**key, h=h,
                                              **{f"strat_{k}": v for k, v in bt.metrics(ps).items()},
                                              **{f"bh_{k}": v for k, v in bt.metrics(pb).items()}))
            log.info("MA%d %s done", w, band.label)

    pooled_df = pd.DataFrame(pooled)
    pooled_df["q_fdr"] = bh_fdr(pooled_df["p"].to_numpy())
    et_df = pd.DataFrame(by_ticker)
    et_df["q_fdr"] = bh_fdr(et_df["p"].to_numpy())

    meta = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sample_start=str(s_idx[0].date()), sample_end=str(s_idx[-1].date()),
        n_bars=int(T), tickers=tickers,
        n_pooled_tests=int(pooled_df["p"].notna().sum()),
        n_ticker_tests=int(et_df["p"].notna().sum()),
        config={k: (str(v) if not isinstance(v, (int, float, str, list, tuple, type(None))) else v)
                for k, v in asdict(cfg).items()},
    )
    return dict(
        event_study_pooled=pooled_df,
        event_study_by_ticker=et_df,
        event_study_by_year=pd.DataFrame(by_year),
        signal_frequency=pd.DataFrame(freq),
        backtest_by_ticker=pd.DataFrame(bt_rows),
        backtest_portfolio=pd.DataFrame(port_rows),
        _meta=pd.DataFrame([{"json": json.dumps(meta, default=str)}]),
    )
