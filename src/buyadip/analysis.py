"""Orchestrates the full grid: signals -> event study -> placebo -> FDR -> backtest."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import Config, band_label
from .data import Panel
from .events import forward_returns, sample_mask
from .signals import atr, detect_events, moving_average, open_distance, regime_above_ma
from .stats import bh_fdr, shift_placebo

log = logging.getLogger(__name__)

BENCHMARKS = ("own", "spy")


def _cell_stats(mask: np.ndarray, fw: dict) -> dict:
    n = int(mask.sum())
    if n == 0:
        return dict(n=0)
    r = fw["R"][mask]
    out = dict(n=n, mean_ret=float(r.mean()), hit_rate=float((r > 0).mean()),
               hit_rate_base=float(np.nanmean(fw["HB"][mask])),
               mae_mean=float(fw["MAE"][mask].mean()),
               mae_p10=float(np.quantile(fw["MAE"][mask], 0.10)),
               mfe_mean=float(fw["MFE"][mask].mean()))
    for b in BENCHMARKS:
        x = fw[f"X_{b}"][mask]
        x = x[np.isfinite(x)]
        out[f"base_{b}"] = float(np.nanmean(fw[f"B_{b}"][mask])) if x.size else np.nan
        out[f"excess_{b}"] = float(x.mean()) if x.size else np.nan
        out[f"median_excess_{b}"] = float(np.median(x)) if x.size else np.nan
    return out


def _placebo(fw: dict, E: np.ndarray, min_gap: int) -> dict:
    out = {}
    for b in BENCHMARKS:
        pl = shift_placebo(fw[f"X_{b}"], E, min_gap)
        out[f"z_{b}"], out[f"p_{b}"] = pl["z"], pl["p"]
    return out


def run(panel: Panel, cfg: Config) -> dict[str, pd.DataFrame]:
    idx = panel.index
    smask = sample_mask(idx, cfg.sample_start, cfg.sample_end)
    if smask.sum() < 250:
        raise ValueError("sample shorter than one year")
    sv = smask.values
    s_idx = idx[sv]
    years = s_idx.year.to_numpy()
    tickers = panel.tickers
    T, N = int(sv.sum()), len(tickers)
    log.info("sample %s -> %s : %d bars x %d tickers", s_idx[0].date(), s_idx[-1].date(), T, N)

    regime = regime_above_ma(panel.close[cfg.market_ticker], cfg.regime_ma)[sv].to_numpy()

    fws = {}
    for h in cfg.horizons:
        f = forward_returns(panel, h, smask, cfg.baseline_window, cfg.market_ticker)
        R = f.ret.to_numpy()
        fws[h] = dict(R=R, X_own=f.excess_own.to_numpy(), X_spy=f.excess_spy.to_numpy(),
                      B_own=f.base_own.to_numpy(), B_spy=f.base_spy.to_numpy(),
                      HB=f.hit_base.to_numpy(), MAE=f.mae.to_numpy(), MFE=f.mfe.to_numpy(),
                      valid=np.isfinite(R))

    atr_df = atr(panel.high, panel.low, panel.close, cfg.atr_window)
    o_np = panel.adj_open[sv].to_numpy()
    c_np = panel.adj_close[sv].to_numpy()
    listed = np.isfinite(o_np) & np.isfinite(c_np)
    cost = cfg.cost_bps_per_side / 1e4
    rng = np.random.default_rng(cfg.seed)
    yrs_listed = listed.sum(axis=0) / bt.TRADING_DAYS

    pooled, by_ticker, by_year, freq, bt_rows, port_rows = [], [], [], [], [], []

    for w in cfg.ma_windows:
        ma = moving_average(panel.close, w, cfg.ma_type)
        for k in cfg.atr_bands:
            label = band_label(k)
            dist_open, _, width = open_distance(panel.open, panel.close, ma, atr_df, k)
            inside = (dist_open.abs() <= width)[sv].to_numpy()
            warm = width[sv].notna().to_numpy()
            ev_sets = {d: detect_events(panel.open, panel.close, ma, atr_df, k, d, cfg.cooldown)[sv].to_numpy()
                       for d in cfg.directions}
            for j, t in enumerate(tickers):
                freq.append(dict(ma=w, band=label, ticker=t,
                                 pct_days_in_zone=float(inside[warm[:, j], j].mean()) if warm[:, j].any() else np.nan,
                                 **{f"events_per_year_{d}": float(E[:, j].sum() / max(yrs_listed[j], 1e-9))
                                    for d, E in ev_sets.items()}))

            for direction, E in ev_sets.items():
                key = dict(ma=w, band=label, direction=direction)
                for h in cfg.horizons:
                    fw = fws[h]
                    m = E & fw["valid"]
                    row = dict(**key, h=h, **_cell_stats(m, fw), **_placebo(fw, E, h + cfg.placebo_min_gap))
                    row.update(n_tickers=int(m.any(axis=0).sum()), n_dates=int(m.any(axis=1).sum()))
                    for name, rmask in (("up", regime == 1), ("down", regime == 0)):
                        mm = m & rmask[:, None]
                        row[f"n_regime_{name}"] = int(mm.sum())
                        row[f"excess_own_regime_{name}"] = float(fw["X_own"][mm].mean()) if mm.any() else np.nan
                    pooled.append(row)

                    for y in np.unique(years):
                        my = m & (years == y)[:, None]
                        if my.any():
                            by_year.append(dict(**key, h=h, year=int(y), n=int(my.sum()),
                                                excess_own=float(fw["X_own"][my].mean()),
                                                excess_spy=float(np.nanmean(fw["X_spy"][my])),
                                                mean_ret=float(fw["R"][my].mean())))

                    for j, t in enumerate(tickers):
                        sub = {kk: v[:, [j]] for kk, v in fw.items()}
                        rowt = dict(**key, h=h, ticker=t, **_cell_stats(m[:, [j]], sub),
                                    **_placebo(sub, E[:, [j]], h + cfg.placebo_min_gap))
                        by_ticker.append(rowt)

                    if direction != "from_above":  # only the dip side is a long strategy
                        continue
                    S = np.full((T, N), np.nan)
                    B = np.full((T, N), np.nan)
                    for j, t in enumerate(tickers):
                        ok = listed[:, j]
                        if ok.sum() < 250:
                            continue
                        rows_ = np.flatnonzero(E[ok, j])
                        res, sr, br = bt.run_ticker(rows_, h, o_np[ok, j], c_np[ok, j], cost,
                                                    cfg.n_random_strategies, rng)
                        bt_rows.append(dict(**key, h=h, ticker=t, **res))
                        S[ok, j], B[ok, j] = sr, br
                    # Equal weight across tickers listed that day; idle sleeve = cash at 0%.
                    port_rows.append(dict(**key, h=h,
                                          **{f"strat_{kk}": v for kk, v in bt.metrics(np.nanmean(S, axis=1)).items()},
                                          **{f"bh_{kk}": v for kk, v in bt.metrics(np.nanmean(B, axis=1)).items()}))
            log.info("MA%d %s done", w, label)

    pooled_df = pd.DataFrame(pooled)
    et_df = pd.DataFrame(by_ticker)
    for b in BENCHMARKS:
        pooled_df[f"q_{b}"] = bh_fdr(pooled_df[f"p_{b}"].to_numpy())
        et_df[f"q_{b}"] = bh_fdr(et_df[f"p_{b}"].to_numpy())

    meta = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sample_start=str(s_idx[0].date()), sample_end=str(s_idx[-1].date()),
        n_bars=T, tickers=tickers,
        first_bar={t: str(s_idx[np.argmax(listed[:, j])].date()) for j, t in enumerate(tickers)},
        n_pooled_tests=int(pooled_df["p_own"].notna().sum()),
        n_ticker_tests=int(et_df["p_own"].notna().sum()),
        config={k: (v if isinstance(v, (int, float, str, list, tuple, type(None))) else str(v))
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
