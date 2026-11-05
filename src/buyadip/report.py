"""Writes CSVs, heatmaps and results/REPORT.md from the analysis tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# Diverging blue (better than benchmark) <-> red (worse), neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "excess", ["#a83232", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
Q_STAR = 0.10
PERIODS = [(2000, 2002, "2000-02 dot-com bust"), (2003, 2007, "2003-07"), (2008, 2009, "2008-09 GFC"),
           (2010, 2019, "2010-19"), (2020, 2026, "2020-26")]
BENCH_NAME = {"own": "same ticker, any day (+/-126 bars)", "spy": "SPY, same dates"}


def _pct(x: float, nd: int = 2) -> str:
    return "" if pd.isna(x) else f"{x * 100:+.{nd}f}%"


def heatmap(df: pd.DataFrame, rows: str, cols: str, value: str, q: str, title: str, cbar: str,
            path: Path) -> None:
    piv = df.pivot_table(index=rows, columns=cols, values=value, aggfunc="first")
    qpv = df.pivot_table(index=rows, columns=cols, values=q, aggfunc="first").reindex_like(piv)
    npv = df.pivot_table(index=rows, columns=cols, values="n", aggfunc="first").reindex_like(piv)
    vals = piv.to_numpy() * 100
    lim = max(np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0, 0.05)
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 2.4, 0.55 * piv.shape[0] + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            star = "*" if np.isfinite(qpv.iat[i, j]) and qpv.iat[i, j] < Q_STAR else ""
            ink = "white" if abs(v) > 0.6 * lim else TEXT
            ax.text(j, i, f"{v:+.2f}{star}\nn={int(npv.iat[i, j])}", ha="center", va="center",
                    fontsize=7.5, color=ink)
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns], color=TEXT)
    ax.set_yticks(range(piv.shape[0]), [str(r) for r in piv.index], color=TEXT)
    ax.set_xlabel(cols, color=MUTED)
    ax.set_ylabel(rows, color=MUTED)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar, color=MUTED, fontsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _grid_table(sub: pd.DataFrame, value: str, q: str) -> list[str]:
    piv = sub.pivot_table(index="ma", columns="h", values=value, aggfunc="first")
    qpv = sub.pivot_table(index="ma", columns="h", values=q, aggfunc="first")
    L = ["| MA \\ hold | " + " | ".join(f"{c}d" for c in piv.columns) + " |",
         "|---|" + "---|" * piv.shape[1]]
    for ma in piv.index:
        cells = []
        for h in piv.columns:
            v, qq = piv.at[ma, h], qpv.at[ma, h]
            s = _pct(v, 1)
            cells.append(f"**{s}**" if np.isfinite(qq) and qq < Q_STAR else s)
        L.append(f"| {ma} | " + " | ".join(cells) + " |")
    return L


def write(tables: dict[str, pd.DataFrame], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()
    meta = json.loads(tables["_meta"]["json"].iat[0])
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    for name, df in tables.items():
        if not name.startswith("_"):
            df.to_csv(results_dir / f"{name}.csv", index=False, float_format="%.6g")

    P, F, Y = tables["event_study_pooled"], tables["signal_frequency"], tables["event_study_by_year"]
    BT, PF, ET = tables["backtest_by_ticker"], tables["backtest_portfolio"], tables["event_study_by_ticker"]
    bands = list(dict.fromkeys(P["band"]))
    nt = len(meta["tickers"])

    L: list[str] = ["# Buy a Dip? - Results\n"]
    L.append(f"Generated {meta['generated_utc']} | sample **{meta['sample_start']} -> {meta['sample_end']}** "
             f"({meta['n_bars']} bars) | {nt} tickers: {', '.join(meta['tickers'])}\n")
    late = {t: d for t, d in meta["first_bar"].items() if d > meta["sample_start"]}
    if late:
        L.append("Later listings enter when their MA is warm: " +
                 ", ".join(f"{t} {d[:4]}" for t, d in late.items()) + ".\n")
    L.append(f"Tests: {meta['n_pooled_tests']} pooled cells x 2 benchmarks, {meta['n_ticker_tests']} per-ticker "
             "cells. Significance = circular time-shift placebo, BH-FDR within each benchmark. "
             f"**Bold / \\*** = q < {Q_STAR}.\n")
    L.append("> Signal: today's **open** within +/-k ATR of yesterday's MA, after yesterday's close was outside "
             "the zone. Buy at that open, sell at the open h trading days later, total-return prices.\n"
             "> Benchmarks (not timing the market): **own** = average h-day return of buying the same ticker on "
             "any day within +/-126 bars; **spy** = SPY over the identical dates.\n")

    # 1. Scorecard
    L.append("## 1. Scorecard - dip from above\n")
    A = P[P.direction == "from_above"]
    L.append("| hold | cells | own: share > 0 | own: median | own: sig + / - | spy: share > 0 | spy: median | spy: sig + / - |")
    L.append("|---|---|---|---|---|---|---|---|")
    for h, g in A.groupby("h"):
        parts = [f"{h}d", str(len(g))]
        for b in ("own", "spy"):
            sig = g[g[f"q_{b}"] < Q_STAR]
            parts += [f"{(g[f'excess_{b}'] > 0).mean():.0%}", _pct(g[f"excess_{b}"].median()),
                      f"{(sig[f'excess_{b}'] > 0).sum()} / {(sig[f'excess_{b}'] < 0).sum()}"]
        L.append("| " + " | ".join(parts) + " |")
    L.append("")

    # 2. Grids + heatmaps
    L.append("## 2. Excess return grids\n")
    for d in ("from_above", "from_below"):
        for b in ("own", "spy"):
            for band in bands:
                sub = P[(P.direction == d) & (P.band == band)]
                L.append(f"**{d.replace('_', ' ')} | {band} | vs {BENCH_NAME[b]}**\n")
                L += _grid_table(sub, f"excess_{b}", f"q_{b}")
                L.append("")
                fn = f"heat_{d}_{band}_{b}.png"
                heatmap(sub, "ma", "h", f"excess_{b}", f"q_{b}",
                        f"{d.replace('_', ' ')} | {band} | pooled {nt} tickers | vs {b}  (* q<{Q_STAR})",
                        f"excess vs {BENCH_NAME[b]} (%)", fig_dir / fn)
    for h in (20, 120):
        sub = ET[(ET.direction == "from_above") & (ET.band == "atr1") & (ET.h == h)]
        if not sub.empty:
            fn = f"heat_ticker_atr1_h{h}_own.png"
            heatmap(sub, "ticker", "ma", "excess_own", "q_own",
                    f"from above | atr1 | {h}d | per ticker vs own (* q<{Q_STAR} over all ticker cells)",
                    "excess vs same-ticker any day (%)", fig_dir / fn)
            L.append(f"![per ticker {h}d](figures/{fn})\n")

    # 3. Top cells
    L.append("## 3. Strongest dip cells (by placebo z, own benchmark)\n")
    L.append("| MA | band | hold | n | raw | own base | excess own | excess spy | hit | base hit | MAE p10 | z | q |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in A.sort_values("z_own", ascending=False).head(10).iterrows():
        L.append(f"| {r.ma} | {r.band} | {r.h} | {r.n} | {_pct(r.mean_ret, 1)} | {_pct(r.base_own, 1)} | "
                 f"{_pct(r.excess_own)} | {_pct(r.excess_spy)} | {r.hit_rate:.0%} | {r.hit_rate_base:.0%} | "
                 f"{_pct(r.mae_p10, 1)} | {r.z_own:.2f} | {r.q_own:.2f} |")
    L.append("")

    # 4. Frequency
    L.append("## 4. How often does each definition fire? (median across tickers)\n")
    L.append("| MA | " + " | ".join(f"{b}: % days in zone / dips per yr" for b in bands) + " |")
    L.append("|---|" + "---|" * len(bands))
    fq = F.groupby(["ma", "band"]).agg(z=("pct_days_in_zone", "median"), e=("events_per_year_from_above", "median"))
    for ma in sorted(F.ma.unique()):
        L.append(f"| {ma} | " + " | ".join(f"{fq.at[(ma, b), 'z']:.0%} / {fq.at[(ma, b), 'e']:.1f}" for b in bands) + " |")
    L.append("")

    # 5. Regime + periods
    L.append("## 5. Regime: SPY above vs below its 200-day SMA (excess vs own, dip from above, atr1)\n")
    R = A[A.band == "atr1"].sort_values(["h", "ma"])
    L.append("| MA | hold | n up | excess up | n down | excess down |")
    L.append("|---|---|---|---|---|---|")
    for _, r in R[R.h.isin([20, 60, 120, 250])].iterrows():
        L.append(f"| {r.ma} | {r.h} | {r.n_regime_up} | {_pct(r.excess_own_regime_up)} | "
                 f"{r.n_regime_down} | {_pct(r.excess_own_regime_down)} |")
    L.append("")
    L.append("## 6. Stability by period (excess vs own, dip from above, atr1)\n")
    yy = Y[(Y.direction == "from_above") & (Y.band == "atr1")].copy()
    if not yy.empty:
        yy["period"] = None
        for a, b, name in PERIODS:
            yy.loc[yy.year.between(a, b), "period"] = name
        yy["w"] = yy.excess_own * yy.n
        g = yy.groupby(["h", "ma", "period"]).agg(w=("w", "sum"), n=("n", "sum"))
        g = (g.w / g.n).unstack("period")
        cols = [p[2] for p in PERIODS if p[2] in g.columns]
        L.append("| hold | MA | " + " | ".join(cols) + " |")
        L.append("|---|---|" + "---|" * len(cols))
        for (h, ma), row in g.iterrows():
            if h in (20, 60, 120, 250):
                L.append(f"| {h} | {ma} | " + " | ".join(_pct(row[c], 1) for c in cols) + " |")
        L.append("")

    # 7. Strategy
    L.append("## 7. Tradable strategy (buy at the dip open, sell at open h days later, 5 bps/side)\n")
    L.append("Equal-weight across tickers listed that day (idle sleeve = cash at 0%) vs equal-weight buy & hold.\n")
    L.append("| MA | band | hold | CAGR | B&H CAGR | Sharpe | B&H Sharpe | MaxDD | B&H MaxDD |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in PF.sort_values(["band", "ma", "h"]).iterrows():
        L.append(f"| {r.ma} | {r.band} | {r.h} | {_pct(r.strat_cagr, 1)} | {_pct(r.bh_cagr, 1)} | "
                 f"{r.strat_sharpe:.2f} | {r.bh_sharpe:.2f} | {_pct(r.strat_max_dd, 0)} | {_pct(r.bh_max_dd, 0)} |")
    L.append("")
    if not BT.empty:
        BT = BT.assign(beat_bh=BT.strat_sharpe > BT.bh_sharpe, beat_rand=BT.sharpe_pctile_vs_random > 0.5,
                       top10=BT.sharpe_pctile_vs_random >= 0.9)
        g = BT.groupby("h").agg(specs=("beat_bh", "size"), beat_bh=("beat_bh", "mean"),
                                beat_random=("beat_rand", "mean"), top_decile=("top10", "mean"),
                                exposure=("strat_exposure", "median")).reset_index()
        L.append("Per ticker x spec: share where the dip strategy's Sharpe beats ...\n")
        L.append("| hold | ticker-specs | buy & hold | median random entry (same #trades, same hold) | top 10% of random | median exposure |")
        L.append("|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            L.append(f"| {int(r.h)} | {int(r.specs)} | {r.beat_bh:.0%} | {r.beat_random:.0%} | "
                     f"{r.top_decile:.0%} | {r.exposure:.0%} |")
        L.append("\nNo timing skill => 'beats median random' ~50% and 'top 10%' ~10%.\n")

    path = results_dir / "REPORT.md"
    path.write_text("\n".join(L))
    return path
