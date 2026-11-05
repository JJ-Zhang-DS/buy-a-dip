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

# Diverging blue (better than random entry) <-> red (worse), neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "excess", ["#a83232", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
TEXT = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
Q_STAR = 0.10  # FDR level for the significance marker


def _pct(x: float, nd: int = 2) -> str:
    return "" if pd.isna(x) else f"{x * 100:.{nd}f}%"


def heatmap(df: pd.DataFrame, rows: str, cols: str, title: str, path: Path,
            value: str = "mean_excess", q: str | None = "q_fdr") -> None:
    piv = df.pivot_table(index=rows, columns=cols, values=value, aggfunc="first")
    qpv = df.pivot_table(index=rows, columns=cols, values=q, aggfunc="first") if q else None
    npv = df.pivot_table(index=rows, columns=cols, values="n", aggfunc="first")
    vals = piv.to_numpy() * 100
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    lim = max(lim, 0.05)
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 2.2, 0.55 * piv.shape[0] + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            star = ""
            if qpv is not None and np.isfinite(qpv.iat[i, j]) and qpv.iat[i, j] < Q_STAR:
                star = "*"
            n = npv.iat[i, j]
            ink = "white" if abs(v) > 0.6 * lim else TEXT
            ax.text(j, i, f"{v:+.2f}{star}\nn={int(n)}", ha="center", va="center", fontsize=7.5, color=ink)
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns], color=TEXT)
    ax.set_yticks(range(piv.shape[0]), [str(r) for r in piv.index], color=TEXT)
    ax.set_xlabel(cols, color=MUTED)
    ax.set_ylabel(rows, color=MUTED)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("excess return vs same ticker-year random entry (%)", color=MUTED, fontsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write(tables: dict[str, pd.DataFrame], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    meta = json.loads(tables["_meta"]["json"].iat[0])
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    for name, df in tables.items():
        if not name.startswith("_"):
            df.to_csv(results_dir / f"{name}.csv", index=False, float_format="%.6g")

    P = tables["event_study_pooled"]
    F = tables["signal_frequency"]
    Y = tables["event_study_by_year"]
    BT = tables["backtest_by_ticker"]
    PF = tables["backtest_portfolio"]
    ET = tables["event_study_by_ticker"]
    bands = list(dict.fromkeys(P["band"]))
    figs = []
    for d in ["from_above", "from_below"]:
        for b in bands:
            sub = P[(P.direction == d) & (P.band == b)]
            if sub.empty:
                continue
            fn = f"heat_{d}_{b}.png"
            heatmap(sub, "ma", "h", f"{d.replace('_', ' ')} | band {b} | pooled {len(meta['tickers'])} tickers  (* = BH q<{Q_STAR})",
                    fig_dir / fn)
            figs.append((d, b, fn))
    # Per-ticker view for the user's proposed spec (5% band) and a tight band
    for b in [x for x in ["pct5", "pct2"] if x in bands]:
        for h in [20, 60]:
            sub = ET[(ET.direction == "from_above") & (ET.band == b) & (ET.h == h)]
            if not sub.empty:
                fn = f"heat_ticker_{b}_h{h}.png"
                heatmap(sub, "ticker", "ma", f"from above | {b} | h={h}d | per ticker (* = BH q<{Q_STAR} over all ticker cells)",
                        fig_dir / fn)
                figs.append(("ticker", f"{b} h{h}", fn))

    L: list[str] = []
    L.append("# Buy a Dip? - Results\n")
    L.append(f"Generated {meta['generated_utc']} | sample **{meta['sample_start']} -> {meta['sample_end']}** "
             f"({meta['n_bars']} bars) | tickers: {', '.join(meta['tickers'])}\n")
    L.append(f"Tests: {meta['n_pooled_tests']} pooled cells, {meta['n_ticker_tests']} per-ticker cells. "
             f"Significance = circular time-shift placebo, BH-FDR across all cells.\n")
    L.append("> Excess return = forward return after the signal minus the average forward return of a random "
             "entry day in the same ticker and calendar year. Entry at next open, exit at close of day h, "
             "total-return prices, no costs (costs enter only in the strategy section).\n")

    # 1. Scorecard
    L.append("## 1. Scorecard - dip from above (the hypothesis)\n")
    A = P[P.direction == "from_above"]
    sc = A.groupby("h").agg(cells=("n", "size"),
                            share_positive=("mean_excess", lambda s: (s > 0).mean()),
                            median_excess=("mean_excess", "median"),
                            ).reset_index()
    sig = A[A.q_fdr < Q_STAR]
    sc["sig_pos"] = sc["h"].map(sig[sig.mean_excess > 0].groupby("h").size()).fillna(0).astype(int)
    sc["sig_neg"] = sc["h"].map(sig[sig.mean_excess < 0].groupby("h").size()).fillna(0).astype(int)
    L.append("| horizon (days) | cells | share of cells with excess > 0 | median cell excess | significant + (q<0.10) | significant - |")
    L.append("|---|---|---|---|---|---|")
    for _, r in sc.iterrows():
        L.append(f"| {int(r.h)} | {int(r.cells)} | {r.share_positive:.0%} | {_pct(r.median_excess)} | "
                 f"{int(r.sig_pos)} | {int(r.sig_neg)} |")
    L.append("")

    top = A.sort_values("z", ascending=False).head(12)
    L.append("**Strongest 12 cells by placebo z (from above)**\n")
    L.append("| MA | band | h | n | raw mean | baseline | excess | hit | base hit | MAE p10 | z | q |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        L.append(f"| {r.ma} | {r.band} | {r.h} | {r.n} | {_pct(r.mean_ret)} | {_pct(r.mean_base)} | "
                 f"{_pct(r.mean_excess)} | {r.hit_rate:.0%} | {r.hit_rate_base:.0%} | {_pct(r.mae_p10, 1)} | "
                 f"{r.z:.2f} | {r.q_fdr:.2f} |")
    L.append("")

    # 2. Signal frequency
    L.append("## 2. How often does each definition fire?\n")
    L.append("Median across tickers. `in zone` = share of all days with the close inside the band "
             "(a band that is almost always 'near' carries no information).\n")
    fq = F.groupby(["ma", "band"]).agg(in_zone=("pct_days_in_zone", "median"),
                                       ev=("events_per_year_from_above", "median")).reset_index()
    pz = fq.pivot(index="ma", columns="band", values="in_zone")[bands]
    pe = fq.pivot(index="ma", columns="band", values="ev")[bands]
    L.append("| MA | " + " | ".join(f"{b} in-zone / events/yr" for b in bands) + " |")
    L.append("|---|" + "---|" * len(bands))
    for ma in pz.index:
        L.append(f"| {ma} | " + " | ".join(f"{pz.at[ma, b]:.0%} / {pe.at[ma, b]:.1f}" for b in bands) + " |")
    L.append("")

    # 3. Heatmaps
    L.append("## 3. Heatmaps (excess %, pooled)\n")
    for d, b, fn in figs:
        L.append(f"![{d} {b}](figures/{fn})\n")

    # 4. Regime
    L.append("## 4. Regime dependence (SPY above vs below its 200-day SMA at signal date)\n")
    R = A[A.band.isin(["pct3", "pct5", "atr1"]) & A.h.isin([20, 60, 120])]
    L.append("| MA | band | h | n up | excess up | n down | excess down |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in R.sort_values(["band", "h", "ma"]).iterrows():
        L.append(f"| {r.ma} | {r.band} | {r.h} | {r.n_regime_up} | {_pct(r.excess_regime_up)} | "
                 f"{r.n_regime_down} | {_pct(r.excess_regime_down)} |")
    L.append("")

    # 5. Year stability
    L.append("## 5. Year-by-year stability (from above, band pct5, excess %)\n")
    yy = Y[(Y.direction == "from_above") & (Y.band == "pct5") & Y.h.isin([20, 60])]
    if not yy.empty:
        pv = yy.pivot_table(index=["h", "ma"], columns="year", values="mean_excess")
        L.append("| h | MA | " + " | ".join(str(c) for c in pv.columns) + " |")
        L.append("|---|---|" + "---|" * len(pv.columns))
        for (h, ma), row in pv.iterrows():
            L.append(f"| {h} | {ma} | " + " | ".join(_pct(v) for v in row.values) + " |")
        L.append("")

    # 6. Strategy
    L.append("## 6. Tradable strategy (buy next open after dip, hold h days, 5 bps/side)\n")
    L.append("Equal-weight portfolio across all tickers (idle sleeve = cash at 0%) vs equal-weight buy & hold.\n")
    pf = PF[PF.band.isin(["pct2", "pct5", "atr1"])].sort_values(["band", "ma", "h"])
    L.append("| MA | band | h | CAGR | B&H CAGR | Sharpe | B&H Sharpe | MaxDD | B&H MaxDD |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in pf.iterrows():
        L.append(f"| {r.ma} | {r.band} | {r.h} | {_pct(r.strat_cagr, 1)} | {_pct(r.bh_cagr, 1)} | "
                 f"{r.strat_sharpe:.2f} | {r.bh_sharpe:.2f} | {_pct(r.strat_max_dd, 0)} | {_pct(r.bh_max_dd, 0)} |")
    L.append("")
    if not BT.empty:
        BT = BT.assign(beat_bh=BT.strat_sharpe > BT.bh_sharpe,
                       beat_rand=BT.sharpe_pctile_vs_random > 0.5,
                       top10=BT.sharpe_pctile_vs_random >= 0.9)
        g = BT.groupby("h").agg(specs=("beat_bh", "size"), beat_bh=("beat_bh", "mean"),
                                beat_random=("beat_rand", "mean"), top_decile=("top10", "mean"),
                                exposure=("strat_exposure", "median")).reset_index()
        L.append("Per ticker x spec (all MAs and bands): share where the dip strategy's Sharpe beats ...\n")
        L.append("| h | ticker-specs | ... buy & hold | ... median random entry (same #trades, same h) | in top 10% of random | median exposure |")
        L.append("|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            L.append(f"| {int(r.h)} | {int(r.specs)} | {r.beat_bh:.0%} | {r.beat_random:.0%} | {r.top_decile:.0%} | {r.exposure:.0%} |")
        L.append("\nUnder the null of no timing skill, 'beats median random' should be ~50% and 'top 10%' ~10%.\n")

    path = results_dir / "REPORT.md"
    path.write_text("\n".join(L))
    return path
