"""Command line entry point.

    buyadip fetch                 # download/refresh price cache into data/
    buyadip run                   # full grid -> results/
    buyadip run --start 2010-01-01 --tickers SPY QQQ   # robustness variants
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="buyadip")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--tickers", nargs="+")
        p.add_argument("--data-start")
        p.add_argument("--data-dir", type=Path)
        if name == "run":
            p.add_argument("--start", help="sample start (default 2020-01-01)")
            p.add_argument("--end")
            p.add_argument("--ma-type", choices=["sma", "ema"])
            p.add_argument("--results-dir", type=Path)
            p.add_argument("--n-random", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = Config()
    upd = {}
    if a.tickers:
        upd["tickers"] = tuple(t.upper() for t in a.tickers)
    if a.data_start:
        upd["data_start"] = a.data_start
    if a.data_dir:
        upd["data_dir"] = a.data_dir
    if a.cmd == "run":
        for arg, key in [("start", "sample_start"), ("end", "sample_end"), ("ma_type", "ma_type"),
                         ("results_dir", "results_dir"), ("n_random", "n_random_strategies")]:
            if getattr(a, arg) is not None:
                upd[key] = getattr(a, arg)
    cfg = replace(cfg, **upd)

    if a.cmd == "fetch":
        from .data import fetch
        fetch(cfg.tickers, cfg.data_start, cfg.data_dir)
        return 0

    from . import analysis, report
    from .data import load_panel
    panel = load_panel(cfg.tickers, cfg.data_dir)
    tables = analysis.run(panel, cfg)
    path = report.write(tables, cfg.results_dir)
    logging.info("report written to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
