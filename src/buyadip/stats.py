"""Inference for the event study.

Primary test = circular time-shift placebo.
  Shift the whole event calendar by s bars (same s for every ticker), recompute the mean
  excess return, repeat for every admissible s. Because the shifted calendar keeps
  (a) the number of events, (b) their clustering in time (e.g. 12 tickers all dipping in
  Mar-2020 or 2022), and (c) the overlap structure of h-day windows, the null distribution
  automatically accounts for overlapping returns and cross-sectional correlation, which a
  naive t-test would overstate by roughly sqrt(h) and sqrt(#tickers per date).

All shifts are evaluated at once with FFT circular cross-correlation (exact, O(T log T)).
"""
from __future__ import annotations

import numpy as np


def _circ_xcorr(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """c[s] = sum_t x[t] * e[(t - s) mod T], for all s, column-wise summed.

    x, e: (T, N). Returns (T,).
    """
    fx = np.fft.rfft(x, axis=0)
    fe = np.fft.rfft(e, axis=0)
    # sum_t x[t] e[t-s]  <=>  ifft( X * conj(E) )[s] with this sign convention
    c = np.fft.irfft(fx * np.conj(fe), n=x.shape[0], axis=0)
    return c.sum(axis=1)


def shift_placebo(excess: np.ndarray, events: np.ndarray, min_gap: int) -> dict:
    """excess: (T, N) float with NaN where undefined; events: (T, N) bool."""
    if excess.shape != events.shape:
        raise ValueError("shape mismatch")
    T = excess.shape[0]
    valid = np.isfinite(excess)
    x0 = np.where(valid, excess, 0.0)
    e = events.astype(float)
    s_sum = _circ_xcorr(x0, e)
    s_cnt = np.rint(_circ_xcorr(valid.astype(float), e))

    obs_cnt = s_cnt[0]
    if obs_cnt < 1:
        return dict(n=0, mean=np.nan, z=np.nan, p=np.nan, null_mean=np.nan, null_sd=np.nan, n_null=0)
    obs = s_sum[0] / obs_cnt

    s = np.arange(T)
    ok = (np.minimum(s, T - s) >= min_gap) & (s_cnt >= max(5, 0.5 * obs_cnt))
    null = s_sum[ok] / s_cnt[ok]
    if null.size < 30:
        return dict(n=int(obs_cnt), mean=obs, z=np.nan, p=np.nan,
                    null_mean=np.nan, null_sd=np.nan, n_null=int(null.size))
    mu, sd = null.mean(), null.std(ddof=1)
    p = (1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (1 + null.size)
    return dict(n=int(obs_cnt), mean=obs, z=(obs - mu) / sd if sd > 0 else np.nan,
                p=float(p), null_mean=mu, null_sd=sd, n_null=int(null.size))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and are excluded from m."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q
