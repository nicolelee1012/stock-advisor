"""Portfolio construction / risk controls.

Turns a ranked list of picks into position WEIGHTS, applying risk controls that
target better risk-adjusted return instead of just more return:

  1. Inverse-volatility weighting   calmer names get more capital than jumpy ones
  2. Per-position cap               no single stock dominates
  3. Sector cap                     no single GICS sector dominates
  4. Volatility targeting           scale total exposure toward a vol target; the
                                    remainder sits in cash (return 0). This is the
                                    main drawdown tamer — in turbulent regimes the
                                    book de-risks automatically.

Simplification (documented honestly): we estimate portfolio vol as the
weight-weighted average of single-name vols, which ignores diversification and
so is conservative (tends to under-lever). Good enough without a covariance
matrix; the backtest tunes TARGET_VOL empirically.

The SAME function is used by the backtest and live prediction, so what we test
is what we run.
"""

import numpy as np
import pandas as pd

import config

VOL_FLOOR = 0.05  # avoid divide-by-zero / absurd weights on ultra-low-vol names


def _normalize(w: dict) -> dict:
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else w


def _apply_caps(weights: dict, sector_map: dict, max_pos, max_sector, iters=5):
    """Iterative water-filling: clip position & sector caps, renormalize, repeat."""
    w = dict(weights)
    for _ in range(iters):
        w = _normalize(w)
        # Position cap
        w = {k: min(v, max_pos) for k, v in w.items()}
        # Sector cap: scale down any sector that exceeds the cap.
        sector_tot = {}
        for k, v in w.items():
            sector_tot.setdefault(sector_map.get(k, "Unknown"), 0.0)
            sector_tot[sector_map.get(k, "Unknown")] += v
        for k in list(w):
            sec = sector_map.get(k, "Unknown")
            if sector_tot[sec] > max_sector and sector_tot[sec] > 0:
                w[k] *= max_sector / sector_tot[sec]
    return _normalize(w)


def build_weights(picks: pd.DataFrame, sector_map: dict = None, cfg=config) -> dict:
    """picks: DataFrame with at least [ticker, vol_1m], already top-N selected.

    Returns {ticker: weight}. Weights sum to the invested fraction (<= 1.0);
    1 - sum is implicit cash. Includes a special key "__cash__" for clarity.
    """
    sector_map = sector_map or {}
    if picks.empty:
        return {"__cash__": 1.0}

    vols = picks["vol_1m"].clip(lower=VOL_FLOOR).fillna(picks["vol_1m"].median())
    raw = {t: 1.0 / v for t, v in zip(picks["ticker"], vols)}
    weights = _apply_caps(_normalize(raw), sector_map,
                          cfg.MAX_POSITION_WEIGHT, cfg.MAX_SECTOR_WEIGHT)

    # Volatility targeting: scale total exposure toward TARGET_VOL.
    # Portfolio vol is far below the average single-name vol because of
    # diversification. Approximate it with an assumed average pairwise
    # correlation rho:  port_vol ~ wavg_vol * sqrt(rho + (1-rho)/n).
    vol_lookup = dict(zip(picks["ticker"], vols))
    wavg_vol = sum(weights[t] * vol_lookup.get(t, cfg.TARGET_VOL) for t in weights)
    n = max(len(weights), 1)
    diversification = np.sqrt(cfg.ASSUMED_CORR + (1 - cfg.ASSUMED_CORR) / n)
    port_vol_est = wavg_vol * diversification
    if port_vol_est > 0:
        exposure = np.clip(cfg.TARGET_VOL / port_vol_est, cfg.MIN_EXPOSURE, 1.0)
    else:
        exposure = 1.0

    out = {t: w * exposure for t, w in weights.items()}
    out["__cash__"] = max(0.0, 1.0 - sum(out.values()))
    return out
