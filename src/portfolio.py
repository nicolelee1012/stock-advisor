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


def _apply_caps(weights: dict, sector_map: dict, max_pos, max_sector):
    """Enforce sector and position caps via water-filling.

    Excess from a capped sector/name is redistributed ONLY to under-cap
    names (not renormalized globally), which avoids the pathology of piling
    weight onto a lone minority-sector name. Sum stays ~1.
    """
    w = _normalize(dict(weights))

    # ---- Sector cap: scale over-cap sectors down, give excess to other sectors.
    for _ in range(10):
        sec_tot = {}
        for k, v in w.items():
            sec_tot[sector_map.get(k, "Unknown")] = sec_tot.get(
                sector_map.get(k, "Unknown"), 0.0) + v
        over = {s for s, t in sec_tot.items() if t > max_sector + 1e-9}
        if not over:
            break
        excess = 0.0
        for k in w:
            if sector_map.get(k, "Unknown") in over:
                scale = max_sector / sec_tot[sector_map.get(k, "Unknown")]
                excess += w[k] * (1 - scale)
                w[k] *= scale
        under = [k for k in w if sector_map.get(k, "Unknown") not in over]
        tot = sum(w[k] for k in under)
        if tot <= 0:
            break
        for k in under:
            w[k] += excess * w[k] / tot

    # ---- Position cap: clip over-cap names, redistribute to under-cap names.
    for _ in range(20):
        over = {k: v for k, v in w.items() if v > max_pos + 1e-9}
        if not over:
            break
        excess = sum(v - max_pos for v in over.values())
        for k in over:
            w[k] = max_pos
        under = [k for k, v in w.items() if v < max_pos - 1e-9]
        tot = sum(w[k] for k in under)
        if tot <= 0:
            break
        for k in under:
            w[k] += excess * w[k] / tot

    return _normalize(w)


def get_profile(profile=None) -> dict:
    """Resolve a profile name (or dict) to its parameter dict."""
    if isinstance(profile, dict):
        return profile
    return config.PROFILES[profile or config.DEFAULT_PROFILE]


def build_weights(picks: pd.DataFrame, sector_map: dict = None,
                  profile=None, cfg=config) -> dict:
    """picks: DataFrame with at least [ticker, vol_1m], already top-N selected.

    `profile` selects the construction style (see config.PROFILES). Returns
    {ticker: weight}; weights sum to the invested fraction (<= 1.0), and the
    special key "__cash__" holds the uninvested residual.
    """
    sector_map = sector_map or {}
    p = get_profile(profile)
    if picks.empty:
        return {"__cash__": 1.0}

    vols = picks["vol_1m"].clip(lower=VOL_FLOOR).fillna(picks["vol_1m"].median())
    if p["scheme"] == "inverse_vol":
        raw = {t: 1.0 / v for t, v in zip(picks["ticker"], vols)}
    else:  # equal weight
        raw = {t: 1.0 for t in picks["ticker"]}
    weights = _apply_caps(_normalize(raw), sector_map,
                          p["max_position"], p["max_sector"])

    # Volatility targeting: scale total exposure toward the profile's target.
    # Portfolio vol is far below the average single-name vol because of
    # diversification. Approximate it with an assumed average pairwise
    # correlation rho:  port_vol ~ wavg_vol * sqrt(rho + (1-rho)/n).
    vol_lookup = dict(zip(picks["ticker"], vols))
    wavg_vol = sum(weights[t] * vol_lookup.get(t, p["target_vol"]) for t in weights)
    n = max(len(weights), 1)
    diversification = np.sqrt(cfg.ASSUMED_CORR + (1 - cfg.ASSUMED_CORR) / n)
    port_vol_est = wavg_vol * diversification
    if port_vol_est > 0:
        exposure = np.clip(p["target_vol"] / port_vol_est, p["min_exposure"], 1.0)
    else:
        exposure = 1.0

    out = {t: w * exposure for t, w in weights.items()}
    out["__cash__"] = max(0.0, 1.0 - sum(out.values()))
    return out
