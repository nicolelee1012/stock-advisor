"""Point-in-time fundamental features derived from EDGAR facts.

The discipline: for any as_of date T, we use only disclosures whose FIRST filing
date is <= T. We collapse each (ticker, concept, period_end) to its earliest
disclosure (when the market first learned it), then derive:

  eps_yoy            year-over-year change in quarterly diluted EPS
  rev_yoy            year-over-year change in quarterly revenue
  eps_accel          change in eps_yoy vs the prior quarter (earnings acceleration)
  days_since_earnings  trading-relevant recency of the last report
  earnings_recent    1 if the last report landed within ~2 weeks (the PEAD window)

These target the post-earnings-announcement drift / revisions effects, which are
well-documented at the ~1-week horizon and are NOT captured by price momentum.
"""

import numpy as np
import pandas as pd

FUND_COLS = ["eps_yoy", "rev_yoy", "eps_accel", "days_since_earnings", "earnings_recent"]


def prepare_facts(facts: pd.DataFrame) -> dict:
    """Collapse to first-disclosure quarterly series per ticker/concept.

    Returns {ticker: {concept: DataFrame[period_end, first_filed, val] sorted by
    period_end}}, ready for fast as-of slicing.
    """
    if facts is None or facts.empty:
        return {}
    # Earliest filing for each distinct (ticker, concept, period_end).
    first = (facts.sort_values("filed")
             .groupby(["ticker", "concept", "period_end"], as_index=False)
             .first()
             .rename(columns={"filed": "first_filed"}))
    out = {}
    for (ticker, concept), g in first.groupby(["ticker", "concept"]):
        out.setdefault(ticker, {})[concept] = (
            g[["period_end", "first_filed", "val"]]
            .sort_values("period_end").reset_index(drop=True))
    return out


def _yoy(curr, prior):
    """Sign-robust YoY change: (curr - prior) / |prior|."""
    if prior is None or prior == 0 or pd.isna(prior) or pd.isna(curr):
        return np.nan
    return (curr - prior) / abs(prior)


def _series_asof(df, as_of):
    """Rows of a concept series whose first disclosure was on/before as_of."""
    if df is None:
        return None
    sub = df[df["first_filed"] <= as_of]
    return sub if not sub.empty else None


def _concept_features(eps, rev, as_of):
    feats = {"eps_yoy": np.nan, "rev_yoy": np.nan, "eps_accel": np.nan,
             "days_since_earnings": np.nan, "earnings_recent": 0}

    if eps is not None and len(eps) >= 5:
        v = eps["val"].to_numpy()
        feats["eps_yoy"] = _yoy(v[-1], v[-5])
        if len(eps) >= 6:
            prev_yoy = _yoy(v[-2], v[-6])
            feats["eps_accel"] = feats["eps_yoy"] - prev_yoy
        last_filed = eps["first_filed"].max()
        days = (pd.to_datetime(as_of) - last_filed).days
        feats["days_since_earnings"] = days
        feats["earnings_recent"] = 1 if 0 <= days <= 14 else 0

    if rev is not None and len(rev) >= 5:
        rv = rev["val"].to_numpy()
        feats["rev_yoy"] = _yoy(rv[-1], rv[-5])

    return feats


def compute_fundamental_features(prepared: dict, tickers, as_of) -> pd.DataFrame:
    """One row of fundamental features per ticker, as of `as_of`."""
    as_of = pd.to_datetime(as_of)
    rows = []
    for t in tickers:
        concepts = prepared.get(t, {})
        eps = _series_asof(concepts.get("eps"), as_of)
        rev = _series_asof(concepts.get("revenue"), as_of)
        rows.append({"ticker": t, **_concept_features(eps, rev, as_of)})
    return pd.DataFrame(rows)
