"""Point-in-time features from earnings surprises and analyst actions.

For any as_of date T we use only events dated <= T:

  surprise_last      most recent earnings surprise (reported vs consensus EPS)
  surprise_avg4      average surprise over the last 4 reports (quality of beats)
  analyst_pt_rev_90  mean analyst price-target revision % in the trailing 90 days
  analyst_breadth_90 (raises - cuts) / actions in the trailing 90 days

surprise_last paired with the existing `days_since_earnings` is the classic PEAD
setup: a big beat tends to drift up over the following days/weeks. analyst breadth
captures revision momentum, a separate short-horizon signal.
"""

import numpy as np
import pandas as pd

EST_COLS = ["surprise_last", "surprise_avg4", "analyst_pt_rev_90", "analyst_breadth_90"]
WINDOW_DAYS = 90


def prepare(surprises: pd.DataFrame, analyst: pd.DataFrame) -> dict:
    """Index events by ticker as sorted numpy arrays for fast as-of slicing."""
    out = {}

    if surprises is not None and not surprises.empty:
        for tk, g in surprises.sort_values("earnings_date").groupby("ticker"):
            out.setdefault(tk, {})["surp_date"] = g["earnings_date"].to_numpy()
            out[tk]["surp_val"] = (g["surprise_pct"].to_numpy() / 100.0)  # -> decimal

    if analyst is not None and not analyst.empty:
        a = analyst.copy()
        prior = a["prior_pt"].replace(0, np.nan)
        a["pt_rev"] = (a["current_pt"] - prior) / prior
        for tk, g in a.sort_values("grade_date").groupby("ticker"):
            out.setdefault(tk, {})["anly_date"] = g["grade_date"].to_numpy()
            out[tk]["anly_rev"] = g["pt_rev"].to_numpy()
    return out


def _surprise_feats(d, as_of):
    dates = d.get("surp_date")
    if dates is None:
        return np.nan, np.nan
    mask = dates <= as_of
    vals = d["surp_val"][mask]
    if len(vals) == 0:
        return np.nan, np.nan
    last = vals[-1]
    avg4 = np.nanmean(vals[-4:])
    return last, avg4


def _analyst_feats(d, as_of, lo):
    dates = d.get("anly_date")
    if dates is None:
        return np.nan, np.nan
    mask = (dates <= as_of) & (dates > lo)
    rev = d["anly_rev"][mask]
    rev = rev[~np.isnan(rev)]
    if len(rev) == 0:
        return np.nan, np.nan
    pt_rev = float(np.mean(rev))
    breadth = float((np.sum(rev > 0) - np.sum(rev < 0)) / len(rev))
    return pt_rev, breadth


def compute(prepared: dict, tickers, as_of) -> pd.DataFrame:
    """One row of surprise/analyst features per ticker, as of `as_of`."""
    as_of = np.datetime64(pd.to_datetime(as_of))
    lo = as_of - np.timedelta64(WINDOW_DAYS, "D")
    rows = []
    for t in tickers:
        d = prepared.get(t, {})
        s_last, s_avg = _surprise_feats(d, as_of)
        pt_rev, breadth = _analyst_feats(d, as_of, lo)
        rows.append({"ticker": t, "surprise_last": s_last, "surprise_avg4": s_avg,
                     "analyst_pt_rev_90": pt_rev, "analyst_breadth_90": breadth})
    return pd.DataFrame(rows)
