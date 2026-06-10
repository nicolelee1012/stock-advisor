"""Free earnings-surprise & analyst-revision data via yfinance.

Two trainable, point-in-time signals (both timestamped, multi-year history):

  earnings surprises   consensus EPS estimate vs. reported EPS, by report date.
                       The surprise magnitude is the actual PEAD driver — distinct
                       from the YoY earnings *growth* we get from EDGAR.
  analyst actions      up/downgrades and price-target changes, by grade date.
                       Net revision flow in a trailing window is a known short-
                       horizon signal.

Each ticker is cached to disk so we fetch it only once. Consolidated into two
CSVs the feature layer reads.

Caveat: yfinance scrapes Yahoo; data can be incomplete or shift, and the
consensus estimate is assumed to be the pre-announcement value (minor look-ahead
risk if Yahoo backfills). Good enough for a free signal; revisit if it proves
flaky.
"""

import time

import pandas as pd

import config

try:
    import yfinance as yf
except ImportError:
    yf = None

EST_DIR = config.DATA_DIR / "estimates"
EST_DIR.mkdir(exist_ok=True)
SURPRISE_CSV = config.DATA_DIR / "earnings_surprises.csv"
ANALYST_CSV = config.DATA_DIR / "analyst_actions.csv"

PAUSE = 0.15  # be polite to Yahoo


def _fetch_one(ticker):
    """Fetch & cache surprises and analyst actions for one ticker.

    Returns (surprises_df, analyst_df). Uses per-ticker CSV caches for resume.
    """
    surp_cache = EST_DIR / f"{ticker}_surprise.csv"
    anly_cache = EST_DIR / f"{ticker}_analyst.csv"

    if surp_cache.exists() and anly_cache.exists():
        s = pd.read_csv(surp_cache, parse_dates=["earnings_date"]) if surp_cache.stat().st_size > 2 else pd.DataFrame()
        a = pd.read_csv(anly_cache, parse_dates=["grade_date"]) if anly_cache.stat().st_size > 2 else pd.DataFrame()
        return s, a

    t = yf.Ticker(ticker)

    # --- surprises ---
    try:
        ed = t.get_earnings_dates(limit=60)
        ed = ed.dropna(subset=["Surprise(%)"]).reset_index()
        s = pd.DataFrame({
            "ticker": ticker,
            "earnings_date": pd.to_datetime(ed["Earnings Date"]).dt.tz_localize(None),
            "eps_estimate": ed["EPS Estimate"].values,
            "reported_eps": ed["Reported EPS"].values,
            "surprise_pct": ed["Surprise(%)"].values,
        })
    except Exception:
        s = pd.DataFrame()
    time.sleep(PAUSE)

    # --- analyst actions ---
    try:
        ud = t.get_upgrades_downgrades().reset_index()
        date_col = "GradeDate" if "GradeDate" in ud.columns else ud.columns[0]
        a = pd.DataFrame({
            "ticker": ticker,
            "grade_date": pd.to_datetime(ud[date_col]).dt.tz_localize(None),
            "action": ud.get("Action"),
            "current_pt": ud.get("currentPriceTarget"),
            "prior_pt": ud.get("priorPriceTarget"),
        })
    except Exception:
        a = pd.DataFrame()
    time.sleep(PAUSE)

    # Cache (write header even when empty so the resume check works).
    (s if not s.empty else pd.DataFrame(
        columns=["ticker", "earnings_date", "eps_estimate", "reported_eps", "surprise_pct"])
     ).to_csv(surp_cache, index=False)
    (a if not a.empty else pd.DataFrame(
        columns=["ticker", "grade_date", "action", "current_pt", "prior_pt"])
     ).to_csv(anly_cache, index=False)
    return s, a


def build_tables(tickers=None, verbose=True):
    """Fetch (cached) surprises + analyst actions for all tickers -> 2 CSVs."""
    tickers = tickers or config.UNIVERSE
    surps, anlys = [], []
    for i, tk in enumerate(tickers, 1):
        s, a = _fetch_one(tk)
        if not s.empty:
            surps.append(s)
        if not a.empty:
            anlys.append(a)
        if verbose and i % 50 == 0:
            print(f"  ...{i}/{len(tickers)} tickers", flush=True)

    surp_df = pd.concat(surps, ignore_index=True) if surps else pd.DataFrame()
    anly_df = pd.concat(anlys, ignore_index=True) if anlys else pd.DataFrame()
    if not surp_df.empty:
        surp_df.to_csv(SURPRISE_CSV, index=False)
    if not anly_df.empty:
        anly_df.to_csv(ANALYST_CSV, index=False)
    if verbose:
        print(f"Surprises: {len(surp_df)} rows / "
              f"{surp_df['ticker'].nunique() if not surp_df.empty else 0} tickers. "
              f"Analyst: {len(anly_df)} rows / "
              f"{anly_df['ticker'].nunique() if not anly_df.empty else 0} tickers.")
    return surp_df, anly_df


def load_surprises():
    if not SURPRISE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SURPRISE_CSV, parse_dates=["earnings_date"])


def load_analyst():
    if not ANALYST_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(ANALYST_CSV, parse_dates=["grade_date"])


if __name__ == "__main__":
    import sys
    build_tables(sys.argv[1:] or None)
