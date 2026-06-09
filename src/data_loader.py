"""Price/volume data access via yfinance.

Design notes on avoiding look-ahead bias:
  * We only ever ask for data up to and including an "as-of" date.
  * Features are computed strictly from rows <= as-of (see features.py).
  * yfinance gives split/dividend-adjusted closes, which is what we want for
    return computation.

Swap this module out for Polygon / Alpha Vantage later — the rest of the code
only depends on get_price_history() returning a tidy DataFrame.
"""

import pandas as pd

import config

try:
    import yfinance as yf
except ImportError:  # keep import-time failures friendly
    yf = None


def get_price_history(tickers=None, period_days=None, as_of=None):
    """Return a long-format DataFrame: [date, ticker, close, volume].

    as_of: optional ISO date string. Rows after this date are dropped so callers
           can simulate "what was knowable on day X".
    """
    if yf is None:
        raise ImportError("yfinance not installed. Run: pip install -r requirements.txt")

    tickers = tickers or config.UNIVERSE
    period_days = period_days or config.HISTORY_LOOKBACK_DAYS

    raw = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=True,       # adjusted OHLC -> clean returns
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames = []
    for t in tickers:
        try:
            sub = raw[t][["Close", "Volume"]].copy()
        except (KeyError, TypeError):
            # Single-ticker downloads aren't grouped by ticker.
            sub = raw[["Close", "Volume"]].copy() if len(tickers) == 1 else None
        if sub is None or sub.empty:
            continue
        sub = sub.rename(columns={"Close": "close", "Volume": "volume"})
        sub["ticker"] = t
        sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
        frames.append(sub)

    if not frames:
        raise RuntimeError("No price data returned. Check tickers / network.")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.dropna(subset=["close"]).sort_values(["ticker", "date"])

    if as_of is not None:
        as_of = pd.to_datetime(as_of)
        df = df[df["date"] <= as_of]

    return df.reset_index(drop=True)
