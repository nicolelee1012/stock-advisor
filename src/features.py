"""Feature engineering — strictly point-in-time (no look-ahead bias).

Given long-format price history, compute one feature row per ticker as of the
latest available date. Every feature uses only past/current rows, never future
ones. The data_loader's `as_of` cutoff plus these backward-only windows are the
two guards that keep the model honest.

Features (all derivable from price/volume — fundamentals come in a later phase):
  ret_1m, ret_3m, ret_6m   trailing total returns over ~21/63/126 trading days
  vol_1m                   annualized rolling volatility (risk proxy)
  dist_sma50, dist_sma200  % distance of price above/below moving averages
  vol_trend                recent vs longer-run average volume (liquidity/interest)
"""

import numpy as np
import pandas as pd

# Trading-day windows
WIN_1M, WIN_3M, WIN_6M = 21, 63, 126
SMA_SHORT, SMA_LONG = 50, 200


def _trailing_return(close: pd.Series, window: int) -> float:
    """Total return over the trailing `window` rows, or NaN if too short."""
    if len(close) <= window:
        return np.nan
    return close.iloc[-1] / close.iloc[-1 - window] - 1.0


def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """price_df: long format [date, ticker, close, volume] (already <= as_of).

    Returns one row per ticker with the feature columns above. Tickers without
    enough history are dropped (can't compute 6m momentum on 3 months of data).
    """
    rows = []
    for ticker, g in price_df.sort_values("date").groupby("ticker"):
        close = g["close"].reset_index(drop=True)
        volume = g["volume"].reset_index(drop=True)
        if len(close) < WIN_6M + 5:
            continue

        daily_ret = close.pct_change()
        sma50 = close.rolling(SMA_SHORT).mean().iloc[-1]
        sma200 = close.rolling(SMA_LONG).mean().iloc[-1]
        last = close.iloc[-1]

        rows.append({
            "ticker": ticker,
            "as_of": g["date"].iloc[-1],
            "close": last,
            "ret_1m": _trailing_return(close, WIN_1M),
            "ret_3m": _trailing_return(close, WIN_3M),
            "ret_6m": _trailing_return(close, WIN_6M),
            "vol_1m": daily_ret.tail(WIN_1M).std() * np.sqrt(252),
            "dist_sma50": last / sma50 - 1.0 if sma50 else np.nan,
            "dist_sma200": last / sma200 - 1.0 if sma200 else np.nan,
            "vol_trend": (volume.tail(WIN_1M).mean() / volume.tail(WIN_3M).mean() - 1.0)
                          if volume.tail(WIN_3M).mean() else np.nan,
        })

    return pd.DataFrame(rows).dropna().reset_index(drop=True)
