"""Universe management — the list of tickers we rank.

We pull the *current* S&P 500 constituents from Wikipedia and cache them to
data/sp500.csv. config.py reads that cache (fast, offline) and falls back to a
curated core list if it's absent.

IMPORTANT caveat — survivorship bias: this is the list of companies in the index
*today*. A backtest run against it silently ignores names that were in the index
historically but later got dropped/delisted (often the losers). That flatters
results. True point-in-time constituents are a paid dataset; until we have one,
treat backtest absolute returns as optimistic and lean on the *relative* ranking
vs the baselines (which share the same bias).
"""

import io

import pandas as pd
import requests

import config

SP500_CSV = config.DATA_DIR / "sp500.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def refresh_sp500():
    """Scrape current S&P 500 constituents and cache to data/sp500.csv."""
    # Wikipedia 403s the default urllib UA; send a real browser-ish header.
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    # yfinance uses '-' where the index uses '.' (e.g. BRK.B -> BRK-B).
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    out = df[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    out.to_csv(SP500_CSV, index=False)
    print(f"Cached {len(out)} S&P 500 constituents -> {SP500_CSV}")
    return out


def load_universe():
    """Return the list of tickers to use (S&P 500 cache, else curated core)."""
    if SP500_CSV.exists():
        return pd.read_csv(SP500_CSV)["ticker"].tolist()
    return config.CORE_UNIVERSE


def load_sectors():
    """Return {ticker: sector} mapping if available, else {}."""
    if SP500_CSV.exists():
        df = pd.read_csv(SP500_CSV)
        return dict(zip(df["ticker"], df["sector"]))
    return {}


if __name__ == "__main__":
    refresh_sp500()
