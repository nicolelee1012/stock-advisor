"""SEC EDGAR fundamentals — free, full-history, point-in-time.

We pull quarterly EPS and revenue facts via the XBRL `companyconcept` API. The
key field is `filed`: the date each number actually became public. Every feature
we derive uses only facts with filed <= as_of, so there's no look-ahead.

SEC fair-access rules: declare a real User-Agent (with contact) and stay under
~10 req/s. We cache every response to disk so we fetch each ticker only once.

Reference: https://www.sec.gov/os/accessing-edgar-data
"""

import json
import time

import pandas as pd
import requests

import config

# SEC asks for a descriptive UA with contact info.
HEADERS = {"User-Agent": "trading-research (nicole@pallet.com)"}
CIK_URL = "https://www.sec.gov/files/company_tickers.json"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

EDGAR_DIR = config.DATA_DIR / "edgar"
EDGAR_DIR.mkdir(exist_ok=True)
CIK_CACHE = EDGAR_DIR / "cik_map.json"
FACTS_CSV = config.DATA_DIR / "edgar_facts.csv"

# Candidate XBRL tags per concept (companies tag revenue differently); we use the
# first that returns data.
CONCEPTS = {
    "eps": (["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares"),
    "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenues", "SalesRevenueNet"], "USD"),
    "net_income": (["NetIncomeLoss"], "USD"),
}

REQUEST_PAUSE = 0.12  # ~8 req/s, comfortably under SEC's limit


def ticker_cik_map():
    """Return {TICKER: cik:int}, cached locally."""
    if CIK_CACHE.exists():
        return {k: int(v) for k, v in json.loads(CIK_CACHE.read_text()).items()}
    resp = requests.get(CIK_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    mapping = {row["ticker"]: int(row["cik_str"]) for row in data.values()}
    CIK_CACHE.write_text(json.dumps(mapping))
    return mapping


def _fetch_concept(cik, tag):
    """Fetch one XBRL concept for one company; return parsed JSON or None."""
    cache = EDGAR_DIR / f"{cik:010d}_{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = CONCEPT_URL.format(cik=cik, tag=tag)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    time.sleep(REQUEST_PAUSE)
    if resp.status_code == 404:
        cache.write_text("{}")  # remember the miss so we don't refetch
        return None
    resp.raise_for_status()
    data = resp.json()
    cache.write_text(json.dumps(data))
    return data


def _extract_quarterly(data, unit):
    """Pull quarterly facts (period <= ~100 days) from a concept payload."""
    if not data or "units" not in data or unit not in data["units"]:
        return []
    rows = []
    for f in data["units"][unit]:
        start, end, filed = f.get("start"), f.get("end"), f.get("filed")
        if not end or not filed:
            continue
        # Keep quarterly observations (skip full-year aggregates).
        if start:
            span = (pd.to_datetime(end) - pd.to_datetime(start)).days
            if span > 110:
                continue
        rows.append({"period_end": end, "filed": filed, "val": f.get("val"),
                     "form": f.get("form")})
    return rows


def build_facts_table(tickers=None, verbose=True):
    """Fetch (cached) EPS/revenue/net_income facts for tickers -> tidy DataFrame.

    Columns: ticker, concept, period_end, filed, val, form. Cached to CSV.
    """
    tickers = tickers or config.UNIVERSE
    cik_map = ticker_cik_map()

    rows = []
    missing = []
    for i, t in enumerate(tickers, 1):
        cik = cik_map.get(t)
        if cik is None:
            missing.append(t)
            continue
        for concept, (tags, unit) in CONCEPTS.items():
            for tag in tags:
                data = _fetch_concept(cik, tag)
                facts = _extract_quarterly(data, unit)
                if facts:
                    for fct in facts:
                        rows.append({"ticker": t, "concept": concept, **fct})
                    break  # first tag that yields data wins
        if verbose and i % 50 == 0:
            print(f"  ...{i}/{len(tickers)} tickers")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["filed"] = pd.to_datetime(df["filed"])
        df = df.drop_duplicates(["ticker", "concept", "period_end", "filed"])
        df.to_csv(FACTS_CSV, index=False)
    if verbose:
        print(f"Built facts table: {len(df)} rows for "
              f"{df['ticker'].nunique() if not df.empty else 0} tickers "
              f"({len(missing)} tickers had no CIK)")
    return df


def load_facts():
    """Load the cached facts table (or empty frame)."""
    if not FACTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(FACTS_CSV, parse_dates=["period_end", "filed"])
    return df


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or None
    build_facts_table(tickers)
