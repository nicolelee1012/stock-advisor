"""Single source of truth for the model's feature matrix.

Both training (dataset.build_panel) and live inference (predict.py) call
build_features() so they assemble features identically — the cardinal rule for
avoiding train/serve skew. Price features + point-in-time fundamentals, merged
on ticker.

If EDGAR facts aren't available, fundamental columns are filled NaN and the model
falls back to price-only behavior gracefully.
"""

import pandas as pd

from src import features, fundamentals, edgar, estimate_features, estimates

PRICE_COLS = ["ret_1m", "ret_3m", "ret_6m", "vol_1m",
              "dist_sma50", "dist_sma200", "vol_trend"]
FUND_COLS = fundamentals.FUND_COLS
EST_COLS = estimate_features.EST_COLS
FEATURE_COLS = PRICE_COLS + FUND_COLS + EST_COLS

# Module-level caches so we prepare the (large) event structures only once per run.
_PREPARED_FACTS = None
_PREPARED_EST = None


def prepared_facts():
    global _PREPARED_FACTS
    if _PREPARED_FACTS is None:
        _PREPARED_FACTS = fundamentals.prepare_facts(edgar.load_facts())
    return _PREPARED_FACTS


def prepared_estimates():
    global _PREPARED_EST
    if _PREPARED_EST is None:
        _PREPARED_EST = estimate_features.prepare(
            estimates.load_surprises(), estimates.load_analyst())
    return _PREPARED_EST


def build_features(prices: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Assemble price + fundamental + surprise/analyst features as of `as_of`
    (default: latest date in `prices`). One row per ticker, all FEATURE_COLS present.
    """
    snapshot = prices if as_of is None else prices[prices["date"] <= pd.to_datetime(as_of)]
    price_feat = features.compute_features(snapshot)
    if price_feat.empty:
        return price_feat

    eff_asof = as_of or price_feat["as_of"].iloc[0]
    tickers = price_feat["ticker"].tolist()

    fund_feat = fundamentals.compute_fundamental_features(prepared_facts(), tickers, eff_asof)
    est_feat = estimate_features.compute(prepared_estimates(), tickers, eff_asof)

    merged = (price_feat.merge(fund_feat, on="ticker", how="left")
              .merge(est_feat, on="ticker", how="left"))
    # earnings_recent is a 0/1 flag; missing -> 0. Other features stay NaN
    # (tree models handle NaN; the panel builder decides whether to drop).
    merged["earnings_recent"] = merged["earnings_recent"].fillna(0)
    return merged
