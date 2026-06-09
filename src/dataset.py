"""Build the supervised training panel from price history.

A "panel" is a tidy table with one row per (rebalance_date, ticker):
    [date, ticker, <feature columns...>, fwd_excess_return]

Look-ahead discipline:
  * Features at date t use ONLY rows <= t (we slice the price frame and reuse
    features.compute_features, the exact same code path as live inference).
  * The label uses the future window t -> t+HORIZON. That is the ONE place
    future data is allowed, and only because it's the supervised target during
    training. At live inference there is no label — we just rank by prediction.

Weekly cadence (every HORIZON trading days) keeps successive samples from
overlapping too heavily, which would otherwise inflate apparent performance.
"""

import pandas as pd

import config
from src import data_loader, featureset

# Centralized in featureset so training and live inference share one definition.
FEATURE_COLS = featureset.FEATURE_COLS
PRICE_COLS = featureset.PRICE_COLS
LABEL_COL = "fwd_excess_return"


def _forward_return(close_by_date: pd.Series, entry_date, exit_date):
    """Return over [entry_date, exit_date] using the first available close on/after
    each date, or None if either side is missing."""
    entry = close_by_date[close_by_date.index >= entry_date]
    exit_ = close_by_date[close_by_date.index >= exit_date]
    if entry.empty or exit_.empty:
        return None
    return exit_.iloc[0] / entry.iloc[0] - 1.0


def build_panel(years=8, horizon=None, cadence=None, prices=None, verbose=True):
    """Construct the training panel.

    years:   how much history to pull (ignored if `prices` is supplied).
    horizon: forward window for the label (defaults to config.HORIZON_DAYS).
    cadence: trading days between rebalance dates (defaults to horizon).
    prices:  optional pre-fetched long-format price frame to reuse (avoids a
             redundant download when the caller — e.g. the backtester — already
             has it).
    """
    horizon = horizon or config.HORIZON_DAYS
    cadence = cadence or horizon

    if prices is None:
        tickers = list(config.UNIVERSE) + [config.BENCHMARK]
        prices = data_loader.get_price_history(tickers=tickers, period_days=years * 365)

    # Pre-index benchmark and per-ticker closes by date for fast forward lookups.
    bench = (prices[prices["ticker"] == config.BENCHMARK]
             .set_index("date")["close"].sort_index())
    close_by_ticker = {
        t: g.set_index("date")["close"].sort_index()
        for t, g in prices.groupby("ticker")
    }

    # Rebalance dates: every `cadence`-th trading day from the benchmark calendar.
    all_dates = bench.index.sort_values()
    rebalance_dates = all_dates[::cadence]

    rows = []
    for t in rebalance_dates:
        # Need enough lookback for 6m momentum, and a full forward window.
        future = all_dates[all_dates >= t]
        if len(future) <= horizon:
            continue  # label window runs off the end of history
        exit_date = future[horizon]

        feat = featureset.build_features(prices, as_of=t)
        if feat.empty:
            continue

        bench_fwd = _forward_return(bench, t, exit_date)
        if bench_fwd is None:
            continue

        for _, r in feat.iterrows():
            tk = r["ticker"]
            if tk == config.BENCHMARK:
                continue  # the benchmark is the yardstick, not a candidate
            fwd = _forward_return(close_by_ticker.get(tk, pd.Series(dtype=float)),
                                  t, exit_date)
            if fwd is None:
                continue
            row = {"date": t, "ticker": tk, LABEL_COL: fwd - bench_fwd}
            for c in FEATURE_COLS:
                row[c] = r[c]
            rows.append(row)

    # Require the label and price features; leave fundamentals NaN where missing
    # (gradient-boosted trees handle NaN natively — dropping them would discard
    # companies that simply haven't filed enough history yet).
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.dropna(subset=[LABEL_COL] + PRICE_COLS).reset_index(drop=True)
    if verbose:
        print(f"Built panel: {len(panel)} rows, "
              f"{panel['date'].nunique()} rebalance dates, "
              f"{panel['ticker'].nunique()} tickers")
    return panel


if __name__ == "__main__":
    p = build_panel()
    print(p.head())
