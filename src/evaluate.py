"""Score matured predictions against the benchmark — the "confirm/test" step.

For every prediction that has no outcome yet, check whether HORIZON_DAYS trading
days have elapsed since the run date. If so, look up the exit price and compute:

    realized_return = exit/entry - 1
    benchmark_return = benchmark exit/entry - 1   (over the SAME window)
    excess_return   = realized - benchmark
    hit             = excess_return > 0

Predictions that haven't matured yet are simply skipped — they'll be picked up on
a future run. This is what lets you back-test honestly: run a prediction with an
old as_of date, then evaluate immediately, and the horizon is already in the past.
"""

import pandas as pd

import config
from src import db, data_loader


def _price_on_or_after(price_df: pd.DataFrame, ticker: str, date) -> tuple:
    """First available (date, close) for ticker on/after `date`, else (None, None)."""
    sub = price_df[(price_df["ticker"] == ticker) & (price_df["date"] >= date)]
    if sub.empty:
        return None, None
    row = sub.sort_values("date").iloc[0]
    return row["date"], float(row["close"])


def evaluate_pending(db_path=None):
    """Mature & score all unscored predictions whose horizon has passed.

    Returns the number of predictions newly scored.
    """
    pending = db.unscored_predictions(db_path)
    if not pending:
        return 0

    # Pull enough history to cover the oldest pending prediction + horizon.
    tickers = sorted({p["ticker"] for p in pending} | {config.BENCHMARK})
    prices = data_loader.get_price_history(tickers=tickers,
                                           period_days=config.HISTORY_LOOKBACK_DAYS)

    scored = 0
    for p in pending:
        run_date = pd.to_datetime(p["run_date"])
        horizon = p["horizon_days"]

        # Target exit = `horizon` trading days after run_date. We approximate by
        # advancing calendar days generously, then taking the first trading day
        # on/after that target (handles weekends/holidays).
        target_exit = run_date + pd.tseries.offsets.BDay(horizon)

        # Has the horizon matured? Only score if we actually have data past it.
        latest = prices[prices["ticker"] == p["ticker"]]["date"].max()
        if pd.isna(latest) or latest < target_exit:
            continue  # not matured yet — try again on a future run

        exit_date, exit_price = _price_on_or_after(prices, p["ticker"], target_exit)
        if exit_price is None or not p["entry_price"]:
            continue

        # Benchmark over the same entry->exit window.
        _, bench_entry = _price_on_or_after(prices, p["benchmark"], run_date)
        _, bench_exit = _price_on_or_after(prices, p["benchmark"], target_exit)
        if not bench_entry or not bench_exit:
            continue

        realized = exit_price / p["entry_price"] - 1.0
        bench = bench_exit / bench_entry - 1.0
        excess = realized - bench

        db.save_outcome(
            pred_id=p["pred_id"],
            evaluated_date=pd.to_datetime(exit_date).date().isoformat(),
            exit_price=exit_price,
            realized_return=realized,
            benchmark_return=bench,
            excess_return=excess,
            db_path=db_path,
        )
        scored += 1

    return scored


if __name__ == "__main__":
    n = evaluate_pending()
    print(f"Scored {n} newly-matured prediction(s).")
