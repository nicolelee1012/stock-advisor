"""Walk-forward backtester with transaction costs.

The honest grader. It simulates periodic rebalancing over history and reports
returns NET OF FEES, so any strategy must earn its keep. Every strategy — the
trained model and all baselines — runs through the SAME engine for a fair fight.

A "strategy" is just a function:
    strategy(snapshot_prices, as_of_date) -> list of tickers to hold (equal weight)
where snapshot_prices contains only rows <= as_of_date (look-ahead safe).

Costs: we charge TRANSACTION_COST_BPS on turnover (fraction of the book that
changes) at each rebalance — buying into new names and selling out of dropped
ones both incur cost.
"""

import numpy as np
import pandas as pd

import config
from src import features, model

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Strategies (each returns the set of tickers to hold this period)
# --------------------------------------------------------------------------
def strat_model(predict_fn, top_n):
    """Wrap a ranking function (e.g. model.rank) into a strategy."""
    def _strat(snapshot, as_of):
        feat = features.compute_features(snapshot)
        if feat.empty:
            return []
        ranked = predict_fn(feat)
        return ranked.head(top_n)["ticker"].tolist()
    return _strat


def strat_momentum(top_n):
    """Simple 3-month momentum — one of our must-beat baselines."""
    def _strat(snapshot, as_of):
        feat = features.compute_features(snapshot)
        feat = feat[feat["ticker"] != config.BENCHMARK]
        if feat.empty:
            return []
        return feat.sort_values("ret_3m", ascending=False).head(top_n)["ticker"].tolist()
    return _strat


def strat_equal_weight(snapshot, as_of):
    """Hold the entire universe equally."""
    return list(config.UNIVERSE)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def run_backtest(prices, strategy, horizon=None, cadence=None, top_n=None,
                 cost_bps=None, start_after=config.HISTORY_LOOKBACK_DAYS):
    """Simulate a strategy over `prices` (long format, full history).

    Returns a dict with the equity curve (Series indexed by date) and metrics.
    start_after skips the initial window where features can't be computed yet.
    """
    horizon = horizon or config.HORIZON_DAYS
    cadence = cadence or horizon
    top_n = top_n or config.TOP_N
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps

    close = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    all_dates = close.index
    rebalance_dates = all_dates[start_after::cadence]

    equity = 1.0
    curve = {}
    prev_holds = set()

    for t in rebalance_dates:
        future = all_dates[all_dates >= t]
        if len(future) <= horizon:
            break
        exit_date = future[horizon]

        snapshot = prices[prices["date"] <= t]
        holds = strategy(snapshot, t)
        holds = [h for h in holds if h in close.columns]
        if not holds:
            curve[t] = equity
            continue

        # Equal-weight period return of the held basket.
        entry = close.loc[t, holds]
        exit_ = close.loc[exit_date, holds]
        period_ret = float((exit_ / entry - 1.0).mean())

        # Turnover cost: fraction of book changed since last rebalance.
        new_set = set(holds)
        if prev_holds:
            turnover = len(new_set.symmetric_difference(prev_holds)) / (2 * max(len(new_set), 1))
        else:
            turnover = 1.0  # initial buy-in
        cost = turnover * (cost_bps / 10_000.0)

        equity *= (1.0 + period_ret - cost)
        curve[exit_date] = equity
        prev_holds = new_set

    return _summarize(pd.Series(curve), cadence, horizon)


def run_walkforward_model(prices, horizon=None, cadence=None, top_n=None,
                          cost_bps=None, retrain_every=4, min_train_dates=60):
    """Leak-free backtest of the TRAINED model.

    The critical difference from naively backtesting a globally-trained model:
    at each rebalance date t we use a model fit ONLY on panel rows from strictly
    earlier rebalance dates (each of whose label windows completed by t). The
    model never sees the future it's being tested on. We retrain every
    `retrain_every` periods to keep it affordable.

    This is the number that actually matters — if it doesn't beat the baselines
    here, the model has no edge, regardless of how good in-sample looks.
    """
    from src import dataset, train

    horizon = horizon or config.HORIZON_DAYS
    cadence = cadence or horizon
    top_n = top_n or config.TOP_N
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps

    panel = dataset.build_panel(horizon=horizon, cadence=cadence,
                                prices=prices, verbose=False)
    close = prices.pivot_table(index="date", columns="ticker",
                               values="close").sort_index()
    all_dates = close.index
    panel_dates = pd.Series(sorted(panel["date"].unique()))

    equity = 1.0
    curve = {}
    prev_holds = set()
    reg = None
    periods_since_fit = 0

    for i, t in enumerate(panel_dates):
        # Train only on rows from strictly-earlier rebalance dates (no leakage).
        train_rows = panel[panel["date"] < t]
        if train_rows["date"].nunique() < min_train_dates:
            continue
        if reg is None or periods_since_fit >= retrain_every:
            reg, _ = train._make_regressor()
            reg.fit(train_rows[dataset.FEATURE_COLS], train_rows[dataset.LABEL_COL])
            periods_since_fit = 0
        periods_since_fit += 1

        feats_t = panel[panel["date"] == t]
        if feats_t.empty:
            continue
        preds = reg.predict(feats_t[dataset.FEATURE_COLS])
        holds = (feats_t.assign(_p=preds).sort_values("_p", ascending=False)
                 .head(top_n)["ticker"].tolist())
        holds = [h for h in holds if h in close.columns]
        if not holds:
            continue

        future = all_dates[all_dates >= t]
        if len(future) <= horizon:
            break
        exit_date = future[horizon]
        period_ret = float((close.loc[exit_date, holds] / close.loc[t, holds] - 1.0).mean())

        new_set = set(holds)
        turnover = (len(new_set.symmetric_difference(prev_holds)) / (2 * max(len(new_set), 1))
                    if prev_holds else 1.0)
        equity *= (1.0 + period_ret - turnover * (cost_bps / 10_000.0))
        curve[exit_date] = equity
        prev_holds = new_set

    return _summarize(pd.Series(curve), cadence, horizon)


def run_buy_and_hold(prices, ticker=None):
    """Buy-and-hold a single ticker (default benchmark) — the simplest baseline."""
    ticker = ticker or config.BENCHMARK
    s = (prices[prices["ticker"] == ticker].set_index("date")["close"].sort_index())
    curve = s / s.iloc[0]
    return _summarize(curve, cadence=1, horizon=1)


def _summarize(curve, cadence, horizon):
    """Compute return/Sharpe/drawdown metrics from an equity curve."""
    curve = curve.sort_index()
    if len(curve) < 2:
        return {"equity_curve": curve, "total_return": 0.0, "ann_return": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "n_periods": len(curve)}

    period_rets = curve.pct_change().dropna()
    total = curve.iloc[-1] / curve.iloc[0] - 1.0

    span_days = (curve.index[-1] - curve.index[0]).days or 1
    years = span_days / 365.25
    ann = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0

    # Annualize Sharpe by the number of rebalances per year.
    periods_per_year = TRADING_DAYS / cadence if cadence else TRADING_DAYS
    vol = period_rets.std()
    sharpe = (period_rets.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0

    running_max = curve.cummax()
    max_dd = float((curve / running_max - 1.0).min())

    return {
        "equity_curve": curve,
        "total_return": float(total),
        "ann_return": float(ann),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_periods": len(curve),
    }


def compare_all(prices=None, top_n=None):
    """Run the model strategy plus all baselines and return a metrics table."""
    if prices is None:
        from src import data_loader
        tickers = list(config.UNIVERSE) + [config.BENCHMARK]
        prices = data_loader.get_price_history(tickers=tickers, period_days=8 * 365)

    baselines = {
        "momentum_baseline": strat_momentum(top_n or config.TOP_N),
        "equal_weight": strat_equal_weight,
    }
    results = {name: run_backtest(prices, strat, top_n=top_n)
               for name, strat in baselines.items()}
    # The trained model uses the LEAK-FREE walk-forward path, not strat_model.
    results["model_walkforward"] = run_walkforward_model(prices, top_n=top_n)
    results["buy_hold_SPY"] = run_buy_and_hold(prices)

    table = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "equity_curve"}
        for name, r in results.items()
    }).T
    return table, results


if __name__ == "__main__":
    table, _ = compare_all()
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n=== Backtest (net of fees) ===\n")
    print(table[["total_return", "ann_return", "sharpe", "max_drawdown", "n_periods"]])
