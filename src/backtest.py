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
                          cost_bps=None, retrain_every=4, min_train_dates=60,
                          risk_managed=False, sector_map=None):
    """Leak-free backtest of the TRAINED model.

    The critical difference from naively backtesting a globally-trained model:
    at each rebalance date t we use a model fit ONLY on panel rows from strictly
    earlier rebalance dates (each of whose label windows completed by t). The
    model never sees the future it's being tested on. We retrain every
    `retrain_every` periods to keep it affordable.

    risk_managed=False: equal-weight top_n (the simple strategy).
    risk_managed=True:  position weights from portfolio.build_weights (inverse-vol,
                        sector/position caps, vol targeting with a cash buffer).

    This is the number that actually matters — if it doesn't beat the baselines
    here, the model has no edge, regardless of how good in-sample looks.
    """
    from src import dataset, train, portfolio

    horizon = horizon or config.HORIZON_DAYS
    cadence = cadence or horizon
    top_n = top_n or (config.RISK_N_HOLDINGS if risk_managed else config.TOP_N)
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    sector_map = sector_map or {}

    panel = dataset.build_panel(horizon=horizon, cadence=cadence,
                                prices=prices, verbose=False)
    close = prices.pivot_table(index="date", columns="ticker",
                               values="close").sort_index()
    all_dates = close.index
    panel_dates = pd.Series(sorted(panel["date"].unique()))

    equity = 1.0
    curve = {}
    prev_w = {}            # ticker -> weight from the prior rebalance (for turnover)
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
        preds = feats_t.assign(_p=reg.predict(feats_t[dataset.FEATURE_COLS]))
        picks = preds.sort_values("_p", ascending=False).head(top_n)
        picks = picks[picks["ticker"].isin(close.columns)]
        if picks.empty:
            continue

        # Position weights (ticker -> weight; cash is the residual to 1.0).
        if risk_managed:
            w = portfolio.build_weights(picks, sector_map)
            w.pop("__cash__", None)
        else:
            n = len(picks)
            w = {tk: 1.0 / n for tk in picks["ticker"]}

        future = all_dates[all_dates >= t]
        if len(future) <= horizon:
            break
        exit_date = future[horizon]

        # Weighted basket return (uninvested weight sits in cash, return 0).
        rets = close.loc[exit_date, list(w)] / close.loc[t, list(w)] - 1.0
        period_ret = float(sum(w[tk] * rets[tk] for tk in w))

        # Turnover = half the sum of absolute weight changes across all names.
        names = set(w) | set(prev_w)
        turnover = 0.5 * sum(abs(w.get(tk, 0.0) - prev_w.get(tk, 0.0)) for tk in names)
        equity *= (1.0 + period_ret - turnover * (cost_bps / 10_000.0))
        curve[exit_date] = equity
        prev_w = w

    return _summarize(pd.Series(curve), cadence, horizon)


def run_walkforward_variants(prices, sector_map=None, horizon=None, cadence=None,
                             cost_bps=None, retrain_every=4, min_train_dates=60):
    """One leak-free walk-forward pass, evaluated under BOTH weighting schemes.

    The model's predictions are identical across schemes; only portfolio
    construction differs. Running a single pass keeps them strictly comparable
    and avoids retraining the model twice. Returns
    {"model_walkforward": summary, "model_risk_managed": summary}.
    """
    from src import dataset, train, portfolio

    horizon = horizon or config.HORIZON_DAYS
    cadence = cadence or horizon
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    sector_map = sector_map or {}
    n_plain, n_risk = config.TOP_N, config.RISK_N_HOLDINGS

    panel = dataset.build_panel(horizon=horizon, cadence=cadence,
                                prices=prices, verbose=False)
    close = prices.pivot_table(index="date", columns="ticker",
                               values="close").sort_index()
    all_dates = close.index
    panel_dates = pd.Series(sorted(panel["date"].unique()))

    eq = {"plain": 1.0, "risk": 1.0}
    curves = {"plain": {}, "risk": {}}
    prev_w = {"plain": {}, "risk": {}}
    reg, periods_since_fit = None, 0

    for t in panel_dates:
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
        ranked = (feats_t.assign(_p=reg.predict(feats_t[dataset.FEATURE_COLS]))
                  .sort_values("_p", ascending=False))
        ranked = ranked[ranked["ticker"].isin(close.columns)]
        if ranked.empty:
            continue

        future = all_dates[all_dates >= t]
        if len(future) <= horizon:
            break
        exit_date = future[horizon]

        # Equal-weight top-N (plain) and risk-managed weights (risk).
        plain_picks = ranked.head(n_plain)
        w_plain = {tk: 1.0 / len(plain_picks) for tk in plain_picks["ticker"]}
        w_risk = portfolio.build_weights(ranked.head(n_risk), sector_map)
        w_risk.pop("__cash__", None)

        for key, w in (("plain", w_plain), ("risk", w_risk)):
            rets = close.loc[exit_date, list(w)] / close.loc[t, list(w)] - 1.0
            period_ret = float(sum(w[tk] * rets[tk] for tk in w))
            names = set(w) | set(prev_w[key])
            turnover = 0.5 * sum(abs(w.get(tk, 0.0) - prev_w[key].get(tk, 0.0))
                                 for tk in names)
            eq[key] *= (1.0 + period_ret - turnover * (cost_bps / 10_000.0))
            curves[key][exit_date] = eq[key]
            prev_w[key] = w

    return {
        "model_walkforward": _summarize(pd.Series(curves["plain"]), cadence, horizon),
        "model_risk_managed": _summarize(pd.Series(curves["risk"]), cadence, horizon),
    }


def run_weighted_strategy(prices, rank_fn, profile, sector_map=None, horizon=None,
                          cadence=None, cost_bps=None,
                          start_after=config.HISTORY_LOOKBACK_DAYS):
    """Backtest a (training-free) ranking under a risk-managed profile.

    rank_fn(feat_df) -> feat_df sorted best-first. Used to apply the risk-control
    layer (portfolio.build_weights) to a simple signal like momentum, so we can
    test "winning signal + winning risk management".
    """
    from src import portfolio

    horizon = horizon or config.HORIZON_DAYS
    cadence = cadence or horizon
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    sector_map = sector_map or {}
    n = portfolio.get_profile(profile)["n_holdings"]

    close = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    all_dates = close.index
    rebalance_dates = all_dates[start_after::cadence]

    equity, curve, prev_w = 1.0, {}, {}
    for t in rebalance_dates:
        future = all_dates[all_dates >= t]
        if len(future) <= horizon:
            break
        exit_date = future[horizon]

        feat = features.compute_features(prices[prices["date"] <= t])
        feat = feat[feat["ticker"] != config.BENCHMARK]
        if feat.empty:
            continue
        picks = rank_fn(feat).head(n)
        picks = picks[picks["ticker"].isin(close.columns)]
        if picks.empty:
            continue

        w = portfolio.build_weights(picks, sector_map, profile)
        w.pop("__cash__", None)
        rets = close.loc[exit_date, list(w)] / close.loc[t, list(w)] - 1.0
        period_ret = float(sum(w[tk] * rets[tk] for tk in w))
        names = set(w) | set(prev_w)
        turnover = 0.5 * sum(abs(w.get(tk, 0.0) - prev_w.get(tk, 0.0)) for tk in names)
        equity *= (1.0 + period_ret - turnover * (cost_bps / 10_000.0))
        curve[exit_date] = equity
        prev_w = w

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

    from src import universe
    sector_map = universe.load_sectors()

    baselines = {
        "momentum_baseline": strat_momentum(top_n or config.TOP_N),
        "equal_weight": strat_equal_weight,
    }
    results = {name: run_backtest(prices, strat, top_n=top_n)
               for name, strat in baselines.items()}
    # One walk-forward pass, evaluated equal-weight AND risk-managed.
    results.update(run_walkforward_variants(prices, sector_map=sector_map))
    # Winning signal (momentum) + winning risk management, both profiles.
    mom_rank = lambda f: f.sort_values("ret_3m", ascending=False)
    results["momentum_balanced"] = run_weighted_strategy(
        prices, mom_rank, "balanced", sector_map)
    results["momentum_aggressive"] = run_weighted_strategy(
        prices, mom_rank, "aggressive", sector_map)
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
