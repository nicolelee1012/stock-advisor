"""Produce and persist the top-N picks for a given run date.

This is the "every day the model says these 10 stocks are high-potential for next
week" step. Entry price = the close on the run date, which becomes the basis for
scoring the realized return later (see evaluate.py).
"""

import pandas as pd

import config
from src import db, data_loader, featureset, model, portfolio, universe


def _momentum_rank(feat: pd.DataFrame) -> pd.DataFrame:
    """Rank by 3-month price momentum — the backtest's strongest simple signal."""
    f = feat.copy()
    f["score"] = f["ret_3m"]
    f = f[f["ticker"] != config.BENCHMARK].sort_values(
        "score", ascending=False).reset_index(drop=True)
    f["thesis"] = f.apply(
        lambda r: f"3m momentum {r['ret_3m']:+.1%}; 6m {r['ret_6m']:+.1%}; "
                  f"{'above' if r['dist_sma200'] > 0 else 'below'} 200d SMA", axis=1)
    return f


def _rank_for(source: str, feat: pd.DataFrame) -> pd.DataFrame:
    return _momentum_rank(feat) if source == "momentum" else model.rank(feat)


def run_prediction(as_of=None, strategy=None, db_path=None, prices=None):
    """Generate one strategy's risk-managed portfolio as of `as_of`.

    `strategy` (config.STRATEGIES) picks the ranking source (model or momentum)
    and the portfolio profile. The strategy name is the run's model_version, so
    each strategy is tracked as its own forward record. `prices` can be passed to
    avoid re-downloading when running multiple strategies.
    """
    strategy = strategy or config.DEFAULT_STRATEGY
    spec = config.STRATEGIES[strategy]
    profile = spec["profile"]

    if prices is None:
        prices = data_loader.get_price_history(as_of=as_of)
    feat = featureset.build_features(prices, as_of=as_of)
    if feat.empty:
        raise RuntimeError("No features computed — not enough price history.")

    ranked = _rank_for(spec["source"], feat)
    picks = ranked.head(portfolio.get_profile(profile)["n_holdings"]).copy()

    # Risk-managed position weights for this profile (cash is the residual).
    weights = portfolio.build_weights(picks, universe.load_sectors(), profile)
    weights.pop("__cash__", 0.0)
    picks["weight"] = picks["ticker"].map(weights).fillna(0.0)

    run_date = pd.to_datetime(picks["as_of"].iloc[0]).date().isoformat()

    db.init_db(db_path)
    run_id = db.save_run(
        run_date=run_date,
        model_version=strategy,          # strategy name is the tracking key
        horizon_days=config.HORIZON_DAYS,
        universe_size=len(config.UNIVERSE),
        benchmark=config.BENCHMARK,
        db_path=db_path,
    )

    pick_records = [
        {
            "ticker": r["ticker"],
            "rank": i + 1,
            "score": float(r["score"]),
            "weight": float(r["weight"]),
            "entry_price": float(r["close"]),
            "thesis": r["thesis"],
        }
        for i, (_, r) in enumerate(picks.iterrows())
    ]
    db.save_predictions(run_id, pick_records, db_path=db_path)
    return run_date, picks


def run_all_strategies(as_of=None, db_path=None):
    """Run every configured strategy off a single price download."""
    prices = data_loader.get_price_history(as_of=as_of)
    out = {}
    for name in config.STRATEGIES:
        out[name] = run_prediction(as_of=as_of, strategy=name,
                                   db_path=db_path, prices=prices)
    return out


if __name__ == "__main__":
    for name, (date, picks) in run_all_strategies().items():
        inv = picks["weight"].sum()
        print(f"\n[{name}] {len(picks)} holdings as of {date} "
              f"— invested {inv:.0%}, cash {1-inv:.0%}:")
        for i, (_, r) in enumerate(picks.iterrows(), 1):
            print(f"{i:2d}. {r['ticker']:6s} w={r['weight']:.1%} "
                  f"score={r['score']:+.3f}  {r['thesis']}")
