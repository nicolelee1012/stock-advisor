"""Produce and persist the top-N picks for a given run date.

This is the "every day the model says these 10 stocks are high-potential for next
week" step. Entry price = the close on the run date, which becomes the basis for
scoring the realized return later (see evaluate.py).
"""

import pandas as pd

import config
from src import db, data_loader, featureset, model, portfolio, universe


def run_prediction(as_of=None, profile=None, db_path=None, prices=None):
    """Generate the risk-managed portfolio for one profile as of `as_of`.

    `profile` selects the construction style (config.PROFILES). The model
    version is tagged with the profile so each profile is tracked as its own
    run. Returns (run_date, picks_df with a `weight` column). `prices` can be
    passed to avoid re-downloading when running multiple profiles.
    """
    profile = profile or config.DEFAULT_PROFILE
    p = portfolio.get_profile(profile)

    if prices is None:
        prices = data_loader.get_price_history(as_of=as_of)
    feat = featureset.build_features(prices, as_of=as_of)
    if feat.empty:
        raise RuntimeError("No features computed — not enough price history.")

    ranked = model.rank(feat)
    picks = ranked.head(p["n_holdings"]).copy()

    # Risk-managed position weights for this profile (cash is the residual).
    weights = portfolio.build_weights(picks, universe.load_sectors(), profile)
    cash = weights.pop("__cash__", 0.0)
    picks["weight"] = picks["ticker"].map(weights).fillna(0.0)

    run_date = pd.to_datetime(picks["as_of"].iloc[0]).date().isoformat()
    model_version = f"{model.MODEL_VERSION}-{profile}"

    db.init_db(db_path)
    run_id = db.save_run(
        run_date=run_date,
        model_version=model_version,
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


def run_all_profiles(as_of=None, db_path=None):
    """Run every configured profile off a single price download."""
    prices = data_loader.get_price_history(as_of=as_of)
    out = {}
    for name in config.PROFILES:
        out[name] = run_prediction(as_of=as_of, profile=name,
                                   db_path=db_path, prices=prices)
    return out


if __name__ == "__main__":
    for name, (date, picks) in run_all_profiles().items():
        inv = picks["weight"].sum()
        print(f"\n[{name}] top {len(picks)} as of {date} "
              f"({model.MODEL_VERSION}) — invested {inv:.0%}, cash {1-inv:.0%}:")
        for i, (_, r) in enumerate(picks.iterrows(), 1):
            print(f"{i:2d}. {r['ticker']:6s} w={r['weight']:.1%} "
                  f"score={r['score']:+.3f}  {r['thesis']}")
