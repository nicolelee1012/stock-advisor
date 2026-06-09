"""Produce and persist the top-N picks for a given run date.

This is the "every day the model says these 10 stocks are high-potential for next
week" step. Entry price = the close on the run date, which becomes the basis for
scoring the realized return later (see evaluate.py).
"""

import pandas as pd

import config
from src import db, data_loader, featureset, model


def run_prediction(as_of=None, top_n=None, db_path=None):
    """Generate top-N picks as of `as_of` (default: latest available date).

    Returns (run_date, picks_df). Persists a run + its predictions to the DB.
    """
    top_n = top_n or config.TOP_N

    prices = data_loader.get_price_history(as_of=as_of)
    feat = featureset.build_features(prices, as_of=as_of)
    if feat.empty:
        raise RuntimeError("No features computed — not enough price history.")

    ranked = model.rank(feat)
    picks = ranked.head(top_n).copy()

    # run_date = the as-of date of the data we actually used.
    run_date = pd.to_datetime(picks["as_of"].iloc[0]).date().isoformat()

    db.init_db(db_path)
    run_id = db.save_run(
        run_date=run_date,
        model_version=model.MODEL_VERSION,
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
            "entry_price": float(r["close"]),
            "thesis": r["thesis"],
        }
        for i, (_, r) in enumerate(picks.iterrows())
    ]
    db.save_predictions(run_id, pick_records, db_path=db_path)

    return run_date, picks


if __name__ == "__main__":
    date, picks = run_prediction()
    print(f"\nTop {len(picks)} picks as of {date} ({model.MODEL_VERSION}):\n")
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        print(f"{i:2d}. {r['ticker']:6s} score={r['score']:+.2f}  {r['thesis']}")
