"""Train the ranking model with walk-forward validation.

Pipeline:
  1. Build the supervised panel (dataset.build_panel).
  2. Walk-forward CV with TimeSeriesSplit (expanding window, NEVER shuffled) to
     estimate out-of-sample skill. Headline metric = Information Coefficient (IC):
     the Spearman rank correlation between predicted and actual forward excess
     return, averaged across folds. Positive & stable IC = the signal generalizes.
  3. Retrain on ALL data and persist to data/model.joblib.

Backend: LightGBM if available (libomp installed), else sklearn
HistGradientBoostingRegressor — same gradient-boosted-trees idea, no native deps.
The saved artifact records which backend produced it.
"""

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

import config
from src import dataset

MODEL_PATH = config.DATA_DIR / "model.joblib"
MODEL_VERSION = "lgbm-excess-ret-0.2"


def _make_regressor():
    """Prefer LightGBM; fall back to sklearn's HistGradientBoosting."""
    try:
        from lightgbm import LGBMRegressor
        reg = LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
        )
        return reg, "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )
        return reg, "sklearn-histgb"


def walk_forward_cv(panel, n_splits=5):
    """Expanding-window CV grouped by rebalance date (no date split across folds)."""
    dates = np.sort(panel["date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(dates), 1):
        train_dates, test_dates = dates[train_idx], dates[test_idx]
        tr = panel[panel["date"].isin(train_dates)]
        te = panel[panel["date"].isin(test_dates)]
        if tr.empty or te.empty:
            continue

        reg, backend = _make_regressor()
        reg.fit(tr[dataset.FEATURE_COLS], tr[dataset.LABEL_COL])
        pred = reg.predict(te[dataset.FEATURE_COLS])

        # IC per test date, then averaged — the standard cross-sectional metric.
        te = te.assign(_pred=pred)
        ics = []
        for _, g in te.groupby("date"):
            if len(g) >= 3:
                ic, _ = spearmanr(g["_pred"], g[dataset.LABEL_COL])
                if not np.isnan(ic):
                    ics.append(ic)
        hit = float((np.sign(pred) == np.sign(te[dataset.LABEL_COL])).mean())
        fold_metrics.append({
            "fold": i,
            "train_dates": len(train_dates),
            "test_dates": len(test_dates),
            "mean_ic": float(np.mean(ics)) if ics else np.nan,
            "dir_hit_rate": hit,
            "backend": backend,
        })

    return pd.DataFrame(fold_metrics)


def train_and_save(years=8, n_splits=5):
    print("Building training panel (this pulls ~8y of prices)...")
    panel = dataset.build_panel(years=years)
    if len(panel) < 100:
        raise RuntimeError(f"Panel too small ({len(panel)} rows) to train.")

    print("\n=== Walk-forward validation (TimeSeriesSplit) ===")
    cv = walk_forward_cv(panel, n_splits=n_splits)
    print(cv.to_string(index=False))
    print(f"\nMean IC across folds: {cv['mean_ic'].mean():+.4f}  "
          f"(positive & stable = signal generalizes out-of-sample)")
    print(f"Mean directional hit rate: {cv['dir_hit_rate'].mean():.1%}")

    print("\nRetraining on ALL data and saving artifact...")
    reg, backend = _make_regressor()
    reg.fit(panel[dataset.FEATURE_COLS], panel[dataset.LABEL_COL])

    artifact = {
        "model": reg,
        "backend": backend,
        "feature_cols": dataset.FEATURE_COLS,
        "model_version": MODEL_VERSION,
        "trained_rows": len(panel),
        "cv_mean_ic": float(cv["mean_ic"].mean()),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved {backend} model -> {MODEL_PATH}")
    return artifact, cv


if __name__ == "__main__":
    train_and_save()
