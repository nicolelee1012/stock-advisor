"""The ranking model.

PHASE 1 (now): a transparent momentum/quality baseline — NO training. It exists
so the whole pipeline runs end-to-end and gives us a benchmark to beat. Finance
data is noisy; a simple, explainable baseline is the honest starting point.

PHASE 2 (later): drop a trained LightGBM ranker behind the SAME `rank()`
interface. Nothing downstream changes — predict.py just calls rank(features).

The baseline blends standardized momentum (favor trending names), a mild
low-volatility preference (penalize the jumpiest), and trend confirmation
(price above its 200-day average). It also writes a human-readable thesis.
"""

import joblib
import pandas as pd

import config

BASELINE_VERSION = "baseline-momentum-0.1"
_MODEL_PATH = config.DATA_DIR / "model.joblib"


def _load_artifact():
    """Load the trained model artifact if one exists, else None."""
    if not _MODEL_PATH.exists():
        return None
    try:
        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


# Resolved at import; rank() picks the trained model when present.
MODEL_VERSION = (_load_artifact() or {}).get("model_version", BASELINE_VERSION)


def rank(features: pd.DataFrame) -> pd.DataFrame:
    """features: output of features.compute_features().

    Returns the same rows plus `score` (higher = more attractive) and `thesis`,
    sorted best-first. Uses the trained model if data/model.joblib exists,
    otherwise falls back to the transparent momentum baseline. This is the one
    function the rest of the system depends on — its signature never changes.
    """
    artifact = _load_artifact()
    if artifact is not None:
        return _rank_trained(features, artifact)
    return _rank_baseline(features)


def _rank_trained(features: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Score by the trained model's predicted forward excess return."""
    f = features.copy()
    cols = artifact["feature_cols"]
    f["score"] = artifact["model"].predict(f[cols])
    f = f.sort_values("score", ascending=False).reset_index(drop=True)
    f["thesis"] = f.apply(
        lambda r: f"pred. 5d excess {r['score']:+.2%} · " + _thesis(r), axis=1)
    return f


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std and std > 0 else s * 0.0


def _rank_baseline(features: pd.DataFrame) -> pd.DataFrame:
    """Transparent momentum/quality blend — no training. The bar to beat."""
    f = features.copy()

    # Blend of standardized signals. Weights are deliberate, not fit — this is a
    # baseline. The trained model learns these relationships from data instead.
    f["score"] = (
        0.40 * _zscore(f["ret_3m"])
        + 0.30 * _zscore(f["ret_6m"])
        + 0.15 * _zscore(f["ret_1m"])
        - 0.10 * _zscore(f["vol_1m"])          # prefer calmer names
        + 0.05 * _zscore(f["dist_sma200"])     # trend confirmation
    )

    f = f.sort_values("score", ascending=False).reset_index(drop=True)
    f["thesis"] = f.apply(_thesis, axis=1)
    return f


def _thesis(row) -> str:
    bits = []
    bits.append(f"3m momentum {row['ret_3m']:+.1%}")
    bits.append(f"6m {row['ret_6m']:+.1%}")
    bits.append("above 200d SMA" if row["dist_sma200"] > 0 else "below 200d SMA")
    bits.append(f"vol {row['vol_1m']:.0%}")
    return "; ".join(bits)
