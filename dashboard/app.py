"""Streamlit dashboard — your human-in-the-loop review surface.

Run with:  streamlit run dashboard/app.py

Three tabs:
  Today's picks   the latest run's top-N with rank, score, thesis, entry price
  Track record    hit rate, avg excess return, equity curve of picks vs benchmark
  All predictions searchable table of every pick joined to its outcome (if matured)

This is decision SUPPORT — it shows ideas, reasons, and the model's real track
record. It does not trade.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

st.set_page_config(page_title="Stock Decision Support", layout="wide")


@st.cache_data(ttl=60)
def load_tables():
    if not config.DB_PATH.exists():
        return None
    conn = sqlite3.connect(config.DB_PATH)
    try:
        runs = pd.read_sql("SELECT * FROM prediction_runs", conn)
        preds = pd.read_sql(
            "SELECT p.*, r.run_date, r.model_version, r.horizon_days, r.benchmark "
            "FROM predictions p JOIN prediction_runs r ON p.run_id = r.run_id", conn)
        outcomes = pd.read_sql("SELECT * FROM outcomes", conn)
    finally:
        conn.close()
    return runs, preds, outcomes


st.title("📈 Stock Decision-Support System")
st.caption("Decision support, not an autopilot. Ranks a universe daily and tracks "
           "how the picks actually performed vs the benchmark. Not financial advice.")

data = load_tables()
if data is None:
    st.warning("No database yet. Run `python scripts/run_daily.py` to generate the "
               "first set of predictions.")
    st.stop()

runs, preds, outcomes = data
if preds.empty:
    st.warning("No predictions recorded yet. Run `python scripts/run_daily.py`.")
    st.stop()

merged = preds.merge(outcomes, on="pred_id", how="left")

tab_today, tab_record, tab_backtest, tab_all = st.tabs(
    ["🎯 Today's picks", "📊 Track record", "🧪 Backtest", "🗂️ All predictions"])

# --------------------------------------------------------------------------
with tab_today:
    latest_date = preds["run_date"].max()
    today = preds[preds["run_date"] == latest_date].sort_values("rank")
    st.subheader(f"Top {len(today)} for the next {config.HORIZON_DAYS} trading days")
    st.caption(f"Run date: {latest_date} · model: {today['model_version'].iloc[0]}")
    st.dataframe(
        today[["rank", "ticker", "score", "entry_price", "thesis"]]
        .rename(columns={"entry_price": "entry $", "thesis": "why"}),
        hide_index=True, use_container_width=True,
    )
    st.info("Each pick matures after the horizon, then gets scored vs "
            f"{config.BENCHMARK}. Check the Track record tab as outcomes fill in.")

# --------------------------------------------------------------------------
with tab_record:
    scored = merged.dropna(subset=["excess_return"])
    if scored.empty:
        st.info("No matured outcomes yet — come back after the horizon passes "
                "(or back-test by predicting with an older as_of date).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Picks scored", len(scored))
        c2.metric("Hit rate", f"{scored['hit'].mean():.0%}",
                  help="Share of picks that beat the benchmark")
        c3.metric("Avg excess return", f"{scored['excess_return'].mean():+.2%}")
        c4.metric("Avg realized return", f"{scored['realized_return'].mean():+.2%}")

        st.subheader("Mean excess return by run date")
        by_date = (scored.groupby("run_date")["excess_return"].mean()
                   .reset_index().set_index("run_date"))
        st.bar_chart(by_date)

        st.subheader("Cumulative avg excess return (picks vs benchmark)")
        cum = (scored.sort_values("run_date").groupby("run_date")["excess_return"]
               .mean().cumsum().reset_index().set_index("run_date"))
        st.line_chart(cum)

# --------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Leak-free walk-forward backtest (net of fees)")
    metrics_path = config.DATA_DIR / "backtest_metrics.csv"
    curves_path = config.DATA_DIR / "backtest_curves.csv"
    if not metrics_path.exists():
        st.info("No backtest yet. Run `python scripts/run_backtest.py` to generate "
                "the comparison vs the baselines.")
    else:
        m = pd.read_csv(metrics_path, index_col=0)
        show = m[["total_return", "ann_return", "sharpe", "max_drawdown", "n_periods"]].copy()
        for c in ["total_return", "ann_return", "max_drawdown"]:
            show[c] = show[c].map(lambda x: f"{x:+.1%}")
        show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}")
        st.dataframe(show, use_container_width=True)
        st.caption("The model must beat momentum / equal-weight / buy-hold SPY "
                   "AFTER fees to be worth using. The model row uses a strictly "
                   "walk-forward (no look-ahead) backtest.")
        if curves_path.exists():
            curves = pd.read_csv(curves_path, index_col=0, parse_dates=True)
            st.subheader("Equity curves (growth of $1)")
            st.line_chart(curves)

# --------------------------------------------------------------------------
with tab_all:
    st.subheader("Every prediction")
    q = st.text_input("Filter by ticker", "").strip().upper()
    view = merged.copy()
    if q:
        view = view[view["ticker"].str.contains(q)]
    view["status"] = view["excess_return"].apply(
        lambda x: "matured" if pd.notna(x) else "pending")
    cols = ["run_date", "rank", "ticker", "score", "entry_price", "exit_price",
            "realized_return", "benchmark_return", "excess_return", "hit", "status"]
    st.dataframe(view[cols].sort_values(["run_date", "rank"], ascending=[False, True]),
                 hide_index=True, use_container_width=True)
