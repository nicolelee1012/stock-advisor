"""Run the full backtest comparison and persist results for the dashboard.

Writes:
  data/backtest_metrics.csv   one row per strategy (return/Sharpe/drawdown)
  data/backtest_curves.csv    equity curves (date x strategy) for charting

Re-run this whenever the model or features change so the dashboard reflects the
latest honest, leak-free comparison.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src import backtest            # noqa: E402


def main():
    print("Running leak-free backtest comparison (this retrains walk-forward)...")
    table, results = backtest.compare_all()

    metrics_path = config.DATA_DIR / "backtest_metrics.csv"
    table.to_csv(metrics_path)
    print(f"\nSaved metrics -> {metrics_path}\n")
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(table[["total_return", "ann_return", "sharpe", "max_drawdown", "n_periods"]])

    curves = pd.DataFrame({name: r["equity_curve"] for name, r in results.items()})
    curves = curves.sort_index().ffill()
    curves_path = config.DATA_DIR / "backtest_curves.csv"
    curves.to_csv(curves_path)
    print(f"\nSaved equity curves -> {curves_path}")


if __name__ == "__main__":
    main()
