# Stock Decision-Support System

A **decision-support tool** — not an autopilot. It never places trades. Every
trading day it ranks a universe of ~30 large-cap stocks and surfaces the **top 10
"high-potential for the next week"** picks, saves them, and **scores them against
the benchmark (SPY)** once the horizon has passed. A Streamlit dashboard shows
today's picks and the model's real track record.

> Not financial advice. This is a learning/research harness. Markets are noisy;
> treat every number here skeptically.

## The daily loop

```
  data (yfinance) → features → model → top-10 picks → SQLite
        ... 5 trading days later ...
  SQLite → fetch actual returns → score vs SPY (hit/miss) → SQLite → dashboard
```

## Layout

| Path | Role |
|------|------|
| `config.py` | All tunables: universe, horizon (5d), top-N (10), benchmark, costs |
| `src/universe.py` | S&P 500 constituents (cached `data/sp500.csv`); fallback core list |
| `src/data_loader.py` | yfinance fetch with an `as_of` cutoff (look-ahead guard) |
| `src/features.py` | Price features: momentum/volatility/trend, backward-only windows |
| `src/edgar.py` | SEC EDGAR fundamentals fetch (EPS/revenue) with filing dates |
| `src/fundamentals.py` | Point-in-time fundamental features (EPS/rev YoY, accel, PEAD recency) |
| `src/featureset.py` | **Single feature-assembly entry point** — price + fundamentals, used by both training and live inference (no train/serve skew) |
| `src/model.py` | LightGBM ranker behind `rank()`; momentum baseline fallback |
| `src/dataset.py` | Builds the supervised panel (features + forward excess-return label) |
| `src/train.py` | Walk-forward (`TimeSeriesSplit`) validation + saves model artifact |
| `src/backtest.py` | Leak-free walk-forward backtest vs baselines, net of fees |
| `src/predict.py` | Run for a date → save top-N picks + entry prices |
| `src/evaluate.py` | Mature & score predictions vs SPY → hit/miss/excess return |
| `src/db.py` | SQLite: `prediction_runs`, `predictions`, `outcomes` |
| `scripts/run_daily.py` | Orchestrator the scheduler calls |
| `dashboard/app.py` | Streamlit UI |

## Data sources & honest caveats

| Source | Used for | Status / caveat |
|--------|----------|-----------------|
| yfinance | Prices/volume (universe + SPY) | Free, adjusted closes |
| Wikipedia | S&P 500 constituents | **Current** list only → survivorship bias in backtests |
| SEC EDGAR | EPS/revenue fundamentals, **point-in-time** via filing dates | Free, no key, full history |
| (later) FMP/Finnhub | Analyst estimates → true earnings *surprise* + revisions | Needs paid key; free tiers don't backfill enough history |

Refresh data:
```bash
python -m src.universe          # re-pull S&P 500 list
python -m src.edgar             # re-pull EDGAR fundamentals (cached per ticker)
```

## Setup

```bash
cd ~/Desktop/trading
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

> **LightGBM note:** it's listed for Phase 2 but needs the OpenMP runtime on
> macOS (`brew install libomp`). The current baseline doesn't use it, so you can
> install libomp later when we add the trained model.

## Run it

```bash
# One full daily cycle (score matured picks, then make today's picks)
.venv/bin/python scripts/run_daily.py

# Just today's picks, printed
.venv/bin/python -m src.predict

# Score anything that has matured
.venv/bin/python -m src.evaluate

# Dashboard
.venv/bin/streamlit run dashboard/app.py
```

### Back-test the scoring loop instantly
You don't have to wait a real week to see scoring work. Predict with an old date,
then evaluate — the horizon is already in the past:

```python
from src import predict, evaluate
predict.run_prediction(as_of="2026-05-15")
evaluate.evaluate_pending()   # scores those picks vs SPY
```

## Automate (launchd, weekdays after close)

```bash
cp scripts/com.user.trading-daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.user.trading-daily.plist
launchctl list | grep trading          # confirm it's scheduled
```

Runs Mon–Fri at 14:30 Pacific (17:30 Eastern). Logs to `data/run.log` and
`data/launchd.*.log`. Edit the `Hour` in the plist if you're not on Pacific time.

## Roadmap (next phases)

1. **Trained model** — LightGBM ranker behind the same `rank()` interface, with
   **walk-forward** validation (`TimeSeriesSplit`), never random splits.
2. **Realistic backtester** — transaction costs, slippage, position/sector caps,
   turnover, max drawdown. Must beat buy-and-hold / equal-weight / simple
   momentum **after fees**.
3. **Richer features** — fundamentals, earnings dates, macro, news sentiment
   (all point-in-time).
4. **Risk controls + confidence** — the model outputs uncertainty, not just "buy".
```
