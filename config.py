"""Central configuration for the trading decision-support system.

Everything tunable lives here so the rest of the code stays clean.
This is a DECISION-SUPPORT tool, not an autopilot. Nothing here places trades.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "trading.db"

DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# The job (kept narrow on purpose)
# ---------------------------------------------------------------------------
# Each run we rank the universe and surface the TOP_N highest-potential names
# for the next HORIZON_DAYS trading days. We predict relative ranking, not price.
TOP_N = 10
HORIZON_DAYS = 5            # ~1 trading week
MODEL_VERSION = "baseline-momentum-0.1"

# Benchmark every pick is measured against (excess return = pick - benchmark).
BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# Universe. CORE_UNIVERSE is the curated fallback; the live UNIVERSE is the
# cached S&P 500 list (data/sp500.csv) when present. Refresh it with:
#     python -m src.universe
# ---------------------------------------------------------------------------
CORE_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Health care
    "UNH", "JNJ", "LLY", "PFE", "ABBV",
    # Consumer
    "WMT", "COST", "PG", "KO", "MCD", "HD",
    # Industrials / energy / other
    "CAT", "XOM", "CVX", "DIS",
]

# Live universe: cached S&P 500 if available, else the curated core. Read the
# CSV directly here (not via src.universe) to keep config import-light and avoid
# a circular import.
_SP500_CSV = DATA_DIR / "sp500.csv"
if _SP500_CSV.exists():
    import csv as _csv
    with open(_SP500_CSV, newline="") as _f:
        UNIVERSE = [row["ticker"] for row in _csv.DictReader(_f)]
else:
    UNIVERSE = CORE_UNIVERSE

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# How much history to pull for feature computation. Features need a lookback
# window (e.g. 6-month momentum), so we keep a generous buffer.
HISTORY_LOOKBACK_DAYS = 400

# ---------------------------------------------------------------------------
# Backtest realism knobs (used by evaluate / future backtester)
# ---------------------------------------------------------------------------
# Round-trip cost assumption in basis points (commissions + half-spread + slippage).
TRANSACTION_COST_BPS = 10  # 0.10% per round trip, conservative for liquid large-caps
