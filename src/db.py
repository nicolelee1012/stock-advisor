"""SQLite persistence: store daily predictions and later score their outcomes.

This is the core of the "save predictions each day, then confirm/test them"
workflow. Three tables:

  prediction_runs  one row per day the model runs
  predictions      the top-N picks for each run (+ entry price, thesis)
  outcomes         filled in LATER, once the horizon has matured (actual vs benchmark)

A prediction is "matured" once HORIZON_DAYS trading days have passed and we can
look up what actually happened. evaluate.py walks unscored matured predictions
and writes their outcomes here.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date      TEXT NOT NULL,            -- ISO date the model ran (data as-of)
    model_version TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    universe_size INTEGER NOT NULL,
    benchmark     TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE (run_date, model_version)        -- one run per day per model version
);

CREATE TABLE IF NOT EXISTS predictions (
    pred_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES prediction_runs(run_id),
    ticker        TEXT NOT NULL,
    rank          INTEGER NOT NULL,         -- 1 = highest conviction
    score         REAL NOT NULL,            -- raw model score (higher = better)
    entry_price   REAL,                     -- close on run_date (basis for return)
    thesis        TEXT,                     -- human-readable "why"
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outcomes (
    pred_id          INTEGER PRIMARY KEY REFERENCES predictions(pred_id),
    evaluated_date   TEXT NOT NULL,         -- the date returns were measured to
    exit_price       REAL,
    realized_return  REAL,                  -- pick's return over the horizon
    benchmark_return REAL,                  -- benchmark return over same window
    excess_return    REAL,                  -- realized - benchmark (the real scorecard)
    hit              INTEGER,               -- 1 if excess_return > 0 else 0
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id);
"""


@contextmanager
def connect(db_path: Path = None):
    """Context-managed connection with foreign keys + row access by name."""
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = None):
    """Create tables if they don't exist. Safe to call repeatedly."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_run(run_date, model_version, horizon_days, universe_size, benchmark, db_path=None):
    """Create (or fetch) a prediction run, return its run_id.

    Idempotent on (run_date, model_version): re-running the same day replaces the
    old picks rather than duplicating them.
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT run_id FROM prediction_runs WHERE run_date=? AND model_version=?",
            (run_date, model_version),
        )
        existing = cur.fetchone()
        if existing:
            run_id = existing["run_id"]
            # Clear prior picks (and their outcomes) so a re-run is clean.
            conn.execute(
                "DELETE FROM outcomes WHERE pred_id IN "
                "(SELECT pred_id FROM predictions WHERE run_id=?)",
                (run_id,),
            )
            conn.execute("DELETE FROM predictions WHERE run_id=?", (run_id,))
            return run_id

        cur = conn.execute(
            "INSERT INTO prediction_runs "
            "(run_date, model_version, horizon_days, universe_size, benchmark) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_date, model_version, horizon_days, universe_size, benchmark),
        )
        return cur.lastrowid


def save_predictions(run_id, picks, db_path=None):
    """picks: list of dicts with ticker, rank, score, entry_price, thesis."""
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO predictions (run_id, ticker, rank, score, entry_price, thesis) "
            "VALUES (:run_id, :ticker, :rank, :score, :entry_price, :thesis)",
            [{"run_id": run_id, **p} for p in picks],
        )


def unscored_predictions(db_path=None):
    """Predictions that don't yet have an outcome row (need maturing/scoring)."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT p.pred_id, p.ticker, p.entry_price, r.run_date, r.horizon_days, r.benchmark "
            "FROM predictions p "
            "JOIN prediction_runs r ON p.run_id = r.run_id "
            "LEFT JOIN outcomes o ON p.pred_id = o.pred_id "
            "WHERE o.pred_id IS NULL "
            "ORDER BY r.run_date",
        ).fetchall()
        return [dict(r) for r in rows]


def save_outcome(pred_id, evaluated_date, exit_price, realized_return,
                 benchmark_return, excess_return, db_path=None):
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO outcomes "
            "(pred_id, evaluated_date, exit_price, realized_return, "
            " benchmark_return, excess_return, hit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pred_id, evaluated_date, exit_price, realized_return,
             benchmark_return, excess_return, 1 if excess_return > 0 else 0),
        )


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {config.DB_PATH}")
