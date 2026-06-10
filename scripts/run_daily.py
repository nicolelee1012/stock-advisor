"""Daily orchestrator — this is what the scheduler (launchd) calls.

Order matters:
  1. init_db          ensure tables exist
  2. evaluate_pending score anything that matured since last run
  3. run_prediction   generate + save today's top-N picks

Logs to data/run.log so you can audit what the scheduler did unattended.
"""

import logging
import sys
from pathlib import Path

# Make `import config` / `from src import ...` work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                  # noqa: E402
from src import db, evaluate, predict          # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(config.DATA_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("run_daily")


def main():
    log.info("=== daily run start ===")
    db.init_db()

    try:
        scored = evaluate.evaluate_pending()
        log.info("scored %d newly-matured prediction(s)", scored)
    except Exception:
        log.exception("evaluate step failed (continuing to prediction)")

    try:
        results = predict.run_all_profiles()
        for name, (run_date, picks) in results.items():
            inv = picks["weight"].sum()
            log.info("[%s] saved %d picks for %s (invested %.0f%%): %s",
                     name, len(picks), run_date, inv * 100,
                     ", ".join(picks["ticker"].tolist()))
    except Exception:
        log.exception("prediction step failed")
        return 1

    log.info("=== daily run done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
