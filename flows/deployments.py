"""Schedules for the two flows.

Run once to register both with a Prefect server:

    python flows/deployments.py

The two cadences are deliberately far apart. Monitoring is cheap and its whole
value is noticing a problem early, so it runs hourly. Retraining is expensive
and disruptive, so it runs weekly and is gated on top of that.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flows.monitor import monitor_flow
from flows.retrain import retrain_flow
from src.logger import get_logger

logger = get_logger(__name__)

# Hourly. Cheap to run and the earliest warning available.
MONITOR_CRON = "0 * * * *"

# Sunday at 02:00, when nothing else is competing for the machine.
RETRAIN_CRON = "0 2 * * 0"

WORK_POOL = "default"


def register() -> None:
    """Register both flows on their schedules."""
    monitor_flow.serve(
        name="freight-monitor-hourly",
        cron=MONITOR_CRON,
        parameters={"days": 7, "report": False},
        tags=["monitoring"],
        description="Check drift and live accuracy, and flag when a retrain is due.",
    )

    retrain_flow.serve(
        name="freight-retrain-weekly",
        cron=RETRAIN_CRON,
        parameters={"force": False, "dry_run": False},
        tags=["training"],
        description="Train a candidate and promote it only if it beats production.",
    )


if __name__ == "__main__":
    logger.info(
        "Registering deployments. Monitor %s, retrain %s", MONITOR_CRON, RETRAIN_CRON
    )
    register()
