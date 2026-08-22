"""The scheduled monitoring flow.

Runs far more often than retraining, because the point is to notice a problem
before the next scheduled retrain rather than after it. Checks drift, checks
accuracy on whatever outcomes have arrived, and decides whether either is bad
enough to warrant retraining early.

    python flows/monitor.py
    python flows/monitor.py --days 14 --report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task
from prefect.logging import get_run_logger

from monitoring.drift import build_evidently_report, compute_drift
from monitoring.performance import snapshot, traffic_summary
from serving.store import PredictionStore, StoreUnavailableError
from src.config import Config, load_config
from src.logger import setup_logging

# Live error this far above the validation figure means the model is no longer
# performing as it was measured to, whatever the drift numbers say.
HOLDOUT_MAPE = 1.88
MAPE_DEGRADATION_FACTOR = 2.0

# Retraining is expensive and disruptive, so it is only recommended when there
# is enough scored traffic for the judgement to be sound.
MIN_SCORED_FOR_ACTION = 200


@task(name="check-drift")
def check_drift(config: Config, store: PredictionStore, days: int) -> dict:
    """Compare recent traffic against the training data.

    Args:
        config: Loaded project configuration.
        store: The prediction store.
        days: How far back to treat as current.

    Returns:
        The drift report.
    """
    logger = get_run_logger()
    report = compute_drift(store, config, days).to_dict()

    logger.info(
        "Drift %s over %s days: unknown cities %.1f%%, drifted features %s",
        report["status"],
        days,
        report["unknown_city_rate"] * 100,
        report["drifted_features"] or "none",
    )

    return report


@task(name="check-performance")
def check_performance(store: PredictionStore, days: int) -> dict:
    """Score the model on whatever outcomes have arrived.

    Args:
        store: The prediction store.
        days: How far back to look.

    Returns:
        The performance snapshot.
    """
    logger = get_run_logger()
    result = snapshot(store, days).to_dict()

    if result["n_scored"]:
        logger.info(
            "Live MAPE %.2f%% on %s scored predictions, %.0f%% coverage",
            result["metrics"].get("mape", 0),
            result["n_scored"],
            result["coverage"] * 100,
        )
    else:
        logger.info("No outcomes reported yet, so nothing can be scored")

    return result


@task(name="decide-action")
def decide_action(drift: dict, performance: dict) -> dict:
    """Work out whether anything needs doing.

    Drift alone is not enough to justify a retrain. Traffic can move without
    the model getting worse, and retraining on a whim churns the service for
    nothing. Degraded accuracy on real outcomes is the stronger signal, so it
    is weighted accordingly.

    Args:
        drift: The drift report.
        performance: The performance snapshot.

    Returns:
        What should happen and why.
    """
    logger = get_run_logger()

    reasons: list[str] = []
    scored = performance.get("n_scored", 0)
    mape = performance.get("metrics", {}).get("mape")

    if (
        mape is not None
        and scored >= MIN_SCORED_FOR_ACTION
        and mape > HOLDOUT_MAPE * MAPE_DEGRADATION_FACTOR
    ):
        reasons.append(
            f"live MAPE {mape:.2f}% is more than {MAPE_DEGRADATION_FACTOR:g} times "
            f"the {HOLDOUT_MAPE}% measured during validation"
        )

    if drift.get("status") == "alert":
        reasons.append(
            f"drift is at alert level, with {drift['unknown_city_rate']:.1%} of "
            "traffic on lanes the model has no history for"
        )

    if drift.get("drifted_features"):
        reasons.append(
            "feature distributions have moved: " + ", ".join(drift["drifted_features"])
        )

    action = "retrain" if reasons else "none"

    if action == "retrain":
        logger.warning("Recommending a retrain: %s", "; ".join(reasons))
    else:
        logger.info("No action needed")

    return {
        "action": action,
        "reasons": reasons,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


@task(name="write-evidently-report")
def write_report(config: Config, store: PredictionStore, days: int) -> str | None:
    """Build and save the full Evidently drift report.

    Args:
        config: Loaded project configuration.
        store: The prediction store.
        days: How far back to treat as current.

    Returns:
        Where the report was written, or None when it could not be built.
    """
    logger = get_run_logger()

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    output = config.paths.figures_dir.parent / "evidently" / f"drift-{stamp}.html"

    if build_evidently_report(store, config, days, output):
        logger.info("Wrote the Evidently report to %s", output)
        return str(output)

    logger.info("Not enough traffic for an Evidently report")
    return None


@flow(name="freight-monitor", log_prints=True)
def monitor_flow(
    config_path: str | None = None,
    days: int = 7,
    report: bool = False,
) -> dict:
    """Check drift and accuracy, and say whether anything needs doing.

    Args:
        config_path: Config file to use.
        days: How far back to treat as current.
        report: Whether to build the full Evidently report as well.

    Returns:
        A summary of the checks and the recommendation.

    Raises:
        StoreUnavailableError: If the prediction store cannot be reached.
    """
    logger = get_run_logger()
    setup_logging(level="INFO")

    config = load_config(config_path, create_dirs=False)
    store = PredictionStore()

    if not store.connect():
        raise StoreUnavailableError(
            "cannot monitor without the prediction store. Check DATABASE_URL."
        )

    traffic = traffic_summary(store, days)

    if not traffic.get("n_predictions"):
        logger.warning(
            "No traffic in the last %s days, so there is nothing to check", days
        )
        return {"action": "none", "reasons": ["no traffic"], "n_predictions": 0}

    drift = check_drift(config, store, days)
    performance = check_performance(store, days)
    decision = decide_action(drift, performance)

    summary = {
        **decision,
        "window_days": days,
        "n_predictions": traffic["n_predictions"],
        "n_scored": performance.get("n_scored", 0),
        "coverage": performance.get("coverage", 0),
        "live_mape": performance.get("metrics", {}).get("mape"),
        "drift_status": drift["status"],
        "unknown_city_rate": drift["unknown_city_rate"],
    }

    if report:
        summary["evidently_report"] = write_report(config, store, days)

    # Written to disk so a scheduler or an alerting rule can read the outcome
    # without parsing the logs.
    output = config.paths.logs_dir / "monitor_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return summary


def main() -> int:
    """Run the flow from the command line.

    Returns:
        0 when nothing needs doing, 1 when a retrain is recommended.
    """
    parser = argparse.ArgumentParser(description="Check drift and live accuracy.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--report", action="store_true", help="also build the full Evidently report"
    )
    args = parser.parse_args()

    result = monitor_flow(args.config, days=args.days, report=args.report)

    print()
    print(f"window          {result.get('window_days', 0)} days")
    print(f"predictions     {result.get('n_predictions', 0):,}")
    print(f"scored          {result.get('n_scored', 0):,}")

    if result.get("live_mape") is not None:
        print(f"live MAPE       {result['live_mape']:.2f}%")

    print(f"drift status    {result.get('drift_status', 'unknown')}")
    print(f"action          {result['action']}")

    for reason in result.get("reasons", []):
        print(f"  - {reason}")

    return 1 if result["action"] == "retrain" else 0


if __name__ == "__main__":
    raise SystemExit(main())
