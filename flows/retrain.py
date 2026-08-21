"""The scheduled retraining flow.

Runs on a timer, trains a candidate in an isolated directory, and only replaces
the live model if the candidate earns it. Training into a staging directory
rather than over the top of production is what makes a failed run harmless: if
anything goes wrong, the live model has not been touched.

    python flows/retrain.py                    run once, now
    python flows/retrain.py --force            promote regardless of the gate
    python flows/retrain.py --dry-run          train and compare, never promote
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task
from prefect.logging import get_run_logger

from flows.promotion import (
    PromotionDecision,
    evaluate_candidate,
    production_metrics,
    write_decision,
)
from src.config import Config, load_config
from src.logger import setup_logging
from src.pipelines.training_pipeline import run_training

STAGING_DIR = "models_staging"
ARCHIVE_DIR = "models_archive"
DECISION_FILE = "promotion.json"


@task(name="train-candidate", retries=1, retry_delay_seconds=30)
def train_candidate(config: Config) -> dict:
    """Train a candidate model into the staging directory.

    Retried once, because a transient failure part way through a long run is
    cheaper to repeat than to page someone about.

    Args:
        config: Configuration pointing at the staging directory.

    Returns:
        The scores the run produced.
    """
    logger = get_run_logger()
    logger.info("Training a candidate into %s", config.paths.model_dir)

    result = run_training(config)

    logger.info(
        "Candidate RMSE $%.2f, MAPE %.2f%%, %.1f%% better than the baseline",
        result.holdout_scores["rmse"],
        result.holdout_scores["mape"],
        result.improvement_over_baseline,
    )

    return {
        "holdout": result.holdout_scores.to_dict(),
        "baseline": result.baseline_scores.to_dict(),
        "cv": result.cv_summary,
    }


@task(name="decide-promotion")
def decide(scores: dict, production_dir: Path) -> PromotionDecision:
    """Compare the candidate against the model currently in production.

    Args:
        scores: What the candidate scored.
        production_dir: Directory holding the live model.

    Returns:
        The decision.
    """
    logger = get_run_logger()

    decision = evaluate_candidate(
        candidate=scores["holdout"],
        baseline=scores["baseline"],
        production=production_metrics(production_dir),
    )

    logger.info("%s — %s", "PROMOTE" if decision.promote else "REJECT", decision.reason)

    for check, passed in decision.checks.items():
        logger.info("  %-28s %s", check, "pass" if passed else "FAIL")

    return decision


@task(name="promote-model")
def promote(staging_dir: Path, production_dir: Path, archive_dir: Path) -> Path:
    """Swap the candidate into production, keeping the old model.

    The outgoing model is archived rather than deleted, so a bad promotion can
    be undone by copying a directory back.

    Args:
        staging_dir: Where the candidate was trained.
        production_dir: Where the live model lives.
        archive_dir: Where superseded models are kept.

    Returns:
        Where the previous model was archived, or the production directory when
        there was nothing to archive.
    """
    logger = get_run_logger()

    archived = production_dir

    if production_dir.is_dir() and any(production_dir.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archived = archive_dir / f"model-{stamp}"
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(production_dir, archived)
        logger.info("Archived the previous model to %s", archived)

    shutil.rmtree(production_dir, ignore_errors=True)
    shutil.copytree(staging_dir, production_dir)
    logger.info("Promoted the candidate into %s", production_dir)

    return archived


@task(name="clean-staging")
def clean_staging(staging_dir: Path) -> None:
    """Remove the staging directory once a run has finished with it.

    Args:
        staging_dir: The directory to remove.
    """
    shutil.rmtree(staging_dir, ignore_errors=True)


@flow(name="freight-retrain", log_prints=True)
def retrain_flow(
    config_path: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Train a candidate model and promote it only if it earns its place.

    Args:
        config_path: Config file to use.
        force: Promote regardless of the gate. For a deliberate rollout of a
            model whose value is not visible in the holdout number.
        dry_run: Train and compare, but never promote.

    Returns:
        A summary of what happened.
    """
    import dataclasses

    logger = get_run_logger()
    setup_logging(level="INFO")

    config = load_config(config_path)
    production_dir = config.paths.model_dir
    staging_dir = production_dir.parent / STAGING_DIR
    archive_dir = production_dir.parent / ARCHIVE_DIR

    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Everything the candidate writes goes to staging, so a failure part way
    # through cannot damage the model currently being served.
    staging_config = dataclasses.replace(
        config,
        paths=dataclasses.replace(
            config.paths,
            model_dir=staging_dir,
            model_file=staging_dir / "model.pkl",
            metadata_file=staging_dir / "metadata.json",
        ),
    )

    scores = train_candidate(staging_config)
    decision = decide(scores, production_dir)
    write_decision(decision, staging_dir / DECISION_FILE)

    promoted = False

    if dry_run:
        logger.info("Dry run, so nothing was promoted")
    elif decision.promote or force:
        if force and not decision.promote:
            logger.warning("Forcing a promotion the gate would have rejected")
        promote(staging_dir, production_dir, archive_dir)
        promoted = True
    else:
        logger.info("Candidate kept in staging at %s for inspection", staging_dir)

    if promoted:
        clean_staging(staging_dir)

    return {
        "promoted": promoted,
        "reason": decision.reason,
        "candidate_rmse": decision.candidate_rmse,
        "production_rmse": decision.production_rmse,
        "improvement_pct": decision.improvement_pct,
        "dry_run": dry_run,
        "forced": force and not decision.promote,
    }


def main() -> int:
    """Run the flow from the command line.

    Returns:
        0 when a model was promoted or the run was a dry run, 1 when a
        candidate was rejected.
    """
    parser = argparse.ArgumentParser(description="Retrain and conditionally promote.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true",
                        help="promote even if the gate would reject")
    parser.add_argument("--dry-run", action="store_true",
                        help="train and compare without promoting")
    args = parser.parse_args()

    result = retrain_flow(args.config, force=args.force, dry_run=args.dry_run)

    print()
    print(f"promoted        {result['promoted']}")
    print(f"reason          {result['reason']}")
    print(f"candidate RMSE  ${result['candidate_rmse']:,.2f}")

    if result["production_rmse"] is not None:
        print(f"production RMSE ${result['production_rmse']:,.2f}")
        print(f"improvement     {result['improvement_pct']:+.2f}%")

    return 0 if result["promoted"] or result["dry_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())