"""Command-line entry point for training.

Run from the project root:

    python entrypoint/train.py
    python entrypoint/train.py --config config/config.yaml --log-level DEBUG
    python entrypoint/train.py --skip-cv          # quicker while iterating

Trains the model, prints how it scored against the baseline, and writes the
model bundle to models/ ready for entrypoint/predict.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

# Lets the script run from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, ConfigError, load_config
from src.logger import get_logger, setup_logging
from src.pipelines.training_pipeline import TrainingResult, run_training

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

RESULTS_FILE = "training_results.json"
RULE = "-" * 78


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the freight rate model and save it to models/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python entrypoint/train.py\n"
            "  python entrypoint/train.py --skip-cv\n"
            "  python entrypoint/train.py --config config/experiment.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config file to use (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="how much detail to print to the console (default: INFO)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="skip cross-validation and score on the holdout only, which is "
             "much faster while iterating",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="skip the baseline model",
    )
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply the command-line flags on top of the config file.

    Args:
        config: Configuration loaded from disk.
        args: Parsed command-line arguments.

    Returns:
        The configuration to run with.
    """
    if args.skip_cv:
        logger.warning("Cross-validation skipped — the holdout is the only estimate left")
        config = dataclasses.replace(
            config, split=dataclasses.replace(config.split, cv_folds=[])
        )

    if args.skip_baseline:
        logger.warning("Baseline skipped — the model score has nothing to beat")
        config = dataclasses.replace(
            config, evaluation=dataclasses.replace(config.evaluation, run_baseline=False)
        )

    return config


def report(result: TrainingResult, config: Config) -> None:
    """Print the headline numbers and save them for the write-up.

    Args:
        result: What the training run produced.
        config: The configuration it ran with.
    """
    print(f"\nHoldout results\n{RULE}")
    print(result.comparison_table().round(3).to_string(index=False))

    if result.fold_scores:
        folds = pd.DataFrame([score.to_dict() for score in result.fold_scores])
        primary = config.evaluation.primary_metric

        print(f"\nCross-validation folds\n{RULE}")
        print(folds.round(3).to_string(index=False))
        print(
            f"\nmean {primary}: {result.cv_summary[primary]:,.2f} "
            f"(+/- {result.cv_summary[f'{primary}_std']:,.2f})"
        )

    print(f"\nTop features by gain\n{RULE}")
    print(result.bundle.model.feature_importance(top=8).to_string(index=False))

    # Written to disk so the report can quote the same numbers the run produced.
    results_path = config.paths.model_dir / RESULTS_FILE
    results_path.write_text(
        json.dumps(result.summary(), indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{RULE}")
    print(f"Model beats the baseline by {result.improvement_over_baseline:.1f}% on RMSE.")
    print(f"Bundle saved to  {config.paths.model_dir}")
    print(f"Scores saved to  {results_path}")
    print("\nNext: python entrypoint/predict.py")


def main() -> int:
    """Run the training pipeline.

    Returns:
        0 when training succeeded, 1 when it did not.
    """
    args = parse_args()
    setup_logging(level=args.log_level)

    try:
        config = apply_overrides(load_config(args.config), args)
        config.check_input_files()
        result = run_training(config)
    except ConfigError as exc:
        logger.error("Configuration problem: %s", exc)
        return EXIT_FAILED
    except Exception:
        logger.exception("Training failed")
        return EXIT_FAILED

    report(result, config)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())