"""Command-line entry point for prediction.

Run from the project root, after training:

    python entrypoint/predict.py
    python entrypoint/predict.py --score        # also run the assessment's scorer
    python entrypoint/predict.py --log-level DEBUG

Writes validation_predictions.csv and fills data/december_chart_inputs.csv,
which are the two files score.py checks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# Lets the script run from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, ConfigError, load_config
from src.logger import get_logger, setup_logging
from src.models.persistence import PersistenceError
from src.pipelines.inference_pipeline import (
    InferenceResult,
    run_inference,
)

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

SCORER = "score.py"
RULE = "-" * 78

# Below this, the December curve is flat enough to suggest the time features
# stopped working past the end of the training data.
FLAT_CURVE_PCT = 0.5


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict every validation load and fill the December chart inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python entrypoint/predict.py\n"
            "  python entrypoint/predict.py --score\n"
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
        "--score",
        action="store_true",
        help="run the assessment's score.py afterwards to validate both files "
             "and draw the December chart",
    )
    return parser.parse_args()


def report(result: InferenceResult, config: Config) -> None:
    """Print what was written and how the December curve behaves.

    Args:
        result: Where the files went and what they contain.
        config: The configuration the run used.
    """
    print(f"\nFiles written\n{RULE}")
    print(result.summary().to_string(index=False))

    december = result.december_predictions
    spread = float(december.max() - december.min())
    relative = spread / float(december.mean()) * 100

    print(f"\nDecember curve\n{RULE}")
    print(f"first day (Dec 1)  ${december[0]:,.2f}")
    print(f"last day  (Dec 31) ${december[-1]:,.2f}")
    print(f"trend across month ${december[-1] - december[0]:+,.2f}")
    print(f"spread             ${spread:,.2f} ({relative:.2f}% of the mean)")
    print(f"distinct values    {len(np.unique(np.round(december, 2)))} of {len(december)}")

    # A flat line here is the classic sign that the seasonal features stopped
    # extrapolating, so it is worth saying loudly rather than only logging.
    if relative < FLAT_CURVE_PCT:
        print("\nWARNING: the curve is nearly flat. Check that the seasonal")
        print("features are still defined past the end of the training window.")

    print(f"\n{RULE}")
    print(f"Submission  {config.paths.submission}")
    print(f"December    {config.paths.december}")


def run_scorer(config: Config) -> int:
    """Run the assessment's own scorer on the two output files.

    Args:
        config: The configuration the run used.

    Returns:
        0 if the scorer accepted both files, 1 otherwise.
    """
    root = config.paths.submission.parent
    scorer = root / SCORER

    if not scorer.is_file():
        logger.warning("%s not found — skipping validation", scorer)
        return EXIT_OK

    command = [
        sys.executable,
        str(scorer),
        "--predictions",
        str(config.paths.submission),
        "--december-predictions",
        str(config.paths.december),
    ]

    print(f"\nRunning {SCORER}\n{RULE}")
    completed = subprocess.run(command, cwd=root, check=False)

    if completed.returncode != EXIT_OK:
        logger.error("%s rejected the output files", SCORER)
        return EXIT_FAILED

    return EXIT_OK


def main() -> int:
    """Produce both output files from the saved model.

    Returns:
        0 when prediction succeeded, 1 when it did not.
    """
    args = parse_args()
    setup_logging(level=args.log_level)

    try:
        config = load_config(args.config)
        result = run_inference(config)
    except ConfigError as exc:
        logger.error("Configuration problem: %s", exc)
        return EXIT_FAILED
    except PersistenceError as exc:
        logger.error("%s", exc)
        logger.error("Train a model first: python entrypoint/train.py")
        return EXIT_FAILED
    except Exception:
        logger.exception("Prediction failed")
        return EXIT_FAILED

    report(result, config)

    if args.score:
        return run_scorer(config)

    print(f"\nNext: python {SCORER} \\")
    print(f"        --predictions {config.paths.submission.name} \\")
    print(f"        --december-predictions data/{config.paths.december.name}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())