"""Turning a saved model into the two files the assessment asks for.

Nothing is refitted here. The imputation medians, city coordinates, category
codes, and seasonal curve all come from the training run, so a load is scored
exactly as the model was trained to expect. The December file is written back
in place with its original seven columns untouched, because score.py checks
them by name and by order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.cleaning import clean
from src.data.loading import load_december, load_validation
from src.logger import get_logger, log_step
from src.models.persistence import ModelBundle, save_predictions

logger = get_logger(__name__)

# score.py requires these seven columns, in this order, in the December file.
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


class InferenceError(Exception):
    """Raised when predictions cannot be produced or written."""


@dataclass
class InferenceResult:
    """Where the two output files were written and what they contain."""

    submission_path: Path
    december_path: Path
    validation_predictions: np.ndarray
    december_predictions: np.ndarray

    def summary(self) -> pd.DataFrame:
        """Describe both prediction sets side by side.

        Returns:
            One row per output file.
        """
        return pd.DataFrame([
            {
                "file": self.submission_path.name,
                "n": len(self.validation_predictions),
                "min": round(float(self.validation_predictions.min()), 2),
                "mean": round(float(self.validation_predictions.mean()), 2),
                "max": round(float(self.validation_predictions.max()), 2),
            },
            {
                "file": self.december_path.name,
                "n": len(self.december_predictions),
                "min": round(float(self.december_predictions.min()), 2),
                "mean": round(float(self.december_predictions.mean()), 2),
                "max": round(float(self.december_predictions.max()), 2),
            },
        ])


def _predict(df: pd.DataFrame, bundle: ModelBundle, config: Config, label: str) -> np.ndarray:
    """Clean, build features, and predict for one set of loads.

    Args:
        df: Raw loads.
        bundle: The saved model and its fitted state.
        config: Loaded project configuration.
        label: Name for this dataset in the log lines.

    Returns:
        Predicted rates in dollars.

    Raises:
        InferenceError: If cleaning changed the number of rows.
    """
    cleaned, _, _ = clean(
        df,
        config,
        is_training=False,
        artifacts=bundle.cleaning,
        label=label,
    )

    # A dropped row here would mean a short submission, so this is worth
    # checking rather than trusting.
    if len(cleaned) != len(df):
        raise InferenceError(
            f"{label}: cleaning changed the row count from {len(df):,} to {len(cleaned):,}"
        )

    features = bundle.features.transform(cleaned)
    return bundle.model.predict(features)


def predict_validation(bundle: ModelBundle, config: Config) -> tuple[pd.Series, np.ndarray]:
    """Score the 12,000 loads that need final predictions.

    Args:
        bundle: The saved model and its fitted state.
        config: Loaded project configuration.

    Returns:
        The load identifiers and their predicted rates.
    """
    validation = load_validation(config)
    predictions = _predict(validation, bundle, config, "validation")
    return validation[config.submission.id_column], predictions


def predict_december(bundle: ModelBundle, config: Config) -> np.ndarray:
    """Score the 31 fixed rows behind the December chart.

    This file has no coordinates and no market columns, which is why the model
    was trained without them and why the coordinate lookup exists.

    Args:
        bundle: The saved model and its fitted state.
        config: Loaded project configuration.

    Returns:
        One predicted rate per day of December.
    """
    december = load_december(config)
    return _predict(december, bundle, config, "december")


def write_december(predictions: np.ndarray, config: Config) -> Path:
    """Fill the December file's predicted_rate column, in place.

    The original file is reread as plain text so dates and numbers keep the
    formatting score.py expects. Only the empty column is filled.

    Args:
        predictions: One rate per day, in file order.
        config: Loaded project configuration.

    Returns:
        The path written.

    Raises:
        InferenceError: If the row count or column layout would break the scorer.
    """
    path = config.paths.december
    original = pd.read_csv(path)

    if len(original) != len(predictions):
        raise InferenceError(
            f"December file has {len(original)} rows but {len(predictions)} predictions"
        )

    original["predicted_rate"] = np.round(predictions, 2)

    if list(original.columns) != DECEMBER_COLUMNS:
        raise InferenceError(
            f"December columns must stay {DECEMBER_COLUMNS}, found {list(original.columns)}"
        )

    if (original["predicted_rate"] <= 0).any():
        raise InferenceError("December predictions must all be positive")

    original.to_csv(path, index=False)

    logger.info(
        "Filled %s December rates in %s ($%.2f to $%.2f)",
        len(original),
        path.name,
        original["predicted_rate"].min(),
        original["predicted_rate"].max(),
    )
    return path


def run_inference(config: Config) -> InferenceResult:
    """Produce both output files from the saved model.

    Args:
        config: Loaded project configuration.

    Returns:
        Where the files were written and what they contain.

    Raises:
        InferenceError: If no trained model is available.
    """
    with log_step(logger, "Loading the model bundle"):
        bundle = ModelBundle.load(config.paths.model_dir, config)

    with log_step(logger, "Predicting the validation loads"):
        load_ids, validation_predictions = predict_validation(bundle, config)

    with log_step(logger, "Writing the submission file"):
        save_predictions(
            load_ids,
            pd.Series(validation_predictions),
            config.paths.submission,
            config,
        )

    with log_step(logger, "Predicting the December chart rows"):
        december_predictions = predict_december(bundle, config)

    with log_step(logger, "Filling the December file"):
        december_path = write_december(december_predictions, config)

    result = InferenceResult(
        submission_path=config.paths.submission,
        december_path=december_path,
        validation_predictions=validation_predictions,
        december_predictions=december_predictions,
    )

    _warn_if_december_is_flat(december_predictions)
    return result


def _warn_if_december_is_flat(predictions: np.ndarray) -> None:
    """Check the December curve actually moves.

    A flat line is the classic sign that time features stopped extrapolating
    past the end of the training data. Better to catch it here than to submit
    a chart that shows it.

    Args:
        predictions: The 31 December rates.
    """
    spread = float(predictions.max() - predictions.min())
    relative = spread / float(predictions.mean()) * 100
    distinct = len(np.unique(np.round(predictions, 2)))

    logger.info(
        "December curve: %s distinct values, $%.2f spread (%.2f%% of the mean)",
        distinct,
        spread,
        relative,
    )

    if distinct <= 2 or relative < 0.5:
        logger.warning(
            "The December curve is nearly flat — check that the seasonal "
            "features are extrapolating past the training window"
        )