"""Saving and restoring everything needed to make a prediction.

A trained model on its own is not enough. Prediction also needs the medians
used for imputation, the city coordinates, the category codes, and the fitted
seasonal curve. If any of those are rebuilt from the data being scored, the
predictions no longer match what the model was trained to expect. They travel
together as one bundle for that reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import Config
from src.data.cleaning import CleaningArtifacts
from src.features.build import FeatureBuilder
from src.logger import get_logger
from src.models.estimator import FreightRateModel

logger = get_logger(__name__)

MODEL_FILE = "model.pkl"
CLEANING_FILE = "cleaning_artifacts.json"
METADATA_FILE = "metadata.json"


class PersistenceError(Exception):
    """Raised when a model bundle cannot be saved or restored."""


@dataclass
class ModelBundle:
    """Everything required to turn a raw load into a predicted rate.

    Args:
        model: The fitted estimator.
        features: The fitted feature builder.
        cleaning: Imputation values learned during training.
        metadata: What was trained, when, and how well it scored.
    """

    model: FreightRateModel
    features: FeatureBuilder
    cleaning: CleaningArtifacts
    metadata: dict[str, Any]

    def save(self, directory: Path) -> None:
        """Write the bundle to a directory.

        Args:
            directory: Where to write. Created if it does not exist.

        Raises:
            PersistenceError: If the model or features are not fitted.
        """
        if not self.model.is_fitted:
            raise PersistenceError("cannot save — the model is not fitted")

        if not self.features.is_fitted:
            raise PersistenceError("cannot save — the feature builder is not fitted")

        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, directory / MODEL_FILE)
        self.features.save(directory)
        self.cleaning.to_json(directory / CLEANING_FILE)

        (directory / METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )

        logger.info("Saved model bundle to %s", directory)

    @classmethod
    def load(cls, directory: Path, config: Config) -> ModelBundle:
        """Restore a bundle written by save().

        Args:
            directory: Directory holding the saved files.
            config: Loaded project configuration.

        Returns:
            The restored bundle.

        Raises:
            PersistenceError: If any file is missing or unreadable.
        """
        model_path = directory / MODEL_FILE
        metadata_path = directory / METADATA_FILE

        if not model_path.is_file():
            raise PersistenceError(
                f"no model at {model_path} — run the training pipeline first"
            )

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise PersistenceError(f"could not load {model_path}: {exc}") from exc

        bundle = cls(
            model=model,
            features=FeatureBuilder.load(directory, config),
            cleaning=CleaningArtifacts.from_json(directory / CLEANING_FILE),
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {},
        )

        logger.info(
            "Loaded model bundle from %s (trained %s)",
            directory,
            bundle.metadata.get("trained_at", "unknown"),
        )
        bundle.check_compatible()
        return bundle

    def check_compatible(self) -> None:
        """Confirm the model and feature builder still agree.

        Guards against a bundle assembled from files written by different runs,
        which would fail later with a confusing column error.

        Raises:
            PersistenceError: If the feature lists do not match.
        """
        expected = self.features.feature_names
        actual = self.model.feature_names

        if expected != actual:
            raise PersistenceError(
                f"model expects {len(actual)} features but the builder produces "
                f"{len(expected)} — these files came from different runs"
            )


def build_metadata(
    config: Config,
    train_data: pd.DataFrame,
    model: FreightRateModel,
    features: FeatureBuilder,
    scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record what this model was trained on and how it performed.

    Written next to the model so a prediction can always be traced back to the
    run and the settings that produced it.

    Args:
        config: Loaded project configuration.
        train_data: The rows the model was finally fitted on.
        model: The fitted estimator.
        features: The fitted feature builder.
        scores: Validation results to store alongside.

    Returns:
        A JSON-serialisable summary.
    """
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": config.project.name,
        "random_seed": config.project.random_seed,
        "config_file": str(config.source_file),
        "training": {
            "n_rows": len(train_data),
            "date_from": str(train_data["date"].min().date()),
            "date_to": str(train_data["date"].max().date()),
        },
        "model": {
            "estimator": config.model.estimator,
            "target_transform": config.model.target_transform,
            "seasonal_offset": config.model.seasonal_offset,
            "smearing": round(model.transform.smearing, 6),
            "n_trees": model.best_iteration or config.model.params.get("n_estimators"),
        },
        "features": {
            "n_features": len(features.feature_names),
            "names": features.feature_names,
            "seasonal_r_squared": round(features.seasonal_index.r_squared, 4),
            "n_known_cities": len(features.coordinates.cities),
        },
        "validation": scores or {},
    }


def save_predictions(
    load_ids: pd.Series,
    predictions: pd.Series,
    path: Path,
    config: Config,
) -> None:
    """Write the submission file in the exact format score.py expects.

    Args:
        load_ids: Identifiers, one per prediction.
        predictions: Predicted rates in dollars.
        path: Where to write the CSV.
        config: Loaded project configuration.

    Raises:
        PersistenceError: If the output would fail the scorer's checks.
    """
    settings = config.submission

    frame = pd.DataFrame({
        settings.id_column: load_ids.to_numpy(),
        settings.prediction_column: predictions.to_numpy(),
    })

    if len(frame) != settings.expected_rows:
        raise PersistenceError(
            f"expected {settings.expected_rows:,} rows, got {len(frame):,}"
        )

    if frame[settings.id_column].duplicated().any():
        raise PersistenceError("submission contains duplicate load_id values")

    if set(frame[settings.id_column]) != settings.expected_ids():
        raise PersistenceError("submission load_id values do not match the scorer's set")

    if frame[settings.prediction_column].isna().any():
        raise PersistenceError("submission contains missing predictions")

    if (frame[settings.prediction_column] <= 0).any():
        raise PersistenceError("submission contains non-positive rates")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    logger.info(
        "Wrote %s predictions to %s (mean $%.2f)",
        f"{len(frame):,}",
        path,
        frame[settings.prediction_column].mean(),
    )