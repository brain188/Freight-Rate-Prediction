"""The full training run, from raw CSV to a saved model bundle.

The order matters. Cross-validation and the holdout are scored on models that
never saw the months they are tested on, so the numbers reported here are the
honest ones. Only after that does the final model refit on everything, which
is worth doing because the last two months are the ones closest to what we
have to predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import Config
from src.data.cleaning import CleaningArtifacts, clean
from src.data.loading import load_train
from src.features.build import FeatureBuilder
from src.logger import get_logger, log_config, log_step
from src.models.baseline import MedianRatePerMile
from src.models.estimator import FreightRateModel
from src.models.persistence import ModelBundle, build_metadata
from src.validation.metrics import Scores, compare, evaluate, summarise_folds
from src.validation.splitters import (
    Split,
    final_training_data,
    rolling_origin_splits,
    temporal_split,
)

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Everything a training run produced, for the report and the log."""

    bundle: ModelBundle
    baseline_scores: Scores
    holdout_scores: Scores
    fold_scores: list[Scores] = field(default_factory=list)
    cv_summary: dict[str, float] = field(default_factory=dict)

    @property
    def improvement_over_baseline(self) -> float:
        """How much better the model is than the baseline, as a percentage."""
        baseline = self.baseline_scores["rmse"]
        model = self.holdout_scores["rmse"]
        return float((baseline - model) / baseline * 100)

    def comparison_table(self) -> pd.DataFrame:
        """Baseline against the model on the holdout.

        Returns:
            One row per model, best first.
        """
        return compare([self.baseline_scores, self.holdout_scores])

    def summary(self) -> dict[str, Any]:
        """Collect the validation results for the metadata file.

        Returns:
            Baseline, holdout, per-fold, and cross-validation numbers.
        """
        return {
            "baseline": self.baseline_scores.to_dict(),
            "holdout": self.holdout_scores.to_dict(),
            "folds": [score.to_dict() for score in self.fold_scores],
            "cv_summary": self.cv_summary,
            "improvement_over_baseline_pct": round(self.improvement_over_baseline, 2),
        }


def _fit_and_score(split: Split, config: Config) -> Scores:
    """Train on one split and score it on the held-out side.

    The feature builder is refitted inside every split. Fitting it once on all
    the data would leak later months into earlier folds through the seasonal
    curve and the category codes.

    Args:
        split: The train/test pair.
        config: Loaded project configuration.

    Returns:
        Scores on the test side.
    """
    features = FeatureBuilder(config)
    X_train = features.fit_transform(split.train)
    X_test = features.transform(split.test)

    model = FreightRateModel(config).fit(X_train, split.train[config.project.target])
    predictions = model.predict(X_test)

    return evaluate(split.test[config.project.target].to_numpy(), predictions, split.name)


def run_training(config: Config) -> TrainingResult:
    """Run the whole training pipeline and save the model.

    Args:
        config: Loaded project configuration.

    Returns:
        The trained bundle and every score produced along the way.
    """
    log_config(logger, config)
    target = config.project.target

    with log_step(logger, "Loading training data"):
        raw = load_train(config)

    with log_step(logger, "Cleaning training data"):
        cleaned, artifacts, _ = clean(raw, config, is_training=True, label="train_test")

    with log_step(logger, "Building the time-based holdout split"):
        holdout = temporal_split(cleaned, config)

    baseline_scores = _score_baseline(holdout, config)

    fold_scores: list[Scores] = []
    cv_summary: dict[str, float] = {}

    if config.split.cv_folds:
        with log_step(logger, "Cross-validating across rolling-origin folds"):
            for split in rolling_origin_splits(cleaned, config):
                fold_scores.append(_fit_and_score(split, config))
            cv_summary = summarise_folds(fold_scores, config.evaluation.primary_metric)

    with log_step(logger, "Scoring the model on the holdout"):
        holdout_scores = _fit_and_score(holdout, config)

    bundle = _fit_final_model(cleaned, artifacts, config, baseline_scores, holdout_scores)

    result = TrainingResult(
        bundle=bundle,
        baseline_scores=baseline_scores,
        holdout_scores=holdout_scores,
        fold_scores=fold_scores,
        cv_summary=cv_summary,
    )

    bundle.metadata["validation"] = result.summary()

    with log_step(logger, "Saving the model bundle"):
        bundle.save(config.paths.model_dir)

    # Logged after the bundle is on disk, so a tracking failure cannot lose a
    # trained model. Returns None when no tracking server is reachable.
    _log_to_mlflow(config, result)

    logger.info(
        "Model beats the baseline by %.1f%% on the holdout (RMSE $%.2f vs $%.2f)",
        result.improvement_over_baseline,
        holdout_scores["rmse"],
        baseline_scores["rmse"],
    )
    return result


def _log_to_mlflow(config: Config, result: TrainingResult) -> None:
    """Record the run in MLflow, if a tracking server is reachable.

    Args:
        config: Loaded project configuration.
        result: What the training run produced.
    """
    try:
        from tracking.mlflow_tracking import get_tracker

        run_id = get_tracker().log_training_run(config, result, config.paths.model_dir)
        if run_id:
            result.bundle.metadata["mlflow_run_id"] = run_id
    except ImportError:
        logger.warning("Could not log this run to MLflow", exc_info=False)


def _score_baseline(holdout: Split, config: Config) -> Scores:
    """Score the simple reference model on the holdout.

    Args:
        holdout: The train/holdout split.
        config: Loaded project configuration.

    Returns:
        Baseline scores, or a placeholder when the baseline is disabled.
    """
    if not config.evaluation.run_baseline:
        logger.info("Baseline disabled in config — skipping")
        return Scores(label="baseline (skipped)", n=0, values={"rmse": float("inf")})

    with log_step(logger, "Scoring the baseline"):
        baseline = MedianRatePerMile(config).fit(holdout.train)
        return evaluate(
            holdout.test[config.project.target].to_numpy(),
            baseline.predict(holdout.test),
            "baseline",
        )


def _fit_final_model(
    cleaned: pd.DataFrame,
    artifacts: CleaningArtifacts,
    config: Config,
    baseline_scores: Scores,
    holdout_scores: Scores,
) -> ModelBundle:
    """Refit on the full labelled data and package everything for prediction.

    Args:
        cleaned: All cleaned training loads.
        artifacts: Imputation values learned during cleaning.
        config: Loaded project configuration.
        baseline_scores: Baseline results, stored in the metadata.
        holdout_scores: Model results, stored in the metadata.

    Returns:
        The bundle ready to be saved.
    """
    with log_step(logger, "Refitting the final model"):
        full = final_training_data(cleaned, config)

        features = FeatureBuilder(config)
        X = features.fit_transform(full)
        model = FreightRateModel(config).fit(X, full[config.project.target])

        metadata = build_metadata(
            config,
            full,
            model,
            features,
            {
                "baseline": baseline_scores.to_dict(),
                "holdout": holdout_scores.to_dict(),
            },
        )

    return ModelBundle(
        model=model,
        features=features,
        cleaning=artifacts,
        metadata=metadata,
    )