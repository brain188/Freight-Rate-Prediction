"""MLflow tracking and model registry integration.

MLflow is treated as an optional observability and model-registry dependency.
If MLflow is unavailable or any tracking operation fails, training and
monitoring continue normally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "freight-rate-prediction"
REGISTERED_MODEL = "freight-rate-model"


@dataclass
class RunSummary:
    """One training run, flattened for display."""

    run_id: str
    run_name: str
    status: str
    started_at: str
    duration_seconds: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """Return the first eight characters of the run identifier."""
        return self.run_id[:8]


@dataclass
class ModelVersionSummary:
    """One registered model version."""

    name: str
    version: str
    stage: str
    run_id: str
    created_at: str
    description: str = ""

    @property
    def is_production(self) -> bool:
        """Return whether this version is marked as production."""
        return self.stage.lower() == "production"


class MLflowTracker:
    """Read and write MLflow while tolerating tracking failures.

    Args:
        tracking_uri: MLflow tracking URI. Uses MLFLOW_TRACKING_URI when
            omitted.
        experiment: Name of the MLflow experiment.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment: str = DEFAULT_EXPERIMENT,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            DEFAULT_TRACKING_URI,
        )
        self.experiment = experiment
        self._available: bool | None = None
        self._client: Any = None

    @property
    def ui_url(self) -> str | None:
        """Return the browser URL for the MLflow UI."""
        override = os.getenv("MLFLOW_UI_URL")

        if override:
            return override.rstrip("/")

        if self.tracking_uri.startswith(("http://", "https://")):
            return self.tracking_uri.rstrip("/")

        return None

    def run_url(self, run_id: str) -> str | None:
        """Build a deep link to an MLflow run."""
        if not self.ui_url:
            return None

        return f"{self.ui_url}/#/experiments/0/runs/{run_id}"

    def model_url(self, name: str, version: str) -> str | None:
        """Build a deep link to a registered model version."""
        if not self.ui_url:
            return None

        return f"{self.ui_url}/#/models/{name}/versions/{version}"

    @property
    def is_available(self) -> bool:
        """Return whether MLflow is currently available."""
        if self._available is None:
            self._available = self._connect()

        return self._available

    def _connect(self) -> bool:
        """Initialize and verify the MLflow client."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)

            self._client = MlflowClient(
                tracking_uri=self.tracking_uri,
            )

            # Lightweight connectivity check.
            self._client.search_experiments(max_results=1)

            logger.info(
                "MLflow tracking is available at %s",
                self.tracking_uri,
            )

            return True

        except ImportError:
            logger.warning(
                "MLflow is not installed; tracking will be skipped."
            )
            return False

        except Exception as exc:
            logger.warning(
                "MLflow is unavailable: %s. "
                "Training and monitoring will continue without it.",
                exc,
            )
            return False

    def log_training_run(
        self,
        config: Any,
        result: Any,
        model_dir: Path,
        register: bool = True,
    ) -> str | None:
        """Log a completed training run to MLflow.

        MLflow failures are intentionally isolated so they cannot interrupt
        the training pipeline.

        Returns:
            MLflow run ID, or None if logging failed.
        """
        if not self.is_available:
            return None

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)

            run_name = (
                f"train-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
            )

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(
                    {
                        "estimator": config.model.estimator,
                        "target_transform": config.model.target_transform,
                        "seasonal_offset": config.model.seasonal_offset,
                        "fourier_order": config.features.fourier_order,
                        "use_day_of_week": config.features.use_day_of_week,
                        "rpm_lower": config.cleaning.rpm_lower,
                        "rpm_upper": config.cleaning.rpm_upper,
                        "train_end": str(config.split.train_end),
                        "holdout_start": str(config.split.holdout_start),
                        "random_seed": config.project.random_seed,
                        **{
                            f"lgbm_{key}": value
                            for key, value in config.model.params.items()
                        },
                    }
                )

                self._log_metrics(
                    "holdout",
                    result.holdout_scores.values,
                )

                self._log_metrics(
                    "baseline",
                    result.baseline_scores.values,
                )

                self._log_metrics(
                    "cv",
                    result.cv_summary,
                )

                mlflow.log_metric(
                    "improvement_over_baseline_pct",
                    result.improvement_over_baseline,
                )

                self._log_artifacts(model_dir)

                if register:
                    self._register(
                        run_id=run.info.run_id,
                        result=result,
                    )

                run_id = run.info.run_id

                logger.info(
                    "Logged MLflow run %s",
                    run_id[:8],
                )

                return run_id

        except ImportError:
            logger.warning(
                "MLflow is unavailable while logging the training run."
            )
            return None

        except Exception as exc:
            logger.exception(
                "Could not log training run to MLflow: %s",
                exc,
            )
            return None

    @staticmethod
    def _log_metrics(
        prefix: str,
        metrics: dict[str, float],
    ) -> None:
        """Log a group of metrics with a namespace prefix."""
        import mlflow

        for metric, value in metrics.items():
            if value is None:
                continue

            mlflow.log_metric(
                f"{prefix}_{metric}",
                float(value),
            )

    @staticmethod
    def _log_artifacts(model_dir: Path) -> None:
        """Log available training artifacts."""
        import mlflow

        artifact_names = (
            "metadata.json",
            "training_results.json",
            "seasonal_index.json",
        )

        for filename in artifact_names:
            path = model_dir / filename

            if path.is_file():
                mlflow.log_artifact(str(path))

    def _register(
        self,
        run_id: str,
        result: Any,
    ) -> None:
        """Register the trained model without failing the training run."""
        try:
            import mlflow

            mlflow.lightgbm.log_model(
                result.bundle.model.model,
                name="model",
                registered_model_name=REGISTERED_MODEL,
            )

            logger.info(
                "Registered a new version of %s",
                REGISTERED_MODEL,
            )

        except ImportError:
            logger.warning(
                "MLflow is unavailable; model registration skipped."
            )

        except Exception as exc:
            logger.warning(
                "Could not register model %s: %s",
                REGISTERED_MODEL,
                exc,
                exc_info=True,
            )

    def recent_runs(
        self,
        limit: int = 20,
    ) -> list[RunSummary]:
        """Return recent training runs, newest first."""
        if not self.is_available:
            return []

        try:
            experiment = self._client.get_experiment_by_name(
                self.experiment,
            )

            if experiment is None:
                return []

            runs = self._client.search_runs(
                [experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=limit,
            )

            summaries: list[RunSummary] = []

            for run in runs:
                started = datetime.fromtimestamp(
                    run.info.start_time / 1000,
                    tz=timezone.utc,
                )

                duration = None

                if run.info.end_time:
                    duration = (
                        run.info.end_time - run.info.start_time
                    ) / 1000

                summaries.append(
                    RunSummary(
                        run_id=run.info.run_id,
                        run_name=(
                            run.info.run_name
                            or run.info.run_id[:8]
                        ),
                        status=run.info.status,
                        started_at=started.strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        duration_seconds=duration,
                        metrics=dict(run.data.metrics),
                        params=dict(run.data.params),
                    )
                )

            return summaries

        except Exception as exc:
            logger.warning(
                "Could not read recent MLflow runs: %s",
                exc,
                exc_info=True,
            )
            return []

    def model_versions(
        self,
        name: str = REGISTERED_MODEL,
    ) -> list[ModelVersionSummary]:
        """Return registered model versions, newest first."""
        if not self.is_available:
            return []

        try:
            versions = self._client.search_model_versions(
                f"name='{name}'"
            )

            summaries = [
                ModelVersionSummary(
                    name=version.name,
                    version=str(version.version),
                    stage=(
                        getattr(
                            version,
                            "current_stage",
                            "None",
                        )
                        or "None"
                    ),
                    run_id=version.run_id,
                    created_at=datetime.fromtimestamp(
                        version.creation_timestamp / 1000,
                        tz=timezone.utc,
                    ).strftime("%Y-%m-%d %H:%M"),
                    description=version.description or "",
                )
                for version in versions
            ]

            return sorted(
                summaries,
                key=lambda item: int(item.version),
                reverse=True,
            )

        except Exception as exc:
            logger.warning(
                "Could not read registered model versions for %s: %s",
                name,
                exc,
                exc_info=True,
            )
            return []


_tracker: MLflowTracker | None = None


def get_tracker() -> MLflowTracker:
    """Return the process-wide MLflow tracker."""
    global _tracker

    if _tracker is None:
        _tracker = MLflowTracker()

    return _tracker