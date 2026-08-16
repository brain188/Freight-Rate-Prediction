"""Logging training runs to MLflow and reading the registry back.

Two jobs. Wrapping a training run so its parameters, metrics and artifacts are
recorded, and reading the registry so the dashboard can show what exists and
link out to the MLflow UI for the detail.

Everything here degrades rather than fails. If no tracking server is reachable
the training pipeline still runs and the dashboard still loads, because a
missing tracking server should not stop a model being trained or monitored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

# Falls back to a local directory so MLflow works with no server to set up.
# MLflow 3 put the filesystem backend into maintenance mode, so SQLite is the
# sensible local default. A server URL overrides it in Docker.
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
        """The first eight characters of the run identifier."""
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
        """Whether this version is the one being served."""
        return self.stage.lower() == "production"


class MLflowTracker:
    """Reads and writes MLflow, tolerating an unreachable server.

    Args:
        tracking_uri: Where MLflow lives. Read from MLFLOW_TRACKING_URI when omitted.
        experiment: Experiment name to log runs under.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment: str = DEFAULT_EXPERIMENT,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
        )
        self.experiment = experiment
        self._available: bool | None = None
        self._client: Any = None

    @property
    def ui_url(self) -> str | None:
        """The browser URL for the tracking server, if it has one.

        Returns:
            The URL, or None when tracking is going to a local directory.
        """
        return self.tracking_uri if self.tracking_uri.startswith("http") else None

    def run_url(self, run_id: str) -> str | None:
        """Build a deep link to one run in the MLflow UI.

        Args:
            run_id: The run to link to.

        Returns:
            A URL, or None when there is no server to link to.
        """
        if not self.ui_url:
            return None
        return f"{self.ui_url}/#/experiments/0/runs/{run_id}"

    def model_url(self, name: str, version: str) -> str | None:
        """Build a deep link to one registered model version.

        Args:
            name: Registered model name.
            version: Version number.

        Returns:
            A URL, or None when there is no server to link to.
        """
        if not self.ui_url:
            return None
        return f"{self.ui_url}/#/models/{name}/versions/{version}"

    @property
    def is_available(self) -> bool:
        """Whether MLflow can be reached.

        Checked once and cached, so a dashboard refresh does not retry a dead
        server on every callback.

        Returns:
            True when tracking is usable.
        """
        if self._available is None:
            self._available = self._connect()
        return self._available

    def _connect(self) -> bool:
        """Try to reach the tracking server.

        Returns:
            True when the connection succeeded.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
            self._client.search_experiments(max_results=1)
            logger.info("MLflow ready at %s", self.tracking_uri)
            return True
        except Exception as exc:
            logger.warning(
                "MLflow unavailable (%s). Training and monitoring continue without it.",
                exc.__class__.__name__,
            )
            return False

    def log_training_run(
        self,
        config: Any,
        result: Any,
        model_dir: Path,
        register: bool = True,
    ) -> str | None:
        """Record a completed training run.

        Args:
            config: The configuration the run used.
            result: The TrainingResult the pipeline produced.
            model_dir: Directory holding the saved bundle.
            register: Whether to add a version to the model registry.

        Returns:
            The run identifier, or None when MLflow could not be reached.
        """
        if not self.is_available:
            return None

        try:
            import mlflow

            mlflow.set_experiment(self.experiment)
            name = f"train-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({
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
                    **{f"lgbm_{k}": v for k, v in config.model.params.items()},
                })

                # Prefixed so holdout, baseline and cross validation stay
                # distinguishable in the MLflow UI.
                for metric, value in result.holdout_scores.values.items():
                    mlflow.log_metric(f"holdout_{metric}", value)
                for metric, value in result.baseline_scores.values.items():
                    mlflow.log_metric(f"baseline_{metric}", value)
                for metric, value in result.cv_summary.items():
                    mlflow.log_metric(f"cv_{metric}", value)

                mlflow.log_metric(
                    "improvement_over_baseline_pct", result.improvement_over_baseline
                )

                for filename in ("metadata.json", "training_results.json", "seasonal_index.json"):
                    path = model_dir / filename
                    if path.is_file():
                        mlflow.log_artifact(str(path))

                if register:
                    self._register(run.info.run_id, result)

                logger.info("Logged run %s to MLflow", run.info.run_id[:8])
                return run.info.run_id

        except Exception:
            logger.exception("Could not log the run to MLflow")
            return None

    def _register(self, run_id: str, result: Any) -> None:
        """Add this run's model to the registry.

        Args:
            run_id: The run the model came from.
            result: The TrainingResult, used for the version description.
        """
        try:
            import mlflow

            model_uri = f"runs:/{run_id}/model"
            mlflow.lightgbm.log_model(
                result.bundle.model.model,
                name="model",
                registered_model_name=REGISTERED_MODEL,
            )
            logger.info("Registered a new version of %s", REGISTERED_MODEL)
        except Exception as exc:
            logger.warning("Could not register the model: %s", exc)

    def recent_runs(self, limit: int = 20) -> list[RunSummary]:
        """Read the most recent training runs.

        Args:
            limit: How many to return.

        Returns:
            Runs, newest first, or an empty list when MLflow is unreachable.
        """
        if not self.is_available:
            return []

        try:
            experiment = self._client.get_experiment_by_name(self.experiment)
            if experiment is None:
                return []

            runs = self._client.search_runs(
                [experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=limit,
            )

            summaries = []
            for run in runs:
                started = datetime.fromtimestamp(
                    run.info.start_time / 1000, tz=timezone.utc
                )
                duration = (
                    (run.info.end_time - run.info.start_time) / 1000
                    if run.info.end_time
                    else None
                )
                summaries.append(RunSummary(
                    run_id=run.info.run_id,
                    run_name=run.info.run_name or run.info.run_id[:8],
                    status=run.info.status,
                    started_at=started.strftime("%Y-%m-%d %H:%M"),
                    duration_seconds=duration,
                    metrics=dict(run.data.metrics),
                    params=dict(run.data.params),
                ))
            return summaries

        except Exception:
            logger.exception("Could not read runs from MLflow")
            return []

    def model_versions(self, name: str = REGISTERED_MODEL) -> list[ModelVersionSummary]:
        """Read the registered versions of a model.

        Args:
            name: Registered model name.

        Returns:
            Versions, newest first, or an empty list when unreachable.
        """
        if not self.is_available:
            return []

        try:
            versions = self._client.search_model_versions(f"name='{name}'")
            summaries = [
                ModelVersionSummary(
                    name=version.name,
                    version=version.version,
                    stage=getattr(version, "current_stage", "None") or "None",
                    run_id=version.run_id,
                    created_at=datetime.fromtimestamp(
                        version.creation_timestamp / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M"),
                    description=version.description or "",
                )
                for version in versions
            ]
            return sorted(summaries, key=lambda v: int(v.version), reverse=True)

        except Exception:
            logger.debug("No registered model named %s", name)
            return []


_tracker: MLflowTracker | None = None


def get_tracker() -> MLflowTracker:
    """Return the process wide tracker, creating it on first use.

    Returns:
        The tracker.
    """
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker