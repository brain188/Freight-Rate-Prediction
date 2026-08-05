"""Typed access to config/config.yaml.

Loads the YAML once, checks the values make sense, and exposes them as frozen
dataclasses so a typo becomes an AttributeError instead of a silent KeyError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Resolved from this file so imports work regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_LOGGING_PATH = PROJECT_ROOT / "config" / "logging.yaml"


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or inconsistent."""


def _to_date(value: str, field_name: str) -> date:
    """Parse a YAML date string into a date object.

    Args:
        value: Date string, expected in YYYY-MM-DD form.
        field_name: Name used in the error message if parsing fails.

    Returns:
        The parsed date.

    Raises:
        ConfigError: If the string is not a valid date.
    """
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{field_name}: '{value}' is not a valid date") from exc


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project settings."""

    name: str
    random_seed: int
    target: str


@dataclass(frozen=True)
class PathsConfig:
    """All file and directory locations, resolved to absolute paths."""

    train: Path
    validation: Path
    december: Path
    submission_template: Path
    preprocessed_dir: Path
    features_dir: Path
    predictions_dir: Path
    model_dir: Path
    model_file: Path
    metadata_file: Path
    submission: Path
    figures_dir: Path
    logs_dir: Path

    @classmethod
    def from_dict(cls, raw: dict[str, str], root: Path) -> PathsConfig:
        """Build the config with every path anchored to the project root.

        Args:
            raw: The `paths` block from the YAML file.
            root: Project root that relative paths are resolved against.

        Returns:
            A PathsConfig holding absolute paths.
        """
        return cls(**{key: (root / value).resolve() for key, value in raw.items()})

    def input_files(self) -> list[Path]:
        """Return the assessment-provided files the pipeline reads."""
        return [self.train, self.validation, self.december]

    def create_directories(self) -> None:
        """Create every output directory the pipeline writes to."""
        for directory in (
            self.preprocessed_dir,
            self.features_dir,
            self.predictions_dir,
            self.model_dir,
            self.figures_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CVFold:
    """One rolling-origin fold: train on everything before, test on what follows."""

    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class SplitConfig:
    """Time-based split settings.

    Random splitting would leak later months into training. Every load we
    predict falls after the training window, so the holdout must sit at the end.
    """

    strategy: str
    train_start: date
    train_end: date
    holdout_start: date
    holdout_end: date
    cv_folds: list[CVFold]
    refit_on_full_data: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SplitConfig:
        """Build the config, parsing every date string.

        Args:
            raw: The `split` block from the YAML file.

        Returns:
            A SplitConfig with dates as date objects.
        """
        folds = [
            CVFold(
                train_end=_to_date(fold["train_end"], "cv_folds.train_end"),
                test_start=_to_date(fold["test_start"], "cv_folds.test_start"),
                test_end=_to_date(fold["test_end"], "cv_folds.test_end"),
            )
            for fold in raw.get("cv_folds", [])
        ]
        return cls(
            strategy=raw["strategy"],
            train_start=_to_date(raw["train_start"], "train_start"),
            train_end=_to_date(raw["train_end"], "train_end"),
            holdout_start=_to_date(raw["holdout_start"], "holdout_start"),
            holdout_end=_to_date(raw["holdout_end"], "holdout_end"),
            cv_folds=folds,
            refit_on_full_data=raw["refit_on_full_data"],
        )

    def validate(self) -> None:
        """Check the windows are ordered and do not overlap.

        Raises:
            ConfigError: If any window is reversed or the holdout overlaps training.
        """
        if self.train_start >= self.train_end:
            raise ConfigError("split.train_start must come before split.train_end")

        if self.holdout_start <= self.train_end:
            raise ConfigError(
                "split.holdout_start must come after split.train_end, "
                "otherwise the holdout leaks into training"
            )

        if self.holdout_start >= self.holdout_end:
            raise ConfigError("split.holdout_start must come before split.holdout_end")

        for index, fold in enumerate(self.cv_folds, start=1):
            if fold.test_start <= fold.train_end:
                raise ConfigError(f"cv fold {index}: test window overlaps training")
            if fold.test_start >= fold.test_end:
                raise ConfigError(f"cv fold {index}: test window is reversed")


@dataclass(frozen=True)
class CleaningConfig:
    """Data-quality rules from notebooks/02_data_quality.ipynb."""

    fix_negative_weight: bool
    impute_weight: bool
    weight_impute_strategy: str
    impute_market_index: bool
    drop_rate_outliers: bool
    rpm_lower: float
    rpm_upper: float
    weight_min: float
    weight_max: float
    distance_min: float

    def validate(self) -> None:
        """Check the thresholds are ordered and positive.

        Raises:
            ConfigError: If a bound is reversed or non-positive.
        """
        if self.rpm_lower <= 0:
            raise ConfigError("cleaning.rpm_lower must be positive")
        if self.rpm_lower >= self.rpm_upper:
            raise ConfigError("cleaning.rpm_lower must be below cleaning.rpm_upper")
        if self.weight_min >= self.weight_max:
            raise ConfigError("cleaning.weight_min must be below cleaning.weight_max")


@dataclass(frozen=True)
class FeaturesConfig:
    """Which features to build and how.

    Args:
        use_market_index: Kept off by default. Weak signal and absent from the
            December file, so leaving it out gives one model for both inputs.
    """

    use_market_index: bool
    use_quote_signal: bool
    fourier_order: int
    fourier_period: float
    use_day_of_week: bool
    use_coordinates: bool
    use_log_distance: bool
    categorical_encoding: str
    unknown_category_value: int

    def validate(self) -> None:
        """Check the seasonal terms are usable.

        Raises:
            ConfigError: If the Fourier settings are out of range.
        """
        if not 1 <= self.fourier_order <= 10:
            raise ConfigError("features.fourier_order must be between 1 and 10")
        if self.fourier_period <= 0:
            raise ConfigError("features.fourier_period must be positive")

        # Coordinates are the only way to place the eight cities that never
        # appear in training, so encoding by name alone would break 12% of rows.
        if not self.use_coordinates:
            raise ConfigError(
                "features.use_coordinates must stay true — unseen cities have "
                "no other fallback"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Estimator choice and hyperparameters."""

    target_transform: str
    estimator: str
    params: dict[str, Any]
    early_stopping_rounds: int
    seasonal_offset: bool = False

    def validate(self) -> None:
        """Check the transform is one we support.

        Raises:
            ConfigError: If the target transform is unrecognised.
        """
        allowed = {"log", "rate_per_mile", "log_rate_per_mile", "none"}
        if self.target_transform not in allowed:
            raise ConfigError(
                f"model.target_transform must be one of {sorted(allowed)}, "
                f"got '{self.target_transform}'"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Which metrics to report and which one decides."""

    metrics: list[str]
    primary_metric: str
    run_baseline: bool

    def validate(self) -> None:
        """Check the primary metric is one we compute.

        Raises:
            ConfigError: If the primary metric is not in the metric list.
        """
        if self.primary_metric not in self.metrics:
            raise ConfigError(
                f"evaluation.primary_metric '{self.primary_metric}' is not in "
                f"evaluation.metrics {self.metrics}"
            )


@dataclass(frozen=True)
class SubmissionConfig:
    """Output format required by the assessment's score.py."""

    id_column: str
    prediction_column: str
    expected_rows: int
    id_prefix: str
    min_rate: float

    def expected_ids(self) -> set[str]:
        """Build the exact load_id set the scorer checks against.

        Returns:
            IDs from TE-000001 through TE-012000.
        """
        return {
            f"{self.id_prefix}{index:06d}"
            for index in range(1, self.expected_rows + 1)
        }


@dataclass(frozen=True)
class Config:
    """The whole configuration, ready to use."""

    project: ProjectConfig
    paths: PathsConfig
    split: SplitConfig
    cleaning: CleaningConfig
    features: FeaturesConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    submission: SubmissionConfig
    source_file: Path = field(repr=False)

    def validate(self) -> None:
        """Run every section's own checks.

        Raises:
            ConfigError: If any section is inconsistent.
        """
        self.split.validate()
        self.cleaning.validate()
        self.features.validate()
        self.model.validate()
        self.evaluation.validate()

    def check_input_files(self) -> None:
        """Confirm the provided CSVs are where the config says they are.

        Raises:
            ConfigError: If any input file is missing.
        """
        missing = [path for path in self.paths.input_files() if not path.is_file()]
        if missing:
            listed = "\n  ".join(str(path) for path in missing)
            raise ConfigError(f"input files not found:\n  {listed}")


def load_config(
    path: str | Path | None = None,
    *,
    create_dirs: bool = True,
) -> Config:
    """Load, validate, and return the project configuration.

    Args:
        path: Config file to read. Defaults to config/config.yaml.
        create_dirs: Whether to create the output directories on load.

    Returns:
        A validated Config.

    Raises:
        ConfigError: If the file is missing, malformed, or fails validation.
    """
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} is empty or not a mapping")

    required = {
        "project",
        "paths",
        "split",
        "cleaning",
        "features",
        "model",
        "evaluation",
        "submission",
    }
    if absent := required - raw.keys():
        raise ConfigError(f"config is missing sections: {sorted(absent)}")

    config = Config(
        project=ProjectConfig(**raw["project"]),
        paths=PathsConfig.from_dict(raw["paths"], PROJECT_ROOT),
        split=SplitConfig.from_dict(raw["split"]),
        cleaning=CleaningConfig(**raw["cleaning"]),
        features=FeaturesConfig(**raw["features"]),
        model=ModelConfig(**raw["model"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        submission=SubmissionConfig(**raw["submission"]),
        source_file=config_path,
    )

    config.validate()

    if create_dirs:
        config.paths.create_directories()

    return config