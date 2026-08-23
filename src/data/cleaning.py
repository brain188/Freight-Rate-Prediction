"""Data-quality fixes established in notebooks/02_data_quality.ipynb.

Two rules shape this module. Row removal only ever applies to training data,
because the scorer demands a rate for all 12,000 validation loads. And whatever
we learn to impute with is fitted on training data only, then reused at
prediction time so nothing leaks back from the scoring set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class CleaningError(Exception):
    """Raised when cleaning cannot proceed or leaves the data unusable."""


@dataclass(frozen=True)
class CleaningArtifacts:
    """Values learned on training data and reused at prediction time.

    Recomputing these on the validation set would leak information from the
    data we are being scored on, so they travel with the model instead.
    """

    weight_median_by_equipment: dict[str, float]
    weight_median_overall: float
    market_index_median: float

    def to_json(self, path: Path) -> None:
        """Write the artifacts alongside the trained model.

        Args:
            path: Destination JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        logger.debug("Saved cleaning artifacts to %s", path)

    @classmethod
    def from_json(cls, path: Path) -> CleaningArtifacts:
        """Load artifacts saved by a previous training run.

        Args:
            path: JSON file written by to_json.

        Returns:
            The stored artifacts.

        Raises:
            CleaningError: If the file is missing or malformed.
        """
        if not path.is_file():
            raise CleaningError(f"cleaning artifacts not found: {path}")

        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise CleaningError(f"could not read {path}: {exc}") from exc


@dataclass
class CleaningReport:
    """What cleaning actually changed, for logging and the written report."""

    label: str
    rows_in: int
    rows_out: int = 0
    negative_weights_fixed: int = 0
    weights_imputed: int = 0
    market_index_imputed: int = 0
    rates_dropped_high: int = 0
    rates_dropped_low: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        """Number of rows removed during cleaning."""
        return self.rows_in - self.rows_out

    def log(self) -> None:
        """Write the report to the log as one block."""
        logger.info("Cleaning report — %s", self.label)
        logger.info(
            "  rows in / out        : %s / %s",
            f"{self.rows_in:,}",
            f"{self.rows_out:,}",
        )
        logger.info("  negative weights fixed: %s", f"{self.negative_weights_fixed:,}")
        logger.info("  weights imputed       : %s", f"{self.weights_imputed:,}")
        logger.info("  market_index imputed  : %s", f"{self.market_index_imputed:,}")

        if self.rates_dropped_high or self.rates_dropped_low:
            logger.info(
                "  rates dropped         : %s high, %s low (%.2f%% of rows)",
                f"{self.rates_dropped_high:,}",
                f"{self.rates_dropped_low:,}",
                self.rows_dropped / self.rows_in * 100,
            )

        for note in self.notes:
            logger.info("  note: %s", note)


def fit_cleaning_artifacts(df: pd.DataFrame, config: Config) -> CleaningArtifacts:
    """Learn the imputation values from training data.

    Weights are imputed by equipment type because the three types differ
    slightly, and the medians are robust to the skew in the column.

    Args:
        df: Raw training data.
        config: Loaded project configuration.

    Returns:
        Artifacts to pass into every later call to clean().

    Raises:
        CleaningError: If no usable weights exist to learn from.
    """
    # Fix signs first, otherwise the negatives drag the medians down.
    weights = df["weight"].abs()

    if weights.notna().sum() == 0:
        raise CleaningError("no usable weight values to fit imputation on")

    by_equipment = weights.groupby(df["equipment"]).median()
    overall = float(weights.median())

    market_index = df["market_index"] if "market_index" in df.columns else pd.Series(dtype=float)
    market_median = float(market_index.median()) if market_index.notna().any() else 0.0

    artifacts = CleaningArtifacts(
        weight_median_by_equipment={str(k): float(v) for k, v in by_equipment.items()},
        weight_median_overall=overall,
        market_index_median=market_median,
    )

    logger.info(
        "Fitted cleaning artifacts on %s rows: weight medians %s",
        f"{len(df):,}",
        {k: round(v) for k, v in artifacts.weight_median_by_equipment.items()},
    )
    return artifacts


def _fix_negative_weight(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Flip negative weights back to positive.

    The negatives are a sign flip, not garbage: their bounds and distribution
    match the valid weights exactly, so abs() recovers the true value.

    Args:
        df: Frame to fix, modified in place.
        report: Report to record the count on.

    Returns:
        The same frame, with weights made positive.
    """
    negative = df["weight"] < 0
    report.negative_weights_fixed = int(negative.sum())

    if report.negative_weights_fixed:
        df["weight"] = df["weight"].abs()

    return df


def _impute_weight(
    df: pd.DataFrame,
    artifacts: CleaningArtifacts,
    report: CleaningReport,
) -> pd.DataFrame:
    """Fill missing weights with the median for that equipment type.

    Args:
        df: Frame to fill, modified in place.
        artifacts: Medians learned on training data.
        report: Report to record the count on.

    Returns:
        The same frame, with no missing weights.
    """
    missing = df["weight"].isna()
    report.weights_imputed = int(missing.sum())

    if not report.weights_imputed:
        return df

    by_equipment = df["equipment"].map(artifacts.weight_median_by_equipment)
    df["weight"] = df["weight"].fillna(by_equipment)

    # Catches an equipment type that never appeared during training.
    df["weight"] = df["weight"].fillna(artifacts.weight_median_overall)

    return df


def _impute_market_index(
    df: pd.DataFrame,
    artifacts: CleaningArtifacts,
    report: CleaningReport,
) -> pd.DataFrame:
    """Fill missing market_index values.

    The feature is excluded from the model by default, but imputing it keeps
    the pipeline working if it is ever switched back on.

    Args:
        df: Frame to fill, modified in place.
        artifacts: Median learned on training data.
        report: Report to record the count on.

    Returns:
        The same frame, with no missing market_index values.
    """
    if "market_index" not in df.columns:
        return df

    missing = df["market_index"].isna()
    report.market_index_imputed = int(missing.sum())

    if report.market_index_imputed:
        df["market_index"] = df["market_index"].fillna(artifacts.market_index_median)

    return df


def _drop_rate_outliers(
    df: pd.DataFrame,
    config: Config,
    report: CleaningReport,
) -> pd.DataFrame:
    """Remove loads whose rate per mile cannot be real.

    Judged per mile, not in dollars: Roughly 1.4% of rows sit far outside the main body.

    Args:
        df: Training frame to filter.
        config: Loaded project configuration.
        report: Report to record the counts on.

    Returns:
        The frame with corrupted rows removed.

    Raises:
        CleaningError: If filtering would leave too little data to train on.
    """
    target = config.project.target
    rate_per_mile = df[target] / df["distance"]

    too_high = rate_per_mile > config.cleaning.rpm_upper
    too_low = rate_per_mile < config.cleaning.rpm_lower

    report.rates_dropped_high = int(too_high.sum())
    report.rates_dropped_low = int(too_low.sum())

    kept = df[~(too_high | too_low)].reset_index(drop=True)

    if len(kept) < 0.5 * len(df):
        raise CleaningError(
            f"rate filter removed {1 - len(kept) / len(df):.0%} of rows — "
            "check cleaning.rpm_lower and cleaning.rpm_upper"
        )

    return kept


def clean(
    df: pd.DataFrame,
    config: Config,
    *,
    is_training: bool,
    artifacts: CleaningArtifacts | None = None,
    label: str | None = None,
) -> tuple[pd.DataFrame, CleaningArtifacts, CleaningReport]:
    """Apply every data-quality fix to a set of loads.

    The same function serves training and prediction. Only the is_training flag
    differs, and it controls the one step that removes rows.

    Args:
        df: Raw loads from any of the input files.
        config: Loaded project configuration.
        is_training: True for labelled data. Must be False at prediction time,
            since every validation load has to receive a rate.
        artifacts: Values learned on training data. Fitted here when training,
            required when predicting.
        label: Name for this dataset in the log lines.

    Returns:
        The cleaned frame, the artifacts used, and a report of what changed.

    Raises:
        CleaningError: If predicting without artifacts, or if cleaning leaves
            missing weights behind.
    """
    label = label or ("train" if is_training else "predict")
    report = CleaningReport(label=label, rows_in=len(df))
    out = df.copy()

    if artifacts is None:
        if not is_training:
            raise CleaningError(
                "artifacts are required when is_training=False — pass the ones "
                "saved during training so imputation stays consistent"
            )
        artifacts = fit_cleaning_artifacts(out, config)

    if config.cleaning.fix_negative_weight:
        out = _fix_negative_weight(out, report)

    if config.cleaning.impute_weight:
        out = _impute_weight(out, artifacts, report)

    if config.cleaning.impute_market_index:
        out = _impute_market_index(out, artifacts, report)

    # Training only. There is no target to check at prediction time, and the
    # scorer rejects a submission that is short even one row.
    if is_training and config.cleaning.drop_rate_outliers:
        out = _drop_rate_outliers(out, config, report)
    elif not is_training:
        report.notes.append("no rows removed — every load must receive a rate")

    report.rows_out = len(out)
    _assert_clean(out, config, is_training)
    report.log()

    return out, artifacts, report


def _assert_clean(df: pd.DataFrame, config: Config, is_training: bool) -> None:
    """Confirm cleaning left the data in the state the model expects.

    Args:
        df: The cleaned frame.
        config: Loaded project configuration.
        is_training: Whether this was the training path.

    Raises:
        CleaningError: If any expected guarantee does not hold.
    """
    if df["weight"].isna().any():
        raise CleaningError("weights are still missing after imputation")

    if (df["weight"] < 0).any():
        raise CleaningError("negative weights survived cleaning")

    if (df["distance"] <= 0).any():
        raise CleaningError("distance contains zero or negative values")

    if is_training:
        rate_per_mile = df[config.project.target] / df["distance"]
        outside = ~rate_per_mile.between(config.cleaning.rpm_lower, config.cleaning.rpm_upper)
        if outside.any():
            raise CleaningError(f"{outside.sum()} corrupted rates survived filtering")
