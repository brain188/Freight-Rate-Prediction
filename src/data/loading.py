"""Reading the CSVs and checking they contain what we expect.

Every file enters the pipeline through this module, so a missing column or a
changed dtype is caught here rather than surfacing as a strange error later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import get_logger, log_dataframe

logger = get_logger(__name__)

# Columns every load carries, whether or not we have its rate.
BASE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]

# What each file must contain. The December file is thin: no
# coordinates, no market_index, no quote_signal.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "train": ["load_id", *BASE_COLUMNS, "posted_rate"],
    "validation": ["load_id", *BASE_COLUMNS],
    "december": [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ],
    "template": ["load_id", "predicted_rate"],
}

# Set explicitly so pandas cannot infer a load_id as an integer or a weight
# as an object because of the missing values.
DTYPES: dict[str, str] = {
    "load_id": "string",
    "pickup": "string",
    "delivery": "string",
    "equipment": "string",
    "pickup_lat": "float64",
    "pickup_lon": "float64",
    "delivery_lat": "float64",
    "delivery_lon": "float64",
    "distance": "float64",
    "weight": "float64",
    "market_index": "float64",
    "quote_signal": "float64",
    "posted_rate": "float64",
    "predicted_rate": "float64",
}


class SchemaError(Exception):
    """Raised when a file is missing columns or holds unusable values."""


@dataclass(frozen=True)
class Datasets:
    """The three input files, loaded and checked."""

    train: pd.DataFrame
    validation: pd.DataFrame
    december: pd.DataFrame


def _check_columns(df: pd.DataFrame, kind: str, path: Path) -> None:
    """Confirm a file has every column the pipeline needs.

    Args:
        df: The loaded dataframe.
        kind: Which schema to check against, e.g. "train".
        path: Source file, used in the error message.

    Raises:
        SchemaError: If any required column is absent.
    """
    missing = [col for col in REQUIRED_COLUMNS[kind] if col not in df.columns]
    if missing:
        raise SchemaError(f"{path.name} is missing columns: {missing}")


def _read_csv(path: Path, kind: str) -> pd.DataFrame:
    """Read a CSV with fixed dtypes and parsed dates.

    Args:
        path: File to read.
        kind: Schema key used to validate the result.

    Returns:
        The loaded dataframe.

    Raises:
        SchemaError: If the file is missing, unreadable, or fails the schema check.
    """
    if not path.is_file():
        raise SchemaError(f"file not found: {path}")

    try:
        df = pd.read_csv(path, dtype=DTYPES, parse_dates=["date"])
    except ValueError as exc:
        raise SchemaError(f"could not parse {path.name}: {exc}") from exc

    _check_columns(df, kind, path)

    if df.empty:
        raise SchemaError(f"{path.name} contains no rows")

    return df


def load_train(config: Config) -> pd.DataFrame:
    """Load the labelled development data.

    Args:
        config: Loaded project configuration.

    Returns:
        The training loads, including posted_rate.

    Raises:
        SchemaError: If the target column holds missing or non-positive values.
    """
    df = _read_csv(config.paths.train, "train")
    target = config.project.target

    if df[target].isna().any():
        raise SchemaError(f"{target} contains missing values — the target must be complete")

    if (df[target] <= 0).any():
        raise SchemaError(f"{target} contains non-positive values")

    log_dataframe(logger, df, "train_test")
    logger.info(
        "Training dates: %s to %s",
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def load_validation(config: Config) -> pd.DataFrame:
    """Load the loads that need final predictions.

    Args:
        config: Loaded project configuration.

    Returns:
        The 12,000 validation loads.

    Raises:
        SchemaError: If the row count or load_id values do not match the scorer.
    """
    df = _read_csv(config.paths.validation, "validation")
    expected = config.submission.expected_rows

    if len(df) != expected:
        raise SchemaError(f"expected {expected:,} validation rows, found {len(df):,}")

    if df["load_id"].duplicated().any():
        raise SchemaError("validation.csv contains duplicate load_id values")

    # Catch an ID mismatch now rather than after a full training run.
    if set(df["load_id"]) != config.submission.expected_ids():
        raise SchemaError("validation load_id values do not match the scorer's expected set")

    log_dataframe(logger, df, "validation")
    logger.info(
        "Validation dates: %s to %s",
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def load_december(config: Config) -> pd.DataFrame:
    """Load the 31 fixed rows behind the December chart.

    Args:
        config: Loaded project configuration.

    Returns:
        One row per day of December 2025, with predicted_rate still empty.

    Raises:
        SchemaError: If the file does not hold exactly 31 unique December days.
    """
    df = _read_csv(config.paths.december, "december")

    if len(df) != 31:
        raise SchemaError(f"expected 31 December rows, found {len(df)}")

    if df["date"].duplicated().any():
        raise SchemaError("december_chart_inputs.csv contains duplicate dates")

    expected_days = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    if set(df["date"]) != set(expected_days):
        raise SchemaError("december_chart_inputs.csv must cover every day of December 2025")

    log_dataframe(logger, df, "december_chart_inputs")
    return df


def load_submission_template(config: Config) -> pd.DataFrame:
    """Load the submission template if it was provided.

    Args:
        config: Loaded project configuration.

    Returns:
        The template, or an empty frame built from the expected IDs when the
        file is absent. Either way the submission ends up correctly formed.
    """
    path = config.paths.submission_template

    if not path.is_file():
        logger.warning("Template not found at %s — building IDs from config", path)
        ids = sorted(config.submission.expected_ids())
        return pd.DataFrame(
            {
                config.submission.id_column: pd.Series(ids, dtype="string"),
                config.submission.prediction_column: pd.Series(dtype="float64"),
            }
        )

    df = pd.read_csv(path, dtype=DTYPES)
    _check_columns(df, "template", path)
    return df


def load_all(config: Config) -> Datasets:
    """Load and check all three input files in one call.

    Args:
        config: Loaded project configuration.

    Returns:
        A Datasets holding the training, validation, and December frames.
    """
    return Datasets(
        train=load_train(config),
        validation=load_validation(config),
        december=load_december(config),
    )


def save_interim(df: pd.DataFrame, path: Path, label: str) -> None:
    """Write an intermediate dataframe to disk.

    Parquet is used when available because it keeps dtypes intact between
    pipeline stages; CSV is a silent fallback.

    Args:
        df: Frame to write.
        path: Destination path.
        label: Name used in the log line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        written = path.with_suffix(".parquet")
    except (ImportError, ValueError):
        df.to_csv(path.with_suffix(".csv"), index=False)
        written = path.with_suffix(".csv")

    logger.debug("Saved %s to %s (%s rows)", label, written.name, f"{len(df):,}")
