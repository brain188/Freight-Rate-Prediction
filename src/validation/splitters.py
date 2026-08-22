"""Splitting the development data for honest validation.

Every load we must price falls after the training window, so the split has to
imitate that. Cutting the data by date means the model is always tested on
loads it could not have seen, which is the situation it will face for real.
A random split would let October teach the model about September and report a
score that does not survive contact with the actual prediction set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class SplitError(Exception):
    """Raised when a split is impossible or would leak."""


@dataclass(frozen=True)
class Split:
    """One train/test pair drawn from the development data."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_end(self) -> date:
        """Last date the model is allowed to learn from."""
        return self.train["date"].max().date()

    @property
    def test_start(self) -> date:
        """First date the model is tested on."""
        return self.test["date"].min().date()

    @property
    def gap_days(self) -> int:
        """Days between the end of training and the start of testing."""
        return (self.test["date"].min() - self.train["date"].max()).days

    def summary(self) -> dict[str, object]:
        """Describe the split in one row.

        Returns:
            Sizes, date ranges, and the forward gap.
        """
        return {
            "split": self.name,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "train_from": self.train["date"].min().date(),
            "train_to": self.train_end,
            "test_from": self.test_start,
            "test_to": self.test["date"].max().date(),
            "gap_days": self.gap_days,
        }

    def log(self) -> None:
        """Write the split details to the log."""
        logger.info(
            "%s | train %s rows (%s to %s) | test %s rows (%s to %s)",
            self.name,
            f"{len(self.train):,}",
            self.train["date"].min().date(),
            self.train_end,
            f"{len(self.test):,}",
            self.test_start,
            self.test["date"].max().date(),
        )


def assert_no_leakage(split: Split) -> None:
    """Confirm no test date reaches back into training.

    Cheap to run and catches the single most damaging mistake in a time-series
    project, so it runs on every split we build.

    Args:
        split: The split to check.

    Raises:
        SplitError: If the windows overlap or either side is empty.
    """
    if split.train.empty:
        raise SplitError(f"{split.name}: training side is empty")

    if split.test.empty:
        raise SplitError(f"{split.name}: test side is empty")

    latest_train = split.train["date"].max()
    earliest_test = split.test["date"].min()

    if earliest_test <= latest_train:
        raise SplitError(
            f"{split.name}: test starts {earliest_test.date()} but training runs to "
            f"{latest_train.date()} — the windows overlap"
        )


def _window(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    """Select rows inside a date window, inclusive at both ends.

    Args:
        df: Frame with a date column.
        start: First date to keep, or None for no lower bound.
        end: Last date to keep, or None for no upper bound.

    Returns:
        The selected rows, with the index reset.
    """
    mask = pd.Series(True, index=df.index)

    if start is not None:
        mask &= df["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["date"] <= pd.Timestamp(end)

    return df[mask].reset_index(drop=True)


def temporal_split(df: pd.DataFrame, config: Config) -> Split:
    """Split into a training window and a later holdout window.

    The holdout is two months long to match the two-month gap between the end
    of the labelled data and the last load we have to price.

    Args:
        df: Cleaned development data.
        config: Loaded project configuration.

    Returns:
        The train/holdout split.

    Raises:
        SplitError: If either window comes back empty or they overlap.
    """
    settings = config.split

    split = Split(
        name="holdout",
        train=_window(df, settings.train_start, settings.train_end),
        test=_window(df, settings.holdout_start, settings.holdout_end),
    )

    assert_no_leakage(split)
    split.log()
    return split


def rolling_origin_splits(df: pd.DataFrame, config: Config) -> list[Split]:
    """Build the cross-validation folds defined in the config.

    Each fold trains on everything up to its cut-off and tests on the two
    months that follow, so all three folds rehearse the same forward jump the
    real task demands.

    Args:
        df: Cleaned development data.
        config: Loaded project configuration.

    Returns:
        One Split per configured fold.

    Raises:
        SplitError: If no folds are configured or a fold leaks.
    """
    if not config.split.cv_folds:
        raise SplitError("no cv_folds configured — add them under split in config.yaml")

    splits = []

    for number, fold in enumerate(config.split.cv_folds, start=1):
        split = Split(
            name=f"fold_{number}",
            train=_window(df, config.split.train_start, fold.train_end),
            test=_window(df, fold.test_start, fold.test_end),
        )
        assert_no_leakage(split)
        split.log()
        splits.append(split)

    logger.info("Built %s rolling-origin folds", len(splits))
    return splits


def random_split(
    df: pd.DataFrame,
    config: Config,
    test_size: float = 0.2,
) -> Split:
    """Split rows at random, ignoring the date.

    Not used for model selection. It exists so the report can show what a
    random split scores and why that number is misleading here.

    Args:
        df: Cleaned development data.
        config: Loaded project configuration.
        test_size: Share of rows held out.

    Returns:
        A shuffled split, which will overlap in time by design.
    """
    rng = np.random.default_rng(config.project.random_seed)
    shuffled = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))

    split = Split(
        name="random",
        train=df.iloc[shuffled[:cut]].reset_index(drop=True),
        test=df.iloc[shuffled[cut:]].reset_index(drop=True),
    )

    logger.warning(
        "Random split built for comparison only — its test window overlaps "
        "training in time and will flatter the score"
    )
    return split


def describe_splits(splits: list[Split]) -> pd.DataFrame:
    """Summarise several splits as one table for the report.

    Args:
        splits: Splits to describe.

    Returns:
        One row per split.
    """
    return pd.DataFrame([split.summary() for split in splits])


def final_training_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Return the data to refit on once the model is settled.

    After validation has done its job, holding back the most recent two months
    would be wasteful — they are the months closest to what we predict.

    Args:
        df: Cleaned development data.
        config: Loaded project configuration.

    Returns:
        All labelled rows if refitting is enabled, otherwise the training window.
    """
    if not config.split.refit_on_full_data:
        logger.info("Refitting on the training window only")
        return _window(df, config.split.train_start, config.split.train_end)

    logger.info("Refitting on all %s labelled rows", f"{len(df):,}")
    return df.reset_index(drop=True)
