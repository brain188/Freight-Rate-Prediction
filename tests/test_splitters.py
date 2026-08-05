"""Checks on how the development data is split.

Every load we predict falls after the training window, so a split that lets
later months teach the model about earlier ones would report a score that does
not survive the real task. These tests guard that.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.validation.splitters import (
    Split,
    SplitError,
    assert_no_leakage,
    describe_splits,
    final_training_data,
    random_split,
    rolling_origin_splits,
    temporal_split,
)


def test_holdout_starts_after_training_ends(clean_loads, config):
    """No holdout date reaches back into the training window."""
    split = temporal_split(clean_loads, config)

    assert split.train["date"].max() < split.test["date"].min()
    assert split.gap_days >= 1


def test_holdout_windows_match_the_config(clean_loads, config):
    """The split respects the dates written in config.yaml."""
    split = temporal_split(clean_loads, config)

    assert split.train["date"].min().date() >= config.split.train_start
    assert split.train["date"].max().date() <= config.split.train_end
    assert split.test["date"].min().date() >= config.split.holdout_start
    assert split.test["date"].max().date() <= config.split.holdout_end


def test_split_keeps_every_row_it_should(clean_loads, config):
    """Rows are never duplicated across the two sides."""
    split = temporal_split(clean_loads, config)

    overlap = set(split.train["load_id"]) & set(split.test["load_id"])

    assert not overlap


def test_folds_never_leak(clean_loads, config):
    """Each cross-validation fold tests on months it did not train on."""
    for split in rolling_origin_splits(clean_loads, config):
        assert split.train["date"].max() < split.test["date"].min()
        assert not split.train.empty
        assert not split.test.empty


def test_folds_move_forward_in_time(clean_loads, config):
    """Later folds train on more history than earlier ones."""
    splits = rolling_origin_splits(clean_loads, config)
    cutoffs = [split.train["date"].max() for split in splits]

    assert cutoffs == sorted(cutoffs)
    assert len(splits) == len(config.split.cv_folds)


def test_leakage_guard_catches_a_random_split(clean_loads, config):
    """A shuffled split is rejected, because its windows overlap in time.

    This is the mistake the guard exists for. It is easy to make and it
    quietly inflates every number that follows.
    """
    split = random_split(clean_loads, config)

    with pytest.raises(SplitError, match="overlap"):
        assert_no_leakage(split)


def test_empty_side_is_rejected(clean_loads):
    """A split with nothing on one side fails rather than scoring on nothing."""
    empty = clean_loads.iloc[0:0]

    with pytest.raises(SplitError, match="empty"):
        assert_no_leakage(Split(name="broken", train=clean_loads, test=empty))

    with pytest.raises(SplitError, match="empty"):
        assert_no_leakage(Split(name="broken", train=empty, test=clean_loads))


def test_reversed_split_is_rejected(clean_loads, config):
    """Training on later data than the test set is caught."""
    cut = pd.Timestamp("2025-06-01")

    reversed_split = Split(
        name="reversed",
        train=clean_loads[clean_loads["date"] >= cut],
        test=clean_loads[clean_loads["date"] < cut],
    )

    with pytest.raises(SplitError, match="overlap"):
        assert_no_leakage(reversed_split)


def test_refit_uses_every_labelled_row(clean_loads, config):
    """The final model trains on all the data once validation is done."""
    full = final_training_data(clean_loads, config)

    if config.split.refit_on_full_data:
        assert len(full) == len(clean_loads)
    else:
        assert len(full) <= len(clean_loads)


def test_describe_splits_summarises_each_one(clean_loads, config):
    """The summary table has one row per split and no missing values."""
    splits = rolling_origin_splits(clean_loads, config)

    table = describe_splits(splits)

    assert len(table) == len(splits)
    assert not table.isna().any().any()
    assert (table["gap_days"] >= 1).all()