"""Checks on the data-quality fixes.

The rule these tests exist to protect: cleaning may drop rows during training,
but never at prediction time, because the scorer demands a rate for all 12,000
validation loads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    CleaningArtifacts,
    CleaningError,
    clean,
    fit_cleaning_artifacts,
)


def test_negative_weights_are_flipped(dirty_loads, config):
    """Sign-flipped weights come back positive rather than being dropped."""
    before = dirty_loads.loc[dirty_loads["weight"] < 0, "weight"].abs().tolist()

    cleaned, _, report = clean(dirty_loads, config, is_training=True)

    assert report.negative_weights_fixed == dirty_loads.attrs["n_negative_weight"]
    assert (cleaned["weight"] >= 0).all()

    # The magnitudes survive, so the true weight is recovered.
    assert set(before).issubset(set(cleaned["weight"]))


def test_missing_weights_are_imputed(dirty_loads, config):
    """No weight is left missing after cleaning."""
    assert dirty_loads["weight"].isna().sum() == dirty_loads.attrs["n_missing_weight"]

    cleaned, _, report = clean(dirty_loads, config, is_training=True)

    assert report.weights_imputed == dirty_loads.attrs["n_missing_weight"]
    assert cleaned["weight"].isna().sum() == 0


def test_weight_imputed_with_equipment_median(clean_loads, config):
    """A missing weight is filled with the median for its equipment type."""
    frame = clean_loads.copy()
    expected = frame.groupby("equipment")["weight"].median()

    row = 0
    equipment = frame.loc[row, "equipment"]
    frame.loc[row, "weight"] = np.nan

    cleaned, _, _ = clean(frame, config, is_training=True)
    filled = cleaned.loc[cleaned["load_id"] == frame.loc[row, "load_id"], "weight"]

    assert filled.iloc[0] == pytest.approx(expected[equipment], rel=0.05)


def test_corrupt_rates_dropped_during_training(dirty_loads, config):
    """Rates far outside the sane per-mile band are removed from training."""
    expected = dirty_loads.attrs["n_rate_high"] + dirty_loads.attrs["n_rate_low"]

    cleaned, _, report = clean(dirty_loads, config, is_training=True)

    assert report.rates_dropped_high == dirty_loads.attrs["n_rate_high"]
    assert report.rates_dropped_low == dirty_loads.attrs["n_rate_low"]
    assert report.rows_dropped == expected

    rate_per_mile = cleaned["posted_rate"] / cleaned["distance"]
    assert rate_per_mile.between(config.cleaning.rpm_lower, config.cleaning.rpm_upper).all()


def test_no_rows_dropped_at_prediction_time(unlabelled_loads, clean_loads, config):
    """Every load survives cleaning when we are predicting.

    This is the guarantee the whole submission depends on. Losing even one row
    means score.py rejects the file.
    """
    artifacts = fit_cleaning_artifacts(clean_loads, config)

    cleaned, _, report = clean(unlabelled_loads, config, is_training=False, artifacts=artifacts)

    assert len(cleaned) == len(unlabelled_loads)
    assert report.rows_dropped == 0
    assert report.rates_dropped_high == 0
    assert report.rates_dropped_low == 0


def test_prediction_requires_artifacts(unlabelled_loads, config):
    """Cleaning refuses to run at prediction time without training artifacts.

    Recomputing the medians on the data being scored would leak information
    from the scoring set into the pipeline.
    """
    with pytest.raises(CleaningError, match="artifacts are required"):
        clean(unlabelled_loads, config, is_training=False)


def test_artifacts_survive_a_round_trip(clean_loads, config, tmp_path):
    """Artifacts saved to disk load back unchanged."""
    artifacts = fit_cleaning_artifacts(clean_loads, config)
    path = tmp_path / "cleaning_artifacts.json"

    artifacts.to_json(path)

    assert CleaningArtifacts.from_json(path) == artifacts


def test_missing_artifacts_file_is_reported(tmp_path):
    """A missing artifacts file fails with a clear message."""
    with pytest.raises(CleaningError, match="not found"):
        CleaningArtifacts.from_json(tmp_path / "nothing.json")


def test_training_artifacts_are_reused_not_refitted(clean_loads, unlabelled_loads, config):
    """Prediction imputes with the training medians, not its own."""
    artifacts = fit_cleaning_artifacts(clean_loads, config)

    frame = unlabelled_loads.copy()
    frame["weight"] = np.nan
    equipment = frame.loc[0, "equipment"]

    cleaned, _, _ = clean(frame, config, is_training=False, artifacts=artifacts)
    expected = artifacts.weight_median_by_equipment[equipment]

    assert cleaned.loc[0, "weight"] == pytest.approx(expected)


def test_absurd_thresholds_are_caught(dirty_loads, config):
    """A filter that would remove most of the data raises rather than running."""
    import dataclasses

    broken = dataclasses.replace(
        config, cleaning=dataclasses.replace(config.cleaning, rpm_upper=1.01)
    )

    with pytest.raises(CleaningError, match="removed"):
        clean(dirty_loads, broken, is_training=True)


def test_cleaning_does_not_mutate_the_input(dirty_loads, config):
    """The caller's frame is left alone."""
    before = dirty_loads.copy()

    clean(dirty_loads, config, is_training=True)

    pd.testing.assert_frame_equal(dirty_loads, before)


def test_rows_survive_even_if_a_target_column_is_present(unlabelled_loads, clean_loads, config):
    """Row removal stays off at prediction time whatever columns are present.

    Without this, a filter written to key off the target column rather than the
    is_training flag would silently shorten the submission.
    """
    artifacts = fit_cleaning_artifacts(clean_loads, config)

    frame = unlabelled_loads.copy()
    frame["posted_rate"] = frame["distance"] * 9.0  # far outside the sane band

    cleaned, _, report = clean(frame, config, is_training=False, artifacts=artifacts)

    assert len(cleaned) == len(frame)
    assert report.rows_dropped == 0


@pytest.mark.integration
def test_real_validation_keeps_all_twelve_thousand_rows(config):
    """Cleaning the real validation file returns every load.

    The end-to-end version of the rule above: score.py rejects a submission
    that is short even one row.
    """
    if not config.paths.validation.is_file():
        pytest.skip("validation.csv not present")

    from src.data.loading import load_train, load_validation

    train = load_train(config)
    validation = load_validation(config)

    artifacts = fit_cleaning_artifacts(train, config)
    cleaned, _, _ = clean(validation, config, is_training=False, artifacts=artifacts)

    assert len(cleaned) == config.submission.expected_rows
    assert cleaned["weight"].notna().all()
    assert (cleaned["weight"] >= 0).all()
