"""Checks on the two output files.

These mirror the rules in the assessment's score.py. Catching a malformed
submission here costs a second; catching it after a full run, or worse after
submitting, costs a lot more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.persistence import PersistenceError, save_predictions
from src.pipelines.inference_pipeline import (
    DECEMBER_COLUMNS,
    InferenceError,
    write_december,
)


@pytest.fixture
def valid_submission(config) -> tuple[pd.Series, pd.Series]:
    """A correctly formed set of predictions.

    Returns:
        Load identifiers and positive rates.
    """
    ids = pd.Series(sorted(config.submission.expected_ids()))
    rates = pd.Series(np.linspace(200.0, 6000.0, len(ids)))
    return ids, rates


def test_valid_submission_is_written(valid_submission, config, tmp_path):
    """A well-formed submission produces the two columns the scorer wants."""
    ids, rates = valid_submission
    path = tmp_path / "validation_predictions.csv"

    save_predictions(ids, rates, path, config)
    written = pd.read_csv(path)

    assert list(written.columns) == ["load_id", "predicted_rate"]
    assert len(written) == config.submission.expected_rows
    assert (written["predicted_rate"] > 0).all()


def test_row_count_must_match(valid_submission, config, tmp_path):
    """A short submission is rejected."""
    ids, rates = valid_submission

    with pytest.raises(PersistenceError, match="rows"):
        save_predictions(ids[:100], rates[:100], tmp_path / "short.csv", config)


def test_duplicate_ids_are_rejected(valid_submission, config, tmp_path):
    """Repeating a load_id fails."""
    ids, rates = valid_submission
    duplicated = ids.copy()
    duplicated.iloc[1] = duplicated.iloc[0]

    with pytest.raises(PersistenceError, match="duplicate"):
        save_predictions(duplicated, rates, tmp_path / "dupes.csv", config)


def test_wrong_ids_are_rejected(valid_submission, config, tmp_path):
    """IDs that do not match the scorer's set fail."""
    ids, rates = valid_submission
    wrong = ids.copy()
    wrong.iloc[0] = "TE-999999"

    with pytest.raises(PersistenceError, match="do not match"):
        save_predictions(wrong, rates, tmp_path / "wrong.csv", config)


def test_non_positive_rates_are_rejected(valid_submission, config, tmp_path):
    """A zero or negative rate fails, as score.py requires."""
    ids, rates = valid_submission
    broken = rates.copy()
    broken.iloc[0] = 0.0

    with pytest.raises(PersistenceError, match="non-positive"):
        save_predictions(ids, broken, tmp_path / "zero.csv", config)


def test_missing_rates_are_rejected(valid_submission, config, tmp_path):
    """A gap in the predictions fails rather than being written."""
    ids, rates = valid_submission
    broken = rates.copy()
    broken.iloc[0] = np.nan

    with pytest.raises(PersistenceError, match="missing"):
        save_predictions(ids, broken, tmp_path / "gap.csv", config)


def test_expected_ids_cover_the_full_range(config):
    """The scorer's ID set runs from TE-000001 to TE-012000."""
    ids = config.submission.expected_ids()

    assert len(ids) == 12_000
    assert "TE-000001" in ids
    assert "TE-012000" in ids
    assert "TE-000000" not in ids


@pytest.mark.integration
def test_december_keeps_its_seven_columns(config, tmp_path, monkeypatch):
    """Filling the December file leaves its layout untouched.

    score.py checks the seven columns by name and by order, so anything that
    reorders or adds a column breaks the submission.
    """
    if not config.paths.december.is_file():
        pytest.skip("december_chart_inputs.csv not present")

    working = tmp_path / "december_chart_inputs.csv"
    working.write_bytes(config.paths.december.read_bytes())

    import dataclasses

    patched = dataclasses.replace(
        config, paths=dataclasses.replace(config.paths, december=working)
    )

    write_december(np.linspace(700.0, 900.0, 31), patched)
    written = pd.read_csv(working)

    assert list(written.columns) == DECEMBER_COLUMNS
    assert len(written) == 31
    assert (written["predicted_rate"] > 0).all()
    assert written["pickup"].eq("Lexington").all()
    assert written["delivery"].eq("Fort Wayne").all()
    assert np.isclose(written["distance"], 360.0).all()
    assert written["equipment"].eq("Dry Van").all()
    assert np.isclose(written["weight"], 32_000.0).all()


@pytest.mark.integration
def test_december_row_count_must_match(config, tmp_path):
    """Passing the wrong number of December predictions fails."""
    if not config.paths.december.is_file():
        pytest.skip("december_chart_inputs.csv not present")

    working = tmp_path / "december_chart_inputs.csv"
    working.write_bytes(config.paths.december.read_bytes())

    import dataclasses

    patched = dataclasses.replace(
        config, paths=dataclasses.replace(config.paths, december=working)
    )

    with pytest.raises(InferenceError, match="rows"):
        write_december(np.ones(10) * 800.0, patched)
