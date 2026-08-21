"""Checks on the model promotion gate.

This is the guard that lets retraining run unattended. Without it a scheduled
retrain ships whatever it produced, and a bad month of data quietly degrades
the service.
"""

from __future__ import annotations

import json

import pytest

from flows.promotion import (
    MAX_ACCEPTABLE_MAPE,
    MAX_ACCEPTABLE_RMSE,
    MIN_IMPROVEMENT_PCT,
    evaluate_candidate,
    production_metrics,
    write_decision,
)

BASELINE = {"rmse": 112.61, "mape": 3.85}
PRODUCTION = {"rmse": 67.65, "mape": 1.88}


def test_first_model_is_promoted():
    """With nothing deployed, the hard floors are the only bar."""
    decision = evaluate_candidate({"rmse": 67.65, "mape": 1.88}, BASELINE, None)

    assert decision.promote is True
    assert decision.production_rmse is None


def test_clearly_better_model_is_promoted():
    """A meaningful improvement goes live."""
    decision = evaluate_candidate({"rmse": 58.0, "mape": 1.60}, BASELINE, PRODUCTION)

    assert decision.promote is True
    assert decision.improvement_pct > MIN_IMPROVEMENT_PCT


def test_identical_model_is_rejected():
    """A swap that buys nothing is pure risk, so it does not happen."""
    decision = evaluate_candidate(dict(PRODUCTION), BASELINE, PRODUCTION)

    assert decision.promote is False
    assert decision.improvement_pct == pytest.approx(0.0, abs=0.01)


def test_marginal_improvement_is_rejected():
    """An improvement inside the noise of the folds is not acted on."""
    marginal = PRODUCTION["rmse"] * (1 - (MIN_IMPROVEMENT_PCT / 2) / 100)
    decision = evaluate_candidate({"rmse": marginal, "mape": 1.87}, BASELINE, PRODUCTION)

    assert decision.promote is False


def test_worse_model_is_rejected():
    """A regression never reaches production."""
    decision = evaluate_candidate({"rmse": 95.0, "mape": 2.8}, BASELINE, PRODUCTION)

    assert decision.promote is False
    assert decision.improvement_pct < 0


def test_model_losing_to_the_baseline_is_rejected():
    """Failing to beat a median lookup means the model is not working."""
    decision = evaluate_candidate({"rmse": 130.0, "mape": 4.2}, BASELINE, None)

    assert decision.promote is False
    assert decision.checks["beats the baseline"] is False


def test_hard_ceiling_is_enforced_even_against_bad_production():
    """A candidate can beat production and still be too poor to ship.

    Without this, a degraded production model would let progressively worse
    candidates through on relative improvement alone.
    """
    decision = evaluate_candidate(
        {"rmse": MAX_ACCEPTABLE_RMSE + 20, "mape": MAX_ACCEPTABLE_MAPE + 1},
        {"rmse": 400.0},
        {"rmse": 300.0},
    )

    assert decision.promote is False
    assert decision.checks["rmse below hard ceiling"] is False


def test_every_check_is_recorded():
    """The decision carries the reasoning, not just the verdict."""
    decision = evaluate_candidate({"rmse": 58.0, "mape": 1.6}, BASELINE, PRODUCTION)

    assert set(decision.checks) >= {
        "rmse below hard ceiling",
        "mape below hard ceiling",
        "beats the baseline",
        "improves on production",
    }
    assert decision.reason


def test_decision_survives_a_round_trip(tmp_path):
    """A decision saved to disk can be read back."""
    decision = evaluate_candidate({"rmse": 58.0, "mape": 1.6}, BASELINE, PRODUCTION)
    path = tmp_path / "promotion.json"

    write_decision(decision, path)
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["promote"] is True
    assert stored["candidate_rmse"] == pytest.approx(58.0)


def test_missing_production_scores_are_handled(tmp_path):
    """An empty model directory reads as no production model, not an error."""
    assert production_metrics(tmp_path) is None


def test_production_scores_are_read_from_disk(tmp_path):
    """The gate reads the live model's own recorded scores."""
    (tmp_path / "training_results.json").write_text(
        json.dumps({"holdout": PRODUCTION}), encoding="utf-8"
    )

    assert production_metrics(tmp_path)["rmse"] == pytest.approx(67.65)