"""Deciding whether a newly trained model is allowed to replace the live one.

This is the piece that makes automated retraining safe. A scheduled retrain
that always ships whatever it produced is worse than no retraining at all,
because a bad month of data silently degrades the service. Every candidate has
to earn its place by beating the model currently in production on the same
holdout, by a margin wide enough not to be noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

# A candidate has to beat production by at least this much to be promoted.
# Anything smaller is within the run to run noise of the folds and is not
# worth the risk of a swap.
MIN_IMPROVEMENT_PCT = 1.0

# Hard floors. A candidate that fails either is rejected regardless of how it
# compares to production, because production itself may have degraded.
MAX_ACCEPTABLE_RMSE = 150.0
MAX_ACCEPTABLE_MAPE = 5.0

# The baseline is the number every model has to beat for its score to mean
# anything, so failing to beat it is an automatic rejection.
MUST_BEAT_BASELINE = True


@dataclass
class PromotionDecision:
    """Whether a candidate model should go live, and why."""

    promote: bool
    reason: str
    candidate_rmse: float
    production_rmse: float | None
    improvement_pct: float | None
    checks: dict[str, bool]
    decided_at: str

    def to_dict(self) -> dict[str, Any]:
        """Flatten for logging or a JSON artifact.

        Returns:
            Every field.
        """
        return asdict(self)

    def log(self) -> None:
        """Write the decision and every check that fed it."""
        verdict = "PROMOTE" if self.promote else "REJECT"
        logger.info("%s — %s", verdict, self.reason)

        for check, passed in self.checks.items():
            logger.info("  %-28s %s", check, "pass" if passed else "FAIL")


def production_metrics(model_dir: Path) -> dict[str, float] | None:
    """Read the holdout scores of the model currently deployed.

    Args:
        model_dir: Directory holding the live model bundle.

    Returns:
        The holdout metrics, or None when there is no live model yet.
    """
    path = model_dir / "training_results.json"

    if not path.is_file():
        return None

    try:
        results = json.loads(path.read_text(encoding="utf-8"))
        return results.get("holdout")
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read production scores from %s", path)
        return None


def evaluate_candidate(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
    production: dict[str, float] | None,
) -> PromotionDecision:
    """Decide whether a candidate model should replace production.

    Args:
        candidate: Holdout metrics for the newly trained model.
        baseline: Holdout metrics for the simple reference model.
        production: Holdout metrics for the model currently deployed.

    Returns:
        The decision, with every individual check recorded.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidate_rmse = float(candidate.get("rmse", float("inf")))
    candidate_mape = float(candidate.get("mape", float("inf")))

    checks: dict[str, bool] = {
        "rmse below hard ceiling": candidate_rmse <= MAX_ACCEPTABLE_RMSE,
        "mape below hard ceiling": candidate_mape <= MAX_ACCEPTABLE_MAPE,
    }

    if MUST_BEAT_BASELINE and baseline:
        checks["beats the baseline"] = candidate_rmse < float(baseline.get("rmse", 0))

    # No production model means this is the first one, so there is nothing to
    # compare against and the hard floors are the only bar.
    if production is None:
        promote = all(checks.values())
        return PromotionDecision(
            promote=promote,
            reason=(
                "first model, promoted on the hard floors alone"
                if promote
                else "first model, but it failed a hard floor"
            ),
            candidate_rmse=candidate_rmse,
            production_rmse=None,
            improvement_pct=None,
            checks=checks,
            decided_at=now,
        )

    production_rmse = float(production.get("rmse", float("inf")))
    improvement = (production_rmse - candidate_rmse) / production_rmse * 100

    checks["improves on production"] = improvement >= MIN_IMPROVEMENT_PCT

    promote = all(checks.values())

    if promote:
        reason = f"beats production by {improvement:.1f}% on RMSE"
    elif improvement < MIN_IMPROVEMENT_PCT:
        reason = (
            f"only {improvement:+.1f}% against production, below the "
            f"{MIN_IMPROVEMENT_PCT:.0f}% needed to justify a swap"
        )
    else:
        failed = [name for name, passed in checks.items() if not passed]
        reason = f"failed: {', '.join(failed)}"

    decision = PromotionDecision(
        promote=promote,
        reason=reason,
        candidate_rmse=candidate_rmse,
        production_rmse=production_rmse,
        improvement_pct=round(improvement, 2),
        checks=checks,
        decided_at=now,
    )
    return decision


def write_decision(decision: PromotionDecision, path: Path) -> None:
    """Save the decision alongside the model.

    Kept so a deployment can be traced back to the comparison that allowed it,
    rather than only to the run that produced it.

    Args:
        decision: What was decided.
        path: Destination JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    logger.debug("Wrote the promotion decision to %s", path)
