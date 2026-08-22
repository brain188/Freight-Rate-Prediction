"""Measuring how the model is doing on real traffic.

Freight rates are quoted now and confirmed later, so accuracy cannot be
computed when a prediction is made. Everything here reads the logged
predictions, joins whatever outcomes have arrived, and reports on the subset
that can be scored. Coverage is reported alongside every metric, because a
figure computed on three percent of traffic should not be read the same way as
one computed on eighty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from serving.store import PredictionStore, actuals, predictions
from src.logger import get_logger
from src.validation.metrics import METRIC_FUNCTIONS

logger = get_logger(__name__)

# Below this many scored rows a metric is too noisy to act on.
MIN_SCORED_ROWS = 30


@dataclass
class PerformanceSnapshot:
    """Model performance over one window of traffic."""

    window_days: int
    n_predictions: int
    n_scored: int
    metrics: dict[str, float] = field(default_factory=dict)
    median_feedback_days: float | None = None
    computed_at: str = ""

    @property
    def coverage(self) -> float:
        """Share of predictions in the window that have an outcome."""
        return self.n_scored / self.n_predictions if self.n_predictions else 0.0

    @property
    def is_reliable(self) -> bool:
        """Whether enough outcomes have arrived to trust the metrics."""
        return self.n_scored >= MIN_SCORED_ROWS

    def to_dict(self) -> dict:
        """Flatten for an API response or a dashboard.

        Returns:
            Every field plus the derived coverage and reliability.
        """
        return {
            **asdict(self),
            "coverage": round(self.coverage, 4),
            "is_reliable": self.is_reliable,
        }


def load_predictions(store: PredictionStore, days: int = 30) -> pd.DataFrame:
    """Read recent predictions with any outcomes attached.

    A left join, deliberately. Predictions without an outcome still matter for
    drift and volume even though they cannot be scored.

    Args:
        store: The prediction store.
        days: How far back to read.

    Returns:
        One row per prediction, with actual_rate null where none has arrived.
    """
    since = datetime.now(UTC) - timedelta(days=days)

    query = (
        select(
            predictions.c.prediction_id,
            predictions.c.load_id,
            predictions.c.pickup,
            predictions.c.delivery,
            predictions.c.distance,
            predictions.c.equipment,
            predictions.c.weight,
            predictions.c.load_date,
            predictions.c.predicted_rate,
            predictions.c.rate_per_mile,
            predictions.c.model_version,
            predictions.c.unknown_city,
            predictions.c.date_beyond_training,
            predictions.c.latency_ms,
            predictions.c.predicted_at,
            actuals.c.actual_rate,
            actuals.c.recorded_at,
        )
        .select_from(
            predictions.outerjoin(actuals, actuals.c.load_id == predictions.c.load_id)
        )
        .where(predictions.c.predicted_at >= since)
        .order_by(predictions.c.predicted_at.desc())
    )

    with store.engine.connect() as connection:
        frame = pd.DataFrame(connection.execute(query).mappings().all())

    if frame.empty:
        return frame

    for column in ("predicted_at", "recorded_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    return frame


def add_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add error columns to the rows that have an outcome.

    Args:
        frame: Predictions joined with actuals.

    Returns:
        The frame with error, absolute error, percentage error and the
        feedback delay added.
    """
    out = frame.copy()
    scored = out["actual_rate"].notna()

    out["error"] = np.where(scored, out["actual_rate"] - out["predicted_rate"], np.nan)
    out["absolute_error"] = out["error"].abs()
    out["absolute_pct_error"] = np.where(
        scored, out["absolute_error"] / out["actual_rate"] * 100, np.nan
    )
    out["feedback_days"] = np.where(
        scored,
        (out["recorded_at"] - out["predicted_at"]).dt.total_seconds() / 86_400,
        np.nan,
    )
    return out


def snapshot(store: PredictionStore, days: int = 30) -> PerformanceSnapshot:
    """Summarise model performance over a window.

    Args:
        store: The prediction store.
        days: How far back to look.

    Returns:
        The snapshot, with empty metrics when nothing can be scored yet.
    """
    frame = load_predictions(store, days)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    if frame.empty:
        return PerformanceSnapshot(
            window_days=days, n_predictions=0, n_scored=0, computed_at=now
        )

    scored = frame[frame["actual_rate"].notna()]

    if scored.empty:
        return PerformanceSnapshot(
            window_days=days,
            n_predictions=len(frame),
            n_scored=0,
            computed_at=now,
        )

    values = {
        name: round(float(function(scored["actual_rate"], scored["predicted_rate"])), 4)
        for name, function in METRIC_FUNCTIONS.items()
    }

    with_errors = add_error_columns(scored)

    result = PerformanceSnapshot(
        window_days=days,
        n_predictions=len(frame),
        n_scored=len(scored),
        metrics=values,
        median_feedback_days=round(float(with_errors["feedback_days"].median()), 2),
        computed_at=now,
    )

    if not result.is_reliable:
        logger.warning(
            "Only %s scored predictions in the last %s days. Metrics are noisy below %s.",
            result.n_scored,
            days,
            MIN_SCORED_ROWS,
        )

    return result


def daily_metrics(
    store: PredictionStore, days: int = 30, metric: str = "mape"
) -> pd.DataFrame:
    """Track one metric day by day.

    Args:
        store: The prediction store.
        days: How far back to look.
        metric: Which metric to compute per day.

    Returns:
        One row per day with volume, scored count and the metric.
    """
    frame = load_predictions(store, days)

    if frame.empty:
        return pd.DataFrame(columns=["day", "n_predictions", "n_scored", metric])

    frame["day"] = frame["predicted_at"].dt.date
    function = METRIC_FUNCTIONS[metric]
    rows = []

    for day, chunk in frame.groupby("day"):
        scored = chunk[chunk["actual_rate"].notna()]
        rows.append(
            {
                "day": day,
                "n_predictions": len(chunk),
                "n_scored": len(scored),
                metric: round(
                    float(function(scored["actual_rate"], scored["predicted_rate"])), 4
                )
                if len(scored)
                else None,
            }
        )

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def segment_metrics(
    store: PredictionStore,
    by: str = "equipment",
    days: int = 30,
    metric: str = "mape",
) -> pd.DataFrame:
    """Break performance down by a feature.

    An overall figure hides a group the model handles badly. This is where a
    problem with one trailer type or one distance band shows up.

    Args:
        store: The prediction store.
        by: Column to group on, or "distance_band".
        days: How far back to look.
        metric: Which metric to report per group.

    Returns:
        One row per group.
    """
    frame = load_predictions(store, days)

    if frame.empty:
        return pd.DataFrame(columns=["group", "n_predictions", "n_scored", metric])

    if by == "distance_band":
        groups = pd.cut(
            frame["distance"],
            [0, 250, 500, 1000, 1500, 2500, 1e9],
            labels=["<250", "250-500", "500-1k", "1k-1.5k", "1.5k-2.5k", "2.5k+"],
        )
    else:
        groups = frame[by]

    frame = frame.assign(group=groups)
    function = METRIC_FUNCTIONS[metric]
    rows = []

    for name, chunk in frame.groupby("group", observed=True):
        scored = chunk[chunk["actual_rate"].notna()]
        rows.append(
            {
                "group": str(name),
                "n_predictions": len(chunk),
                "n_scored": len(scored),
                metric: round(
                    float(function(scored["actual_rate"], scored["predicted_rate"])), 4
                )
                if len(scored)
                else None,
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def traffic_summary(store: PredictionStore, days: int = 7) -> dict:
    """Describe recent traffic, whether or not outcomes have arrived.

    Unknown city rate is the single most useful number here. It rises when
    traffic moves into markets the model has no history for, and it is
    available immediately rather than waiting on feedback.

    Args:
        store: The prediction store.
        days: How far back to look.

    Returns:
        Volume, warning rates, latency and the predicted rate distribution.
    """
    frame = load_predictions(store, days)

    if frame.empty:
        return {"window_days": days, "n_predictions": 0}

    return {
        "window_days": days,
        "n_predictions": len(frame),
        "unknown_city_rate": round(float(frame["unknown_city"].mean()), 4),
        "date_beyond_training_rate": round(
            float(frame["date_beyond_training"].mean()), 4
        ),
        "mean_predicted_rate": round(float(frame["predicted_rate"].mean()), 2),
        "median_predicted_rate": round(float(frame["predicted_rate"].median()), 2),
        "mean_rate_per_mile": round(float(frame["rate_per_mile"].mean()), 3),
        "p50_latency_ms": round(float(frame["latency_ms"].median()), 2),
        "p95_latency_ms": round(float(frame["latency_ms"].quantile(0.95)), 2),
        "model_versions": sorted(frame["model_version"].dropna().unique().tolist()),
    }
