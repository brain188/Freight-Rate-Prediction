"""Scoring functions for freight rate predictions.

Everything is reported in dollars, not log units. The model trains on a
transformed target, but a broker cares how many dollars the quote is out by,
so the numbers here are converted back before they are measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# Distance bands used for segment reporting, chosen to match how the trade
# talks about hauls rather than to split the data evenly.
DISTANCE_BANDS = [0, 250, 500, 1000, 1500, 2500, 5000]
DISTANCE_LABELS = ["<250", "250-500", "500-1k", "1k-1.5k", "1.5k-2.5k", "2.5k+"]


class MetricError(Exception):
    """Raised when predictions and actuals cannot be compared."""


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to aligned float arrays and check they are usable.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        The two arrays.

    Raises:
        MetricError: If the lengths differ or either holds missing values.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape != predicted.shape:
        raise MetricError(
            f"length mismatch: {actual.shape[0]:,} actuals vs {predicted.shape[0]:,} predictions"
        )

    if actual.size == 0:
        raise MetricError("nothing to score — the arrays are empty")

    if np.isnan(actual).any() or np.isnan(predicted).any():
        raise MetricError("predictions or actuals contain missing values")

    return actual, predicted


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in dollars.

    Squaring makes this sensitive to large misses, which is what we want when
    a single badly priced load can cost more than many small errors.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        RMSE in dollars.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in dollars.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        MAE in dollars.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error.

    The most readable metric here: rates span an order of magnitude, so a
    percentage compares a short haul and a long one on equal terms.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        MAPE as a percentage.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def r2(y_true, y_pred) -> float:
    """Share of variance explained.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        R squared, where 1.0 is perfect and 0.0 matches predicting the mean.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    residual = ((actual - predicted) ** 2).sum()
    total = ((actual - actual.mean()) ** 2).sum()
    return float(1.0 - residual / total) if total else 0.0


def median_ae(y_true, y_pred) -> float:
    """Median absolute error, in dollars.

    Reported next to MAE because a wide gap between them says the errors are
    driven by a few large misses rather than being spread evenly.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        Median absolute error in dollars.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.median(np.abs(actual - predicted)))


def bias(y_true, y_pred) -> float:
    """Average signed error, in dollars.

    Positive means the model quotes high on average. Worth watching when
    predicting forward, since a drifting seasonal curve shows up here first.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates.

    Returns:
        Mean signed error in dollars.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.mean(predicted - actual))


METRIC_FUNCTIONS = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "r2": r2,
    "median_ae": median_ae,
    "bias": bias,
}


@dataclass(frozen=True)
class Scores:
    """Metrics for one model on one dataset."""

    label: str
    n: int
    values: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> float:
        """Read one metric by name.

        Args:
            metric: Metric name.

        Returns:
            The value.

        Raises:
            KeyError: If the metric was not computed.
        """
        return self.values[metric]

    def log(self) -> None:
        """Write the scores to the log as one line."""
        parts = []
        for name, value in self.values.items():
            if name == "mape":
                parts.append(f"{name} {value:.2f}%")
            elif name == "r2":
                parts.append(f"{name} {value:.4f}")
            else:
                parts.append(f"{name} ${value:,.2f}")

        logger.info("%s (n=%s): %s", self.label, f"{self.n:,}", " | ".join(parts))

    def to_dict(self) -> dict[str, object]:
        """Flatten to a plain dictionary, for JSON or a table.

        Returns:
            Label, row count, and every metric.
        """
        return {"model": self.label, "n": self.n, **self.values}


def evaluate(
    y_true,
    y_pred,
    label: str = "model",
    metrics: list[str] | None = None,
) -> Scores:
    """Score predictions against actuals.

    Args:
        y_true: Observed rates, in dollars.
        y_pred: Predicted rates, in dollars.
        label: Name for this model or dataset.
        metrics: Which metrics to compute. Defaults to all of them.

    Returns:
        The scores.

    Raises:
        MetricError: If a requested metric is not recognised.
    """
    actual, predicted = _as_arrays(y_true, y_pred)
    names = metrics or list(METRIC_FUNCTIONS)

    if unknown := [name for name in names if name not in METRIC_FUNCTIONS]:
        raise MetricError(
            f"unknown metrics {unknown} — available: {sorted(METRIC_FUNCTIONS)}"
        )

    scores = Scores(
        label=label,
        n=int(actual.size),
        values={name: METRIC_FUNCTIONS[name](actual, predicted) for name in names},
    )
    scores.log()
    return scores


def evaluate_by_segment(
    df: pd.DataFrame,
    y_true,
    y_pred,
    by: str,
    metric: str = "mape",
) -> pd.DataFrame:
    """Score predictions within slices of the data.

    An overall number can hide a group the model handles badly. This is where
    a problem with unseen cities or with the newest month would show up.

    Args:
        df: Frame the predictions came from, used for the grouping column.
        y_true: Observed rates.
        y_pred: Predicted rates.
        by: Column to group on, or one of "month" or "distance_band".
        metric: Metric to report per group.

    Returns:
        One row per group, with the row count and the metric.

    Raises:
        MetricError: If the grouping column cannot be built.
    """
    actual, predicted = _as_arrays(y_true, y_pred)

    if by == "month":
        groups = df["date"].dt.to_period("M").astype(str)
    elif by == "distance_band":
        groups = pd.cut(df["distance"], DISTANCE_BANDS, labels=DISTANCE_LABELS)
    elif by in df.columns:
        groups = df[by]
    else:
        raise MetricError(f"cannot group by '{by}' — column not found")

    function = METRIC_FUNCTIONS.get(metric)
    if function is None:
        raise MetricError(f"unknown metric '{metric}'")

    rows = []
    frame = pd.DataFrame(
        {"group": groups.to_numpy(), "actual": actual, "predicted": predicted}
    )

    for name, chunk in frame.groupby("group", observed=True):
        rows.append(
            {
                "group": name,
                "n": len(chunk),
                "mean_actual": round(chunk["actual"].mean(), 2),
                metric: round(function(chunk["actual"], chunk["predicted"]), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compare(results: list[Scores], sort_by: str = "rmse") -> pd.DataFrame:
    """Put several models side by side.

    Args:
        results: Scores to compare.
        sort_by: Metric to order on. Higher is better only for r2.

    Returns:
        One row per model, best first.
    """
    table = pd.DataFrame([score.to_dict() for score in results])

    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=sort_by != "r2")

    return table.reset_index(drop=True)


def summarise_folds(results: list[Scores], primary: str = "rmse") -> dict[str, float]:
    """Average the metrics across cross-validation folds.

    The spread matters as much as the mean: folds that disagree mean the score
    depends on which months the model happened to see.

    Args:
        results: One Scores per fold.
        primary: Metric to report the spread for.

    Returns:
        Mean of every metric, plus the standard deviation of the primary one.

    Raises:
        MetricError: If no fold results were given.
    """
    if not results:
        raise MetricError("no fold results to summarise")

    table = pd.DataFrame([score.values for score in results])
    summary = {name: float(table[name].mean()) for name in table.columns}
    summary[f"{primary}_std"] = float(table[primary].std())

    logger.info(
        "Cross-validation over %s folds: %s %.2f (+/- %.2f)",
        len(results),
        primary,
        summary[primary],
        summary[f"{primary}_std"],
    )
    return summary
