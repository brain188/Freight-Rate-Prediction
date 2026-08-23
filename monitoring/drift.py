"""Detecting when live traffic stops resembling the training data.

Drift matters here more than in most projects because the model cannot see the
period it is asked to predict. Training stops on 31 October and every load
priced afterwards is a forecast, so traffic moving away from what the model
learned is the earliest warning available.

The most useful signal is also the cheapest. The unknown city rate is computed
from a flag already raised at prediction time, needs no reference dataset, and
is available immediately rather than waiting on outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.performance import load_predictions
from serving.store import PredictionStore
from src.config import Config
from src.data.cleaning import clean
from src.data.loading import load_train
from src.logger import get_logger

logger = get_logger(__name__)

# Columns compared against training. Only ones that exist on both sides.
DRIFT_COLUMNS = ["distance", "weight"]
CATEGORICAL_COLUMNS = ["equipment"]

# Share of unfamiliar cities above which traffic has moved somewhere the model
# has no history for. Training saw none, so anything sustained is meaningful.
UNKNOWN_CITY_WARN = 0.05
UNKNOWN_CITY_ALERT = 0.15

# Population Stability Index thresholds, the usual industry reading.
PSI_WARN = 0.10
PSI_ALERT = 0.25

MIN_ROWS_FOR_DRIFT = 100


@dataclass
class FeatureDrift:
    """How far one feature has moved from the training data."""

    feature: str
    psi: float
    reference_mean: float | None = None
    current_mean: float | None = None
    status: str = "ok"

    def to_dict(self) -> dict:
        """Flatten for a table or an API response.

        Returns:
            Every field.
        """
        return asdict(self)


@dataclass
class DriftReport:
    """Everything known about drift over one window of traffic."""

    computed_at: str
    window_days: int
    n_current: int
    n_reference: int
    unknown_city_rate: float = 0.0
    date_beyond_training_rate: float = 0.0
    features: list[FeatureDrift] = field(default_factory=list)
    evidently_html: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """The worst status across every signal.

        Returns:
            One of "ok", "warn" or "alert".
        """
        if self.n_current < MIN_ROWS_FOR_DRIFT:
            return "unknown"

        if self.unknown_city_rate >= UNKNOWN_CITY_ALERT:
            return "alert"
        if any(f.status == "alert" for f in self.features):
            return "alert"
        if self.unknown_city_rate >= UNKNOWN_CITY_WARN:
            return "warn"
        if any(f.status == "warn" for f in self.features):
            return "warn"

        return "ok"

    @property
    def drifted_features(self) -> list[str]:
        """Features that have moved enough to matter."""
        return [f.feature for f in self.features if f.status != "ok"]

    def to_dict(self) -> dict:
        """Flatten for the dashboard.

        Returns:
            Every field plus the derived status.
        """
        return {
            "computed_at": self.computed_at,
            "window_days": self.window_days,
            "n_current": self.n_current,
            "n_reference": self.n_reference,
            "unknown_city_rate": round(self.unknown_city_rate, 4),
            "date_beyond_training_rate": round(self.date_beyond_training_rate, 4),
            "status": self.status,
            "drifted_features": self.drifted_features,
            "features": [f.to_dict() for f in self.features],
            "notes": self.notes,
        }


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Measure how far a distribution has moved.

    PSI is used rather than a statistical test because it does not grow with
    sample size. A test on ten thousand rows calls every tiny shift
    significant, which is not the question being asked.

    Args:
        reference: Values from the training data.
        current: Values from live traffic.
        bins: How many quantile buckets to compare.

    Returns:
        The index. Below 0.1 is stable, above 0.25 is a real shift.
    """
    reference = reference.dropna()
    current = current.dropna()

    if reference.empty or current.empty:
        return 0.0

    # Quantile edges from the reference, so buckets hold equal training mass.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    # Floor at a small value so an empty bucket does not produce infinity.
    floor = 1e-6
    reference_share = np.clip(reference_share, floor, None)
    current_share = np.clip(current_share, floor, None)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def _status_for(psi: float) -> str:
    """Turn a PSI value into a status.

    Args:
        psi: The computed index.

    Returns:
        One of "ok", "warn" or "alert".
    """
    if psi >= PSI_ALERT:
        return "alert"
    if psi >= PSI_WARN:
        return "warn"
    return "ok"


def reference_data(config: Config) -> pd.DataFrame:
    """Load the training data drift is measured against.

    Cleaned the same way training cleaned it, so a difference reflects real
    traffic rather than a difference in preparation.

    Args:
        config: Loaded project configuration.

    Returns:
        The cleaned training loads.
    """
    cleaned, _, _ = clean(load_train(config), config, is_training=True, label="reference")
    return cleaned


def compute_drift(
    store: PredictionStore,
    config: Config,
    days: int = 7,
    reference: pd.DataFrame | None = None,
) -> DriftReport:
    """Compare recent traffic against the training data.

    Args:
        store: The prediction store.
        config: Loaded project configuration.
        days: How far back to treat as current.
        reference: Training data, loaded when not supplied.

    Returns:
        The drift report.
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    current = load_predictions(store, days)

    if current.empty:
        return DriftReport(
            computed_at=now,
            window_days=days,
            n_current=0,
            n_reference=0,
            notes=["No traffic in this window."],
        )

    if reference is None:
        reference = reference_data(config)

    report = DriftReport(
        computed_at=now,
        window_days=days,
        n_current=len(current),
        n_reference=len(reference),
        unknown_city_rate=float(current["unknown_city"].mean()),
        date_beyond_training_rate=float(current["date_beyond_training"].mean()),
    )

    for column in DRIFT_COLUMNS:
        if column not in current.columns or column not in reference.columns:
            continue

        # Weights are sign flipped in this data, so compare magnitudes.
        reference_values = reference[column].abs()
        current_values = current[column].abs()
        psi = population_stability_index(reference_values, current_values)

        report.features.append(
            FeatureDrift(
                feature=column,
                psi=round(psi, 4),
                reference_mean=round(float(reference_values.mean()), 2),
                current_mean=round(float(current_values.mean()), 2),
                status=_status_for(psi),
            )
        )

    for column in CATEGORICAL_COLUMNS:
        if column not in current.columns or column not in reference.columns:
            continue

        reference_share = reference[column].value_counts(normalize=True)
        current_share = current[column].value_counts(normalize=True)
        categories = reference_share.index.union(current_share.index)

        floor = 1e-6
        expected = reference_share.reindex(categories, fill_value=floor).clip(lower=floor)
        actual = current_share.reindex(categories, fill_value=floor).clip(lower=floor)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))

        report.features.append(
            FeatureDrift(
                feature=column,
                psi=round(psi, 4),
                status=_status_for(psi),
            )
        )

    if report.n_current < MIN_ROWS_FOR_DRIFT:
        report.notes.append(
            f"Only {report.n_current} predictions in this window. "
            f"Drift needs at least {MIN_ROWS_FOR_DRIFT} to be meaningful."
        )

    if report.unknown_city_rate >= UNKNOWN_CITY_WARN:
        report.notes.append(
            f"{report.unknown_city_rate:.1%} of traffic involves a city the model "
            "has no history for. Those loads are priced from position alone."
        )

    logger.info(
        "Drift over %s days: status %s, unknown city %.1f%%, drifted %s",
        days,
        report.status,
        report.unknown_city_rate * 100,
        report.drifted_features or "none",
    )
    return report


def build_evidently_report(
    store: PredictionStore,
    config: Config,
    days: int = 7,
    output: Path | None = None,
) -> str | None:
    """Produce the full Evidently drift report as HTML.

    Kept separate from compute_drift because it is far slower and is only
    wanted when someone opens the drift tab, not on every refresh.

    Args:
        store: The prediction store.
        config: Loaded project configuration.
        days: How far back to treat as current.
        output: Where to write the HTML, if anywhere.

    Returns:
        The report HTML, or None when it could not be produced.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        logger.warning("Evidently is not installed. Skipping the full report.")
        return None

    current = load_predictions(store, days)

    if len(current) < MIN_ROWS_FOR_DRIFT:
        logger.info("Too little traffic for an Evidently report")
        return None

    reference = reference_data(config)
    columns = [c for c in DRIFT_COLUMNS + CATEGORICAL_COLUMNS if c in current.columns]

    reference_frame = reference[columns].copy()
    current_frame = current[columns].copy()

    for frame in (reference_frame, current_frame):
        if "weight" in frame.columns:
            frame["weight"] = frame["weight"].abs()

    try:
        definition = DataDefinition(
            numerical_columns=[c for c in DRIFT_COLUMNS if c in columns],
            categorical_columns=[c for c in CATEGORICAL_COLUMNS if c in columns],
        )

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(
            current_data=Dataset.from_pandas(current_frame, data_definition=definition),
            reference_data=Dataset.from_pandas(reference_frame, data_definition=definition),
        )

        html = result.get_html_str(as_iframe=False)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(html, encoding="utf-8")
            logger.info("Wrote the Evidently report to %s", output)

        return html

    except Exception:
        logger.exception("Could not build the Evidently report")
        return None
