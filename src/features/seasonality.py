"""Time features that stay valid past the end of the training data.

Training stops on 31 October but every load we predict falls in November or
December. A tree can only split on values it has seen, so a raw date or month
number would push every December row into the October leaf and produce a flat
chart. Fourier terms avoid that: they are smooth functions of the calendar and
are defined everywhere, so December sits further along the same curve.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# Anchors day-of-year so the terms are identical across files and runs.
REFERENCE_DATE = pd.Timestamp("2025-01-01")


class SeasonalityError(Exception):
    """Raised when time features cannot be built from the given dates."""


def _day_of_year(dates: pd.Series) -> pd.Series:
    """Convert dates to a continuous position within the year.

    Args:
        dates: Datetime series.

    Returns:
        Days elapsed since 1 January, as a float.
    """
    return (dates - REFERENCE_DATE).dt.total_seconds() / 86_400.0


def fourier_feature_names(order: int) -> list[str]:
    """List the Fourier column names for a given order.

    Args:
        order: Number of sine/cosine pairs.

    Returns:
        Column names in the order they are created.
    """
    names = []
    for k in range(1, order + 1):
        names.extend([f"season_sin_{k}", f"season_cos_{k}"])
    return names


def add_fourier_terms(
    df: pd.DataFrame,
    order: int,
    period: float,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add smooth annual waves describing where a date sits in the year.

    Each pair k captures a cycle k times per year: k=1 is the broad annual
    shape, higher terms add the finer bends the data actually shows.

    Args:
        df: Frame containing the date column.
        order: Number of sine/cosine pairs to add.
        period: Length of one full cycle in days, normally 365.25.
        date_column: Name of the datetime column.

    Returns:
        The frame with 2 x order new columns.

    Raises:
        SeasonalityError: If the date column is missing or holds bad values.
    """
    if date_column not in df.columns:
        raise SeasonalityError(f"'{date_column}' not found — cannot build time features")

    if df[date_column].isna().any():
        raise SeasonalityError(f"'{date_column}' contains missing values")

    out = df.copy()
    position = _day_of_year(out[date_column])

    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * position / period
        out[f"season_sin_{k}"] = np.sin(angle)
        out[f"season_cos_{k}"] = np.cos(angle)

    return out


def add_calendar_features(df: pd.DataFrame, use_day_of_week: bool = True) -> pd.DataFrame:
    """Add plain calendar features that are safe to extrapolate.

    Only weekday is included. Month and day-of-year are deliberately left out:
    training never sees months 11 or 12, so a tree splitting on them would
    reuse the October answer for every prediction we make.

    Args:
        df: Frame containing a date column.
        use_day_of_week: Whether to add the weekday features.

    Returns:
        The frame with the calendar columns added.
    """
    out = df.copy()

    if use_day_of_week:
        out["day_of_week"] = out["date"].dt.dayofweek.astype("int8")
        out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")

    return out


def calendar_feature_names(use_day_of_week: bool = True) -> list[str]:
    """List the calendar column names.

    Args:
        use_day_of_week: Whether weekday features are in use.

    Returns:
        Column names, empty if weekday features are switched off.
    """
    return ["day_of_week", "is_weekend"] if use_day_of_week else []


def add_time_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add every time feature the model uses.

    Args:
        df: Frame containing a date column.
        config: Loaded project configuration.

    Returns:
        The frame with Fourier and calendar columns added.
    """
    out = add_fourier_terms(
        df,
        order=config.features.fourier_order,
        period=config.features.fourier_period,
    )
    return add_calendar_features(out, use_day_of_week=config.features.use_day_of_week)


def time_feature_names(config: Config) -> list[str]:
    """List every time feature name, in build order.

    Args:
        config: Loaded project configuration.

    Returns:
        Fourier names followed by calendar names.
    """
    return [
        *fourier_feature_names(config.features.fourier_order),
        *calendar_feature_names(config.features.use_day_of_week),
    ]


@dataclass(frozen=True)
class SeasonalIndex:
    """A smooth annual curve fitted by least squares on the Fourier terms.

    Feeding the Fourier columns straight to a tree is not enough on its own.
    Sine and cosine stay inside their trained range, but the November-December
    arc of the circle never appears in training, so the tree can only match
    those points to whatever nearby region it happens to have seen.

    A fitted curve has no such gap. It is a closed-form function of the date,
    so it returns a genuine value for December rather than a guess. I use it
    as an offset and leave the tree to learn everything that is not seasonal.
    """

    coefficients: list[float]
    intercept: float
    order: int
    period: float
    r_squared: float

    def _design_matrix(self, dates: pd.Series) -> np.ndarray:
        """Build the Fourier design matrix for a set of dates.

        Args:
            dates: Datetime series.

        Returns:
            Array of shape (n_dates, 2 * order).
        """
        position = _day_of_year(dates).to_numpy()
        columns = []
        for k in range(1, self.order + 1):
            angle = 2.0 * np.pi * k * position / self.period
            columns.extend([np.sin(angle), np.cos(angle)])
        return np.column_stack(columns)

    @classmethod
    def fit(
        cls,
        dates: pd.Series,
        values: pd.Series,
        order: int,
        period: float,
    ) -> SeasonalIndex:
        """Fit the seasonal curve to observed values.

        Fitted on daily averages rather than individual loads, so a busy day
        does not outweigh a quiet one when shaping the curve.

        Args:
            dates: Dates of the observations.
            values: What to fit, normally log rate per mile.
            order: Number of sine/cosine pairs.
            period: Cycle length in days.

        Returns:
            The fitted index.

        Raises:
            SeasonalityError: If there are too few distinct days to fit.
        """
        daily = pd.DataFrame({"date": dates.to_numpy(), "value": values.to_numpy()})
        daily = daily.groupby("date", as_index=False)["value"].mean()

        if len(daily) < 4 * order:
            raise SeasonalityError(
                f"need at least {4 * order} distinct days to fit order {order}, "
                f"found {len(daily)}"
            )

        blank = cls([], 0.0, order, period, 0.0)
        design = blank._design_matrix(daily["date"])
        centred = daily["value"] - daily["value"].mean()

        # Least squares on a small, well-conditioned basis. No regularisation
        # needed and none wanted, since it would flatten the curve.
        coefficients, *_ = np.linalg.lstsq(design, centred.to_numpy(), rcond=None)

        fitted = design @ coefficients
        residual = centred.to_numpy() - fitted
        total = centred.to_numpy() - centred.mean()
        r_squared = float(1.0 - (residual**2).sum() / (total**2).sum())

        index = cls(
            coefficients=[float(c) for c in coefficients],
            intercept=float(daily["value"].mean()),
            order=order,
            period=period,
            r_squared=r_squared,
        )

        logger.info(
            "Fitted seasonal index: order %s, R2 %.4f on %s daily averages",
            order,
            r_squared,
            f"{len(daily):,}",
        )
        return index

    def predict(self, dates: pd.Series) -> np.ndarray:
        """Return the seasonal offset for each date.

        Args:
            dates: Datetime series, which may fall outside the fitted window.

        Returns:
            Centred offsets — positive above the annual average, negative below.

        Raises:
            SeasonalityError: If the index has not been fitted.
        """
        if not self.coefficients:
            raise SeasonalityError("seasonal index has not been fitted")

        return self._design_matrix(dates) @ np.asarray(self.coefficients)

    def curve(self, start: str, end: str) -> pd.Series:
        """Return the fitted curve over a date range, for plotting.

        Args:
            start: First date, as YYYY-MM-DD.
            end: Last date, as YYYY-MM-DD.

        Returns:
            Daily offsets indexed by date.
        """
        dates = pd.Series(pd.date_range(start, end, freq="D"))
        return pd.Series(self.predict(dates), index=dates)

    def to_json(self, path: Path) -> None:
        """Save the fitted curve next to the model.

        Args:
            path: Destination JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        logger.debug("Saved seasonal index to %s", path)

    @classmethod
    def from_json(cls, path: Path) -> SeasonalIndex:
        """Load a curve saved by a previous training run.

        Args:
            path: JSON file written by to_json.

        Returns:
            The stored index.

        Raises:
            SeasonalityError: If the file is missing or malformed.
        """
        if not path.is_file():
            raise SeasonalityError(f"seasonal index not found: {path}")

        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise SeasonalityError(f"could not read {path}: {exc}") from exc


def check_extrapolation(train_dates: pd.Series, predict_dates: pd.Series) -> dict[str, float]:
    """Measure how far the prediction dates fall outside the training window.

    Worth logging on every run: it is the reason the seasonal features are
    built the way they are.

    Args:
        train_dates: Dates the model learns from.
        predict_dates: Dates the model must score.

    Returns:
        Gap in days to the first and last prediction date, and how much of one
        seasonal period the prediction window sits beyond training.
    """
    train_end = train_dates.max()
    gap_first = (predict_dates.min() - train_end).days
    gap_last = (predict_dates.max() - train_end).days

    summary = {
        "days_beyond_training_first": float(gap_first),
        "days_beyond_training_last": float(gap_last),
        "fraction_of_year_beyond": round(gap_last / 365.25, 3),
    }

    logger.info(
        "Extrapolating %s to %s days past %s (%.1f%% of a year)",
        gap_first,
        gap_last,
        train_end.date(),
        summary["fraction_of_year_beyond"] * 100,
    )
    return summary