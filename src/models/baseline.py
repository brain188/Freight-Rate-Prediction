"""A simple rate model that everything else has to beat.

Roughly how a broker prices by hand: take the going rate per mile for a haul
of this length and multiply by the distance. It takes minutes to write and
already reaches R squared around 0.95, which is the point. A score means
nothing on its own, so this is the number the real model is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# Bands follow how the trade talks about hauls, not an even split of the data.
DISTANCE_BANDS = [0, 250, 500, 1000, 1500, 2500, 1e9]
BAND_LABELS = ["<250", "250-500", "500-1k", "1k-1.5k", "1.5k-2.5k", "2.5k+"]


class BaselineError(Exception):
    """Raised when the baseline cannot be fitted or applied."""


@dataclass
class MedianRatePerMile:
    """Predicts rate as distance times a median rate per mile.

    The median is taken per distance band and equipment type, which captures
    the two effects that matter most: rate per mile falls as hauls get longer,
    and refrigerated trailers cost more than dry vans.

    Args:
        config: Loaded project configuration.
    """

    config: Config
    rates: dict[tuple[str, str], float] = field(default_factory=dict)
    fallback_rate: float = 0.0

    @staticmethod
    def _bands(distance: pd.Series) -> pd.Series:
        """Assign each load to a distance band.

        Args:
            distance: Distances in miles.

        Returns:
            Band labels as strings.
        """
        return pd.cut(distance, DISTANCE_BANDS, labels=BAND_LABELS).astype(str)

    def fit(self, df: pd.DataFrame) -> MedianRatePerMile:
        """Learn the median rate per mile for each band and equipment type.

        Args:
            df: Cleaned training loads, including the target.

        Returns:
            The fitted baseline.

        Raises:
            BaselineError: If the target column is absent.
        """
        target = self.config.project.target
        if target not in df.columns:
            raise BaselineError(f"'{target}' is required to fit the baseline")

        rate_per_mile = df[target] / df["distance"]
        grouped = rate_per_mile.groupby([self._bands(df["distance"]), df["equipment"]])

        self.rates = {
            (str(band), str(equipment)): float(value)
            for (band, equipment), value in grouped.median().items()
        }
        self.fallback_rate = float(rate_per_mile.median())

        logger.info(
            "Baseline fitted: %s band/equipment combinations, fallback $%.3f/mile",
            len(self.rates),
            self.fallback_rate,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict a rate for each load.

        Args:
            df: Loads with distance and equipment columns.

        Returns:
            Predicted rates in dollars.

        Raises:
            BaselineError: If the baseline has not been fitted.
        """
        if not self.rates:
            raise BaselineError("call fit() before predict()")

        keys = list(zip(self._bands(df["distance"]), df["equipment"].astype(str)))

        # Any combination missing from training falls back to the overall
        # median, so an unfamiliar band still returns a usable number.
        per_mile = np.array([self.rates.get(key, self.fallback_rate) for key in keys])

        return per_mile * df["distance"].to_numpy()

    def as_table(self) -> pd.DataFrame:
        """Show the learned rates as a band-by-equipment grid.

        Returns:
            Median rate per mile, bands as rows and equipment as columns.
        """
        rows = [
            {"band": band, "equipment": equipment, "rate_per_mile": round(rate, 3)}
            for (band, equipment), rate in self.rates.items()
        ]
        table = pd.DataFrame(rows)
        return table.pivot(index="band", columns="equipment", values="rate_per_mile")
