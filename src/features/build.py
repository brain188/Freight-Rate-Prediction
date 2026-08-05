"""Assembles the model's feature matrix from cleaned loads.

Everything stateful is fitted here on training data and saved alongside the
model, so prediction reuses the exact same lookups, codes, and seasonal curve
rather than recomputing them from the data being scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.features.encoders import OrdinalEncoder, encoded_feature_names
from src.features.geo import (
    CityCoordinates,
    add_distance_features,
    add_geo_features,
    attach_coordinates,
    geo_feature_names,
)
from src.features.seasonality import (
    SeasonalIndex,
    add_time_features,
    time_feature_names,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Filenames used when the fitted state is written to disk.
COORDINATES_FILE = "city_coordinates.json"
ENCODER_FILE = "encoder.json"
SEASONAL_FILE = "seasonal_index.json"
FEATURE_NAMES_FILE = "feature_names.json"

# Features taken straight from the cleaned data, no transformation needed.
PASSTHROUGH_FEATURES = ["weight"]

# The seasonal curve, added as a column. It turns a December date into a value
# that sits inside the range the model trained on, which is what lets a tree
# use it at all.
SEASONAL_FEATURE = "seasonal_index"


class FeatureError(Exception):
    """Raised when the feature matrix cannot be built."""


class FeatureBuilder:
    """Fits feature state on training data and applies it everywhere else.

    Args:
        config: Loaded project configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.coordinates: CityCoordinates | None = None
        self.encoder: OrdinalEncoder | None = None
        self.seasonal_index: SeasonalIndex | None = None
        self.feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        """Whether the builder is ready to transform data."""
        return all((self.coordinates, self.encoder, self.seasonal_index))

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        """Learn every piece of feature state from cleaned training data.

        Args:
            df: Cleaned training loads, including the target column.

        Returns:
            The fitted builder.

        Raises:
            FeatureError: If the target column is absent.
        """
        target = self.config.project.target
        if target not in df.columns:
            raise FeatureError(f"'{target}' is required to fit the seasonal curve")

        self.coordinates = CityCoordinates.fit(df)
        self.encoder = OrdinalEncoder.fit(
            df, unknown_value=self.config.features.unknown_category_value
        )

        # Fitted on log rate per mile so distance does not drown out the
        # seasonal shape we are trying to isolate.
        self.seasonal_index = SeasonalIndex.fit(
            dates=df["date"],
            values=np.log(df[target] / df["distance"]),
            order=self.config.features.fourier_order,
            period=self.config.features.fourier_period,
        )

        self.feature_names = self._build_feature_names()
        logger.info("Feature builder fitted: %s features", len(self.feature_names))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the feature matrix for any set of cleaned loads.

        Args:
            df: Cleaned loads from training, validation, or the December file.

        Returns:
            A frame holding exactly the model's features, in a fixed order.

        Raises:
            FeatureError: If the builder is unfitted or a feature is missing.
        """
        if not self.is_fitted:
            raise FeatureError("call fit() before transform()")

        out = df.copy()

        # The December file names its cities but carries no coordinates.
        out = attach_coordinates(out, self.coordinates)

        out = add_distance_features(out, self.config)
        out = add_geo_features(out, self.config)
        out = add_time_features(out, self.config)
        out = self.encoder.transform(out)

        out[SEASONAL_FEATURE] = self.seasonal_index.predict(out["date"])

        if missing := [name for name in self.feature_names if name not in out.columns]:
            raise FeatureError(f"features could not be built: {missing}")

        features = out[self.feature_names]

        if features.isna().any().any():
            empty = features.columns[features.isna().any()].tolist()
            raise FeatureError(f"features contain missing values: {empty}")

        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return its feature matrix.

        Args:
            df: Cleaned training loads.

        Returns:
            The training feature matrix.
        """
        return self.fit(df).transform(df)

    def _build_feature_names(self) -> list[str]:
        """List every feature the model uses, in a fixed order.

        Order matters because some estimators care about column position.

        Returns:
            Feature names.
        """
        names = [
            *geo_feature_names(self.config),
            *PASSTHROUGH_FEATURES,
            *encoded_feature_names(self.config),
            *time_feature_names(self.config),
            SEASONAL_FEATURE,
        ]

        # Both are weak predictors and neither exists in the December file, so
        # they stay off unless the config explicitly turns them back on.
        if self.config.features.use_market_index:
            names.append("market_index")
        if self.config.features.use_quote_signal:
            names.append("quote_signal")

        return names

    def save(self, directory: Path) -> None:
        """Write the fitted state so prediction can reuse it.

        Args:
            directory: Where to write the artifact files.

        Raises:
            FeatureError: If the builder has not been fitted.
        """
        if not self.is_fitted:
            raise FeatureError("nothing to save — the builder is not fitted")

        directory.mkdir(parents=True, exist_ok=True)
        self.coordinates.to_json(directory / COORDINATES_FILE)
        self.encoder.to_json(directory / ENCODER_FILE)
        self.seasonal_index.to_json(directory / SEASONAL_FILE)

        (directory / FEATURE_NAMES_FILE).write_text(
            json.dumps(self.feature_names, indent=2), encoding="utf-8"
        )
        logger.info("Saved feature state to %s", directory)

    @classmethod
    def load(cls, directory: Path, config: Config) -> FeatureBuilder:
        """Restore a builder saved by a previous training run.

        Args:
            directory: Directory written by save().
            config: Loaded project configuration.

        Returns:
            A fitted builder.

        Raises:
            FeatureError: If the saved feature list is missing.
        """
        builder = cls(config)
        builder.coordinates = CityCoordinates.from_json(directory / COORDINATES_FILE)
        builder.encoder = OrdinalEncoder.from_json(directory / ENCODER_FILE)
        builder.seasonal_index = SeasonalIndex.from_json(directory / SEASONAL_FILE)

        names_path = directory / FEATURE_NAMES_FILE
        if not names_path.is_file():
            raise FeatureError(f"feature list not found: {names_path}")

        builder.feature_names = json.loads(names_path.read_text(encoding="utf-8"))
        logger.info("Loaded feature state from %s", directory)
        return builder

    def describe(self) -> pd.DataFrame:
        """Summarise the features by group, for the write-up.

        Returns:
            One row per feature group with its count and column names.
        """
        groups = {
            "geographic": geo_feature_names(self.config),
            "load": PASSTHROUGH_FEATURES,
            "categorical": encoded_feature_names(self.config),
            "calendar": time_feature_names(self.config),
            "seasonal": [SEASONAL_FEATURE],
        }
        return pd.DataFrame([
            {"group": name, "n": len(cols), "features": ", ".join(cols)}
            for name, cols in groups.items()
        ])