"""The gradient-boosted model and the target transform around it.

Two decisions are worth spelling out. The model learns a transformed target
rather than raw dollars, because rates span an order of magnitude and a long
haul would otherwise dominate every split. And the seasonal curve reaches the
model as a fitted feature, not a raw date, so December is a value the trees
have actually seen rather than a point off the end of the training range.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# Distance is needed to convert a per-mile target back to dollars.
DISTANCE_COLUMN = "distance"

# The fitted seasonal curve, applied as an offset rather than a tree feature.
SEASONAL_COLUMN = "seasonal_index"


class EstimatorError(Exception):
    """Raised when the model cannot be fitted or applied."""


@dataclass(frozen=True)
class TargetTransform:
    """Moves the target in and out of the space the model learns in.

    Args:
        kind: One of "log", "rate_per_mile", "log_rate_per_mile", or "none".
        smearing: Correction applied when undoing a log. Taking exp of an
            average log under-predicts the average dollar value, so this
            factor puts the mean back where it belongs.
    """

    kind: str
    smearing: float = 1.0

    @property
    def is_logarithmic(self) -> bool:
        """Whether the transform involves a log."""
        return "log" in self.kind

    @property
    def is_per_mile(self) -> bool:
        """Whether the transform divides by distance."""
        return "rate_per_mile" in self.kind

    def forward(self, y: pd.Series, distance: pd.Series) -> np.ndarray:
        """Convert dollars into the space the model learns in.

        Args:
            y: Rates in dollars.
            distance: Distance for each load, in miles.

        Returns:
            The transformed target.

        Raises:
            EstimatorError: If the transform is unrecognised.
        """
        values = np.asarray(y, dtype=float)

        if self.is_per_mile:
            values = values / np.asarray(distance, dtype=float)

        if self.is_logarithmic:
            values = np.log(values)

        if self.kind == "none":
            return np.asarray(y, dtype=float)

        return values

    def inverse(self, predictions: np.ndarray, distance: pd.Series) -> np.ndarray:
        """Convert model output back into dollars.

        Args:
            predictions: Raw model output.
            distance: Distance for each load, in miles.

        Returns:
            Predicted rates in dollars.
        """
        values = np.asarray(predictions, dtype=float)

        if self.is_logarithmic:
            values = np.exp(values) * self.smearing

        if self.is_per_mile:
            values = values * np.asarray(distance, dtype=float)

        return values

    def fit_smearing(
        self,
        y_true: pd.Series,
        raw_predictions: np.ndarray,
        distance: pd.Series,
    ) -> TargetTransform:
        """Measure the correction needed when undoing the log.

        Args:
            y_true: Observed rates in dollars.
            raw_predictions: Model output on the training data.
            distance: Distance for each load.

        Returns:
            A copy of the transform carrying the fitted factor.
        """
        if not self.is_logarithmic:
            return self

        actual = self.forward(y_true, distance)
        residuals = actual - np.asarray(raw_predictions, dtype=float)
        factor = float(np.mean(np.exp(residuals)))

        logger.info("Smearing factor: %.5f", factor)
        return TargetTransform(kind=self.kind, smearing=factor)


class FreightRateModel:
    """Gradient-boosted trees predicting the rate for a load.

    By default the fitted seasonal curve is applied as an offset rather than
    handed to the trees as one feature among many. As a feature it competes
    with distance and equipment for splits and ends up contributing almost
    nothing, which leaves the December curve shaped by noise. As an offset it
    is guaranteed to come through at full strength, and the trees are left to
    learn everything that is not seasonal.

    Args:
        config: Loaded project configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.transform = TargetTransform(kind=config.model.target_transform)
        self.use_seasonal_offset = config.model.seasonal_offset
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names: list[str] = []
        self.tree_features: list[str] = []
        self.best_iteration: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the model is ready to predict."""
        return self.model is not None

    def _offset(self, X: pd.DataFrame) -> np.ndarray:
        """Return the seasonal offset for each row.

        Args:
            X: Feature matrix containing the seasonal column.

        Returns:
            Offsets in log space, or zeros when offset mode is off.

        Raises:
            EstimatorError: If offset mode is on but the column is absent.
        """
        if not self.use_seasonal_offset:
            return np.zeros(len(X))

        if SEASONAL_COLUMN not in X.columns:
            raise EstimatorError(
                f"'{SEASONAL_COLUMN}' is required when model.seasonal_offset is true"
            )

        return X[SEASONAL_COLUMN].to_numpy(dtype=float)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> FreightRateModel:
        """Train the model, optionally stopping early on a validation set.

        Args:
            X: Training features, including the distance column.
            y: Training rates in dollars.
            X_valid: Features for early stopping. Training runs to the full
                number of rounds when this is absent.
            y_valid: Rates for the early-stopping set.

        Returns:
            The fitted model.

        Raises:
            EstimatorError: If distance is missing, or offset mode is combined
                with a target transform that is not in log space.
        """
        if DISTANCE_COLUMN not in X.columns:
            raise EstimatorError(
                f"'{DISTANCE_COLUMN}' must be a feature — it is needed to convert "
                "predictions back to dollars"
            )

        # The offset only adds cleanly in log space, where a seasonal shift is
        # a constant rather than something that scales with the rate.
        if self.use_seasonal_offset and not self.transform.is_logarithmic:
            raise EstimatorError(
                "model.seasonal_offset requires a log target transform — set "
                "model.target_transform to log or log_rate_per_mile"
            )

        self.feature_names = list(X.columns)

        # Held out of the trees so the offset is not counted twice.
        self.tree_features = [
            name for name in self.feature_names
            if not (self.use_seasonal_offset and name == SEASONAL_COLUMN)
        ]

        y_transformed = self.transform.forward(y, X[DISTANCE_COLUMN]) - self._offset(X)

        params = dict(self.config.model.params)
        params["random_state"] = self.config.project.random_seed
        self.model = lgb.LGBMRegressor(**params)

        callbacks = []
        eval_set = None

        if X_valid is not None and y_valid is not None:
            valid_target = (
                self.transform.forward(y_valid, X_valid[DISTANCE_COLUMN])
                - self._offset(X_valid)
            )
            eval_set = [(X_valid[self.tree_features], valid_target)]
            callbacks = [
                lgb.early_stopping(self.config.model.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X[self.tree_features], y_transformed, eval_set=eval_set, callbacks=callbacks)
        self.best_iteration = getattr(self.model, "best_iteration_", None)

        # Fitted after training because it depends on the model's own residuals.
        raw = self.model.predict(X[self.tree_features]) + self._offset(X)
        self.transform = self.transform.fit_smearing(y, raw, X[DISTANCE_COLUMN])

        logger.info(
            "Model fitted on %s rows, %s tree features, %s trees, seasonal offset %s",
            f"{len(X):,}",
            len(self.tree_features),
            self.best_iteration or params.get("n_estimators"),
            "on" if self.use_seasonal_offset else "off",
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict rates in dollars.

        Args:
            X: Features with the same columns used for training.

        Returns:
            Predicted rates, always strictly positive as the scorer requires.

        Raises:
            EstimatorError: If the model is unfitted or the columns do not match.
        """
        if not self.is_fitted:
            raise EstimatorError("call fit() before predict()")

        if list(X.columns) != self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            raise EstimatorError(
                f"feature mismatch — missing {sorted(missing)}, unexpected {sorted(extra)}"
            )

        raw = self.model.predict(X[self.tree_features]) + self._offset(X)
        rates = self.transform.inverse(raw, X[DISTANCE_COLUMN])

        # score.py rejects any non-positive rate, so this guarantee is worth
        # enforcing here rather than discovering at submission time.
        if (rates <= 0).any():
            count = int((rates <= 0).sum())
            logger.warning("Clipped %s non-positive predictions to $1.00", count)
            rates = np.clip(rates, 1.0, None)

        return rates

    def feature_importance(self, top: int | None = None) -> pd.DataFrame:
        """Rank the features by how often the trees split on them.

        Args:
            top: Return only this many rows. Defaults to all.

        Returns:
            Features and their gain, most important first.

        Raises:
            EstimatorError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise EstimatorError("no importances — the model is not fitted")

        gains = self.model.booster_.feature_importance(importance_type="gain")
        table = pd.DataFrame({"feature": self.tree_features, "gain": gains})
        table["share"] = (table["gain"] / table["gain"].sum() * 100).round(2)
        table = table.sort_values("gain", ascending=False).reset_index(drop=True)

        return table.head(top) if top else table