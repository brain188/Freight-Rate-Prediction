"""Loading the model once and turning requests into predictions.

The bundle is loaded at startup and held for the life of the process. Loading
it per request would cost roughly a hundred milliseconds and, more importantly,
would let a mid flight retrain change the model between two calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from serving.schemas import (
    MAX_DISTANCE,
    MIN_DISTANCE,
    LoadRequest,
    Warnings,
)
from src.config import Config, load_config
from src.data.cleaning import clean
from src.logger import get_logger
from src.models.persistence import ModelBundle, PersistenceError

logger = get_logger(__name__)

# Columns the model needs. Coordinates are added by the feature builder from
# the city lookup, so callers never have to supply them.
REQUEST_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is asked for before the model is ready."""


class UnknownCityError(ValueError):
    """Raised when a city has no coordinates and none were supplied.

    The model places a city by its position, so an unfamiliar name with no
    coordinates cannot be priced. Guessing a location would return a confident
    number built on nothing, so the request is refused instead.
    """

    def __init__(self, cities: list[str]) -> None:
        self.cities = cities
        super().__init__(
            f"no coordinates for {cities}. These cities were not seen during "
            "training, so supply pickup_lat, pickup_lon, delivery_lat and "
            "delivery_lon for them."
        )


@dataclass
class ServingContext:
    """The model and everything needed to use it, held for the process lifetime."""

    config: Config
    bundle: ModelBundle
    started_at: float

    @property
    def version(self) -> str:
        """When the serving model was trained, used as its version string."""
        return str(self.bundle.metadata.get("trained_at", "unknown"))

    @property
    def training_end(self) -> pd.Timestamp:
        """The last date the model saw during training."""
        end = self.bundle.metadata.get("training", {}).get("date_to")
        return pd.Timestamp(end) if end else pd.Timestamp.max

    @property
    def known_cities(self) -> set[str]:
        """Cities the model has history for."""
        return self.bundle.features.coordinates.cities

    @property
    def uptime_seconds(self) -> float:
        """How long the service has been up."""
        return time.monotonic() - self.started_at


_context: ServingContext | None = None


def load_context(config_path: str | None = None) -> ServingContext:
    """Load the model and build the serving context.

    Args:
        config_path: Config file to use. Defaults to config/config.yaml.

    Returns:
        The loaded context.

    Raises:
        ModelNotLoadedError: If no trained model is available on disk.
    """
    global _context

    config = load_config(config_path, create_dirs=False)

    try:
        bundle = ModelBundle.load(config.paths.model_dir, config)
    except PersistenceError as exc:
        raise ModelNotLoadedError(
            f"{exc}. Train a model first: python entrypoint/train.py"
        ) from exc

    _context = ServingContext(config=config, bundle=bundle, started_at=time.monotonic())

    logger.info(
        "Model ready: trained %s, %s features, %s known cities",
        _context.version,
        len(bundle.features.feature_names),
        len(_context.known_cities),
    )
    return _context


def get_context() -> ServingContext:
    """Return the loaded serving context.

    Returns:
        The context built at startup.

    Raises:
        ModelNotLoadedError: If startup has not completed.
    """
    if _context is None:
        raise ModelNotLoadedError("the model is not loaded")
    return _context


def clear_context() -> None:
    """Drop the loaded context. Used by tests."""
    global _context
    _context = None


def to_frame(loads: list[LoadRequest]) -> pd.DataFrame:
    """Turn requests into the frame the pipeline expects.

    Args:
        loads: Validated requests.

    Returns:
        One row per load, with dates parsed.
    """
    frame = pd.DataFrame([
        {
            "pickup": load.pickup,
            "delivery": load.delivery,
            "distance": float(load.distance),
            "equipment": load.equipment,
            "weight": load.weight,
            "date": pd.Timestamp(load.date),
            "pickup_lat": load.pickup_lat,
            "pickup_lon": load.pickup_lon,
            "delivery_lat": load.delivery_lat,
            "delivery_lon": load.delivery_lon,
        }
        for load in loads
    ])

    # A batch where every weight is missing arrives as object dtype, which the
    # model rejects. Coercing keeps it numeric so imputation can do its job.
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")

    for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def check_locatable(loads: list[LoadRequest], context: ServingContext) -> None:
    """Confirm every city can be placed before we try to price it.

    Args:
        loads: The submitted requests.
        context: The serving context.

    Raises:
        UnknownCityError: If a city is unfamiliar and no coordinates were given.
    """
    known = context.known_cities
    unplaceable: set[str] = set()

    for load in loads:
        if load.pickup not in known and load.pickup_lat is None:
            unplaceable.add(load.pickup)
        if load.delivery not in known and load.delivery_lat is None:
            unplaceable.add(load.delivery)

    if unplaceable:
        raise UnknownCityError(sorted(unplaceable))


def build_warnings(
    loads: list[LoadRequest],
    context: ServingContext,
) -> list[Warnings]:
    """Flag anything about a request that makes its answer less certain.

    None of these refuse the request. The model returns a number either way;
    the caller is told when that number rests on thinner ground.

    Args:
        loads: The submitted requests.
        context: The serving context.

    Returns:
        One set of warnings per load.
    """
    known = context.known_cities
    training_end = context.training_end

    return [
        Warnings(
            unknown_city=load.pickup not in known or load.delivery not in known,
            date_beyond_training=pd.Timestamp(load.date) > training_end,
            distance_out_of_range=not MIN_DISTANCE <= load.distance <= MAX_DISTANCE,
            weight_imputed=load.weight is None,
        )
        for load in loads
    ]


def predict(loads: list[LoadRequest], context: ServingContext) -> list[float]:
    """Price a set of loads.

    Runs the same cleaning and feature building as the offline pipeline, with
    is_training set to false so nothing is dropped and the training medians are
    reused rather than recomputed.

    Args:
        loads: Validated requests.
        context: The serving context.

    Returns:
        Predicted rates in dollars, in the order submitted.
    """
    check_locatable(loads, context)
    frame = to_frame(loads)

    cleaned, _, _ = clean(
        frame,
        context.config,
        is_training=False,
        artifacts=context.bundle.cleaning,
        label="api",
    )

    features = context.bundle.features.transform(cleaned)
    return [float(rate) for rate in context.bundle.model.predict(features)]


@lru_cache(maxsize=1)
def model_info_payload() -> dict:
    """Build the model description served by the info endpoint.

    Cached because it never changes while the process is running.

    Returns:
        A dictionary matching the ModelInfo schema.
    """
    context = get_context()
    meta = context.bundle.metadata

    return {
        "trained_at": context.version,
        "estimator": meta.get("model", {}).get("estimator", "unknown"),
        "target_transform": meta.get("model", {}).get("target_transform", "unknown"),
        "n_features": meta.get("features", {}).get("n_features", 0),
        "features": meta.get("features", {}).get("names", []),
        "training_rows": meta.get("training", {}).get("n_rows", 0),
        "training_from": meta.get("training", {}).get("date_from", ""),
        "training_to": meta.get("training", {}).get("date_to", ""),
        "known_cities": len(context.known_cities),
        "seasonal_r_squared": meta.get("features", {}).get("seasonal_r_squared", 0.0),
        "validation": meta.get("validation", {}),
    }