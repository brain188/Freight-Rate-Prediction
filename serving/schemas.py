"""Request and response shapes for the prediction API.

The request carries the same six fields as december_chart_inputs.csv: pickup,
delivery, distance, equipment, weight and date. That is deliberate. It is the
thinnest input the model was ever designed to work from, so anything the model
can price offline it can price here, and coordinates are filled in from the
city lookup rather than asked for.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The three trailer types the model was trained on.
Equipment = Literal["Dry Van", "Reefer", "Flatbed"]

# Bounds taken from the training data. A request outside them is not refused,
# but it is flagged in the response so the caller knows the answer is a stretch.
MIN_DISTANCE = 70.0
MAX_DISTANCE = 4000.0
MIN_WEIGHT = 1000.0
MAX_WEIGHT = 50_000.0

MAX_BATCH = 1000


class LoadRequest(BaseModel):
    """One freight load to be priced."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pickup": "Lexington",
                "delivery": "Fort Wayne",
                "distance": 360.0,
                "equipment": "Dry Van",
                "weight": 32000.0,
                "date": "2025-12-15",
            }
        }
    )

    load_id: str | None = Field(
        default=None,
        max_length=64,
        description="Your identifier for this load. Supply it to record the "
                    "actual rate later and have the prediction scored.",
    )
    pickup: str = Field(min_length=1, description="Origin city")
    delivery: str = Field(min_length=1, description="Destination city")
    distance: float = Field(gt=0, le=10_000, description="Trip length in miles")
    equipment: Equipment = Field(description="Trailer type")
    weight: float | None = Field(
        default=None,
        description="Cargo weight in pounds. Imputed when missing.",
    )
    date: Date = Field(description="Date the load moves")

    # Only needed for a city the model has no history for. Known cities are
    # looked up automatically, so callers normally leave these out.
    pickup_lat: float | None = Field(default=None, ge=-90, le=90)
    pickup_lon: float | None = Field(default=None, ge=-180, le=180)
    delivery_lat: float | None = Field(default=None, ge=-90, le=90)
    delivery_lon: float | None = Field(default=None, ge=-180, le=180)

    @field_validator("weight")
    @classmethod
    def weight_may_be_sign_flipped(cls, value: float | None) -> float | None:
        """Accept a negative weight rather than rejecting the request.

        Sign flips are a known defect in this data and the cleaning step
        repairs them, so refusing here would reject a load we can price.

        Args:
            value: The submitted weight, possibly negative or missing.

        Returns:
            The value unchanged.

        Raises:
            ValueError: If the magnitude is implausible for road freight.
        """
        if value is not None and abs(value) > 200_000:
            raise ValueError("weight is implausible for a road freight load")
        return value

    @field_validator("pickup", "delivery")
    @classmethod
    def tidy_city_name(cls, value: str) -> str:
        """Trim surrounding whitespace from a city name.

        Args:
            value: The submitted city name.

        Returns:
            The trimmed name.
        """
        return value.strip()


class BatchRequest(BaseModel):
    """Several loads priced in one call."""

    loads: list[LoadRequest] = Field(min_length=1, max_length=MAX_BATCH)


class Warnings(BaseModel):
    """Reasons to treat a prediction with more caution than usual."""

    unknown_city: bool = Field(
        default=False,
        description="Origin or destination was not seen during training",
    )
    date_beyond_training: bool = Field(
        default=False,
        description="The date falls after the last date the model was trained on",
    )
    distance_out_of_range: bool = Field(
        default=False,
        description="Distance is outside the range seen during training",
    )
    weight_imputed: bool = Field(
        default=False,
        description="Weight was missing and filled from the training median",
    )

    @property
    def any(self) -> bool:
        """Whether any warning is set."""
        return any(vars(self).values())


class PredictionResponse(BaseModel):
    """A priced load."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_rate": 793.35,
                "rate_per_mile": 2.204,
                "model_version": "2026-08-05T20:12:03+00:00",
                "warnings": {
                    "unknown_city": False,
                    "date_beyond_training": True,
                    "distance_out_of_range": False,
                    "weight_imputed": False,
                },
                "latency_ms": 8.4,
            }
        }
    )

    predicted_rate: float = Field(description="Predicted rate in dollars")
    rate_per_mile: float = Field(description="Predicted rate divided by distance")
    model_version: str = Field(description="When the serving model was trained")
    warnings: Warnings
    latency_ms: float


class BatchResponse(BaseModel):
    """Several priced loads, in the order they were submitted."""

    predictions: list[PredictionResponse]
    count: int
    latency_ms: float


class ActualRequest(BaseModel):
    """The rate a load actually went for."""

    load_id: str = Field(min_length=1, max_length=64)
    actual_rate: float = Field(gt=0, le=1_000_000, description="Confirmed rate in dollars")
    source: str = Field(default="api", max_length=32)


class BatchActualRequest(BaseModel):
    """Several outcomes reported in one call."""

    actuals: list[ActualRequest] = Field(min_length=1, max_length=10_000)


class BatchActualResponse(BaseModel):
    """How many outcomes were stored."""

    recorded: int
    failed: int


class ActualResponse(BaseModel):
    """Confirmation that an outcome was stored."""

    load_id: str
    actual_rate: float
    recorded: bool


class HealthResponse(BaseModel):
    """Whether the service is up and able to serve."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float

    # Logging being down is not a serving failure, so it is reported rather
    # than folded into the overall status.
    store_available: bool = False


class ModelInfo(BaseModel):
    """What the loaded model is and how it scored."""

    model_config = ConfigDict(protected_namespaces=())

    trained_at: str
    estimator: str
    target_transform: str
    n_features: int
    features: list[str]
    training_rows: int
    training_from: str
    training_to: str
    known_cities: int
    seasonal_r_squared: float
    validation: dict


class ErrorResponse(BaseModel):
    """A failed request."""

    detail: str
    error_type: str