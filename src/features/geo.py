"""Geographic features: coordinates, distance transforms, and lane direction.

Coordinates matter more than they first appear. Eight cities show up only in
the validation set, covering 12% of the loads we must price, and the December
file arrives with no coordinates at all. Latitude and longitude are continuous,
so an unfamiliar city still has a usable position — city names alone do not.
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

EARTH_RADIUS_MILES = 3958.8

COORDINATE_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


class GeoError(Exception):
    """Raised when geographic features cannot be built."""


@dataclass(frozen=True)
class CityCoordinates:
    """Latitude and longitude for every city seen during training.

    Needed because december_chart_inputs.csv gives city names but no
    coordinates, so we look them up from what training already told us.
    """

    latitude: dict[str, float]
    longitude: dict[str, float]

    @property
    def cities(self) -> set[str]:
        """Every city with a known position."""
        return set(self.latitude)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> CityCoordinates:
        """Build the lookup from a frame that carries coordinates.

        Each city is taken from both the pickup and delivery columns, which
        agree throughout the data.

        Args:
            df: Frame with city names and coordinate columns.

        Returns:
            The fitted lookup.

        Raises:
            GeoError: If the required columns are absent.
        """
        missing = [c for c in COORDINATE_COLUMNS if c not in df.columns]
        if missing:
            raise GeoError(f"cannot build coordinate lookup, missing: {missing}")

        origins = df[["pickup", "pickup_lat", "pickup_lon"]].rename(
            columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
        )
        destinations = df[["delivery", "delivery_lat", "delivery_lon"]].rename(
            columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
        )
        combined = pd.concat([origins, destinations]).drop_duplicates("city")

        lookup = cls(
            latitude={str(r.city): float(r.lat) for r in combined.itertuples()},
            longitude={str(r.city): float(r.lon) for r in combined.itertuples()},
        )
        logger.info("Built coordinate lookup for %s cities", len(lookup.cities))
        return lookup

    def to_json(self, path: Path) -> None:
        """Save the lookup next to the model.

        Args:
            path: Destination JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        logger.debug("Saved coordinate lookup to %s", path)

    @classmethod
    def from_json(cls, path: Path) -> CityCoordinates:
        """Load a lookup saved by a previous training run.

        Args:
            path: JSON file written by to_json.

        Returns:
            The stored lookup.

        Raises:
            GeoError: If the file is missing or malformed.
        """
        if not path.is_file():
            raise GeoError(f"coordinate lookup not found: {path}")

        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise GeoError(f"could not read {path}: {exc}") from exc


def attach_coordinates(df: pd.DataFrame, lookup: CityCoordinates) -> pd.DataFrame:
    """Fill in coordinates from city names where they are missing.

    This is what makes the December file usable: it names Lexington and Fort
    Wayne but ships no coordinates.

    Args:
        df: Frame with pickup and delivery city names.
        lookup: Coordinates learned during training.

    Returns:
        The frame with all four coordinate columns present.
    """
    out = df.copy()

    for role in ("pickup", "delivery"):
        for axis, table in (("lat", lookup.latitude), ("lon", lookup.longitude)):
            column = f"{role}_{axis}"
            mapped = out[role].map(table)
            out[column] = out[column].fillna(mapped) if column in out.columns else mapped

    unknown = out[COORDINATE_COLUMNS].isna().any(axis=1)
    if unknown.any():
        names = (
            sorted(set(out.loc[unknown, "pickup"]) | set(out.loc[unknown, "delivery"]))
            - lookup.cities
        )
        logger.warning(
            "%s rows have no coordinates — unknown cities: %s",
            f"{int(unknown.sum()):,}",
            sorted(names),
        )

    return out


def haversine_miles(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Great-circle distance in miles between two coordinate pairs.

    Args:
        lat1: Origin latitudes.
        lon1: Origin longitudes.
        lat2: Destination latitudes.
        lon2: Destination longitudes.

    Returns:
        Straight-line distance in miles.
    """
    phi1, lam1, phi2, lam2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    inner = (
        np.sin((phi2 - phi1) / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(inner))


def add_distance_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add transforms of the distance column.

    Rate per mile falls with distance along a curve, not a straight line, so
    the log gives the model a shape closer to what the data actually does.

    Args:
        df: Frame with a distance column.
        config: Loaded project configuration.

    Returns:
        The frame with distance features added.

    Raises:
        GeoError: If distance holds zero or negative values.
    """
    if (df["distance"] <= 0).any():
        raise GeoError("distance contains zero or negative values")

    out = df.copy()

    if config.features.use_log_distance:
        out["log_distance"] = np.log(out["distance"])

    return out


def add_geo_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add features derived from the origin and destination positions.

    Latitude and longitude place a load in space, which is how an unseen city
    still gets a sensible prediction. Direction and span describe the lane.

    Args:
        df: Frame with coordinate columns already attached.
        config: Loaded project configuration.

    Returns:
        The frame with geographic features added.

    Raises:
        GeoError: If coordinates are missing.
    """
    if not config.features.use_coordinates:
        return df

    missing = [c for c in COORDINATE_COLUMNS if c not in df.columns]
    if missing:
        raise GeoError(f"coordinates not attached, missing: {missing}")

    out = df.copy()

    # Straight-line distance and how much the road detours around it. Circuity
    # sits near 1.18 throughout, so a departure from it flags an unusual lane.
    out["great_circle"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["circuity"] = out["distance"] / out["great_circle"].clip(lower=1.0)

    # Which way the load travels. North-south and east-west movement price
    # differently because return freight is easier to find in some directions.
    out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]

    # Midpoint stands in for the broad region a lane sits in.
    out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2
    out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2

    return out


def geo_feature_names(config: Config) -> list[str]:
    """List every geographic and distance feature, in build order.

    Args:
        config: Loaded project configuration.

    Returns:
        Column names the model will use.
    """
    names = ["distance"]

    if config.features.use_log_distance:
        names.append("log_distance")

    if config.features.use_coordinates:
        names.extend(
            [
                *COORDINATE_COLUMNS,
                "great_circle",
                "circuity",
                "delta_lat",
                "delta_lon",
                "mid_lat",
                "mid_lon",
            ]
        )

    return names
