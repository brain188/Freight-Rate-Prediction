"""Recording what the model predicted and what actually happened.

Written with SQLAlchemy Core rather than an ORM so the SQL stays visible, and
so the same code runs against Postgres in Docker and SQLite in tests.

The rule that shapes this module: logging must never break a prediction. A
database that is down, slow or misconfigured degrades monitoring, not serving.
Every write is wrapped and failures are logged rather than raised.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy import (
    Date as SADate,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.logger import get_logger

logger = get_logger(__name__)

# Falls back to a local SQLite file so the API runs with no database to set up.
DEFAULT_DATABASE_URL = "sqlite:///./freight_monitoring.db"

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", String(36), primary_key=True),
    Column("load_id", String(64), nullable=True),
    Column("pickup", String(128), nullable=False),
    Column("delivery", String(128), nullable=False),
    Column("distance", Float, nullable=False),
    Column("equipment", String(32), nullable=False),
    Column("weight", Float, nullable=True),
    Column("load_date", SADate, nullable=False),
    Column("predicted_rate", Float, nullable=False),
    Column("rate_per_mile", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("unknown_city", Boolean, nullable=False, default=False),
    Column("date_beyond_training", Boolean, nullable=False, default=False),
    Column("distance_out_of_range", Boolean, nullable=False, default=False),
    Column("weight_imputed", Boolean, nullable=False, default=False),
    Column("latency_ms", Float, nullable=True),
    Column("predicted_at", DateTime(timezone=True), nullable=False),
    Index("idx_predictions_predicted_at", "predicted_at"),
    Index("idx_predictions_load_id", "load_id"),
    Index("idx_predictions_model_version", "model_version"),
)

actuals = Table(
    "actuals",
    metadata,
    Column("load_id", String(64), primary_key=True),
    Column("actual_rate", Float, nullable=False),
    Column("source", String(32), nullable=False, default="api"),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Index("idx_actuals_recorded_at", "recorded_at"),
)


class StoreUnavailableError(RuntimeError):
    """Raised when a read needs the database and it cannot be reached."""


@dataclass
class PredictionRecord:
    """One prediction, ready to be written."""

    pickup: str
    delivery: str
    distance: float
    equipment: str
    weight: float | None
    load_date: Date
    predicted_rate: float
    rate_per_mile: float
    model_version: str
    unknown_city: bool = False
    date_beyond_training: bool = False
    distance_out_of_range: bool = False
    weight_imputed: bool = False
    latency_ms: float | None = None
    load_id: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Build the row to insert.

        Returns:
            Column values, with a fresh identifier and timestamp.
        """
        return {
            "prediction_id": str(uuid.uuid4()),
            "load_id": self.load_id,
            "pickup": self.pickup,
            "delivery": self.delivery,
            "distance": self.distance,
            "equipment": self.equipment,
            "weight": self.weight,
            "load_date": self.load_date,
            "predicted_rate": self.predicted_rate,
            "rate_per_mile": self.rate_per_mile,
            "model_version": self.model_version,
            "unknown_city": self.unknown_city,
            "date_beyond_training": self.date_beyond_training,
            "distance_out_of_range": self.distance_out_of_range,
            "weight_imputed": self.weight_imputed,
            "latency_ms": self.latency_ms,
            "predicted_at": datetime.now(timezone.utc),
        }


class PredictionStore:
    """Reads and writes the monitoring tables.

    Args:
        url: SQLAlchemy database URL. Read from DATABASE_URL when omitted.
        echo: Whether to log the SQL, useful when debugging locally.
    """

    def __init__(self, url: str | None = None, echo: bool = False) -> None:
        self.url = url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        self._engine: Engine | None = None
        self._echo = echo
        self._available = False

    @property
    def is_available(self) -> bool:
        """Whether the last connection attempt succeeded."""
        return self._available

    @property
    def engine(self) -> Engine:
        """The connection pool, created on first use.

        Returns:
            The engine.
        """
        if self._engine is None:
            # pool_pre_ping costs one round trip and saves the stale connection
            # errors that appear when the database restarts under a live API.
            self._engine = create_engine(
                self.url, echo=self._echo, pool_pre_ping=True, future=True
            )
        return self._engine

    def connect(self) -> bool:
        """Create the tables and confirm the database is reachable.

        Called once at startup. A failure is reported rather than raised, so
        the API still serves predictions with logging switched off.

        Returns:
            True when the store is usable.
        """
        try:
            metadata.create_all(self.engine)
            with self.engine.connect() as connection:
                connection.execute(select(func.count()).select_from(predictions))
            self._available = True
            logger.info("Prediction store ready: %s", self._safe_url())
        except SQLAlchemyError as exc:
            self._available = False
            logger.warning(
                "Prediction store unavailable (%s). Serving continues without logging.",
                exc.__class__.__name__,
            )
        return self._available

    def close(self) -> None:
        """Dispose of the connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._available = False

    def _safe_url(self) -> str:
        """Return the URL with any password removed.

        Returns:
            A URL safe to write to logs.
        """
        if "@" in self.url and "://" in self.url:
            scheme, rest = self.url.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return self.url

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Yield a connection inside a transaction.

        Yields:
            An open connection.

        Raises:
            StoreUnavailableError: If the database cannot be reached.
        """
        try:
            with self.engine.begin() as connection:
                yield connection
        except SQLAlchemyError as exc:
            raise StoreUnavailableError(str(exc)) from exc

    def log_predictions(self, records: list[PredictionRecord]) -> int:
        """Write predictions, swallowing any failure.

        Runs after the response has been sent, so an error here must never
        surface to the caller. It is logged and the count returned is zero.

        Args:
            records: Predictions to write.

        Returns:
            How many rows were written.
        """
        if not records or not self._available:
            return 0

        try:
            with self._connection() as connection:
                connection.execute(
                    predictions.insert(), [record.to_row() for record in records]
                )
            return len(records)
        except StoreUnavailableError as exc:
            logger.error("Could not log %s predictions: %s", len(records), exc)
            return 0

    def record_actual(self, load_id: str, actual_rate: float, source: str = "api") -> bool:
        """Store the rate a load actually went for.

        Upserts, because a corrected figure should replace an earlier one
        rather than fail on the primary key.

        Args:
            load_id: The load the outcome belongs to.
            actual_rate: The confirmed rate in dollars.
            source: Where the figure came from.

        Returns:
            True when the row was written.

        Raises:
            StoreUnavailableError: If the database cannot be reached. Unlike
                logging, this is a direct request, so the caller is told.
        """
        row = {
            "load_id": load_id,
            "actual_rate": actual_rate,
            "source": source,
            "recorded_at": datetime.now(timezone.utc),
        }

        with self._connection() as connection:
            existing = connection.execute(
                select(actuals.c.load_id).where(actuals.c.load_id == load_id)
            ).first()

            if existing:
                connection.execute(
                    actuals.update().where(actuals.c.load_id == load_id).values(**row)
                )
                logger.info("Updated actual for %s", load_id)
            else:
                connection.execute(actuals.insert().values(**row))

        return True

    def record_actuals(self, rows: list[tuple[str, float, str]]) -> int:
        """Store many outcomes in one transaction.

        A replay reports thousands at a time, and one round trip each is the
        difference between seconds and minutes.

        Args:
            rows: Load identifier, settled rate and source for each outcome.

        Returns:
            How many were written.

        Raises:
            StoreUnavailableError: If the database cannot be reached.
        """
        if not rows:
            return 0

        now = datetime.now(timezone.utc)
        incoming = {load_id: (rate, source) for load_id, rate, source in rows}

        with self._connection() as connection:
            existing = {
                row[0]
                for row in connection.execute(
                    select(actuals.c.load_id).where(actuals.c.load_id.in_(incoming))
                ).all()
            }

            fresh = [
                {"load_id": lid, "actual_rate": rate, "source": source, "recorded_at": now}
                for lid, (rate, source) in incoming.items()
                if lid not in existing
            ]
            if fresh:
                connection.execute(actuals.insert(), fresh)

            for lid in existing:
                rate, source = incoming[lid]
                connection.execute(
                    actuals.update().where(actuals.c.load_id == lid).values(
                        actual_rate=rate, source=source, recorded_at=now
                    )
                )

        return len(incoming)

    def count_predictions(self) -> int:
        """Count every prediction logged.

        Returns:
            The row count.

        Raises:
            StoreUnavailableError: If the database cannot be reached.
        """
        with self._connection() as connection:
            return int(
                connection.execute(select(func.count()).select_from(predictions)).scalar_one()
            )

    def count_actuals(self) -> int:
        """Count every outcome recorded.

        Returns:
            The row count.

        Raises:
            StoreUnavailableError: If the database cannot be reached.
        """
        with self._connection() as connection:
            return int(
                connection.execute(select(func.count()).select_from(actuals)).scalar_one()
            )


_store: PredictionStore | None = None


def get_store() -> PredictionStore:
    """Return the process wide store, creating it on first use.

    Returns:
        The store.
    """
    global _store
    if _store is None:
        _store = PredictionStore()
    return _store


def reset_store(url: str | None = None) -> PredictionStore:
    """Replace the store, used by tests and at startup.

    Args:
        url: Database URL for the new store.

    Returns:
        The new store.
    """
    global _store
    if _store is not None:
        _store.close()
    _store = PredictionStore(url)
    return _store