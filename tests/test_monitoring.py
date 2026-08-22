"""Checks on prediction logging and the metrics built from it.

The rule these protect: logging must never break a prediction. A store that is
down degrades monitoring, not serving.
"""

from __future__ import annotations

import datetime as dt

import pytest

from monitoring.performance import (
    MIN_SCORED_ROWS,
    daily_metrics,
    load_predictions,
    segment_metrics,
    snapshot,
    traffic_summary,
)
from serving.store import PredictionRecord, PredictionStore, StoreUnavailableError


def make_record(index: int, **overrides) -> PredictionRecord:
    """Build a prediction record for testing.

    Args:
        index: Used to build a unique load identifier.
        **overrides: Fields to change from the defaults.

    Returns:
        The record.
    """
    defaults = {
        "load_id": f"L-{index:04d}",
        "pickup": "Atlanta",
        "delivery": "Dallas",
        "distance": 800.0,
        "equipment": "Reefer",
        "weight": 30000.0,
        "load_date": dt.date(2025, 11, 20),
        "predicted_rate": 1900.0,
        "rate_per_mile": 2.375,
        "model_version": "test-v1",
        "latency_ms": 12.0,
    }
    return PredictionRecord(**{**defaults, **overrides})


@pytest.fixture
def store(tmp_path) -> PredictionStore:
    """A store backed by a temporary SQLite file.

    Returns:
        A connected store.
    """
    instance = PredictionStore(f"sqlite:///{tmp_path / 'test.db'}")
    instance.connect()
    return instance


def test_predictions_are_written(store):
    """Logged predictions land in the table."""
    assert store.log_predictions([make_record(i) for i in range(10)]) == 10
    assert store.count_predictions() == 10


def test_logging_never_raises_when_the_store_is_down(store):
    """A dead database returns zero rather than breaking the request.

    This is the guarantee that keeps a monitoring outage from becoming a
    serving outage.
    """
    broken = PredictionStore("postgresql://nobody@127.0.0.1:1/missing")
    broken.connect()

    assert broken.is_available is False
    assert broken.log_predictions([make_record(1)]) == 0


def test_reads_do_raise_when_the_store_is_down():
    """A metrics read fails loudly, unlike a write.

    A caller asking for metrics needs to know the answer is missing, rather
    than receiving an empty result that looks like real data.
    """
    broken = PredictionStore("postgresql://nobody@127.0.0.1:1/missing")
    broken.connect()

    with pytest.raises(StoreUnavailableError):
        broken.count_predictions()


def test_actual_is_recorded_and_updated(store):
    """An outcome can be stored and later corrected."""
    store.log_predictions([make_record(1)])

    store.record_actual("L-0001", 1950.0)
    assert store.count_actuals() == 1

    store.record_actual("L-0001", 1975.0, source="correction")
    assert store.count_actuals() == 1


def test_snapshot_is_empty_before_any_outcome(store):
    """Predictions with no outcomes give volume but no metrics.

    The normal state for a freshly deployed model, since rates take days to
    confirm.
    """
    store.log_predictions([make_record(i) for i in range(50)])

    result = snapshot(store)

    assert result.n_predictions == 50
    assert result.n_scored == 0
    assert result.coverage == 0.0
    assert result.metrics == {}


def test_snapshot_scores_only_matched_rows(store):
    """Metrics come from the predictions that have outcomes."""
    store.log_predictions([make_record(i) for i in range(100)])

    for index in range(40):
        store.record_actual(f"L-{index:04d}", 1900.0)

    result = snapshot(store)

    assert result.n_predictions == 100
    assert result.n_scored == 40
    assert result.coverage == pytest.approx(0.4)
    assert result.metrics["mae"] == pytest.approx(0.0, abs=0.01)


def test_snapshot_flags_thin_coverage(store):
    """Too few outcomes marks the metrics unreliable rather than hiding them."""
    store.log_predictions([make_record(i) for i in range(50)])

    for index in range(MIN_SCORED_ROWS - 1):
        store.record_actual(f"L-{index:04d}", 1900.0)

    assert snapshot(store).is_reliable is False


def test_error_is_computed_correctly(store):
    """A known error comes back as the expected metric."""
    store.log_predictions([make_record(i, predicted_rate=1000.0) for i in range(40)])

    for index in range(40):
        store.record_actual(f"L-{index:04d}", 1100.0)

    metrics = snapshot(store).metrics

    assert metrics["mae"] == pytest.approx(100.0, abs=0.01)
    assert metrics["mape"] == pytest.approx(100 / 1100 * 100, abs=0.01)
    assert metrics["bias"] == pytest.approx(-100.0, abs=0.01)


def test_unknown_city_rate_tracks_traffic(store):
    """The drift signal reflects what was actually sent.

    Available without waiting for outcomes, which is what makes it the first
    place to look when traffic changes.
    """
    store.log_predictions([
        make_record(i, unknown_city=i < 25) for i in range(100)
    ])

    assert traffic_summary(store)["unknown_city_rate"] == pytest.approx(0.25)


def test_segments_split_the_traffic(store):
    """Grouping covers every row without inventing any."""
    store.log_predictions([
        make_record(i, equipment="Dry Van" if i % 2 else "Reefer") for i in range(60)
    ])

    table = segment_metrics(store, "equipment")

    assert set(table["group"]) == {"Dry Van", "Reefer"}
    assert table["n_predictions"].sum() == 60


def test_daily_metrics_have_one_row_per_day(store):
    """The daily table matches the days present in the data."""
    store.log_predictions([make_record(i) for i in range(20)])

    table = daily_metrics(store)

    assert len(table) == 1
    assert table["n_predictions"].iloc[0] == 20


def test_load_predictions_keeps_unmatched_rows(store):
    """The join is a left join, so unscored traffic is still visible."""
    store.log_predictions([make_record(i) for i in range(30)])
    store.record_actual("L-0000", 1900.0)

    frame = load_predictions(store)

    assert len(frame) == 30
    assert frame["actual_rate"].notna().sum() == 1


def test_empty_store_returns_empty_results(store):
    """Nothing logged yet gives empty results rather than an error."""
    assert load_predictions(store).empty
    assert traffic_summary(store)["n_predictions"] == 0
    assert snapshot(store).n_predictions == 0


def test_sqlite_uses_write_ahead_logging(tmp_path):
    """SQLite is configured so a reader does not block a writer.

    Without this, the dashboard polling the same file the API logs to produces
    "database is locked" errors under any real load.
    """
    from sqlalchemy import text

    instance = PredictionStore(f"sqlite:///{tmp_path / 'wal.db'}")
    instance.connect()

    with instance.engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert connection.execute(text("PRAGMA busy_timeout")).scalar() > 0


def test_concurrent_writers_and_readers_do_not_lock(tmp_path):
    """Several processes can write and read at once without failing.

    This reproduces the situation that breaks a naive SQLite setup: the API
    logging predictions while the dashboard polls the same file.
    """
    import threading

    from monitoring.performance import load_predictions

    url = f"sqlite:///{tmp_path / 'concurrent.db'}"
    PredictionStore(url).connect()

    failures = []

    def write(tag: str) -> None:
        store = PredictionStore(url)
        store.connect()
        for batch in range(5):
            written = store.log_predictions(
                [make_record(f"{tag}{batch}{i}") for i in range(50)]
            )
            if written == 0:
                failures.append(f"write {tag}{batch}")

    def read() -> None:
        store = PredictionStore(url)
        store.connect()
        for _ in range(10):
            try:
                load_predictions(store, 90)
            except Exception as exc: # noqa: BLE001
                failures.append(f"read: {exc}")

    threads = [
        threading.Thread(target=write, args=("a",)),
        threading.Thread(target=write, args=("b",)),
        threading.Thread(target=read),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"concurrent access failed: {failures[:3]}"