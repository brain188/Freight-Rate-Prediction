"""Checks on the prediction API.

The most important one is parity. A served prediction must equal the offline
one to the cent, because the whole point of reusing the pipeline rather than
reimplementing it is that the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from serving.app import app

KNOWN_LOAD = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32000.0,
    "date": "2025-12-15",
}


@pytest.fixture(scope="module")
def client():
    """A test client with the model loaded.

    Yields:
        The client, or skips the module when no model has been trained.
    """
    with TestClient(app) as test_client:
        if not test_client.get("/health").json()["model_loaded"]:
            pytest.skip("no trained model — run entrypoint/train.py first")
        yield test_client


def test_health_reports_a_loaded_model(client):
    """The health endpoint says the service can serve."""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info_matches_the_trained_model(client):
    """The info endpoint reports what was actually trained."""
    body = client.get("/model/info").json()

    assert body["estimator"] == "lightgbm"
    assert body["n_features"] == len(body["features"])
    assert body["known_cities"] > 0


def test_single_prediction_is_positive(client):
    """A normal load returns a usable rate."""
    body = client.post("/predict", json=KNOWN_LOAD).json()

    assert body["predicted_rate"] > 0
    assert body["rate_per_mile"] == pytest.approx(
        body["predicted_rate"] / KNOWN_LOAD["distance"], rel=0.01
    )


@pytest.mark.integration
def test_api_matches_the_offline_pipeline(client, config):
    """A served prediction equals the offline one to the cent.

    This is the test that protects the reason for wrapping the existing
    pipeline instead of writing a second one for serving.
    """
    if not config.paths.december.is_file():
        pytest.skip("december_chart_inputs.csv not present")

    december = pd.read_csv(config.paths.december)

    if december["predicted_rate"].isna().all():
        pytest.skip("December file not yet filled — run entrypoint/predict.py")

    loads = [
        {
            "pickup": row.pickup,
            "delivery": row.delivery,
            "distance": float(row.distance),
            "equipment": row.equipment,
            "weight": float(row.weight),
            "date": str(pd.Timestamp(row.date).date()),
        }
        for row in december.itertuples()
    ]

    response = client.post("/predict/batch", json={"loads": loads}).json()
    served = np.array([p["predicted_rate"] for p in response["predictions"]])

    assert np.allclose(served, december["predicted_rate"].to_numpy(), atol=0.005)


def test_batch_returns_one_prediction_per_load(client):
    """Order and count are preserved through a batch."""
    loads = [
        KNOWN_LOAD,
        {**KNOWN_LOAD, "distance": 1200.0},
        {**KNOWN_LOAD, "equipment": "Reefer"},
    ]

    body = client.post("/predict/batch", json={"loads": loads}).json()

    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert all(p["predicted_rate"] > 0 for p in body["predictions"])


def test_missing_weight_is_imputed_not_rejected(client):
    """A load with no weight is priced and the response says so."""
    body = client.post("/predict", json={**KNOWN_LOAD, "weight": None}).json()

    assert body["predicted_rate"] > 0
    assert body["warnings"]["weight_imputed"] is True


def test_negative_weight_is_repaired(client):
    """A sign flipped weight gives the same answer as the correct one.

    Sign flips are a known defect in this data, so the API repairs them rather
    than refusing a load it can price.
    """
    positive = client.post("/predict", json=KNOWN_LOAD).json()
    negative = client.post("/predict", json={**KNOWN_LOAD, "weight": -32000.0}).json()

    assert negative["predicted_rate"] == pytest.approx(positive["predicted_rate"])


def test_future_date_is_flagged_but_served(client):
    """A date past the training window is priced and warned about."""
    body = client.post("/predict", json={**KNOWN_LOAD, "date": "2026-03-01"}).json()

    assert body["predicted_rate"] > 0
    assert body["warnings"]["date_beyond_training"] is True


def test_unknown_city_without_coordinates_is_refused(client):
    """A city the model cannot place is refused with a usable message.

    Guessing a location would return a confident number resting on nothing,
    so the caller is told what to supply instead.
    """
    response = client.post("/predict", json={**KNOWN_LOAD, "pickup": "Chicago"})

    assert response.status_code == 422
    assert response.json()["error_type"] == "unknown_city"
    assert "Chicago" in response.json()["detail"]


def test_unknown_city_with_coordinates_is_served(client):
    """Supplying coordinates lets an unfamiliar city be priced."""
    body = client.post(
        "/predict",
        json={
            **KNOWN_LOAD,
            "pickup": "Chicago",
            "pickup_lat": 41.88,
            "pickup_lon": -87.63,
        },
    ).json()

    assert body["predicted_rate"] > 0
    assert body["warnings"]["unknown_city"] is True


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"equipment": "Tanker"}, "trailer type not in the training data"),
        ({"distance": 0}, "distance must be positive"),
        ({"distance": -100}, "distance must be positive"),
        ({"weight": 500_000}, "weight is implausible"),
        ({"date": "not-a-date"}, "unparseable date"),
    ],
)
def test_bad_input_is_rejected(client, payload, reason):
    """Malformed requests fail validation rather than reaching the model."""
    assert client.post("/predict", json={**KNOWN_LOAD, **payload}).status_code == 422, reason


def test_missing_field_is_rejected(client):
    """An incomplete request is rejected."""
    incomplete = {k: v for k, v in KNOWN_LOAD.items() if k != "date"}

    assert client.post("/predict", json=incomplete).status_code == 422


def test_empty_batch_is_rejected(client):
    """A batch has to contain something."""
    assert client.post("/predict/batch", json={"loads": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    """Batches are capped so one request cannot tie up the process."""
    assert client.post("/predict/batch", json={"loads": [KNOWN_LOAD] * 1001}).status_code == 422


def test_latency_header_is_present(client):
    """Every response carries its processing time."""
    response = client.post("/predict", json=KNOWN_LOAD)

    assert "X-Process-Time-Ms" in response.headers
    assert float(response.headers["X-Process-Time-Ms"]) >= 0
