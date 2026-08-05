"""Shared fixtures.

Most tests run on small synthetic frames so they stay fast and do not depend on
the data being present. The few that need the real files are marked
`integration` and skip themselves when the CSVs are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config

# Two cities the synthetic frames use, with fixed coordinates.
CITIES = {
    "Atlanta": (33.75, -84.39),
    "Dallas": (32.78, -96.80),
    "Denver": (39.74, -104.99),
}


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom marker so pytest does not warn about it.

    Args:
        config: The pytest configuration object.
    """
    config.addinivalue_line(
        "markers", "integration: needs the assessment CSVs to be present"
    )


@pytest.fixture(scope="session")
def config() -> Config:
    """The project configuration, loaded once per test session.

    Returns:
        The validated config.
    """
    return load_config(create_dirs=False)


def make_loads(n: int = 400, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    """Build a small, clean set of synthetic loads.

    Args:
        n: How many rows to create.
        seed: Random seed, so tests are repeatable.
        with_target: Whether to include the posted_rate column.

    Returns:
        A frame shaped like train_test.csv.
    """
    rng = np.random.default_rng(seed)
    names = list(CITIES)

    pickup = rng.choice(names, n)
    delivery = np.array([
        rng.choice([c for c in names if c != origin]) for origin in pickup
    ])

    distance = rng.uniform(100, 2000, n).round(1)
    equipment = rng.choice(["Dry Van", "Reefer", "Flatbed"], n)

    frame = pd.DataFrame({
        "load_id": [f"TE-{i:06d}" for i in range(1, n + 1)],
        "pickup": pickup,
        "delivery": delivery,
        "pickup_lat": [CITIES[c][0] for c in pickup],
        "pickup_lon": [CITIES[c][1] for c in pickup],
        "delivery_lat": [CITIES[c][0] for c in delivery],
        "delivery_lon": [CITIES[c][1] for c in delivery],
        "distance": distance,
        "equipment": equipment,
        "weight": rng.uniform(5_000, 45_000, n).round(0),
        "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D"),
        "market_index": rng.normal(100, 10, n).round(2),
        "quote_signal": rng.normal(0, 1, n).round(3),
    })

    if with_target:
        # Roughly $2.15 a mile with a little noise, so rate per mile lands
        # inside the range the cleaning rules keep.
        frame["posted_rate"] = (distance * rng.uniform(1.8, 2.6, n)).round(2)

    return frame


@pytest.fixture
def clean_loads() -> pd.DataFrame:
    """Synthetic loads with no defects.

    Returns:
        A clean frame.
    """
    return make_loads()


@pytest.fixture
def dirty_loads() -> pd.DataFrame:
    """Synthetic loads carrying every defect the real data contains.

    Returns:
        A frame with known counts of each defect, listed in the attrs dict.
    """
    frame = make_loads()

    # 10 sign-flipped weights, 5 missing weights, 5 missing market_index.
    frame.loc[0:9, "weight"] = -frame.loc[0:9, "weight"]
    frame.loc[10:14, "weight"] = np.nan
    frame.loc[15:19, "market_index"] = np.nan

    # 8 rates far above the sane band and 6 far below.
    frame.loc[20:27, "posted_rate"] = frame.loc[20:27, "distance"] * 9.0
    frame.loc[28:33, "posted_rate"] = frame.loc[28:33, "distance"] * 0.4

    frame.attrs["n_negative_weight"] = 10
    frame.attrs["n_missing_weight"] = 5
    frame.attrs["n_missing_market_index"] = 5
    frame.attrs["n_rate_high"] = 8
    frame.attrs["n_rate_low"] = 6
    return frame


@pytest.fixture
def unlabelled_loads() -> pd.DataFrame:
    """Synthetic loads with no target, as at prediction time.

    Returns:
        A frame without posted_rate, carrying a few defects.
    """
    frame = make_loads(n=200, seed=7, with_target=False)
    frame.loc[0:4, "weight"] = -frame.loc[0:4, "weight"]
    frame.loc[5:7, "weight"] = np.nan
    return frame