"""Checks on feature building.

Two failure modes matter most here. A city the model has never seen must not
produce a crash or a NaN, because 12% of the validation loads involve one. And
the time features must still be defined in December, or every prediction
collapses onto the last date the model saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean
from src.features.build import FeatureBuilder, FeatureError
from src.features.encoders import OrdinalEncoder
from src.features.geo import CityCoordinates, GeoError, attach_coordinates
from src.features.seasonality import SeasonalIndex, add_time_features

DECEMBER = pd.date_range("2025-12-01", "2025-12-31", freq="D")


def test_unseen_city_gets_a_code_not_a_crash(clean_loads, config):
    """An unfamiliar city encodes to the reserved value instead of NaN."""
    encoder = OrdinalEncoder.fit(clean_loads, unknown_value=config.features.unknown_category_value)

    unfamiliar = clean_loads.head(5).copy()
    unfamiliar["pickup"] = "Chicago"

    encoded = encoder.transform(unfamiliar)

    assert encoded["pickup_code"].notna().all()
    assert (encoded["pickup_code"] == config.features.unknown_category_value).all()
    assert (encoded["has_unknown_city"] == 1).all()


def test_known_cities_are_not_flagged(clean_loads, config):
    """Familiar cities leave the unknown flag clear."""
    encoder = OrdinalEncoder.fit(clean_loads, unknown_value=config.features.unknown_category_value)

    encoded = encoder.transform(clean_loads)

    assert (encoded["has_unknown_city"] == 0).all()
    assert (encoded["pickup_code"] >= 0).all()


def test_cities_share_one_vocabulary(clean_loads, config):
    """A city keeps the same code at either end of a lane."""
    encoder = OrdinalEncoder.fit(clean_loads, unknown_value=-1)

    for city in encoder.known_categories("pickup"):
        assert encoder.mapping["pickup"][city] == encoder.mapping["delivery"][city]


def test_coordinates_are_attached_from_city_names(clean_loads):
    """A file with names but no coordinates gets them filled in.

    This is what makes december_chart_inputs.csv usable — it ships without
    latitude or longitude.
    """
    lookup = CityCoordinates.fit(clean_loads)

    thin = clean_loads[["pickup", "delivery", "distance", "equipment", "weight", "date"]].head(10)
    filled = attach_coordinates(thin, lookup)

    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        assert column in filled.columns
        assert filled[column].notna().all()


def test_coordinate_lookup_round_trip(clean_loads, tmp_path):
    """A saved coordinate lookup loads back unchanged."""
    lookup = CityCoordinates.fit(clean_loads)
    path = tmp_path / "city_coordinates.json"

    lookup.to_json(path)

    assert CityCoordinates.from_json(path) == lookup


def test_missing_coordinates_are_reported(clean_loads):
    """Building geo features without coordinates fails loudly."""
    with pytest.raises(GeoError, match="missing"):
        CityCoordinates.fit(clean_loads.drop(columns=["pickup_lat"]))


def test_seasonal_index_is_defined_past_the_training_window(clean_loads, config):
    """The seasonal curve still returns values for December.

    Training stops in October. A curve that stopped there would leave every
    December prediction pinned to the last date the model saw.
    """
    index = SeasonalIndex.fit(
        clean_loads["date"],
        np.log(clean_loads["posted_rate"] / clean_loads["distance"]),
        order=config.features.fourier_order,
        period=config.features.fourier_period,
    )

    offsets = index.predict(pd.Series(DECEMBER))

    assert len(offsets) == 31
    assert np.isfinite(offsets).all()
    assert len(np.unique(offsets.round(8))) == 31


def test_seasonal_values_stay_inside_the_trained_range(clean_loads, config):
    """December offsets land within the range seen during training.

    That is what lets a tree split on them. A value beyond the training range
    would just fall into the outermost leaf.
    """
    index = SeasonalIndex.fit(
        clean_loads["date"],
        np.log(clean_loads["posted_rate"] / clean_loads["distance"]),
        order=config.features.fourier_order,
        period=config.features.fourier_period,
    )

    trained = index.predict(clean_loads["date"])
    december = index.predict(pd.Series(DECEMBER))

    assert december.min() >= trained.min()
    assert december.max() <= trained.max()


def test_fourier_terms_are_bounded(clean_loads, config):
    """Sine and cosine columns stay within their natural range."""
    with_time = add_time_features(clean_loads, config)

    for column in [c for c in with_time.columns if c.startswith("season_")]:
        assert with_time[column].between(-1.0, 1.0).all()


def test_month_is_not_a_feature(clean_loads, config):
    """No raw month or day-of-year column reaches the model.

    Training never sees months 11 or 12, so a tree splitting on them would
    reuse the October answer for every load we predict.
    """
    builder = FeatureBuilder(config).fit(clean_loads)

    for name in builder.feature_names:
        assert name not in {"month", "day_of_year", "year", "week_of_year"}


def _cleaned_pair(clean_loads, unlabelled_loads, config):
    """Clean both frames the way the pipeline does before building features.

    Args:
        clean_loads: Labelled synthetic loads.
        unlabelled_loads: Unlabelled synthetic loads.
        config: Loaded project configuration.

    Returns:
        The cleaned training and prediction frames.
    """
    train, artifacts, _ = clean(clean_loads, config, is_training=True)
    predict, _, _ = clean(unlabelled_loads, config, is_training=False, artifacts=artifacts)
    return train, predict


def test_feature_columns_match_across_datasets(clean_loads, unlabelled_loads, config):
    """Training and prediction produce the same columns in the same order."""
    train, predict = _cleaned_pair(clean_loads, unlabelled_loads, config)
    builder = FeatureBuilder(config)

    train_features = builder.fit_transform(train)
    predict_features = builder.transform(predict)

    assert list(train_features.columns) == list(predict_features.columns)


def test_features_are_never_missing(clean_loads, unlabelled_loads, config):
    """No feature comes out as NaN, including for unfamiliar cities."""
    train, predict = _cleaned_pair(clean_loads, unlabelled_loads, config)
    builder = FeatureBuilder(config).fit(train)

    unfamiliar = predict.copy()
    unfamiliar.loc[0:9, "pickup"] = "Laredo"

    features = builder.transform(unfamiliar)

    assert not features.isna().any().any()


def test_builder_round_trip_gives_identical_features(
    clean_loads, unlabelled_loads, config, tmp_path
):
    """A reloaded builder produces exactly what the original did."""
    train, predict = _cleaned_pair(clean_loads, unlabelled_loads, config)
    builder = FeatureBuilder(config).fit(train)
    expected = builder.transform(predict)

    builder.save(tmp_path)
    restored = FeatureBuilder.load(tmp_path, config)

    pd.testing.assert_frame_equal(restored.transform(predict), expected)


def test_transform_before_fit_is_rejected(clean_loads, config):
    """Transforming without fitting fails with a clear message."""
    with pytest.raises(FeatureError, match="fit"):
        FeatureBuilder(config).transform(clean_loads)


def test_excluded_columns_stay_out(clean_loads, config):
    """market_index and quote_signal are absent while the config says so.

    Neither exists in december_chart_inputs.csv, so including them would mean
    two different models for the two output files.
    """
    builder = FeatureBuilder(config).fit(clean_loads)

    if not config.features.use_market_index:
        assert "market_index" not in builder.feature_names
    if not config.features.use_quote_signal:
        assert "quote_signal" not in builder.feature_names
