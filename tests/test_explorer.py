"""Contracts at the boundary between the shared query, models and public rows."""

from dataclasses import replace
from math import fsum

import numpy as np
import pandas as pd
import pytest

from comparables import ComparableModel, Filters, Query
from data import LAT_BOUNDS, LON_BOUNDS, load_demo, split_data
from explorer import (PropertyQuery, compare_property, display_rows,
                      neighborhood_location, prepare_demo)
from linear import fit_model


@pytest.fixture(scope="module")
def bundle():
    return prepare_demo()


@pytest.fixture
def query(bundle):
    _, train, _, _, _, _ = bundle
    lat, lon = neighborhood_location(train, "Palermo")
    return PropertyQuery("apartment", "Palermo", 3, 70.0, 60.0, lat, lon)


def test_display_ids_preserve_the_original_split_and_exact_public_rows(bundle):
    frame, train, test, fitted, _, _ = bundle
    original_train, original_test, _ = split_data(load_demo()[0], test_fraction=.2, seed=42)
    assert (len(frame), len(train), len(test)) == (7495, 6540, 955)
    assert frame.row_id.tolist() == list(range(7495))
    assert set(train.row_id).isdisjoint(test.row_id)
    assert set(train.spatial_group).isdisjoint(test.spatial_group)
    assert fitted.training_rows == len(train)
    assert fitted.training_groups == frozenset(train.spatial_group)
    # Reordering training rows would change distance ties. IDs may not do that.
    pd.testing.assert_frame_equal(train.drop(columns="row_id").reset_index(drop=True), original_train.reset_index(drop=True))
    pd.testing.assert_frame_equal(test.drop(columns="row_id").reset_index(drop=True), original_test.reset_index(drop=True))
    pd.testing.assert_frame_equal(frame.set_index("row_id", drop=False).loc[train.row_id], train.set_index("row_id", drop=False))


def test_shared_query_preserves_each_original_method_and_selected_training_ids(bundle, query):
    frame, train, test, fitted, _, _ = bundle
    result = compare_property(train, fitted, query, Filters())
    expected = ComparableModel(train).predict(Query("apartment", 70, 60, 3, query.lat, query.lon))
    assert result["comparables"] == expected
    linear_input = pd.DataFrame([{"property_type": "apartment", "neighborhood": "Palermo",
                                  "rooms": 3, "area_total": 70., "area_covered": 60.,
                                  "ratio": 60 / 70}])
    assert result["linear"] == fitted.predict(linear_input).iloc[0].to_dict()
    selected = frame.set_index("row_id").loc[list(expected.comparable_row_ids)]
    assert set(selected.index).issubset(train.row_id)
    assert set(selected.index).isdisjoint(test.row_id)
    assert list(zip(selected.lat, selected.lon)) == list(expected.comparable_locations)
    assert expected.usd_m2 == pytest.approx(fsum(selected.usd_m2) / len(selected), rel=1e-12)
    assert expected.asking_price_usd == pytest.approx(expected.usd_m2 * query.area_total)


def test_regression_unsupported_pair_does_not_hide_valid_comparables(bundle):
    _, train, _, fitted, _, _ = bundle
    lat, lon = neighborhood_location(train, "Agronomía")
    result = compare_property(train, fitted, PropertyQuery("PH", "Agronomía", 3, 100, 80, lat, lon), Filters())
    assert result["comparables"].sufficient
    assert result["linear"]["supported"] is False
    assert result["linear"]["reason"] == "unseen_category_combination"
    assert result["difference"] is None


def test_strict_geographic_filters_do_not_change_regression_or_relax_themselves(bundle, query):
    _, train, _, fitted, _, _ = bundle
    default = compare_property(train, fitted, query, Filters())
    strict_filters = Filters(rooms_tolerance=0, area_tolerance=0, ratio_tolerance=0, radius_km=.1)
    strict = compare_property(train, fitted, query, strict_filters)
    assert strict["comparables"].n_comparables < 3
    assert not strict["comparables"].sufficient
    assert strict["linear"] == default["linear"]
    assert strict["filters"] == strict_filters
    assert strict["difference"] is None


@pytest.mark.parametrize("field,value", [
    ("area_total", 0), ("area_total", -1), ("area_total", np.nan),
    ("area_covered", 0), ("area_covered", 71), ("area_covered", np.inf),
    ("rooms", 0), ("rooms", 2.5), ("rooms", 21),
    ("lat", np.nan), ("lat", LAT_BOUNDS[0] - .01),
    ("lon", np.inf), ("lon", LON_BOUNDS[1] + .01),
    ("property_type", "garage"), ("neighborhood", " "),
])
def test_invalid_shared_inputs_fail_before_either_model_returns_a_price(bundle, query, field, value):
    _, train, _, fitted, _, _ = bundle
    invalid = replace(query, **{field: value})
    with pytest.raises(ValueError):
        compare_property(train, fitted, invalid, Filters())


def test_physically_valid_extrapolation_remains_unsupported_without_clipping(bundle, query):
    _, train, _, fitted, _, _ = bundle
    extreme = replace(query, area_total=10000, area_covered=9000, rooms=20)
    result = compare_property(train, fitted, extreme, Filters())
    assert result["query"] == extreme
    assert not result["comparables"].sufficient
    assert not result["linear"]["supported"]
    assert result["linear"]["reason"] == "outside_training_range"
    assert result["difference"] is None


@pytest.mark.parametrize("latitude,expected_delta", [(-34.600, -80.), (-34.598, 100.)])
def test_difference_is_signed_comparables_minus_regression_without_averaging(latitude, expected_delta):
    area = np.arange(80., 104.)
    training = pd.DataFrame({
        "property_type": "apartment", "neighborhood": "A", "rooms": 3,
        "area_total": area, "area_covered": area * .8, "ratio": .8,
        "lat": -34.600 + np.arange(len(area)) * .0001, "lon": -58.441,
        "usd_m2": 1000 + 10 * area, "spatial_group": "training-block",
        "row_id": np.arange(len(area)) * 7 + 10,
    })
    query = PropertyQuery("apartment", "A", 3, 90., 72., latitude, -58.441)
    result = compare_property(training, fit_model(training), query, Filters())
    assert result["linear"]["predicted_usd_m2"] == pytest.approx(1900)
    assert result["difference"]["usd_m2"] == pytest.approx(expected_delta)
    assert result["difference"]["total_usd"] == pytest.approx(expected_delta * 90)


def test_location_reference_uses_only_supplied_training_examples_and_rounds():
    train = pd.DataFrame({"neighborhood": ["A", "A", "B"],
                          "lat": [-34.6012, -34.6022, -34.68],
                          "lon": [-58.4421, -58.4431, -58.53]})
    assert neighborhood_location(train, "A") == (-34.602, -58.443)
    with pytest.raises(ValueError):
        neighborhood_location(train, "unknown")


def test_public_record_table_excludes_extra_private_fields(bundle):
    frame = bundle[0].iloc[:3].copy()
    frame["description"] = "Do not disclose"
    frame["contact"] = "private@example.test"
    frame["source_id"] = "private-provider-id"
    rendered = display_rows(frame)
    assert len(rendered.columns) == 8
    assert rendered.iloc[:, 0].tolist() == frame.row_id.tolist()
    assert not {"description", "contact", "source_id", "lat", "lon", "spatial_group"} & set(rendered.columns)
    assert not any("private" in str(value).lower() for value in rendered.to_numpy().flat)
