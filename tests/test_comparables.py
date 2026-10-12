import numpy as np
import pandas as pd
import pytest

from comparables import ComparableModel, Filters, Query, evaluate, haversine_km


def rows(prices=(1000, 2000, 3000, 4000, 5000), group="train", area=100.0):
    n = len(prices)
    return pd.DataFrame({
        "property_type": ["apartment"] * n, "area_total": [area] * n,
        "area_covered": [area * .8] * n, "rooms": [3] * n, "ratio": [0.8] * n,
        "lat": [-34.60 + i * 0.001 for i in range(n)], "lon": [-58.44] * n,
        "usd_m2": list(prices), "price_usd": np.asarray(prices) * area,
        "spatial_group": [group] * n,
    })


def query(**overrides):
    return Query(**dict(property_type="apartment", area_total=100.0,
                        area_covered=80.0, rooms=3, lat=-34.60, lon=-58.44) | overrides)


def test_haversine_uses_kilometres_and_latitude():
    assert haversine_km(0, 0, [0, 1], [0, 0]) == pytest.approx([0, 111.19508], rel=1e-5)
    equator = float(haversine_km(0, 0, 0, 1))
    sixty = float(haversine_km(60, 0, 60, 1))
    assert sixty == pytest.approx(equator / 2, rel=1e-4)
    assert np.isfinite(haversine_km(90, 180, -90, 0))


def test_nearest_selected_and_mean_and_range_are_observed():
    result = ComparableModel(rows(), Filters(k=3)).predict(query())
    assert result.n_within_filters == 5
    assert result.n_comparables == 3
    assert result.usd_m2 == 2000
    assert result.asking_price_usd == 200000
    assert (result.observed_min_usd_m2, result.observed_max_usd_m2) == (1000, 3000)
    assert result.distance_max_km == pytest.approx(0.22239, rel=1e-4)
    assert result.comparable_locations == ((-34.600, -58.440), (-34.599, -58.440), (-34.598, -58.440))


def test_map_locations_are_only_selected_training_points_and_are_coarsened():
    frame = rows(tuple(range(1000, 8000, 1000)))
    frame["lat"] += .00012345
    frame["lon"] += .00023456
    frame.loc[0, "property_type"] = "house"
    frame.loc[1, "rooms"] = 10
    result = ComparableModel(frame, Filters(k=3)).predict(query())
    assert result.n_comparables == 3
    assert result.comparable_locations == ((-34.598, -58.440), (-34.597, -58.440), (-34.596, -58.440))
    assert result.usd_m2 == pytest.approx(4000)  # Same selected rows supply the displayed points and price.
    precise_locations = set(zip(frame["lat"], frame["lon"]))
    assert precise_locations.isdisjoint(result.comparable_locations)


def test_selected_ids_preserve_distinct_records_at_coincident_locations():
    frame = rows((1000, 2000, 3000, 9000))
    frame["lat"] = -34.60
    frame["row_id"] = [71, 92, 48, 15]
    result = ComparableModel(frame, Filters(k=3)).predict(query())
    assert result.comparable_row_ids == (71, 92, 48)
    assert result.comparable_locations == ((-34.60, -58.44),) * 3
    assert result.comparable_distances_km == (0.0, 0.0, 0.0)
    assert result.usd_m2 == 2000
    # Stable distance ties follow training order, never ID or price order.
    assert result.n_within_filters == 4


def test_selected_ids_follow_filters_and_use_the_index_when_no_id_is_supplied():
    frame = rows()
    frame.index = [101, 102, 104, 108, 116]
    frame.loc[101, "property_type"] = "house"
    result = ComparableModel(frame, Filters(k=3)).predict(query())
    assert result.comparable_row_ids == (102, 104, 108)
    assert result.comparable_distances_km == pytest.approx((0.111195, 0.222390, 0.333585), rel=1e-5)
    selected = frame.loc[list(result.comparable_row_ids)]
    assert result.usd_m2 == pytest.approx(selected.usd_m2.mean())
    assert result.comparable_locations == tuple(zip(selected.lat.round(3), selected.lon.round(3)))


@pytest.mark.parametrize("identifiers", [[1, 1, 3, 4, 5], ["1", "2", "3", "4", "5"],
                                         [1.0, 2.0, 3.0, 4.0, 5.0], [True, 2, 3, 4, 5]])
def test_display_ids_must_be_unique_integers(identifiers):
    frame = rows()
    frame["row_id"] = identifiers
    with pytest.raises(ValueError, match="identificadores"):
        ComparableModel(frame)


def test_selected_identity_is_copied_at_fit_time_and_available_without_a_price():
    frame = rows((1000, 2000))
    frame["row_id"] = [53, 54]
    model = ComparableModel(frame)
    frame["row_id"] = [0, 1]
    result = model.predict(query())
    assert not result.sufficient
    assert result.comparable_row_ids == (53, 54)
    assert len(result.comparable_distances_km) == 2


@pytest.mark.parametrize("count", [0, 1, 2])
def test_map_includes_available_locations_without_predicting(count):
    frame = rows()
    frame.loc[count:, "property_type"] = "house"
    result = ComparableModel(frame).predict(query())
    assert not result.sufficient
    assert len(result.comparable_locations) == count
    assert result.comparable_locations == ((-34.600, -58.440), (-34.599, -58.440))[:count]


def test_map_has_no_points_for_an_unseen_property_type():
    result = ComparableModel(rows()).predict(query(property_type="PH"))
    assert result.comparable_locations == ()


@pytest.mark.parametrize("column,value", [
    ("property_type", "house"), ("rooms", 5), ("area_total", 151.0),
    ("ratio", 0.2), ("lat", -35.0),
])
def test_every_filter_actually_changes_the_selected_set(column, value):
    frame = rows((1000, 1000, 1000, 10000))
    frame.loc[3, column] = value
    result = ComparableModel(frame, Filters(rooms_tolerance=0, area_tolerance=.5,
                                            ratio_tolerance=.1)).predict(query())
    assert result.n_comparables == 3
    assert result.usd_m2 == 1000


def test_not_enough_evidence_never_relaxes_or_predicts():
    result = ComparableModel(rows((1000, 2000))).predict(query())
    assert result.n_comparables == 2
    assert not result.sufficient
    assert result.usd_m2 is None
    assert result.observed_min_usd_m2 is None


def test_fit_copies_training_arrays_and_query_has_no_target():
    frame = rows()
    model = ComparableModel(frame)
    frame.loc[:, "usd_m2"] = 999999
    assert model.predict(query()).usd_m2 == 3000
    with pytest.raises(TypeError):
        query(usd_m2=999999)


@pytest.mark.parametrize("overrides", [
    {"area_total": 0}, {"area_covered": 0}, {"area_covered": 101},
    {"rooms": 1.5}, {"rooms": 0}, {"lat": float("nan")},
    {"lon": 181}, {"property_type": ""},
])
def test_invalid_queries_rejected(overrides):
    with pytest.raises(ValueError):
        query(**overrides)


@pytest.mark.parametrize("overrides", [
    {"k": 2}, {"k": 5.5}, {"radius_km": 0}, {"area_tolerance": -1},
    {"ratio_tolerance": 1.1}, {"rooms_tolerance": 0.5}, {"radius_km": float("nan")},
])
def test_invalid_filters_rejected(overrides):
    with pytest.raises(ValueError):
        Filters(**overrides)


def test_evaluation_baseline_uses_training_and_same_covered_subset():
    train = rows((1000, 1000, 4000, 10000, 10000))
    test = rows((3000, 5000), "test")
    test.loc[1, "property_type"] = "house"  # Only the first row is covered.
    result = evaluate(train, test)
    assert result["coverage"] == .5
    assert result["covered_rows"] == 1
    assert result["mae"] == 2200  # Mean of train = 5200, test target = 3000.
    assert result["baseline_train_median_usd_m2"] == 4000
    assert result["baseline_mae"] == 1000
    assert result["median_absolute_error"] == 2200
    test.loc[1, "usd_m2"] = 9_999_999
    assert evaluate(train, test) == result  # Uncovered targets cannot affect errors.


def test_evaluation_targets_affect_errors_but_not_predictions_or_coverage():
    train = rows((1000, 1000, 1000))
    test = rows((2000, 2000), "test")
    before = evaluate(train, test)
    test["usd_m2"] = 3000
    after = evaluate(train, test)
    assert before["mae"] == 1000
    assert after["mae"] == 2000
    assert before["coverage"] == after["coverage"] == 1
    assert before["baseline_train_median_usd_m2"] == after["baseline_train_median_usd_m2"] == 1000


def test_evaluation_is_reproducibly_capped_and_handles_no_coverage():
    train = rows()
    test = rows(tuple(range(1000, 1400)), "test")
    first = evaluate(train, test, max_samples=15)
    assert first == evaluate(train, test, max_samples=15)
    assert first["sample_rows"] == 15
    assert first["holdout_rows"] == 400
    test["property_type"] = "house"
    result = evaluate(train, test)
    assert result["coverage"] == 0
    assert result["mae"] is None
    assert result["baseline_mae"] is None


def test_overlapping_spatial_groups_rejected():
    with pytest.raises(ValueError, match="grupos espaciales"):
        evaluate(rows(), rows())


def test_large_valid_holdout_area_does_not_crash_evaluation():
    train = rows()
    test = rows((2000,), "test")
    test["area_total"] = 20000.0
    test["area_covered"] = 16000.0
    result = evaluate(train, test)
    assert result["coverage"] == 0
    assert result["mae"] is None


def test_extreme_finite_targets_keep_mean_and_metrics_finite():
    train = rows((1e308,) * 5, area=1.0)
    result = ComparableModel(train).predict(query(area_total=1.0, area_covered=.8))
    assert result.sufficient
    assert np.isfinite(result.usd_m2)
    assert result.usd_m2 == pytest.approx(1e308)
    test = train.iloc[:2].copy()
    test["spatial_group"] = "test"
    test["usd_m2"] = 1.0
    metrics = evaluate(train, test)
    for name in ("mae", "median_absolute_error", "baseline_mae", "baseline_median_absolute_error"):
        assert np.isfinite(metrics[name])
        assert metrics[name] == pytest.approx(1e308)


def test_extreme_even_training_median_and_total_overflow_controlled():
    train = rows((1.7e308,) * 4, area=1.0)
    result = ComparableModel(train).predict(query(area_total=1.1, area_covered=.88))
    assert not result.sufficient
    assert result.insufficient_reason
    assert result.asking_price_usd is None
    test = train.iloc[:1].copy()
    test["spatial_group"] = "test"
    metrics = evaluate(train, test)
    assert metrics["baseline_train_median_usd_m2"] == pytest.approx(1.7e308)
