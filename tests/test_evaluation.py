"""Joint holdout evaluation must compare both methods on identical cases."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from comparables import Filters
from evaluation import evaluate_joint
from linear import fit_model


PREDICTION_COLUMNS = [
    "comparable_usd_m2", "linear_usd_m2", "comparable_supported",
    "linear_supported", "common_supported", "linear_reason",
]


@pytest.fixture
def training():
    """Real OLS fit with a known solution and geographically ordered neighbours."""
    area = np.arange(80.0, 104.0)
    return pd.DataFrame({
        "property_type": "apartment", "neighborhood": "A", "rooms": 3,
        "area_total": area, "area_covered": area * .8, "ratio": .8,
        "lat": -34.600 + np.arange(len(area)) * .0001, "lon": -58.441,
        "usd_m2": 1000 + 10 * area, "price_usd": (1000 + 10 * area) * area,
        "spatial_group": "training-block",
    })


@pytest.fixture
def heldout():
    # The first three cases are common. Then: comparables only, linear only,
    # neither. Their large targets must not influence the common-case metrics.
    area = np.array([90., 100., 95., 90., 90., 90.])
    target = np.array([2000., 1800., 1900., 50000., 60000., 70000.])
    return pd.DataFrame({
        "property_type": "apartment", "neighborhood": ["A", "A", "A", "B", "A", "B"],
        "rooms": 3, "area_total": area, "area_covered": area * .8, "ratio": .8,
        "lat": [-34.601, -34.590, -34.601, -34.601, -34.520, -34.520],
        "lon": -58.441, "usd_m2": target, "price_usd": target * area,
        "spatial_group": ["south", "north", "south", "south", "far", "far"],
    }, index=pd.Index([41, 9, 83, 2, 57, 16], name="property_id"))


def metric_rows(report):
    rows = report["metrics"]
    assert len(rows) == 3
    assert {row["method"] for row in rows} == {"comparables", "linear", "training_median"}
    return {row["method"]: row for row in rows}


def test_common_subset_metrics_have_manual_expected_values(training, heldout):
    report = evaluate_joint(training, heldout)
    assert report["total_rows"] == 6
    assert report["comparable_supported_rows"] == 4
    assert report["linear_supported_rows"] == 4
    assert report["common_rows"] == 3
    assert report["comparable_coverage"] == pytest.approx(4 / 6)
    assert report["linear_coverage"] == pytest.approx(4 / 6)
    assert report["common_coverage"] == .5
    assert report["baseline_train_median_usd_m2"] == 1915

    predictions = report["predictions"]
    pd.testing.assert_index_equal(predictions.index, heldout.index)
    assert set(PREDICTION_COLUMNS) <= set(predictions.columns)
    np.testing.assert_allclose(predictions.comparable_usd_m2.iloc[:4], [1820, 2010, 1820, 1820])
    np.testing.assert_allclose(predictions.linear_usd_m2.iloc[[0, 1, 2, 4]], [1900, 2000, 1950, 1900], atol=1e-8)
    assert predictions.comparable_supported.tolist() == [True, True, True, True, False, False]
    assert predictions.linear_supported.tolist() == [True, True, True, False, True, False]
    assert predictions.common_supported.tolist() == [True, True, True, False, False, False]
    assert predictions.comparable_usd_m2.iloc[4:].isna().all()
    assert predictions.linear_usd_m2.iloc[[3, 5]].isna().all()
    assert predictions.linear_reason.iloc[[3, 5]].tolist() == ["unknown_neighborhood"] * 2

    metrics = metric_rows(report)
    # Common-case absolute errors: comparables 180, 210, 80; OLS 100, 200,
    # 50; training median 85, 115, 15. Neither method gets an easier subset.
    for method, expected_mae, expected_median in [
        ("comparables", 470 / 3, 180),
        ("linear", 350 / 3, 100),
        ("training_median", 215 / 3, 85),
    ]:
        assert metrics[method]["mae"] == pytest.approx(expected_mae)
        assert metrics[method]["median_ae"] == pytest.approx(expected_median)


def test_explicit_filters_reach_comparables_without_changing_linear(training, heldout):
    default = evaluate_joint(training, heldout)
    three = evaluate_joint(training, heldout, filters=Filters(k=3))
    np.testing.assert_allclose(three["predictions"].comparable_usd_m2.iloc[:4], [1810, 2020, 1810, 1810])
    pd.testing.assert_series_equal(default["predictions"].linear_usd_m2, three["predictions"].linear_usd_m2)
    assert metric_rows(three)["comparables"]["mae"] == pytest.approx(500 / 3)


def test_all_holdout_rows_are_evaluated_even_above_old_300_case_cap(training, heldout):
    count = 357
    test = pd.concat([heldout.iloc[[0]]] * count, ignore_index=True)
    test.index = pd.Index(range(2000, 2000 - count, -1), name="unsorted_id")
    test["usd_m2"] = 1900.0
    test.iloc[-1, test.columns.get_loc("usd_m2")] = 1_000_000.0
    test["price_usd"] = test.usd_m2 * test.area_total
    report = evaluate_joint(training, test)
    assert report["total_rows"] == report["common_rows"] == count
    assert report["comparable_supported_rows"] == report["linear_supported_rows"] == count
    assert report["comparable_coverage"] == report["linear_coverage"] == report["common_coverage"] == 1
    pd.testing.assert_index_equal(report["predictions"].index, test.index)
    assert len(report["predictions"]) == count
    metrics = metric_rows(report)
    assert metrics["comparables"]["mae"] == pytest.approx((356 * 80 + 1_000_000 - 1820) / count)
    assert metrics["linear"]["mae"] == pytest.approx((1_000_000 - 1900) / count)
    assert metrics["training_median"]["mae"] == pytest.approx((356 * 15 + 1_000_000 - 1915) / count)


def test_changing_heldout_targets_cannot_change_predictions_or_support(training, heldout):
    fitted = fit_model(training)
    original_coefficients = fitted.coefficients.copy()
    before = evaluate_joint(training, heldout, fitted=fitted)
    changed = heldout.copy()
    changed["usd_m2"] *= 2
    changed["price_usd"] = changed.usd_m2 * changed.area_total
    after = evaluate_joint(training, changed, fitted=fitted)
    pd.testing.assert_frame_equal(before["predictions"][PREDICTION_COLUMNS], after["predictions"][PREDICTION_COLUMNS])
    np.testing.assert_array_equal(fitted.coefficients, original_coefficients)
    assert before["baseline_train_median_usd_m2"] == after["baseline_train_median_usd_m2"] == 1915
    assert before["common_coverage"] == after["common_coverage"] == .5
    assert metric_rows(before)["linear"]["mae"] != metric_rows(after)["linear"]["mae"]


def test_valid_prefitted_model_matches_fresh_fit_without_mutating_inputs(training, heldout):
    train_copy, test_copy = training.copy(deep=True), heldout.copy(deep=True)
    fresh = evaluate_joint(training, heldout)
    cached = evaluate_joint(training, heldout, fitted=fit_model(training))
    pd.testing.assert_frame_equal(fresh["predictions"], cached["predictions"])
    assert fresh["metrics"] == cached["metrics"]
    pd.testing.assert_frame_equal(training, train_copy)
    pd.testing.assert_frame_equal(heldout, test_copy)


@pytest.mark.parametrize("which", ["train", "test", "both"])
def test_spatial_groups_are_required_on_both_sides(training, heldout, which):
    train = training.drop(columns="spatial_group") if which in ("train", "both") else training
    test = heldout.drop(columns="spatial_group") if which in ("test", "both") else heldout
    with pytest.raises(ValueError):
        evaluate_joint(train, test)


@pytest.mark.parametrize("which", ["train", "test"])
@pytest.mark.parametrize("invalid_group", [None, "", 123])
def test_spatial_groups_must_be_nonempty_strings(training, heldout, which, invalid_group):
    frame = training if which == "train" else heldout
    frame["spatial_group"] = frame["spatial_group"].astype(object)
    frame.loc[frame.index[0], "spatial_group"] = invalid_group
    with pytest.raises(ValueError):
        evaluate_joint(training, heldout)


def test_overlapping_spatial_groups_rejected_even_with_explicit_filters(training, heldout):
    heldout.loc[heldout.index[0], "spatial_group"] = "training-block"
    with pytest.raises(ValueError):
        evaluate_joint(training, heldout, filters=Filters(k=3))


def test_duplicate_row_ids_cannot_hide_behind_different_spatial_labels(training, heldout):
    training["row_id"] = np.arange(len(training))
    heldout["row_id"] = 1000 + np.arange(len(heldout))
    assert evaluate_joint(training, heldout)["total_rows"] == len(heldout)
    heldout.loc[heldout.index[-1], "row_id"] = 0
    with pytest.raises(ValueError):
        evaluate_joint(training, heldout)


@pytest.mark.parametrize("mismatch", ["groups", "rows"])
def test_prefitted_model_must_match_training_groups_and_row_count(training, heldout, mismatch):
    fitted = fit_model(training)
    if mismatch == "groups":
        fitted = replace(fitted, training_groups=frozenset({"some-other-training-block"}))
    else:
        fitted = replace(fitted, training_rows=len(training) + 1)
    with pytest.raises(ValueError):
        evaluate_joint(training, heldout, fitted=fitted)


@pytest.mark.parametrize("invalid_target", [0.0, -1.0, np.nan, np.inf, 1e101])
@pytest.mark.parametrize("which", ["train", "test"])
def test_invalid_targets_rejected_without_changing_fitted_predictions(training, heldout, invalid_target, which):
    fitted = fit_model(training)
    expected = fitted.predict(heldout)
    invalid = (training if which == "train" else heldout).copy()
    invalid.loc[invalid.index[-1], "usd_m2"] = invalid_target
    train, test = (invalid, heldout) if which == "train" else (training, invalid)
    with pytest.raises(ValueError):
        evaluate_joint(train, test, fitted=fitted)
    pd.testing.assert_frame_equal(fitted.predict(heldout), expected)


@pytest.mark.parametrize("which", ["train", "test"])
def test_missing_targets_are_rejected(training, heldout, which):
    train = training.drop(columns="usd_m2") if which == "train" else training
    test = heldout.drop(columns="usd_m2") if which == "test" else heldout
    with pytest.raises(ValueError):
        evaluate_joint(train, test)


def test_no_common_cases_retains_individual_coverage_but_has_no_errors(training, heldout):
    report = evaluate_joint(training, heldout.iloc[[3, 4]])
    assert report["total_rows"] == 2
    assert report["comparable_supported_rows"] == report["linear_supported_rows"] == 1
    assert report["comparable_coverage"] == report["linear_coverage"] == .5
    assert report["common_rows"] == report["common_coverage"] == 0
    assert not report["predictions"].common_supported.any()
    for metric in metric_rows(report).values():
        assert metric["mae"] is None
        assert metric["median_ae"] is None


def test_empty_holdout_returns_empty_predictions_and_no_fabricated_metrics(training, heldout):
    empty = heldout.iloc[:0]
    report = evaluate_joint(training, empty)
    for key in ("total_rows", "comparable_supported_rows", "linear_supported_rows", "common_rows",
                "comparable_coverage", "linear_coverage", "common_coverage"):
        assert report[key] == 0
    assert report["baseline_train_median_usd_m2"] == 1915
    assert report["predictions"].empty
    assert set(PREDICTION_COLUMNS) <= set(report["predictions"].columns)
    pd.testing.assert_index_equal(report["predictions"].index, empty.index)
    for metric in metric_rows(report).values():
        assert metric["mae"] is None
        assert metric["median_ae"] is None
