"""Behavioral tests for fitting, alignment, support, and honest evaluation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from linear import ModelValidationError, fit_model


@pytest.fixture
def frame():
    rng = np.random.default_rng(2026)
    n = 160
    data = pd.DataFrame({
        "rooms": rng.integers(1, 6, n),
        "area_total": rng.uniform(40, 180, n),
        "ratio": rng.uniform(0.5, 1.0, n),
        "property_type": rng.choice(["house", "apartment"], n),
        "neighborhood": rng.choice(["A", "B", "C"], n),
    })
    data["usd_m2"] = (
        1800 + 80 * data["rooms"] - 2 * data["area_total"]
        + 300 * data["ratio"] + 100 * data["neighborhood"].eq("B")
        - 100 * data["property_type"].eq("house")
    )
    return data


def test_named_features_ignore_input_order_and_extra_target(frame):
    model = fit_model(frame)
    first = model.predict(frame.iloc[:10])
    shuffled = frame.iloc[:10, ::-1].copy()
    shuffled["usd_m2"] = -999_999  # Target never enters the prediction matrix.
    second = model.predict(shuffled)
    np.testing.assert_allclose(first.predicted_usd_m2, second.predicted_usd_m2)
    np.testing.assert_allclose(first.predicted_usd_m2, frame.usd_m2.iloc[:10], atol=1e-9)
    assert model.feature_names.count("intercept") == 1
    assert len(model.feature_names) == 4 + (2 - 1) + (3 - 1)


def test_rank_deficient_fit_is_stable_and_reported(frame):
    frame["area_total"] = 20.0 * frame["rooms"]
    frame["ratio"] = 0.8
    model = fit_model(frame)
    assert model.rank_deficient
    assert np.isinf(model.condition_number)
    assert np.isfinite(model.coefficients).all()
    assert model.predict(frame).supported.all()


@pytest.mark.parametrize("column, value, reason", [
    ("neighborhood", "unseen", "unknown_neighborhood"),
    ("property_type", "castle", "unknown_property_type"),
    ("area_total", 0, "invalid_features"),
    ("area_total", np.inf, "invalid_features"),
    ("area_total", 10000, "outside_training_range"),
    ("ratio", 1.1, "invalid_features"),
    ("ratio", 0, "invalid_features"),
    ("rooms", 2.5, "invalid_features"),
])
def test_unsupported_query_is_explicit(frame, column, value, reason):
    model = fit_model(frame)
    query = frame.iloc[[0]].copy()
    query[column] = value
    prediction = model.predict(query).iloc[0]
    assert not prediction.supported
    assert prediction.reason == reason
    assert np.isnan(prediction.predicted_usd_m2)
    assert np.isnan(prediction.predicted_price_usd)


def test_nonpositive_output_is_not_displayed_as_price(frame):
    model = fit_model(frame)
    coefficients = np.zeros_like(model.coefficients)
    coefficients[0] = -1.0
    deliberately_bad_fit = replace(model, coefficients=coefficients)
    prediction = deliberately_bad_fit.predict(frame.iloc[[0]]).iloc[0]
    assert not prediction.supported
    assert prediction.reason == "nonpositive_or_nonfinite_prediction"


def test_covered_area_must_match_ratio_and_not_exceed_total(frame):
    model = fit_model(frame)
    query = frame.iloc[[0]].copy()
    query["area_covered"] = query["area_total"] + 1
    assert model.predict(query).iloc[0].reason == "invalid_features"
    query["area_covered"] = query["area_total"] * 0.1
    assert model.predict(query).iloc[0].reason == "invalid_features"


def test_unseen_category_combination_is_not_supported(frame):
    train = frame[~(frame.property_type.eq("house") & frame.neighborhood.eq("C"))]
    model = fit_model(train)
    query = frame.iloc[[0]].copy()
    query["property_type"] = "house"
    query["neighborhood"] = "C"
    assert model.predict(query).iloc[0].reason == "unseen_category_combination"


def test_evaluation_uses_same_supported_subset_for_baseline(frame):
    model = fit_model(frame)
    test = frame.iloc[:5].copy()
    test.loc[test.index[0], "neighborhood"] = "unseen"
    test.loc[test.index[0], "usd_m2"] = 1e9
    report = model.evaluate(test)
    assert report["coverage"] == 0.8
    assert report["supported_rows"] == 4
    assert report["mae_usd_m2"] < 1e-8
    expected = np.abs(test.usd_m2.iloc[1:] - np.median(frame.usd_m2))
    assert report["baseline_mae_usd_m2"] == pytest.approx(expected.mean())
    assert report["baseline_median_ae_usd_m2"] == pytest.approx(np.median(expected))
    assert report["excluded_reasons"] == {"unknown_neighborhood": 1}


def test_heldout_targets_do_not_change_fit_or_support(frame):
    train = frame.iloc[:120].copy()
    test = frame.iloc[120:].copy()
    model = fit_model(train)
    before = model.predict(test)
    coefficients = model.coefficients.copy()
    test["usd_m2"] *= 100
    model.evaluate(test)
    after = model.predict(test)
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(coefficients, model.coefficients)
    assert model.baseline_usd_m2 == np.median(train.usd_m2)


def test_no_supported_rows_has_no_fabricated_metrics(frame):
    model = fit_model(frame)
    test = frame.iloc[:3].copy()
    test["neighborhood"] = "unseen"
    report = model.evaluate(test)
    assert report["coverage"] == 0.0
    assert report["mae_usd_m2"] is None
    assert report["baseline_mae_usd_m2"] is None


def test_missing_columns_and_invalid_training_are_rejected(frame):
    with pytest.raises(ModelValidationError, match="Missing required"):
        fit_model(frame.drop(columns="rooms"))
    frame.loc[0, "usd_m2"] = 0
    with pytest.raises(ModelValidationError, match="finite positive"):
        fit_model(frame)


def test_evaluation_rejects_training_spatial_groups(frame):
    frame["spatial_group"] = [f"group-{i // 10}" for i in range(len(frame))]
    train, test = frame.iloc[:120], frame.iloc[120:]
    model = fit_model(train)
    assert model.evaluate(test)["total_rows"] == 40
    with pytest.raises(ModelValidationError, match="overlap"):
        model.evaluate(train)
    with pytest.raises(ModelValidationError, match="required"):
        model.evaluate(test.drop(columns="spatial_group"))


def test_extreme_finite_targets_are_rejected_before_reduction(frame):
    frame["usd_m2"] = 1.7e308
    with pytest.raises(ModelValidationError, match="numerical safety limit"):
        fit_model(frame)


def test_overflowing_implied_total_is_rejected(frame):
    model = fit_model(frame)
    coefficients = np.zeros_like(model.coefficients)
    coefficients[0] = 1e308
    enormous = replace(model, coefficients=coefficients)
    prediction = enormous.predict(frame.iloc[[0]]).iloc[0]
    assert not prediction.supported
    assert np.isnan(prediction.predicted_price_usd)


def test_extreme_heldout_targets_are_not_in_metrics(frame):
    model = fit_model(frame)
    test = frame.iloc[:3].copy()
    test["usd_m2"] = 1.7e308
    metrics = model.evaluate(test)
    assert metrics["supported_rows"] == 0
    assert metrics["baseline_mae_usd_m2"] is None
    assert metrics["excluded_reasons"] == {"invalid_target": 3}
