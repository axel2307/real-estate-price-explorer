"""Exercise integrated Streamlit state and independently supported estimates."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from comparables import ComparableModel, Query
from evaluation import evaluate_joint
from explorer import money, prepare_demo


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def bundle():
    return prepare_demo()


@pytest.fixture
def app():
    value = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not value.exception
    return value


def layers(app):
    chart = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
    return {layer["id"]: layer for layer in chart["layers"]}


def estimates(app):
    return {metric.label: metric.value for metric in app.metric
            if metric.label.startswith(("Comparables ·", "Regresión ·"))}


def values(app, kind):
    return "\n".join(element.value for element in app.get(kind))


def submit(app):
    app.button(key="compare").click().run()
    assert not app.exception


def test_initial_map_exposes_every_public_record_and_common_holdout_without_uploads(app, bundle):
    assert values(app, "caption").splitlines().count("Demo educativa con datos ficticios") == 1
    assert not app.get("file_uploader")
    assert not estimates(app)
    assert not app.warning
    assert not app.error
    chart_layers = layers(app)
    groups = chart_layers["data-locations"]["data"]
    assert sum(group["count"] for group in groups) == 7495
    assert len(groups) == 3544
    assert sorted(row_id for group in groups for row_id in group["row_ids"]) == list(range(7495))
    assert not chart_layers["submitted-query"]["data"]
    assert not chart_layers["selected-comparables"]["data"]
    assert app.button(key="view_query").disabled
    assert "teclado" in app.get("deck_gl_json_chart")[0].proto.alt
    _, train, test, fitted, _, _ = bundle
    evaluation = evaluate_joint(train, test, fitted=fitted)
    assert evaluation["total_rows"] == 955
    assert evaluation["common_rows"] == 936
    assert "936 casos admitidos por ambos" in values(app, "markdown")
    assert "955 reservados" in values(app, "markdown")
    metrics_table = next(table.value for table in app.dataframe if len(table.value) == 3 and "Método" in table.value)
    assert metrics_table.iloc[:, 1].tolist() == pytest.approx([row["mae"] for row in evaluation["metrics"]])


def test_one_submission_passes_identical_features_to_both_original_methods(app, bundle):
    submit(app)
    result = app.session_state["result"]
    query = result["query"]
    _, train, _, fitted, _, _ = bundle
    comparable = ComparableModel(train).predict(Query(query.property_type, query.area_total,
        query.area_covered, query.rooms, query.lat, query.lon))
    linear_input = pd.DataFrame([{"property_type": query.property_type,
        "neighborhood": query.neighborhood, "rooms": query.rooms,
        "area_total": query.area_total, "area_covered": query.area_covered,
        "ratio": query.area_covered / query.area_total}])
    linear = fitted.predict(linear_input).iloc[0]
    assert estimates(app) == {
        "Comparables · USD/m²": money(comparable.usd_m2),
        "Comparables · Total": money(comparable.asking_price_usd),
        "Regresión · USD/m²": money(linear.predicted_usd_m2),
        "Regresión · Total": money(linear.predicted_price_usd),
    }
    assert "por debajo" in values(app, "markdown")
    assert "por encima" not in values(app, "markdown")
    assert len(estimates(app)) == 4  # No silently averaged third price.
    assert not app.button(key="view_query").disabled
    chart_layers = layers(app)
    selected_ids = [i for group in chart_layers["selected-comparables"]["data"] for i in group["row_ids"]]
    assert sorted(selected_ids) == sorted(comparable.comparable_row_ids)
    assert chart_layers["submitted-query"]["data"][0]["lat"] == query.lat


def test_query_edits_keep_submitted_prices_and_marker_until_explicit_resubmission(app):
    submit(app)
    original_prices = estimates(app)
    original_point = layers(app)["submitted-query"]["data"]
    original_inputs = app.session_state["result_inputs"]
    app.number_input(key="query_total").set_value(80.).run()
    assert not app.exception
    assert "cambios pendientes" in values(app, "info")
    assert estimates(app) == original_prices
    assert layers(app)["submitted-query"]["data"] == original_point
    assert app.session_state["result_inputs"] == original_inputs
    assert app.session_state["result"]["query"].area_total == 70
    submit(app)
    assert "cambios pendientes" not in values(app, "info")
    assert app.session_state["result"]["query"].area_total == 80
    assert estimates(app) != original_prices


def test_map_filters_color_and_layer_controls_do_not_change_estimates_or_neighbors(app, bundle):
    submit(app)
    original_prices = estimates(app)
    original_ids = app.session_state["result"]["comparables"].comparable_row_ids
    original_point = layers(app)["submitted-query"]["data"]
    app.multiselect(key="map_types").set_value(["house"])
    app.multiselect(key="map_neighborhoods").set_value(["Palermo"])
    app.selectbox(key="map_metric").set_value("price_usd")
    app.checkbox(key="map_show_data").uncheck()
    app.checkbox(key="map_show_query").uncheck().run()
    assert not app.exception
    assert estimates(app) == original_prices
    assert "cambios pendientes" not in values(app, "info")
    chart_layers = layers(app)
    assert chart_layers["data-locations"]["visible"] is False
    assert chart_layers["submitted-query"]["visible"] is False
    assert chart_layers["submitted-query"]["data"] == original_point
    assert app.session_state["result"]["comparables"].comparable_row_ids == original_ids
    assert sorted(i for group in chart_layers["selected-comparables"]["data"] for i in group["row_ids"]) == sorted(original_ids)
    frame = bundle[0]
    expected = frame.loc[frame.property_type.eq("house") & frame.neighborhood.eq("Palermo")]
    assert sorted(i for group in chart_layers["data-locations"]["data"] for i in group["row_ids"]) == sorted(expected.row_id)
    assert app.number_input(key="map_price_max").value == float(frame.price_usd.max())
    app.number_input(key="map_price_min").set_value(0.)
    app.number_input(key="map_price_max").set_value(0.).run()
    assert not app.exception
    assert not layers(app)["data-locations"]["data"]
    assert "No hay registros con estos filtros" in values(app, "info")
    assert estimates(app) == original_prices
    # Selected neighbors remain available independently of the filtered data layer.
    group = layers(app)["selected-comparables"]["data"][0]
    app.selectbox(key="detail_location").set_value("selected-comparables:" + group["location_id"]).run()
    assert not app.exception
    detail = [table.value for table in app.dataframe if "Registro" in table.value][-1]
    assert sorted(detail.Registro) == sorted(group["row_ids"])
    assert estimates(app) == original_prices


def test_unsupported_regression_pair_still_shows_comparables(app):
    app.selectbox(key="query_type").set_value("PH")
    app.selectbox(key="query_neighborhood").set_value("Agronomía")
    app.number_input(key="query_total").set_value(100.)
    app.number_input(key="query_covered").set_value(80.)
    app.button(key="use_neighborhood").click().run()
    assert not app.exception
    submit(app)
    result = app.session_state["result"]
    assert result["comparables"].sufficient
    assert not result["linear"]["supported"]
    assert result["linear"]["reason"] == "unseen_category_combination"
    assert all(label.startswith("Comparables") for label in estimates(app))
    assert len(estimates(app)) == 2
    assert "No hay ejemplos de ese tipo de propiedad" in values(app, "warning")
    assert "**Diferencia:**" not in values(app, "markdown")


def test_strict_comparable_filters_keep_regression_and_evaluate_same_955_rows(app):
    submit(app)
    regression = {key: value for key, value in estimates(app).items() if key.startswith("Regresión")}
    app.slider(key="filter_rooms").set_value(0)
    app.slider(key="filter_area").set_value(0)
    app.slider(key="filter_ratio").set_value(0)
    app.slider(key="filter_radius").set_value(.1)
    submit(app)
    result = app.session_state["result"]
    assert not result["comparables"].sufficient
    assert estimates(app) == regression
    assert "Evidencia insuficiente" in values(app, "warning")
    assert "955 reservados" in values(app, "markdown")
    assert "**Diferencia:**" not in values(app, "markdown")


def test_invalid_surface_clears_old_result_prices_and_map_marker(app):
    submit(app)
    assert len(estimates(app)) == 4
    app.number_input(key="query_covered").set_value(100.)
    submit(app)
    assert "no superar la total" in values(app, "error")
    assert app.session_state["result"] is None
    assert not estimates(app)
    assert not layers(app)["submitted-query"]["data"]
    assert not layers(app)["selected-comparables"]["data"]
    assert sum(group["count"] for group in layers(app)["data-locations"]["data"]) == 7495


def test_invalid_submission_preserves_last_valid_three_neighbor_evaluation(app):
    default_table = next(table.value for table in app.dataframe if "Método" in table.value).copy()
    app.slider(key="filter_k").set_value(3)
    submit(app)
    assert app.session_state["result"]["comparables"].n_comparables == 3
    assert app.session_state["evaluation_filters"].k == 3
    valid_table = next(table.value for table in app.dataframe if "Método" in table.value).copy()
    assert not valid_table.equals(default_table)
    valid_coverage = {metric.label: metric.value for metric in app.metric
                      if metric.label.startswith("Cobertura")}
    app.number_input(key="query_covered").set_value(80.)  # Total remains 70 m².
    submit(app)
    assert "no superar la total" in values(app, "error")
    assert app.session_state["result"] is None
    assert not estimates(app)
    assert app.slider(key="filter_k").value == 3
    assert app.session_state["evaluation_filters"].k == 3
    current_table = next(table.value for table in app.dataframe if "Método" in table.value)
    pd.testing.assert_frame_equal(current_table, valid_table)
    assert {metric.label: metric.value for metric in app.metric
            if metric.label.startswith("Cobertura")} == valid_coverage
    assert "955 reservados" in values(app, "markdown")


@pytest.mark.parametrize("key,invalid_value,error", [
    ("query_total", 0.0, "La superficie total debe ser mayor que 0"),
    ("query_total", -1.0, "La superficie total debe ser mayor que 0"),
    ("query_covered", 0.0, "La superficie cubierta debe ser positiva"),
    ("query_covered", -1.0, "La superficie cubierta debe ser positiva"),
    ("query_rooms", 0, "Los ambientes deben ser un entero entre 1 y 20"),
    ("query_rooms", 21, "Los ambientes deben ser un entero entre 1 y 20"),
    ("query_lat", -35.0, "La ubicación debe estar dentro del área disponible de CABA"),
    ("query_lon", -59.0, "La ubicación debe estar dentro del área disponible de CABA"),
])
def test_numeric_query_edits_reach_validation_without_reusing_stale_values(app, key, invalid_value, error):
    app.slider(key="filter_k").set_value(3)
    submit(app)
    original_prices = estimates(app)
    original_point = layers(app)["submitted-query"]["data"]
    original_evaluation = next(table.value for table in app.dataframe if "Método" in table.value).copy()
    assert len(original_prices) == 4

    app.number_input(key=key).set_value(invalid_value).run()
    assert not app.exception
    assert app.session_state[key] == invalid_value
    assert "cambios pendientes" in values(app, "info")
    assert estimates(app) == original_prices
    assert layers(app)["submitted-query"]["data"] == original_point

    submit(app)
    assert error in values(app, "error")
    assert app.session_state["result"] is None
    assert not estimates(app)
    assert not layers(app)["submitted-query"]["data"]
    assert not layers(app)["selected-comparables"]["data"]
    assert sum(group["count"] for group in layers(app)["data-locations"]["data"]) == 7495
    assert app.session_state["evaluation_filters"].k == 3
    current_evaluation = next(table.value for table in app.dataframe if "Método" in table.value)
    pd.testing.assert_frame_equal(current_evaluation, original_evaluation)


def test_location_selector_exposes_all_coincident_records_without_hover(app):
    group = max(layers(app)["data-locations"]["data"], key=lambda item: item["count"])
    assert group["count"] > 1
    app.selectbox(key="detail_location").set_value("data-locations:" + group["location_id"]).run()
    assert not app.exception
    records = next(table.value for table in app.dataframe if "Registro" in table.value)
    assert sorted(records.Registro) == sorted(group["row_ids"])
    assert len(records) == group["count"]
    assert len(records.columns) == 8
    # A selection hidden by visual filters must not expose stale detail rows.
    app.number_input(key="map_price_max").set_value(0.)
    app.number_input(key="map_price_min").set_value(0.).run()
    assert not app.exception
    assert app.selectbox(key="detail_location").value is None
    assert not any("Registro" in table.value for table in app.dataframe)
