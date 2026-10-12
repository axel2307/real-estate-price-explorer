"""Map invariants: exact model rows, independent filters and safe display data."""
import json
from math import asinh, pi, radians, tan

import numpy as np
import pandas as pd
import pytest

import map_view
from data import LAT_BOUNDS, LON_BOUNDS
from map_view import build_map, filter_map_data, fit_viewport, group_locations


@pytest.fixture
def frame():
    return pd.DataFrame({
        "row_id": [40, 90, 12, 150],
        "lat": [-34.60, -34.60, -34.61, -34.62],
        "lon": [-58.44, -58.44, -58.45, -58.46],
        "property_type": ["apartment", "house", "apartment", "PH"],
        "neighborhood": ["Palermo", "Palermo", "Belgrano", "Caballito"],
        "price_usd": [100_000.0, 300_000.0, 200_000.0, 400_000.0],
        "usd_m2": [2000.0, 1000.0, 4000.0, 5000.0],
        "private_description": ["DO_NOT_SERIALIZE"] * 4,
    })


def test_coincident_locations_keep_exact_record_ids_without_jitter(frame):
    groups = group_locations(frame)
    coincident = groups.loc[groups["location_id"].eq("-34.600,-58.440")].iloc[0]
    assert len(groups) == 3
    assert coincident["count"] == 2
    assert coincident["row_ids"] == (40, 90)
    assert coincident["color_value"] == 1500
    assert coincident["lat"] == -34.60 and coincident["lon"] == -58.44
    shuffled = group_locations(frame.sample(frac=1, random_state=17))
    pd.testing.assert_frame_equal(groups, shuffled)


def test_visual_filters_do_not_mutate_input_or_neighbor_selection(frame):
    original = frame.copy(deep=True)
    visible = filter_map_data(frame, types=["PH"], neighborhoods=["Caballito"],
                              price_range=(4500, 5500), color_metric="usd_m2")
    assert visible["row_id"].tolist() == [150]
    deck = json.loads(build_map(frame, filtered_frame=visible,
                                comparable_row_ids=(40, 12), query={"lat": -34.6, "lon": -58.44}).to_json())
    layers = {layer["id"]: layer for layer in deck["layers"]}
    assert layers["data-locations"]["data"][0]["row_ids"] == [150]
    neighbor_ids = {row_id for row in layers["selected-comparables"]["data"] for row_id in row["row_ids"]}
    assert neighbor_ids == {40, 12}
    assert 90 not in neighbor_ids  # Same coordinate as 40, but not selected by model.
    assert "DO_NOT_SERIALIZE" not in json.dumps(deck)
    pd.testing.assert_frame_equal(frame, original)


def test_layers_independent_and_camera_does_not_follow_visual_filters(frame):
    normal = build_map(frame)
    state = fit_viewport(frame.iloc[[0]])
    hidden = build_map(frame, filtered_frame=frame.iloc[0:0], comparable_row_ids=(90,),
                       query={"lat": -34.6014, "lon": -58.4414},
                       show_data=False, show_query=False, view_state=state)
    assert [layer.id for layer in normal.layers] == [layer.id for layer in hidden.layers]
    assert hidden.layers[0].visible is False and hidden.layers[1].visible is True
    assert hidden.layers[2].visible is False
    assert hidden.layers[1].data[0]["row_ids"] == (90,)
    assert hidden.layers[2].data[0]["lat"] == -34.601
    assert hidden.initial_view_state is state
    empty_base = build_map(frame, filtered_frame=frame.iloc[0:0])
    assert empty_base.initial_view_state.to_json() == normal.initial_view_state.to_json()
    assert hidden._tooltip == {"text": "{tooltip_text}"}


def test_price_filter_uses_selected_units_and_empty_selection_is_empty(frame):
    totals = filter_map_data(frame, price_range=(200_000, 300_000), color_metric="price_usd")
    assert totals["row_id"].tolist() == [90, 12]
    assert filter_map_data(frame, types=[]).empty
    assert filter_map_data(frame, neighborhoods=[]).empty
    empty_groups = group_locations(frame.iloc[0:0])
    assert empty_groups.empty and empty_groups.attrs["color_min"] is None


def test_extreme_finite_prices_have_finite_medians_and_rgb(frame):
    frame["usd_m2"] = [1e308, 1.6e308, 1e-300, 1.7e308]
    groups = group_locations(frame)
    assert groups.attrs["color_scale"] == "log"
    assert np.isfinite(groups["color_value"]).all()
    assert groups.loc[groups["count"].eq(2), "color_value"].iloc[0] == pytest.approx(1.3e308)
    for rgb in groups["color"]:
        assert len(rgb) == 3 and all(isinstance(value, int) and 0 <= value <= 255 for value in rgb)
    frame["usd_m2"] = 7.0
    equal = group_locations(frame)
    assert all(rgb == equal["color"].iloc[0] for rgb in equal["color"])


def test_colour_scale_uses_logarithms_without_clipping_data(frame):
    frame = frame.iloc[[0, 2, 3]].copy()
    frame["usd_m2"] = [10.0, 100.0, 1000.0]
    groups = group_locations(frame).set_index("color_value")
    low, middle, high = [groups.loc[value, "color"] for value in (10, 100, 1000)]
    assert middle == [round((a + b) / 2) for a, b in zip(low, high)]
    assert set(groups.index) == {10, 100, 1000}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1, 0])
def test_invalid_prices_rejected(frame, bad_value):
    frame.loc[0, "usd_m2"] = bad_value
    with pytest.raises(ValueError):
        group_locations(frame)


def test_boundary_viewport_fits_mobile_and_bad_query_is_rejected(frame):
    points = pd.DataFrame({"lat": LAT_BOUNDS, "lon": LON_BOUNDS})
    state = fit_viewport(points)
    world = 512 * 2 ** state.zoom
    center_y = asinh(tan(radians(state.latitude))) / (2 * pi)
    for point in points.itertuples():
        x_offset = abs(point.lon - state.longitude) / 360 * world
        y_offset = abs(asinh(tan(radians(point.lat))) / (2 * pi) - center_y) * world
        assert x_offset <= 300 * .3 + 1e-7
        assert y_offset <= 400 * .3 + 1e-7
    assert fit_viewport(points.iloc[0:0]).zoom == state.zoom
    assert fit_viewport(points.iloc[[0]]).zoom == 15
    with pytest.raises(ValueError):
        build_map(frame, query={"lat": LAT_BOUNDS[0] - .001, "lon": -58.44})
    with pytest.raises(ValueError):
        fit_viewport(pd.DataFrame({"lat": [-90], "lon": [-58.44]}))


def test_ids_and_ranges_must_be_unambiguous(frame):
    with pytest.raises(ValueError):
        build_map(frame, comparable_row_ids=(40, 40))
    with pytest.raises(ValueError):
        build_map(frame, comparable_row_ids=(999,))
    with pytest.raises(ValueError):
        group_locations(frame.assign(row_id=40))
    with pytest.raises(ValueError):
        filter_map_data(frame, price_range=(3000, 1000))
    with pytest.raises(ValueError):
        group_locations(frame, color_metric="private_description")


def test_prepared_base_groups_are_reused_and_match_normal_map(frame, monkeypatch):
    groups = group_locations(frame)
    original = groups.copy(deep=True)
    normal = build_map(frame, comparable_row_ids=(40, 12))
    grouped_row_counts = []
    real_group_locations = group_locations

    def observe_grouping(rows, color_metric):
        grouped_row_counts.append(len(rows))
        return real_group_locations(rows, color_metric)

    monkeypatch.setattr(map_view, "group_locations", observe_grouping)
    reused = build_map(frame, grouped_data=groups, comparable_row_ids=(40, 12))
    assert grouped_row_counts == [2]  # Only the selected neighbours are grouped.
    assert json.loads(normal.to_json()) == json.loads(reused.to_json())
    pd.testing.assert_frame_equal(groups, original)


def test_prepared_groups_reject_incomplete_or_incorrect_global_ids(frame):
    groups = group_locations(frame)
    with pytest.raises(ValueError):
        build_map(frame, grouped_data=groups.iloc[:-1])
    with pytest.raises(ValueError):
        build_map(frame, filtered_frame=frame.iloc[[0]], grouped_data=groups)
    wrong = groups.copy(deep=True)
    wrong.at[0, "row_ids"] = (999,)
    with pytest.raises(ValueError):
        build_map(frame, grouped_data=wrong)
    duplicated = pd.concat([groups, groups.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_map(frame, grouped_data=duplicated)
    shifted = groups.copy(deep=True)
    shifted.at[0, "lat"] -= .001
    shifted.at[0, "location_id"] = f"{shifted.at[0, 'lat']:.3f},{shifted.at[0, 'lon']:.3f}"
    with pytest.raises(ValueError):
        build_map(frame, grouped_data=shifted)
    with pytest.raises(ValueError):
        build_map(frame, grouped_data=groups, color_metric="price_usd")


def test_prepared_empty_groups_and_fractional_mobile_zoom(frame):
    empty = frame.iloc[0:0]
    deck = build_map(frame, filtered_frame=empty, grouped_data=group_locations(empty))
    assert deck.layers[0].data == []
    points = pd.DataFrame({"lat": [-34.692, -34.536], "lon": [-58.539, -58.358]})
    state = fit_viewport(points)
    assert 9 < state.zoom < 10


def test_webgl_dimension_units_are_literals_not_javascript_accessors(frame):
    # Regression for a visually solid map: '@@=pixels' becomes a function,
    # not a deck.gl unit, causing marker radii to cover the entire canvas.
    deck = json.loads(build_map(frame, comparable_row_ids=(40,),
                                query={"lat": -34.6, "lon": -58.44}).to_json())
    for layer in deck["layers"]:
        assert layer["radiusUnits"] == "pixels"
        if layer["stroked"]:
            assert layer["lineWidthUnits"] == "pixels"
