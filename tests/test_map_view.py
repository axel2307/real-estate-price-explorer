"""Map invariants: exact model rows, independent filters and safe display data."""
import json
from dataclasses import FrozenInstanceError
from math import asinh, pi, radians, tan

import numpy as np
import pandas as pd
import pytest

import map_view_v1
from data import LAT_BOUNDS, LON_BOUNDS
from map_view_v1 import (ColorScale, VIRIDIS_COLORS, build_map, filter_map_data,
                      fit_color_scale, fit_viewport, group_locations)


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
    assert hidden._tooltip["text"] == "{tooltip_text}"


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
    assert groups.attrs["color_scale"] == "linear"
    assert np.isfinite(groups["color_value"]).all()
    assert groups.loc[groups["count"].eq(2), "color_value"].iloc[0] == pytest.approx(1.3e308)
    for rgb in groups["color"]:
        assert len(rgb) == 3 and all(isinstance(value, int) and 0 <= value <= 255 for value in rgb)
    frame["usd_m2"] = 7.0
    equal = group_locations(frame)
    assert all(rgb == equal["color"].iloc[0] for rgb in equal["color"])


def test_total_colour_scale_uses_logarithms_without_clipping_data(frame):
    frame = frame.iloc[[0, 2, 3]].copy()
    frame["price_usd"] = [10.0, 100.0, 1000.0]
    scale = ColorScale("price_usd", 10, 1000, "log")
    groups = group_locations(frame, "price_usd", color_scale=scale).set_index("color_value")
    low, middle, high = [groups.loc[value, "color"] for value in (10, 100, 1000)]
    assert low == list(VIRIDIS_COLORS[0])
    assert middle == list(VIRIDIS_COLORS[128])
    assert high == list(VIRIDIS_COLORS[-1])
    assert scale.proportion(100) == .5
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

    def observe_grouping(rows, color_metric, **kwargs):
        grouped_row_counts.append(len(rows))
        return real_group_locations(rows, color_metric, **kwargs)

    monkeypatch.setattr(map_view_v1, "group_locations", observe_grouping)
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


def test_reference_uses_location_medians_with_equal_location_weight(frame):
    # Location medians: 1500, 4000, 5000. Raw rows would give different limits.
    reference = fit_color_scale(frame)
    assert reference.lower == pytest.approx(1600)
    assert reference.upper == pytest.approx(4960)
    assert reference.transform == "linear"
    assert fit_color_scale(frame, "price_usd").transform == "log"
    assert fit_color_scale(frame.sample(frac=1, random_state=4)) == reference
    with pytest.raises(FrozenInstanceError):
        reference.lower = 0


def test_filters_keep_fixed_domain_and_same_values_the_same_colour(frame):
    reference = fit_color_scale(frame)
    all_groups = group_locations(frame, color_scale=reference)
    subset = filter_map_data(frame, neighborhoods=["Belgrano"])
    groups = group_locations(subset, color_scale=reference)
    assert groups.attrs["color_min"] == all_groups.attrs["color_min"]
    assert groups.attrs["color_max"] == all_groups.attrs["color_max"]
    assert groups.attrs["legend_values"] == all_groups.attrs["legend_values"]
    assert groups.attrs["palette_colors"] == all_groups.attrs["palette_colors"]
    assert groups.iloc[0]["color"] == all_groups.loc[all_groups["color_value"].eq(4000), "color"].iloc[0]
    assert groups.attrs["observed_min"] == groups.attrs["observed_max"] == 4000
    assert groups.attrs["below_count"] == groups.attrs["above_count"] == 0


def test_build_map_anchors_filtered_and_reused_groups_to_full_reference(frame):
    subset = frame.loc[frame.row_id.eq(12)]
    prepared = group_locations(subset)  # Its standalone constant domain is wrong for full map.
    original = prepared.copy(deep=True)
    expected = fit_color_scale(frame).color(4000)
    assert prepared.iloc[0]["color"] != expected
    for deck in (build_map(frame, filtered_frame=subset),
                 build_map(frame, filtered_frame=subset, grouped_data=prepared)):
        assert deck.layers[0].data[0]["color"] == expected
        assert deck.layers[0].data[0]["color_value"] == 4000
        assert deck.layers[0].data[0]["row_ids"] == (12,)
    pd.testing.assert_frame_equal(prepared, original)


def test_extremes_preserve_values_and_ids_and_disclose_only_colour_saturation(frame):
    original = frame.copy(deep=True)
    reference = fit_color_scale(frame)
    groups = group_locations(frame, color_scale=reference)
    assert sorted(value for ids in groups.row_ids for value in ids) == sorted(frame.row_id)
    assert set(groups.color_value) == {1500, 4000, 5000}
    assert groups.attrs["below_count"] == groups.attrs["above_count"] == 1
    low = groups.loc[groups.color_value.eq(1500)].iloc[0]
    high = groups.loc[groups.color_value.eq(5000)].iloc[0]
    assert low.color == list(VIRIDIS_COLORS[0])
    assert high.color == list(VIRIDIS_COLORS[-1])
    assert "1.500,00" in low.tooltip_text and "por debajo de P2" in low.tooltip_text
    assert "5.000,00" in high.tooltip_text and "por encima de P98" in high.tooltip_text
    assert "valor real sin recortar" in high.tooltip_text
    assert "no identifica un inmueble" in high.tooltip_text
    assert "DO_NOT_SERIALIZE" not in " ".join(groups.tooltip_text)
    pd.testing.assert_frame_equal(frame, original)


def test_legend_and_palette_match_normalization_in_both_metrics(frame):
    assert len(VIRIDIS_COLORS) == 256
    assert VIRIDIS_COLORS[0] == (68, 1, 84)
    assert VIRIDIS_COLORS[-1] == (253, 231, 37)
    for metric in ("usd_m2", "price_usd"):
        scale = fit_color_scale(frame, metric)
        groups = group_locations(frame, metric, color_scale=scale)
        ticks = groups.attrs["legend_values"]
        assert len(ticks) == 5
        assert np.allclose([scale.proportion(v) for v in ticks], [0, .25, .5, .75, 1])
        assert groups.attrs["legend_colors"] == tuple(tuple(scale.color(v)) for v in ticks)
        assert groups.attrs["palette_colors"] == VIRIDIS_COLORS
        assert all(0 <= channel <= 255 for color in VIRIDIS_COLORS for channel in color)


@pytest.mark.parametrize("lower,upper", [(0, 2), (-1, 2), (3, 2), (float("nan"), 2),
                                        (1, float("inf")), (None, 2), (1, None), (True, 2)])
def test_reference_rejects_invalid_limits(lower, upper):
    with pytest.raises(ValueError):
        ColorScale("usd_m2", lower, upper, "linear")


def test_reference_rejects_wrong_metric_transform_palette_and_empty_nonempty_use(frame):
    with pytest.raises(ValueError):
        ColorScale("usd_m2", 1, 2, "log")
    with pytest.raises(ValueError):
        ColorScale("price_usd", 1, 2, "linear")
    with pytest.raises(ValueError):
        ColorScale("usd_m2", 1, 2, "linear", palette="other")
    wrong_metric = fit_color_scale(frame, "price_usd")
    with pytest.raises(ValueError):
        group_locations(frame, color_scale=wrong_metric)
    with pytest.raises(ValueError):
        build_map(frame, color_scale=wrong_metric)
    with pytest.raises(ValueError):
        group_locations(frame, color_scale=fit_color_scale(frame.iloc[:0]))


def test_empty_and_constant_references_have_safe_legends(frame):
    empty = frame.iloc[:0]
    reference = fit_color_scale(empty)
    assert reference.lower is reference.upper is None
    groups = group_locations(empty, color_scale=reference)
    assert groups.attrs["legend_values"] == groups.attrs["palette_colors"] == ()
    assert build_map(empty).layers[0].data == []
    fixed_empty = group_locations(empty, color_scale=fit_color_scale(frame))
    assert len(fixed_empty.attrs["legend_values"]) == 5
    assert fixed_empty.attrs["observed_min"] is fixed_empty.attrs["observed_max"] is None
    constant = frame.assign(usd_m2=7.0)
    groups = group_locations(constant)
    assert groups.attrs["legend_values"] == (7.0,)
    assert groups.attrs["palette_colors"] == (VIRIDIS_COLORS[128],)
    assert groups.attrs["below_count"] == groups.attrs["above_count"] == 0
    assert all(color == list(VIRIDIS_COLORS[128]) for color in groups.color)


def test_log_scale_remains_finite_across_extreme_positive_domain(frame):
    scale = ColorScale("price_usd", 1e-300, 1.7e308, "log")
    assert all(np.isfinite(scale.legend_values))
    for value in (1e-310, 1e-300, 1.0, 1e308, 1.7e308):
        assert 0 <= scale.proportion(value) <= 1
        assert all(isinstance(channel, int) for channel in scale.color(value))
    frame["price_usd"] = [1e308, 1.6e308, 1e-300, 1.7e308]
    assert np.isfinite(group_locations(frame, "price_usd").color_value).all()


def test_collapsed_quantiles_keep_tail_colours_consistent_with_single_colour_legend():
    rows = pd.DataFrame({
        "row_id": range(100), "lat": [-34.6] * 100,
        "lon": np.linspace(-58.5, -58.401, 100),
        "usd_m2": [1000.0] * 99 + [1_000_000.0],
    })
    scale = fit_color_scale(rows)
    assert scale.lower == scale.upper == 1000
    groups = group_locations(rows, color_scale=scale)
    assert groups.attrs["color_constant"] is True
    assert groups.attrs["palette_colors"] == (VIRIDIS_COLORS[128],)
    assert all(rgb == list(VIRIDIS_COLORS[128]) for rgb in groups.color)
    assert groups.attrs["above_count"] == 1  # Actual tail value remains disclosed.
    extreme = groups.loc[groups.color_value.eq(1_000_000)].iloc[0]
    assert extreme.row_ids == (99,)
    assert "1.000.000,00" in extreme.tooltip_text and "color único" in extreme.tooltip_text
    assert "Color saturado" not in extreme.tooltip_text


def test_adjacent_largest_floats_have_finite_log_legend():
    upper = np.finfo(float).max
    lower = np.nextafter(upper, 0)
    scale = ColorScale("price_usd", lower, upper, "log")
    values = scale.legend_values
    assert len(values) == 5 and all(np.isfinite(values))
    assert all(lower <= value <= upper for value in values)
    assert values[0] == lower and values[-1] == upper
    assert all(isinstance(channel, int) for value in values for channel in scale.color(value))


def test_tooltip_is_plain_text_wrapped_and_anchored_inside_the_map(frame):
    # The bundled deck widget misplaces cursor-relative cards after a scroll.
    # Its getTooltip style is applied after the faulty transform, overriding it.
    tooltip = build_map(frame)._tooltip
    assert tooltip["text"] == "{tooltip_text}" and "html" not in tooltip
    assert tooltip["style"]["transform"] == "translate(8px, 8px)"
    assert tooltip["style"]["maxWidth"] == "min(360px, calc(100% - 16px))"
    assert tooltip["style"]["whiteSpace"] == "pre-line"
    assert tooltip["style"]["overflowWrap"] == "anywhere"
    assert tooltip["style"]["pointerEvents"] == "none"
