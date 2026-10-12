"""Pure, token-free map preparation for the public demonstration projection.

The map's visual filters never select model training rows. Location groups retain
the exact global row IDs, so coincident markers can be explored without inventing
coordinates or looking up model neighbours by an ambiguous latitude/longitude.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import asinh, isfinite, log2, log10, pi, radians, sqrt, tan

import numpy as np
import pandas as pd
import pydeck as pdk

from data import LAT_BOUNDS, LON_BOUNDS


LOW_COLOR = (95, 160, 155)
HIGH_COLOR = (125, 56, 147)
COMPARABLE_COLOR = (5, 113, 176)
QUERY_COLOR = (230, 97, 1)
COLOR_LABELS = {"price_usd": "Precio total (USD)", "usd_m2": "USD/m²"}
GROUP_COLUMNS = (
    "location_id", "lat", "lon", "count", "color_value", "color",
    "row_ids", "radius", "tooltip_text",
)


def _metric_name(color_metric: str) -> str:
    if color_metric not in COLOR_LABELS:
        raise ValueError("Elegí price_usd o usd_m2 para colorear el mapa.")
    return color_metric


def _map_frame(frame: pd.DataFrame, color_metric: str) -> pd.DataFrame:
    """Copy and validate only map inputs; never mutate a caller's model frame."""
    metric = _metric_name(color_metric)
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("El mapa requiere una tabla con columnas únicas.")
    missing = {"lat", "lon", metric} - set(frame.columns)
    if missing:
        raise ValueError("Faltan columnas del mapa: " + ", ".join(sorted(missing)))
    result = frame.copy(deep=True)
    if "row_id" not in result:
        result["row_id"] = result.index
    ids = result["row_id"]
    if (ids.isna().any() or ids.duplicated().any()
            or not ids.map(lambda value: isinstance(value, (int, np.integer))
                           and not isinstance(value, (bool, np.bool_)) and value >= 0).all()):
        raise ValueError("Cada registro del mapa necesita un row_id entero único.")
    for name in ("lat", "lon", metric):
        try:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("El mapa requiere coordenadas y precios finitos.") from exc
        if not np.isfinite(result[name]).all():
            raise ValueError("El mapa requiere coordenadas y precios finitos.")
    if (not result["lat"].between(*LAT_BOUNDS).all()
            or not result["lon"].between(*LON_BOUNDS).all()):
        raise ValueError("La ubicación queda fuera del rango admitido para CABA.")
    if not result[metric].gt(0).all():
        raise ValueError("Los precios del mapa deben ser positivos.")
    result[["lat", "lon"]] = result[["lat", "lon"]].round(3)
    return result


def filter_map_data(
    frame: pd.DataFrame,
    types: Sequence[str] | None = None,
    neighborhoods: Sequence[str] | None = None,
    price_range: tuple[float, float] | None = None,
    color_metric: str = "usd_m2",
) -> pd.DataFrame:
    """Filter the base layer only, using prices in the chosen colour's units.

    None leaves a filter open; an empty type/neighbourhood selection shows no
    records. The returned copy preserves global IDs for map-to-table selection.
    """
    result = _map_frame(frame, color_metric)
    keep = pd.Series(True, index=result.index)
    for selection, name in ((types, "property_type"), (neighborhoods, "neighborhood")):
        if selection is not None:
            if name not in result:
                raise ValueError(f"Falta la columna {name} para filtrar el mapa.")
            keep &= result[name].isin(selection)
    if price_range is not None:
        if len(price_range) != 2:
            raise ValueError("El rango de precios requiere mínimo y máximo.")
        lower, upper = map(float, price_range)
        if not (isfinite(lower) and isfinite(upper) and 0 <= lower <= upper):
            raise ValueError("El rango de precios debe ser finito, positivo y ordenado.")
        keep &= result[color_metric].between(lower, upper)
    return result.loc[keep].copy()


def _safe_median(values: np.ndarray) -> float:
    values = np.sort(values.astype(float, copy=True))
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    lower, upper = float(values[mid - 1]), float(values[mid])
    return lower + (upper - lower) / 2


def _color(value: float, lower: float, upper: float) -> list[int]:
    # A logarithmic colour scale retains every positive price while making the
    # wide price distribution readable. Taking logs before subtraction also
    # avoids overflow for very large finite prices; no data values are clipped.
    log_lower, log_upper = log10(lower), log10(upper)
    if log_lower == log_upper:
        proportion = 0.5
    else:
        proportion = (log10(value) - log_lower) / (log_upper - log_lower)
    proportion = min(1.0, max(0.0, proportion))
    return [round(a + (b - a) * proportion) for a, b in zip(LOW_COLOR, HIGH_COLOR)]


def group_locations(frame: pd.DataFrame, color_metric: str = "usd_m2") -> pd.DataFrame:
    """Group exact three-decimal locations, with a median for colour only.

    Groups never merge noncoincident coordinates and never jitter points. The
    colour domain is logarithmic over visible groups' medians, available through
    DataFrame.attrs for a matching legend. Values are not model predictions.
    """
    source = _map_frame(frame, color_metric)
    records = []
    for (lat, lon), rows in source.groupby(["lat", "lon"], sort=True):
        median = _safe_median(rows[color_metric].to_numpy())
        count = len(rows)
        records.append({
            "location_id": f"{lat:.3f},{lon:.3f}", "lat": float(lat), "lon": float(lon),
            "count": count, "color_value": median,
            "row_ids": tuple(sorted(int(value) for value in rows["row_id"])),
            "radius": min(8.0, 3.0 + sqrt(count)),
            "tooltip_text": (
                f"Ubicación aproximada: {lat:.3f}, {lon:.3f}\n"
                f"Registros visibles: {count}\n"
                f"Mediana · {COLOR_LABELS[color_metric]}: {median:,.0f}"
            ),
        })
    result = pd.DataFrame(records)
    if records:
        lower, upper = float(result["color_value"].min()), float(result["color_value"].max())
        result["color"] = [_color(value, lower, upper) for value in result["color_value"]]
        result = result.loc[:, GROUP_COLUMNS]
    else:
        lower, upper = None, None
        result = pd.DataFrame(columns=GROUP_COLUMNS)
    result.attrs.update(color_metric=color_metric, color_scale="log", color_min=lower, color_max=upper,
                        low_color=LOW_COLOR, high_color=HIGH_COLOR)
    return result


def fit_viewport(points: pd.DataFrame, *, width: int = 300, height: int = 400) -> pdk.ViewState:
    """Fit a flat mobile map with 40% total padding, capped at zoom 15."""
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        raise ValueError("El tamaño del mapa debe ser positivo.")
    if not {"lat", "lon"}.issubset(points.columns):
        raise ValueError("Faltan las coordenadas para encuadrar el mapa.")
    if points.empty:
        points = pd.DataFrame({"lat": LAT_BOUNDS, "lon": LON_BOUNDS})
    coordinates = points[["lat", "lon"]].to_numpy(dtype=float)
    if (not np.isfinite(coordinates).all()
            or not points["lat"].between(*LAT_BOUNDS).all()
            or not points["lon"].between(*LON_BOUNDS).all()):
        raise ValueError("Las coordenadas del encuadre quedan fuera de CABA.")
    min_lat, max_lat = float(points["lat"].min()), float(points["lat"].max())
    min_lon, max_lon = float(points["lon"].min()), float(points["lon"].max())
    latitude, longitude = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    center_y = asinh(tan(radians(latitude))) / (2 * pi)
    y_span = 2 * max(abs(asinh(tan(radians(lat))) / (2 * pi) - center_y)
                     for lat in (min_lat, max_lat))
    x_span = (max_lon - min_lon) / 360
    zooms = [log2(pixels * 0.6 / (512 * span)) for pixels, span in
             ((width, x_span), (height, y_span)) if span > 0]
    # Fractional zoom avoids wasting up to half the viewport while preserving
    # exactly the same mobile padding as the calculated geographic bounds.
    zoom = max(0, min(15, min(zooms))) if zooms else 15
    return pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom,
                         min_zoom=0, max_zoom=18, pitch=0, bearing=0)


def _query_point(query: object | None) -> list[dict]:
    if query is None:
        return []
    try:
        lat, lon = (query["lat"], query["lon"]) if isinstance(query, Mapping) else (query.lat, query.lon)
        lat, lon = float(lat), float(lon)
    except (KeyError, AttributeError, ValueError, TypeError) as exc:
        raise ValueError("La consulta necesita latitud y longitud válidas.") from exc
    if (not isfinite(lat) or not isfinite(lon) or not LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1]
            or not LON_BOUNDS[0] <= lon <= LON_BOUNDS[1]):
        raise ValueError("La consulta queda fuera del rango admitido para CABA.")
    lat, lon = round(lat, 3), round(lon, 3)
    return [{"lat": lat, "lon": lon, "location_id": f"{lat:.3f},{lon:.3f}",
             "tooltip_text": f"Consulta enviada\nUbicación aproximada: {lat:.3f}, {lon:.3f}"}]


def _reuse_groups(groups: pd.DataFrame, base: pd.DataFrame, color_metric: str) -> pd.DataFrame:
    """Validate an already prepared public-data grouping without regrouping."""
    if (not isinstance(groups, pd.DataFrame) or groups.columns.duplicated().any()
            or not set(GROUP_COLUMNS).issubset(groups.columns)
            or groups.attrs.get("color_metric") != color_metric):
        raise ValueError("Los grupos preparados no corresponden a la escala del mapa.")
    if groups["location_id"].isna().any() or groups["location_id"].duplicated().any():
        raise ValueError("Las ubicaciones preparadas deben ser únicas.")
    expected = {row.row_id: (row.lat, row.lon) for row in base.itertuples()}
    seen = set()
    for group in groups.itertuples():
        if (not isinstance(group.row_ids, (tuple, list)) or not group.row_ids
                or group.count != len(group.row_ids)
                or group.location_id != f"{group.lat:.3f},{group.lon:.3f}"):
            raise ValueError("Los grupos preparados contienen ubicaciones ambiguas.")
        for row_id in group.row_ids:
            if (not isinstance(row_id, (int, np.integer)) or isinstance(row_id, (bool, np.bool_))
                    or row_id in seen or expected.get(row_id) != (group.lat, group.lon)):
                raise ValueError("Los grupos preparados no coinciden con los registros visibles.")
            seen.add(row_id)
    if seen != set(expected):
        raise ValueError("Los grupos preparados no cubren todos los registros visibles.")
    # The app can keep extra lookup columns; only our display fields reach Deck.
    return groups.loc[:, GROUP_COLUMNS].copy(deep=True)


def build_map(
    frame: pd.DataFrame,
    *,
    filtered_frame: pd.DataFrame | None = None,
    grouped_data: pd.DataFrame | None = None,
    color_metric: str = "usd_m2",
    query: object | None = None,
    comparable_row_ids: Sequence[int] = (),
    show_data: bool = True,
    show_query: bool = True,
    show_comparables: bool = True,
    view_state: pdk.ViewState | Mapping | None = None,
) -> pdk.Deck:
    """Create independent stable-ID layers; the caller owns camera persistence.

    Pass the same view_state for visual-filter changes; replace it only for an
    explicit recenter action. The default fits the full dataset, not its visual
    subset. Supply grouped_data from group_locations(filtered_frame) to reuse
    the same public-data preparation as the UI's lookup/legend. Exact global-ID
    coverage and coordinate membership are checked before reuse. Only
    allowlisted map fields are serialized into the browser. No cache is used.
    """
    full = _map_frame(frame, color_metric)
    base = full if filtered_frame is None else _map_frame(filtered_frame, color_metric)
    if not set(base["row_id"]).issubset(set(full["row_id"])):
        raise ValueError("Los registros visibles no pertenecen al conjunto del mapa.")
    selected_ids = tuple(comparable_row_ids)
    if (len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids).issubset(set(full["row_id"]))):
        raise ValueError("Los comparables deben referir a registros únicos del conjunto.")
    base_groups = (group_locations(base, color_metric) if grouped_data is None
                   else _reuse_groups(grouped_data, base, color_metric))
    # Exact IDs preserve model selection even when many rows share coordinates.
    neighbors = full.loc[full["row_id"].isin(selected_ids)]
    neighbor_groups = group_locations(neighbors, color_metric)
    if not neighbor_groups.empty:
        neighbor_groups["tooltip_text"] = neighbor_groups.apply(
            lambda row: f"Comparables utilizados en esta ubicación: {row['count']}\n"
                        f"Ubicación aproximada: {row['lat']:.3f}, {row['lon']:.3f}", axis=1)
    point = _query_point(query)
    # Pydeck treats bare strings as JavaScript accessors. Units must be quoted
    # literals; otherwise radiusUnits becomes a function and WebGL can draw
    # enormous circles instead of the requested screen-pixel markers.
    layers = [
        pdk.Layer("ScatterplotLayer", id="data-locations", data=base_groups.to_dict("records"),
                  visible=show_data, pickable=True, auto_highlight=True,
                  get_position="[lon, lat]", get_fill_color="color", get_radius="radius",
                  radius_units="'pixels'", opacity=0.7, stroked=False),
        pdk.Layer("ScatterplotLayer", id="selected-comparables", data=neighbor_groups.to_dict("records"),
                  visible=show_comparables, pickable=True, auto_highlight=True,
                  get_position="[lon, lat]", get_fill_color=list(COMPARABLE_COLOR), get_radius=12,
                  radius_units="'pixels'", opacity=0.95, stroked=True,
                  get_line_color=[255, 255, 255], line_width_units="'pixels'", get_line_width=1),
        pdk.Layer("ScatterplotLayer", id="submitted-query", data=point,
                  visible=show_query, pickable=True,
                  get_position="[lon, lat]", get_fill_color=list(QUERY_COLOR), get_radius=8,
                  radius_units="'pixels'", opacity=1, stroked=True,
                  get_line_color=[255, 255, 255], line_width_units="'pixels'", get_line_width=1),
    ]
    if view_state is None:
        view_state = fit_viewport(full)
    elif isinstance(view_state, Mapping):
        view_state = pdk.ViewState(**view_state)
    return pdk.Deck(layers=layers, initial_view_state=view_state,
                    map_provider="carto", map_style="light", height=400,
                    tooltip={"text": "{tooltip_text}"},
                    description="Datos de la demo, consulta enviada y comparables; ubicaciones aproximadas.")
