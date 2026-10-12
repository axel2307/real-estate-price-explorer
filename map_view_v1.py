"""Pure, token-free map preparation for the public demonstration projection.

The map's visual filters never select model training rows. Location groups retain
the exact global row IDs, so coincident markers can be explored without inventing
coordinates or looking up model neighbours by an ambiguous latitude/longitude.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import asinh, isfinite, log2, log10, pi, radians, sqrt, tan

import numpy as np
import pandas as pd
import pydeck as pdk

from data import LAT_BOUNDS, LON_BOUNDS


# Viridis by Nathaniel J. Smith, Stefan van der Walt and Eric Firing, CC0.
# Source: https://github.com/BIDS/colormap/blob/master/colormaps.py
# License: https://creativecommons.org/publicdomain/zero/1.0/
# 256 RGB entries rounded from the original palette; no runtime plotting dependency.
VIRIDIS_COLORS = (
    (68, 1, 84), (68, 2, 86), (69, 4, 87), (69, 5, 89),
    (70, 7, 90), (70, 8, 92), (70, 10, 93), (70, 11, 94),
    (71, 13, 96), (71, 14, 97), (71, 16, 99), (71, 17, 100),
    (71, 19, 101), (72, 20, 103), (72, 22, 104), (72, 23, 105),
    (72, 24, 106), (72, 26, 108), (72, 27, 109), (72, 28, 110),
    (72, 29, 111), (72, 31, 112), (72, 32, 113), (72, 33, 115),
    (72, 35, 116), (72, 36, 117), (72, 37, 118), (72, 38, 119),
    (72, 40, 120), (72, 41, 121), (71, 42, 122), (71, 44, 122),
    (71, 45, 123), (71, 46, 124), (71, 47, 125), (70, 48, 126),
    (70, 50, 126), (70, 51, 127), (70, 52, 128), (69, 53, 129),
    (69, 55, 129), (69, 56, 130), (68, 57, 131), (68, 58, 131),
    (68, 59, 132), (67, 61, 132), (67, 62, 133), (66, 63, 133),
    (66, 64, 134), (66, 65, 134), (65, 66, 135), (65, 68, 135),
    (64, 69, 136), (64, 70, 136), (63, 71, 136), (63, 72, 137),
    (62, 73, 137), (62, 74, 137), (62, 76, 138), (61, 77, 138),
    (61, 78, 138), (60, 79, 138), (60, 80, 139), (59, 81, 139),
    (59, 82, 139), (58, 83, 139), (58, 84, 140), (57, 85, 140),
    (57, 86, 140), (56, 88, 140), (56, 89, 140), (55, 90, 140),
    (55, 91, 141), (54, 92, 141), (54, 93, 141), (53, 94, 141),
    (53, 95, 141), (52, 96, 141), (52, 97, 141), (51, 98, 141),
    (51, 99, 141), (50, 100, 142), (50, 101, 142), (49, 102, 142),
    (49, 103, 142), (49, 104, 142), (48, 105, 142), (48, 106, 142),
    (47, 107, 142), (47, 108, 142), (46, 109, 142), (46, 110, 142),
    (46, 111, 142), (45, 112, 142), (45, 113, 142), (44, 113, 142),
    (44, 114, 142), (44, 115, 142), (43, 116, 142), (43, 117, 142),
    (42, 118, 142), (42, 119, 142), (42, 120, 142), (41, 121, 142),
    (41, 122, 142), (41, 123, 142), (40, 124, 142), (40, 125, 142),
    (39, 126, 142), (39, 127, 142), (39, 128, 142), (38, 129, 142),
    (38, 130, 142), (38, 130, 142), (37, 131, 142), (37, 132, 142),
    (37, 133, 142), (36, 134, 142), (36, 135, 142), (35, 136, 142),
    (35, 137, 142), (35, 138, 141), (34, 139, 141), (34, 140, 141),
    (34, 141, 141), (33, 142, 141), (33, 143, 141), (33, 144, 141),
    (33, 145, 140), (32, 146, 140), (32, 146, 140), (32, 147, 140),
    (31, 148, 140), (31, 149, 139), (31, 150, 139), (31, 151, 139),
    (31, 152, 139), (31, 153, 138), (31, 154, 138), (30, 155, 138),
    (30, 156, 137), (30, 157, 137), (31, 158, 137), (31, 159, 136),
    (31, 160, 136), (31, 161, 136), (31, 161, 135), (31, 162, 135),
    (32, 163, 134), (32, 164, 134), (33, 165, 133), (33, 166, 133),
    (34, 167, 133), (34, 168, 132), (35, 169, 131), (36, 170, 131),
    (37, 171, 130), (37, 172, 130), (38, 173, 129), (39, 173, 129),
    (40, 174, 128), (41, 175, 127), (42, 176, 127), (44, 177, 126),
    (45, 178, 125), (46, 179, 124), (47, 180, 124), (49, 181, 123),
    (50, 182, 122), (52, 182, 121), (53, 183, 121), (55, 184, 120),
    (56, 185, 119), (58, 186, 118), (59, 187, 117), (61, 188, 116),
    (63, 188, 115), (64, 189, 114), (66, 190, 113), (68, 191, 112),
    (70, 192, 111), (72, 193, 110), (74, 193, 109), (76, 194, 108),
    (78, 195, 107), (80, 196, 106), (82, 197, 105), (84, 197, 104),
    (86, 198, 103), (88, 199, 101), (90, 200, 100), (92, 200, 99),
    (94, 201, 98), (96, 202, 96), (99, 203, 95), (101, 203, 94),
    (103, 204, 92), (105, 205, 91), (108, 205, 90), (110, 206, 88),
    (112, 207, 87), (115, 208, 86), (117, 208, 84), (119, 209, 83),
    (122, 209, 81), (124, 210, 80), (127, 211, 78), (129, 211, 77),
    (132, 212, 75), (134, 213, 73), (137, 213, 72), (139, 214, 70),
    (142, 214, 69), (144, 215, 67), (147, 215, 65), (149, 216, 64),
    (152, 216, 62), (155, 217, 60), (157, 217, 59), (160, 218, 57),
    (162, 218, 55), (165, 219, 54), (168, 219, 52), (170, 220, 50),
    (173, 220, 48), (176, 221, 47), (178, 221, 45), (181, 222, 43),
    (184, 222, 41), (186, 222, 40), (189, 223, 38), (192, 223, 37),
    (194, 223, 35), (197, 224, 33), (200, 224, 32), (202, 225, 31),
    (205, 225, 29), (208, 225, 28), (210, 226, 27), (213, 226, 26),
    (216, 226, 25), (218, 227, 25), (221, 227, 24), (223, 227, 24),
    (226, 228, 24), (229, 228, 25), (231, 228, 25), (234, 229, 26),
    (236, 229, 27), (239, 229, 28), (241, 229, 29), (244, 230, 30),
    (246, 230, 32), (248, 230, 33), (251, 231, 35), (253, 231, 37),
)
COMPARABLE_COLOR = (5, 113, 176)
QUERY_COLOR = (230, 97, 1)
COLOR_LABELS = {"price_usd": "Precio total (USD)", "usd_m2": "USD/m²"}
COLOR_TRANSFORMS = {"price_usd": "log", "usd_m2": "linear"}
GROUP_COLUMNS = (
    "location_id", "lat", "lon", "count", "color_value", "color",
    "row_ids", "radius", "tooltip_text",
)


def _metric_name(color_metric: str) -> str:
    if color_metric not in COLOR_LABELS:
        raise ValueError("Elegí price_usd o usd_m2 para colorear el mapa.")
    return color_metric


@dataclass(frozen=True)
class ColorScale:
    """Immutable visual reference, fitted on all public location medians.

    Empty references have both limits None. A constant reference uses one
    middle colour. Quantile limits affect colour only, never retained prices.
    """

    metric: str
    lower: float | None
    upper: float | None
    transform: str
    palette: str = "viridis"

    def __post_init__(self):
        _metric_name(self.metric)
        if self.transform != COLOR_TRANSFORMS[self.metric] or self.palette != "viridis":
            raise ValueError("La escala debe usar Viridis y la transformación de su métrica.")
        if self.lower is None and self.upper is None:
            return
        if (isinstance(self.lower, (bool, np.bool_)) or isinstance(self.upper, (bool, np.bool_))
                or not isinstance(self.lower, (int, float, np.integer, np.floating))
                or not isinstance(self.upper, (int, float, np.integer, np.floating))
                or not isfinite(self.lower) or not isfinite(self.upper)
                or not 0 < self.lower <= self.upper):
            raise ValueError("Los límites de color deben ser positivos, finitos y ordenados.")

    def proportion(self, value: float) -> float:
        """Clamp just the colour position, without changing the source value."""
        if not isfinite(value) or value <= 0 or self.lower is None:
            raise ValueError("El color requiere un valor positivo y una referencia no vacía.")
        if self.lower == self.upper:
            # A collapsed robust domain cannot encode variation. The one-colour
            # legend must describe every marker, including rare tail values.
            return 0.5
        if value <= self.lower:
            return 0.0
        if value >= self.upper:
            return 1.0
        if self.transform == "log":
            low, high = log10(self.lower), log10(self.upper)
            # Adjacent large floats can have identical rounded logarithms.
            if low != high:
                return (log10(value) - low) / (high - low)
        return (value - self.lower) / (self.upper - self.lower)

    def color(self, value: float) -> list[int]:
        return list(VIRIDIS_COLORS[round(self.proportion(value) * (len(VIRIDIS_COLORS) - 1))])

    @property
    def legend_values(self) -> tuple[float, ...]:
        if self.lower is None:
            return ()
        if self.lower == self.upper:
            return (float(self.lower),)
        if self.transform == "log":
            low, high = log10(self.lower), log10(self.upper)
            def interpolate(fraction):
                if low == high:
                    return self.lower + (self.upper - self.lower) * fraction
                try:
                    value = 10 ** (low + (high - low) * fraction)
                except OverflowError:
                    # Rounded log10(max_float) can exceed the exponentiation
                    # boundary although the original upper limit is finite.
                    return float(self.upper)
                return min(self.upper, max(self.lower, value))
            middle = tuple(interpolate(fraction) for fraction in (.25, .5, .75))
        else:
            middle = tuple(self.lower + (self.upper - self.lower) * fraction for fraction in (.25, .5, .75))
        return (float(self.lower), *middle, float(self.upper))


def _checked_scale(scale: ColorScale, color_metric: str) -> ColorScale:
    if not isinstance(scale, ColorScale) or scale.metric != color_metric:
        raise ValueError("La referencia de color no corresponde a la métrica del mapa.")
    return scale


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


def _fit_scale_values(values: Sequence[float], color_metric: str) -> ColorScale:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return ColorScale(color_metric, None, None, COLOR_TRANSFORMS[color_metric])

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        # Positive finite inputs make this interpolation overflow-safe.
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return ColorScale(color_metric, quantile(.02), quantile(.98), COLOR_TRANSFORMS[color_metric])


def fit_color_scale(frame: pd.DataFrame, color_metric: str = "usd_m2") -> ColorScale:
    """Fit fixed P2/P98 limits on all approximate locations, equally weighted.

    Call with the full public frame before applying any visual filters and
    cache by source fingerprint plus metric. No model/query inputs are used.
    """
    source = _map_frame(frame, color_metric)
    medians = [_safe_median(values.to_numpy()) for _, values in
               source.groupby(["lat", "lon"], sort=True)[color_metric]]
    return _fit_scale_values(medians, color_metric)


def _display_number(value: float) -> str:
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _apply_scale(groups: pd.DataFrame, scale: ColorScale) -> pd.DataFrame:
    """Colour a copy of precomputed groups; source medians/IDs stay intact."""
    result = groups.copy(deep=True)
    below, above = 0, 0
    if len(result):
        if scale.lower is None:
            raise ValueError("Una referencia vacía no puede colorear registros visibles.")
        values = result["color_value"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("Las medianas del mapa deben ser positivas y finitas.")
        below = int((values < scale.lower).sum())
        above = int((values > scale.upper).sum())
        result["color"] = [scale.color(value) for value in values]
        tooltips = []
        for row in result.itertuples():
            tooltip = (
                f"Ubicación aproximada (no identifica un inmueble): {row.lat:.3f}, {row.lon:.3f}\n"
                f"Registros visibles: {row.count}\n"
                f"Mediana · {COLOR_LABELS[scale.metric]}: {_display_number(row.color_value)}"
            )
            if scale.lower == scale.upper:
                tooltip += "\nReferencia P2–P98 constante; color único; valor real sin recortar."
            elif row.color_value < scale.lower:
                tooltip += f"\nColor saturado por debajo de P2 ({_display_number(scale.lower)}); valor real sin recortar."
            elif row.color_value > scale.upper:
                tooltip += f"\nColor saturado por encima de P98 ({_display_number(scale.upper)}); valor real sin recortar."
            tooltips.append(tooltip)
        result["tooltip_text"] = tooltips
        observed_min, observed_max = float(values.min()), float(values.max())
    else:
        observed_min, observed_max = None, None
    values = scale.legend_values
    colors = tuple(tuple(scale.color(value)) for value in values)
    result.attrs.update(
        color_metric=scale.metric, color_scale=scale.transform, color_reference=scale,
        color_constant=scale.lower is not None and scale.lower == scale.upper,
        color_palette=scale.palette, color_min=scale.lower, color_max=scale.upper,
        color_quantiles=(.02, .98), observed_min=observed_min, observed_max=observed_max,
        below_count=below, above_count=above, legend_values=values, legend_colors=colors,
        palette_colors=VIRIDIS_COLORS if len(values) > 1 else colors,
        low_color=colors[0] if colors else None, high_color=colors[-1] if colors else None,
    )
    return result.loc[:, GROUP_COLUMNS]


def group_locations(
    frame: pd.DataFrame, color_metric: str = "usd_m2", *, color_scale: ColorScale | None = None,
) -> pd.DataFrame:
    """Group exact three-decimal locations, with a median for colour only.

    Groups never merge noncoincident coordinates and never jitter points. The
    colour reference should come from fit_color_scale(full_frame), keeping the
    same values the same colour after filtering. Without an explicit reference,
    this standalone helper fits the supplied frame. Values are not predictions.
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
        })
    result = pd.DataFrame(records)
    if not records:
        result = pd.DataFrame(columns=GROUP_COLUMNS)
    scale = (_fit_scale_values(result["color_value"], color_metric) if color_scale is None
             else _checked_scale(color_scale, color_metric))
    return _apply_scale(result, scale)


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
    color_scale: ColorScale | None = None,
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
    the same public-data preparation as the UI's lookup/legend. Pass the cached
    full-frame color_scale too; otherwise it is fitted on frame, never on the
    visual subset. Reused groups are recoloured to this reference. Exact global-ID
    coverage and coordinate membership are checked before reuse. Only
    allowlisted map fields are serialized into the browser. No cache is used.
    """
    full = _map_frame(frame, color_metric)
    scale = (fit_color_scale(full, color_metric) if color_scale is None
             else _checked_scale(color_scale, color_metric))
    base = full if filtered_frame is None else _map_frame(filtered_frame, color_metric)
    if not set(base["row_id"]).issubset(set(full["row_id"])):
        raise ValueError("Los registros visibles no pertenecen al conjunto del mapa.")
    selected_ids = tuple(comparable_row_ids)
    if (len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids).issubset(set(full["row_id"]))):
        raise ValueError("Los comparables deben referir a registros únicos del conjunto.")
    base_groups = (group_locations(base, color_metric, color_scale=scale) if grouped_data is None
                   else _apply_scale(_reuse_groups(grouped_data, base, color_metric), scale))
    # Exact IDs preserve model selection even when many rows share coordinates.
    neighbors = full.loc[full["row_id"].isin(selected_ids)]
    neighbor_groups = group_locations(neighbors, color_metric, color_scale=scale)
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
                    # Streamlit's nested deck widget can offset cursor tooltips
                    # outside its clipped map after scrolling. Anchor the hover
                    # card inside the map; text still comes only from safe fields.
                    tooltip={"text": "{tooltip_text}", "style": {
                        "transform": "translate(8px, 8px)",
                        "maxWidth": "min(360px, calc(100% - 16px))",
                        "boxSizing": "border-box", "whiteSpace": "pre-line",
                        "overflowWrap": "anywhere", "fontSize": "12px",
                        "lineHeight": "1.4", "color": "#ffffff",
                        "backgroundColor": "rgba(17, 24, 39, 0.95)",
                        "borderRadius": "6px", "pointerEvents": "none",
                    }},
                    description="Datos de la demo, consulta enviada y comparables; ubicaciones aproximadas.")
