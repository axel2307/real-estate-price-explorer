"""One property, two independent estimates, and a public-data map explorer."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from comparables import Filters
from data import DataValidationError, LAT_BOUNDS, LON_BOUNDS
from evaluation import evaluate_joint
from explorer import (PROPERTY_LABELS, REASONS, PropertyQuery, compare_property,
                      display_rows, money, neighborhood_location, prepare_demo)
from linear import ModelValidationError
from map_view import build_map, filter_map_data, fit_color_scale, fit_viewport, group_locations


st.set_page_config(page_title="Explorador y comparador inmobiliario", page_icon="🏘️", layout="wide")
# Constant, trusted script: use the interface language for assistive technology.
st.html("<script>document.documentElement.lang = 'es-AR';</script>", unsafe_allow_javascript=True)
st.markdown("""<style>
.block-container {max-width: 1160px; padding-top: 2rem; padding-bottom: 2rem;}
h1 {font-size: clamp(1.75rem, 4vw, 2.65rem) !important; line-height: 1.16 !important;}
[data-testid="stMetricValue"] {font-size: clamp(1.45rem, 3vw, 2rem);}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_bundle(fingerprint):
    return prepare_demo()


@st.cache_data(show_spinner=False)
def get_evaluation(fingerprint, filters):
    _, train, test, fitted, _, _ = get_bundle(fingerprint)
    return evaluate_joint(train, test, filters, fitted=fitted)


@st.cache_data(show_spinner=False)
def get_color_scale(fingerprint, metric):
    return fit_color_scale(get_bundle(fingerprint)[0], metric)


@st.cache_data(show_spinner=False, max_entries=32)
def get_map_subset(fingerprint, types, neighborhoods, price_range, metric):
    public_frame = get_bundle(fingerprint)[0]
    subset = filter_map_data(public_frame, types=types or None, neighborhoods=neighborhoods or None,
                             price_range=price_range, color_metric=metric)
    return subset, group_locations(subset, color_metric=metric,
                                   color_scale=get_color_scale(fingerprint, metric))


st.title("Explorador y comparador inmobiliario")
st.caption("Demo educativa con datos ficticios")
st.write("Completá una propiedad de CABA, compará dos estimaciones y explorá los registros del mapa.")
try:
    fingerprint = hashlib.sha256((Path(__file__).parent / "data/demo.csv").read_bytes()).hexdigest()
    frame, train, test, fitted, report, split_info = get_bundle(fingerprint)
except (OSError, DataValidationError, ModelValidationError, ValueError):
    st.error("No pudimos preparar la demo. Revisá el archivo y los requisitos del proyecto.")
    st.stop()

if st.session_state.get("dataset_fingerprint") != fingerprint:
    # This public-only app does not persist query data outside this session.
    for key in list(st.session_state):
        if key.startswith(("query_", "filter_", "map_", "detail_")) or key in ("result", "result_inputs", "query_error", "evaluation_filters"):
            del st.session_state[key]
    st.session_state.dataset_fingerprint = fingerprint

default_neighborhood = "Palermo" if "Palermo" in fitted.categories["neighborhood"] else fitted.categories["neighborhood"][0]
default_lat, default_lon = neighborhood_location(train, default_neighborhood)
defaults = {
    "query_type": "apartment", "query_neighborhood": default_neighborhood,
    "query_rooms": 3, "query_total": 70.0, "query_covered": 60.0,
    "query_lat": default_lat, "query_lon": default_lon,
    "filter_rooms": 1, "filter_area": 35, "filter_ratio": 25, "filter_radius": 5.0, "filter_k": 5,
    "result": None, "query_error": None, "evaluation_filters": Filters(),
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

st.subheader("Tu propiedad")
with st.container(border=True):
    first = st.columns(3)
    first[0].selectbox("Tipo de propiedad", list(PROPERTY_LABELS), key="query_type", format_func=PROPERTY_LABELS.get)
    first[1].selectbox("Barrio", fitted.categories["neighborhood"], key="query_neighborhood")
    # Let numeric edits reach shared validation, rather than retain stale widget values.
    first[2].number_input("Ambientes", step=1, key="query_rooms", help="Entero entre 1 y 20.")
    second = st.columns(2)
    second[0].number_input("Superficie total (m²)", step=1.0, key="query_total", help="Mayor que 0; se valida al comparar.")
    second[1].number_input("Superficie cubierta (m²)", step=1.0, key="query_covered", help="Mayor que 0 y no mayor que la superficie total.")
    with st.expander("Ubicación aproximada", expanded=False):
        st.caption("La regresión usa el barrio. Los comparables usan estas coordenadas; pueden cruzar un límite barrial.")
        if st.button("Usar ubicación de los ejemplos del barrio", key="use_neighborhood"):
            st.session_state.query_lat, st.session_state.query_lon = neighborhood_location(train, st.session_state.query_neighborhood)
        location = st.columns(2)
        location[0].number_input("Latitud aproximada", step=0.001, format="%.3f", key="query_lat",
                                 help=f"Rango disponible: {LAT_BOUNDS[0]} a {LAT_BOUNDS[1]}.")
        location[1].number_input("Longitud aproximada", step=0.001, format="%.3f", key="query_lon",
                                 help=f"Rango disponible: {LON_BOUNDS[0]} a {LON_BOUNDS[1]}.")
        st.caption("El botón usa un punto de referencia de entrenamiento. Podés editarlo; no representa un domicilio.")
    st.caption(f"Ubicación elegida: {st.session_state.query_lat:.3f}, {st.session_state.query_lon:.3f}")
    with st.expander("Ajustar comparables", expanded=False):
        advanced = st.columns(2)
        advanced[0].slider("Tolerancia de ambientes (±)", 0, 5, key="filter_rooms")
        advanced[0].slider("Tolerancia de superficie total (± %)", 0, 100, step=5, key="filter_area")
        advanced[1].slider("Tolerancia de proporción cubierta (± puntos porcentuales)", 0, 100, step=5, key="filter_ratio")
        advanced[1].slider("Radio máximo (km)", 0.1, 20.0, step=0.1, key="filter_radius")
        st.slider("Cantidad máxima de comparables", 3, 5, key="filter_k")
        st.caption("Se requieren al menos 3. Los filtros no se amplían automáticamente.")
    submitted = st.button("Comparar estimaciones", type="primary", width="stretch", key="compare")

query = PropertyQuery(st.session_state.query_type, st.session_state.query_neighborhood,
                      st.session_state.query_rooms, st.session_state.query_total,
                      st.session_state.query_covered, st.session_state.query_lat, st.session_state.query_lon)
filters = Filters(st.session_state.filter_rooms, st.session_state.filter_area / 100,
                  st.session_state.filter_ratio / 100, st.session_state.filter_radius, st.session_state.filter_k)
inputs = {"query": asdict(query), "filters": asdict(filters)}
if submitted:
    st.session_state.result = None
    st.session_state.query_error = None
    st.session_state.pop("result_inputs", None)
    try:
        st.session_state.result = compare_property(train, fitted, query, filters)
        st.session_state.result_inputs = inputs
        st.session_state.evaluation_filters = filters
    except (ValueError, ModelValidationError) as error:
        st.session_state.query_error = str(error)

result = st.session_state.result
# A stable parent keeps the map mounted when results or validation notices change.
with st.container(key="query_results"):
    if st.session_state.query_error:
        st.error(st.session_state.query_error)
    if result:
        if inputs != st.session_state.result_inputs:
            st.info("Tenés cambios pendientes. Tocá «Comparar estimaciones» para actualizar los resultados y el marcador.")
        sent = result["query"]
        st.subheader("Dos estimaciones para tu consulta")
        st.caption(f"{PROPERTY_LABELS[sent.property_type]} en {sent.neighborhood} · {sent.rooms} ambientes · {sent.area_total:g} m² totales · {sent.area_covered:g} m² cubiertos · {sent.lat:.3f}, {sent.lon:.3f}")
        cards = st.columns(2)
        comparable = result["comparables"]
        with cards[0].container(border=True):
            st.markdown("### Comparables geográficos")
            if comparable.sufficient:
                st.metric("Comparables · USD/m²", money(comparable.usd_m2))
                st.metric("Comparables · Total", money(comparable.asking_price_usd))
                st.caption(f"{comparable.n_comparables} vecinos · {comparable.distance_min_km:.2f}–{comparable.distance_max_km:.2f} km")
            else:
                st.warning(comparable.insufficient_reason or f"Evidencia insuficiente: {comparable.n_comparables} comparables; hacen falta al menos 3.")
        with cards[1].container(border=True):
            st.markdown("### Regresión lineal")
            linear_result = result["linear"]
            if linear_result["supported"]:
                st.metric("Regresión · USD/m²", money(linear_result["predicted_usd_m2"]))
                st.metric("Regresión · Total", money(linear_result["predicted_price_usd"]))
                st.caption("Modelo ajustado con el mismo conjunto de entrenamiento.")
            else:
                st.warning("Sin estimación. " + REASONS.get(linear_result["reason"], "Revisá los valores ingresados."))
        if result["difference"]:
            difference = result["difference"]
            direction = "por encima" if difference["usd_m2"] >= 0 else "por debajo"
            st.write(f"**Diferencia:** comparables queda {money(abs(difference['usd_m2']))}/m² {direction} de regresión; {money(abs(difference['total_usd']))} de diferencia en el total.")
        with st.expander("Ver los comparables utilizados"):
            if comparable.n_comparables:
                selected = frame.set_index("row_id", drop=False).loc[list(comparable.comparable_row_ids)]
                table = display_rows(selected)
                table["Distancia (km)"] = comparable.comparable_distances_km
                st.dataframe(table, hide_index=True, width="stretch")
            if comparable.sufficient:
                st.caption(f"Rango observado de vecinos: {money(comparable.observed_min_usd_m2)}–{money(comparable.observed_max_usd_m2)} por m². {comparable.n_within_filters} registros cumplen los filtros.")
            elif not comparable.n_comparables:
                st.write("No se encontraron vecinos con estos filtros.")

st.subheader("Explorá los datos")
with st.expander("Filtros y capas del mapa", expanded=False):
    map_controls = st.columns(2)
    map_types = map_controls[0].multiselect("Tipo en el mapa", list(PROPERTY_LABELS), format_func=PROPERTY_LABELS.get, placeholder="Todos los tipos", key="map_types")
    map_neighborhoods = map_controls[1].multiselect("Barrio en el mapa", sorted(frame.neighborhood.unique()), placeholder="Todos los barrios", key="map_neighborhoods")
    metric = st.selectbox("Color y filtro de precio", ["usd_m2", "price_usd"], format_func=lambda value: "Precio por m² (USD/m²)" if value == "usd_m2" else "Precio total (USD)", key="map_metric")
    minimum, maximum = float(frame[metric].min()), float(frame[metric].max())
    if st.session_state.get("map_price_unit") != metric:
        st.session_state.map_price_unit = metric
        st.session_state.map_price_min = minimum
        st.session_state.map_price_max = maximum
    prices = st.columns(2)
    price_min = prices[0].number_input("Precio mínimo del mapa", min_value=0.0, max_value=maximum, step=100.0, key="map_price_min")
    price_max = prices[1].number_input("Precio máximo del mapa", min_value=0.0, max_value=maximum, step=100.0, key="map_price_max")
    layers = st.columns(3)
    show_data = layers[0].checkbox("Datos", value=True, key="map_show_data")
    show_query = layers[1].checkbox("Consulta", value=True, key="map_show_query")
    show_neighbors = layers[2].checkbox("Comparables", value=True, key="map_show_neighbors")
    st.caption("Estos filtros cambian sólo los datos visibles. La consulta, los vecinos y los modelos mantienen su configuración.")

color_scale = get_color_scale(fingerprint, metric)
with st.container(key="map_status"):
    if price_min > price_max:
        st.warning("En el mapa, el precio mínimo no puede superar al máximo.")
        filtered = frame.iloc[:0].copy()
        locations = group_locations(filtered, color_metric=metric, color_scale=color_scale)
    else:
        filtered, locations = get_map_subset(fingerprint, tuple(map_types), tuple(map_neighborhoods),
                                             (price_min, price_max), metric)
    st.caption(f"{len(filtered):,} de {len(frame):,} registros · {len(locations):,} ubicaciones".replace(",", "."))
    if filtered.empty and price_min <= price_max:
        st.info("No hay registros con estos filtros. Podés ampliarlos; las estimaciones no cambian.")

map_query = result["query"] if result else None
neighbor_ids = result["comparables"].comparable_row_ids if result else ()
view_controls = st.columns(2)
if view_controls[0].button("Ver todos los datos", key="view_all", width="stretch"):
    st.session_state.map_view_state = fit_viewport(frame)
    st.session_state.map_view_revision = st.session_state.get("map_view_revision", 0) + 1
if view_controls[1].button("Enfocar consulta", key="view_query", disabled=not bool(result), width="stretch"):
    focus = frame.loc[frame.row_id.isin(neighbor_ids), ["lat", "lon"]]
    focus = pd.concat([focus, pd.DataFrame([{"lat": map_query.lat, "lon": map_query.lon}])], ignore_index=True)
    st.session_state.map_view_state = fit_viewport(focus)
    st.session_state.map_view_revision = st.session_state.get("map_view_revision", 0) + 1
deck = build_map(frame, filtered_frame=filtered, grouped_data=locations, color_metric=metric,
                 color_scale=color_scale, query=map_query,
                 comparable_row_ids=neighbor_ids, show_data=show_data, show_query=show_query,
                 show_comparables=show_neighbors, view_state=st.session_state.get("map_view_state"))
map_event = st.pydeck_chart(deck, width="stretch", height=420, on_select="rerun", selection_mode="single-object",
    key=f"map_canvas_{st.session_state.get('map_view_revision', 0)}",
    alt="Mapa de CABA con registros agrupados por ubicación, consulta naranja y comparables azules. El selector de ubicación permite consultar los mismos registros con teclado.")
st.caption("Naranja: consulta · Azul: comparables · Los puntos coincidentes agrupan varios registros.")
legend = locations.attrs
if legend["legend_values"]:
    unit = "USD/m²" if metric == "usd_m2" else "USD total"
    transform = "lineal" if legend["color_scale"] == "linear" else "logarítmica"
    st.caption(f"Color de los datos · {unit} · Escala {transform} fija · P2–P98 de medianas por ubicación.")
    # Trusted numeric metadata and the same full palette used by the map.
    gradient = ",".join(f"rgb({int(red)},{int(green)},{int(blue)})"
                        for red, green, blue in legend["palette_colors"])
    background = f"linear-gradient(to right,{gradient})" if len(legend["palette_colors"]) > 1 else gradient
    ticks = legend["legend_values"]
    positions = [0, len(ticks) // 2, len(ticks) - 1] if len(ticks) > 1 else [0]
    labels = [money(ticks[index]).removeprefix("USD ") for index in positions]
    if len(labels) > 1:
        labels[0], labels[-1] = "≤ " + labels[0], "≥ " + labels[-1]
    tick_html = "".join(f"<span>{label}</span>" for label in labels)
    st.html(f'<div style="padding:4px 0 2px"><div aria-hidden="true" '
            f'style="height:16px;border-radius:4px;border:1px solid #6b7280;'
            f'background:{background}"></div>'
            f'<div style="display:flex;justify-content:space-between;gap:8px;'
            f'font-size:0.85rem;margin-top:4px">{tick_html}</div></div>')
    if len(locations):
        color_note = ("Referencia constante: todos los puntos comparten color; los valores se conservan."
                      if legend.get("color_constant") else
                      "Sólo se satura el color; todos los valores se conservan.")
        st.caption(f"Valores reales visibles: {money(legend['observed_min'])} a {money(legend['observed_max'])}"
                   f"{'/m²' if metric == 'usd_m2' else ''}. "
                   f"{legend['below_count']} ubicaciones por debajo y {legend['above_count']} por encima de la escala. "
                   + color_note)

with st.expander("Registros de una ubicación", expanded=False):
    # Keep exact comparable rows accessible even outside the base-layer filters.
    neighbor_locations = group_locations(frame.loc[frame.row_id.isin(neighbor_ids)], metric,
                                         color_scale=color_scale)
    detail_lookup = {}
    for layer_id, groups, label in (("data-locations", locations, "Datos"),
                                     ("selected-comparables", neighbor_locations, "Comparables")):
        for row in groups.to_dict("records"):
            detail_lookup[f"{layer_id}:{row['location_id']}"] = {**row, "source": label}
    options = list(detail_lookup)
    if st.session_state.get("detail_location") not in options:
        st.session_state.detail_location = None
    # The map is an enhancement; the same locations have a keyboard selector.
    selections = map_event.get("selection", {}).get("objects", {}) if map_event else {}
    picked = next((f"{layer_id}:{item.get('location_id')}" for layer_id, items in selections.items()
                   for item in items if f"{layer_id}:{item.get('location_id')}" in detail_lookup), None)
    if picked != st.session_state.get("detail_last_map_selection"):
        if picked:
            st.session_state.detail_location = picked
        st.session_state.detail_last_map_selection = picked
    def location_label(value):
        if value is None:
            return "Elegí una ubicación o seleccioná un punto del mapa"
        row = detail_lookup[value]
        return f"{row['source']} · {row['lat']:.3f}, {row['lon']:.3f} · {int(row['count'])} registros"
    selected_location = st.selectbox("Ubicación para ver sus registros", [None, *options], format_func=location_label, key="detail_location")
    if selected_location:
        ids = detail_lookup[selected_location]["row_ids"]
        rows = frame.set_index("row_id", drop=False).loc[list(ids)]
        st.dataframe(display_rows(rows), hide_index=True, width="stretch")
    else:
        st.caption("Podés buscar por coordenadas en el selector. No necesitás usar el cursor sobre el mapa.")

with st.expander("Comparación de los modelos", expanded=False):
    evaluation_filters = st.session_state.evaluation_filters
    evaluation = get_evaluation(fingerprint, evaluation_filters)
    st.write(f"Entrenamiento: **{len(train):,}** registros. Evaluación reservada: **{len(test):,}** registros.".replace(",", "."))
    st.caption("Se usan los ajustes de la última comparación válida; al iniciar, los predeterminados.")
    coverage = st.columns(2)
    coverage[0].metric("Cobertura · Comparables", f"{evaluation['comparable_coverage']:.1%}")
    coverage[1].metric("Cobertura · Regresión", f"{evaluation['linear_coverage']:.1%}")
    st.write(f"Comparación de errores sobre **{evaluation['common_rows']} casos admitidos por ambos** de los {evaluation['total_rows']} reservados.")
    if evaluation["common_rows"]:
        labels = {"comparables": "Comparables geográficos", "linear": "Regresión lineal", "training_median": "Mediana del entrenamiento"}
        metrics = pd.DataFrame(evaluation["metrics"])
        metrics["method"] = metrics["method"].map(labels)
        st.dataframe(metrics.rename(columns={"method": "Método", "mae": "Error absoluto medio (USD/m²)", "median_ae": "Mediana del error absoluto (USD/m²)"}),
                     hide_index=True, width="stretch")
        if evaluation["common_rows"] < 30:
            st.warning("Hay menos de 30 casos comunes; la comparación tiene poca evidencia.")
    else:
        st.info("No hay casos admitidos por ambos. No se calculan errores comparativos para un conjunto vacío.")
    st.caption("Menor error es mejor dentro de este conjunto común. La cobertura indica cuántos casos puede resolver cada método.")
    st.markdown("**Rangos de entrenamiento de la regresión**")
    st.dataframe(pd.DataFrame([
        {"Variable": name, "Mínimo": fitted.lower[key], "Máximo": fitted.upper[key]}
        for key, name in (("rooms", "Ambientes"), ("area_total", "Superficie total (m²)"), ("ratio", "Proporción cubierta"))
    ]), hide_index=True, width="stretch")
    if fitted.rank_deficient or fitted.condition_number > 1e8:
        st.warning("Hay redundancia o inestabilidad en la matriz del modelo; los coeficientes no deben interpretarse como efectos independientes.")

st.caption("Los detalles de metodología, datos y privacidad están en el README del proyecto.")
