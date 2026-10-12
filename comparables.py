"""Filtered comparables followed by geographic neighbours; no learned weights."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
FEATURES = ("property_type", "area_total", "rooms", "ratio", "lat", "lon")
MIN_COMPARABLES = 3


@dataclass(frozen=True)
class Query:
    property_type: str
    area_total: float
    area_covered: float
    rooms: int
    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not isinstance(self.property_type, str) or not self.property_type.strip():
            raise ValueError("Elegí un tipo de propiedad.")
        numbers = (self.area_total, self.area_covered, self.rooms, self.lat, self.lon)
        if not all(isfinite(float(value)) for value in numbers):
            raise ValueError("Todos los valores deben ser números finitos.")
        if not 0 < self.area_total:
            raise ValueError("La superficie total debe ser mayor que 0.")
        if not 0 < self.area_covered <= self.area_total:
            raise ValueError("La superficie cubierta debe ser positiva y no superar la total.")
        if int(self.rooms) != self.rooms or not 1 <= self.rooms <= 20:
            raise ValueError("Los ambientes deben ser un entero entre 1 y 20.")
        if not -90 <= self.lat <= 90 or not -180 <= self.lon <= 180:
            raise ValueError("Las coordenadas no son válidas.")

    @property
    def ratio(self) -> float:
        return self.area_covered / self.area_total


@dataclass(frozen=True)
class Filters:
    rooms_tolerance: int = 1
    area_tolerance: float = 0.35
    ratio_tolerance: float = 0.25
    radius_km: float = 5.0
    k: int = 5

    def __post_init__(self) -> None:
        if not all(isfinite(float(value)) for value in (
            self.rooms_tolerance, self.area_tolerance, self.ratio_tolerance,
            self.radius_km, self.k,
        )):
            raise ValueError("Los filtros deben ser números finitos.")
        if int(self.rooms_tolerance) != self.rooms_tolerance or not 0 <= self.rooms_tolerance <= 5:
            raise ValueError("La tolerancia de ambientes debe estar entre 0 y 5.")
        if not 0 <= self.area_tolerance <= 1 or not 0 <= self.ratio_tolerance <= 1:
            raise ValueError("Las tolerancias de superficie y proporción deben estar entre 0 y 1.")
        if not 0 < self.radius_km <= 50:
            raise ValueError("El radio debe ser mayor que 0 y hasta 50 km.")
        if int(self.k) != self.k or not MIN_COMPARABLES <= self.k <= 5:
            raise ValueError("La cantidad máxima debe ser un entero entre 3 y 5.")


@dataclass(frozen=True)
class Estimate:
    n_within_filters: int
    n_comparables: int
    usd_m2: float | None = None
    asking_price_usd: float | None = None
    observed_min_usd_m2: float | None = None
    observed_max_usd_m2: float | None = None
    distance_min_km: float | None = None
    distance_max_km: float | None = None
    insufficient_reason: str | None = None
    # Only selected training locations, coarsened for display; no source rows.
    comparable_locations: tuple[tuple[float, float], ...] = ()
    # Internal display identities, never original provider/listing identifiers.
    comparable_row_ids: tuple[int, ...] = ()
    comparable_distances_km: tuple[float, ...] = ()

    @property
    def sufficient(self) -> bool:
        return self.usd_m2 is not None


def haversine_km(lat: float, lon: float, other_lat, other_lon) -> np.ndarray:
    """Great-circle distance in kilometres, robust to floating point roundoff."""
    lat1, lon1 = np.radians([lat, lon])
    lat2 = np.radians(np.asarray(other_lat, dtype=float))
    lon2 = np.radians(np.asarray(other_lon, dtype=float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _nonnegative_mean(values) -> float:
    """A scaled mean avoids overflow when finite positive values are summed."""
    values = np.asarray(values, dtype=float)
    scale = float(values.max())
    return 0.0 if scale == 0 else float((values / scale).mean() * scale)


def _nonnegative_median(values) -> float:
    """Avoid overflowing the average of the two central positive values."""
    values = np.sort(np.asarray(values, dtype=float))
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    lower, upper = float(values[midpoint - 1]), float(values[midpoint])
    return lower + (upper - lower) / 2


def _validate_frame(frame: pd.DataFrame) -> None:
    required = set(FEATURES) | {"usd_m2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas del modelo: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise ValueError("No hay observaciones válidas para el modelo.")
    numeric = frame[[*FEATURES[1:], "usd_m2"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("El modelo requiere números finitos.")
    if (frame["area_total"] <= 0).any() or (frame["usd_m2"] <= 0).any():
        raise ValueError("La superficie y el precio por m² deben ser positivos.")
    if not frame["ratio"].between(0, 1, inclusive="right").all():
        raise ValueError("La proporción cubierta debe ser mayor que 0 y hasta 1.")
    if not frame["rooms"].between(1, 20).all() or not (frame["rooms"] % 1 == 0).all():
        raise ValueError("Los ambientes deben ser enteros entre 1 y 20.")
    if not frame["lat"].between(-90, 90).all() or not frame["lon"].between(-180, 180).all():
        raise ValueError("Las coordenadas no son válidas.")
    if frame["property_type"].isna().any() or not frame["property_type"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Falta el tipo de propiedad.")


class ComparableModel:
    """Retain only training features and targets; never accept a query target."""

    def __init__(self, train: pd.DataFrame, filters: Filters | None = None):
        _validate_frame(train)
        self.filters = filters or Filters()
        row_ids = train["row_id"].tolist() if "row_id" in train else list(train.index)
        if not all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
                   for value in row_ids) or len(set(row_ids)) != len(row_ids):
            raise ValueError("Los identificadores internos de las filas deben ser enteros únicos.")
        self._types: dict[str, dict[str, np.ndarray]] = {}
        indexed_train = train.assign(__comparable_row_id=row_ids)
        for property_type, rows in indexed_train.groupby("property_type", sort=True):
            self._types[property_type] = {
                column: rows[column].to_numpy(dtype=float, copy=True)
                for column in (*FEATURES[1:], "usd_m2")
            }
            self._types[property_type]["row_id"] = rows["__comparable_row_id"].to_numpy(copy=True)

    def predict(self, query: Query) -> Estimate:
        values = self._types.get(query.property_type)
        if values is None:
            return Estimate(0, 0)
        f = self.filters
        keep = (
            (np.abs(values["rooms"] - query.rooms) <= f.rooms_tolerance)
            & (np.abs(values["area_total"] - query.area_total) <= query.area_total * f.area_tolerance + 1e-10)
            & (np.abs(values["ratio"] - query.ratio) <= f.ratio_tolerance + 1e-10)
        )
        positions = np.flatnonzero(keep)
        distances = haversine_km(query.lat, query.lon, values["lat"][positions], values["lon"][positions])
        in_radius = distances <= f.radius_km + 1e-10
        positions, distances = positions[in_radius], distances[in_radius]
        count = int(len(positions))
        # Stable distance order gives a reproducible choice at equal distances.
        selected = np.argsort(distances, kind="stable")[:f.k]
        selected_positions = positions[selected]
        locations = tuple(
            (round(float(values["lat"][position]), 3), round(float(values["lon"][position]), 3))
            for position in selected_positions
        )
        selected_ids = tuple(int(values["row_id"][position]) for position in selected_positions)
        selected_distances = distances[selected]
        display = {
            "comparable_locations": locations,
            "comparable_row_ids": selected_ids,
            "comparable_distances_km": tuple(float(value) for value in selected_distances),
        }
        if len(selected) < MIN_COMPARABLES:
            return Estimate(count, int(len(selected)), **display)
        targets = values["usd_m2"][selected_positions]
        prediction = _nonnegative_mean(targets)
        total = prediction * float(query.area_total)
        if not isfinite(total) or total <= 0:
            return Estimate(count, int(len(selected)), insufficient_reason=
                            "No se puede calcular una referencia finita y positiva con estos valores extremos. Revisá las unidades y los datos.",
                            **display)
        return Estimate(
            n_within_filters=count, n_comparables=int(len(selected)),
            usd_m2=prediction, asking_price_usd=total,
            observed_min_usd_m2=float(targets.min()), observed_max_usd_m2=float(targets.max()),
            distance_min_km=float(selected_distances.min()), distance_max_km=float(selected_distances.max()),
            **display,
        )


def evaluate(train: pd.DataFrame, test: pd.DataFrame, filters: Filters | None = None,
             max_samples: int = 300, seed: int = 42) -> dict:
    """Held-out errors in USD/m²; baseline uses the identical covered subset.

    A caller must split before calling. Overlapping supplied spatial groups are
    rejected. Sampling is uniform without replacement and independent of price.
    """
    if not isinstance(max_samples, int) or not 1 <= max_samples <= 10_000:
        raise ValueError("La cantidad de casos de evaluación debe ser un entero positivo.")
    _validate_frame(test)
    if "spatial_group" in train and "spatial_group" in test:
        if set(train["spatial_group"]) & set(test["spatial_group"]):
            raise ValueError("Entrenamiento y evaluación comparten grupos espaciales.")
    model = ComparableModel(train, filters)
    sample = test.sample(n=min(max_samples, len(test)), random_state=seed)
    baseline = _nonnegative_median(train["usd_m2"])
    actual, predictions = [], []
    for row in sample.itertuples(index=False):
        query = Query(row.property_type, row.area_total, row.area_total * row.ratio,
                      row.rooms, row.lat, row.lon)
        prediction = model.predict(query)
        if prediction.sufficient:
            actual.append(float(row.usd_m2))
            predictions.append(prediction.usd_m2)
    result = {
        "holdout_rows": len(test), "sample_rows": len(sample), "covered_rows": len(actual),
        "coverage": len(actual) / len(sample), "seed": seed,
        "baseline_train_median_usd_m2": baseline, "target": "USD/m²",
        "mae": None, "median_absolute_error": None,
        "baseline_mae": None, "baseline_median_absolute_error": None,
    }
    if actual:
        errors = np.abs(np.asarray(predictions) - np.asarray(actual))
        baseline_errors = np.abs(baseline - np.asarray(actual))
        result.update(
            mae=_nonnegative_mean(errors), median_absolute_error=_nonnegative_median(errors),
            baseline_mae=_nonnegative_mean(baseline_errors),
            baseline_median_absolute_error=_nonnegative_median(baseline_errors),
        )
    return result
